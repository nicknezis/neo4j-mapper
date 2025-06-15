"""SQLite database reader with JOIN support and chunked processing."""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator, Tuple, NamedTuple
import logging
import re
from dataclasses import dataclass


@dataclass
class QueryPlanStep:
    """Represents a step in the SQLite query execution plan."""
    id: int
    parent: int
    notused: int
    detail: str
    
    
@dataclass 
class QueryPlanAnalysis:
    """Analysis results for a query execution plan."""
    steps: List[QueryPlanStep]
    has_table_scan: bool
    table_scans: List[str]
    has_nested_loop: bool
    uses_index: bool
    indexed_tables: List[str]
    warnings: List[str]
    estimated_cost: str
    complexity_score: int


class SQLiteReader:
    """Reads data from SQLite databases with support for complex joins and chunked processing."""

    def __init__(self, databases: List[Dict[str, str]], chunk_size: int = 10000):
        """Initialize with database configurations.

        Args:
            databases: List of database configurations
            chunk_size: Default chunk size for streaming queries
        """
        self.databases = {}
        self.connections = {}
        self.chunk_size = chunk_size

        for db_config in databases:
            alias = db_config["alias"]
            path = db_config["path"]

            if not Path(path).exists():
                raise FileNotFoundError(f"Database file not found: {path}")

            self.databases[alias] = path

        logging.info(f"Initialized SQLiteReader with {len(databases)} databases")

    def connect_databases(self):
        """Establish connections to all databases."""
        for alias, path in self.databases.items():
            try:
                conn = sqlite3.connect(path)
                conn.row_factory = sqlite3.Row  # Enable column access by name
                self.connections[alias] = conn
                logging.info(f"Connected to database: {alias} ({path})")
            except sqlite3.Error as e:
                raise RuntimeError(f"Failed to connect to {alias}: {e}")

    def close_connections(self):
        """Close all database connections."""
        for alias, conn in self.connections.items():
            try:
                conn.close()
                logging.info(f"Closed connection to {alias}")
            except sqlite3.Error as e:
                logging.warning(f"Error closing connection to {alias}: {e}")

        self.connections.clear()

    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        if not self.connections:
            raise RuntimeError("No database connections available")

        # Determine which connection to use
        if len(self.connections) == 1:
            # Single database query
            conn = next(iter(self.connections.values()))
        else:
            # For multi-database queries, we'll need to attach databases
            # Use the first connection as primary
            conn = next(iter(self.connections.values()))

        try:
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)

            logging.info(f"Query executed successfully, returned {len(df)} rows")
            return df

        except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
            raise RuntimeError(f"Query execution failed: {e}")

    def execute_query_chunked(
        self,
        query: str,
        chunk_size: Optional[int] = None,
        params: Optional[Dict] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """Execute query and return results in chunks for memory-efficient processing.

        Args:
            query: SQL query to execute
            chunk_size: Size of each chunk (uses default if None)
            params: Query parameters

        Yields:
            DataFrame chunks containing query results
        """
        chunk_size = chunk_size or self.chunk_size

        # Determine which connection to use
        if len(self.connections) == 1:
            conn = next(iter(self.connections.values()))
        else:
            # For multi-database queries, use the first connection as primary
            conn = next(iter(self.connections.values()))

        try:
            logging.info(f"Executing chunked query with chunk_size={chunk_size}")

            if params:
                chunk_iter = pd.read_sql_query(
                    query, conn, params=params, chunksize=chunk_size
                )
            else:
                chunk_iter = pd.read_sql_query(query, conn, chunksize=chunk_size)

            total_rows = 0
            chunk_count = 0

            for chunk in chunk_iter:
                chunk_count += 1
                total_rows += len(chunk)
                logging.debug(f"Processing chunk {chunk_count} with {len(chunk)} rows")
                yield chunk

            logging.info(
                f"Chunked query completed: {total_rows} total rows in {chunk_count} chunks"
            )

        except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
            raise RuntimeError(f"Chunked query execution failed: {e}")

    def read_table(
        self,
        alias: str,
        table_name: str,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read data from a specific table."""
        if alias not in self.connections:
            raise ValueError(f"Database alias '{alias}' not found")

        conn = self.connections[alias]

        # Build query
        if columns:
            column_list = ", ".join(columns)
        else:
            column_list = "*"

        query = f"SELECT {column_list} FROM {table_name}"

        if where_clause:
            query += f" WHERE {where_clause}"

        try:
            df = pd.read_sql_query(query, conn)
            logging.info(f"Read {len(df)} rows from {alias}.{table_name}")
            return df

        except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
            raise RuntimeError(f"Failed to read from {alias}.{table_name}: {e}")

    def read_table_chunked(
        self,
        alias: str,
        table_name: str,
        chunk_size: Optional[int] = None,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """Read table data in chunks for memory-efficient processing.

        Args:
            alias: Database alias
            table_name: Name of the table to read
            chunk_size: Size of each chunk (uses default if None)
            columns: Optional list of columns to select
            where_clause: Optional WHERE clause to filter data

        Yields:
            DataFrame chunks containing table data
        """
        if alias not in self.connections:
            raise ValueError(f"Database '{alias}' not connected")

        # Build query
        if columns:
            column_list = ", ".join(columns)
        else:
            column_list = "*"

        query = f"SELECT {column_list} FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"

        yield from self.execute_query_chunked(query, chunk_size)

    def get_table_row_count(self, alias: str, table_name: str) -> int:
        """Get the number of rows in a table for memory planning.

        Args:
            alias: Database alias
            table_name: Name of the table

        Returns:
            Number of rows in the table
        """
        if alias not in self.connections:
            raise ValueError(f"Database '{alias}' not connected")

        query = f"SELECT COUNT(*) as row_count FROM {table_name}"
        result = self.execute_query(query)
        return result.iloc[0]["row_count"]

    def should_use_chunking(
        self, alias: str, table_name: str, memory_threshold: int = 100000
    ) -> bool:
        """Determine if chunking should be used based on table size.

        Args:
            alias: Database alias
            table_name: Name of the table
            memory_threshold: Row count threshold for using chunking

        Returns:
            True if chunking should be used
        """
        try:
            row_count = self.get_table_row_count(alias, table_name)
            return row_count > memory_threshold
        except Exception as e:
            logging.warning(f"Could not determine table size for {table_name}: {e}")
            return False  # Default to non-chunked for safety

    def get_optimal_chunk_size(
        self, estimated_row_count: int, available_memory_mb: int = 512
    ) -> int:
        """Calculate optimal chunk size based on estimated data size and available memory.

        Args:
            estimated_row_count: Estimated number of rows in the result
            available_memory_mb: Available memory in MB

        Returns:
            Optimal chunk size
        """
        # Rough estimate: 1KB per row on average
        estimated_size_mb = estimated_row_count / 1024

        if estimated_size_mb <= available_memory_mb:
            # Data fits in memory, no chunking needed
            return estimated_row_count

        # Calculate chunk size to use roughly half of available memory
        target_chunk_size_mb = available_memory_mb / 2
        chunk_size = int((target_chunk_size_mb * 1024))

        # Ensure minimum and maximum chunk sizes
        chunk_size = max(1000, min(chunk_size, 100000))

        logging.info(
            f"Calculated optimal chunk size: {chunk_size} "
            f"(estimated {estimated_row_count} rows, {estimated_size_mb:.1f}MB)"
        )

        return chunk_size

    def analyze_query_plan(self, query: str, params: Optional[Dict] = None) -> QueryPlanAnalysis:
        """Analyze the execution plan for a query to identify potential performance issues.
        
        Args:
            query: SQL query to analyze
            params: Query parameters
            
        Returns:
            QueryPlanAnalysis with performance warnings and recommendations
        """
        if not self.connections:
            raise RuntimeError("No database connections available")
            
        # Use the first connection for plan analysis
        conn = next(iter(self.connections.values()))
        
        try:
            cursor = conn.cursor()
            
            # Get the query execution plan
            explain_query = f"EXPLAIN QUERY PLAN {query}"
            
            if params:
                cursor.execute(explain_query, params)
            else:
                cursor.execute(explain_query)
                
            plan_rows = cursor.fetchall()
            
            # Parse plan steps
            steps = []
            for row in plan_rows:
                steps.append(QueryPlanStep(
                    id=row[0], 
                    parent=row[1], 
                    notused=row[2], 
                    detail=row[3]
                ))
            
            # Analyze the plan
            analysis = self._analyze_plan_steps(steps)
            
            # Get table sizes for additional context
            table_sizes = self._get_table_sizes()
            analysis = self._add_table_size_warnings(analysis, table_sizes)
            
            logging.info(f"Query plan analyzed: {len(analysis.warnings)} warnings found")
            
            return analysis
            
        except sqlite3.Error as e:
            logging.error(f"Failed to analyze query plan: {e}")
            # Return basic analysis if explain fails
            return QueryPlanAnalysis(
                steps=[],
                has_table_scan=False,
                table_scans=[],
                has_nested_loop=False,
                uses_index=False,
                indexed_tables=[],
                warnings=[f"Could not analyze query plan: {e}"],
                estimated_cost="Unknown",
                complexity_score=0
            )

    def _analyze_plan_steps(self, steps: List[QueryPlanStep]) -> QueryPlanAnalysis:
        """Analyze query plan steps to identify performance issues."""
        table_scans = []
        indexed_tables = []
        warnings = []
        has_nested_loop = False
        complexity_score = 0
        
        for step in steps:
            detail = step.detail.upper()
            
            # Check for table scans
            if "SCAN TABLE" in detail:
                # Extract table name
                table_match = re.search(r'SCAN TABLE (\w+)', detail)
                if table_match:
                    table_name = table_match.group(1)
                    table_scans.append(table_name)
                    complexity_score += 10
                    
                    # Check if using index
                    if "USING INDEX" not in detail:
                        warnings.append(
                            f"Table scan detected on '{table_name}' without index usage. "
                            f"Consider adding indexes on JOIN/WHERE columns."
                        )
                    else:
                        # Extract index name
                        index_match = re.search(r'USING INDEX (\w+)', detail)
                        if index_match:
                            indexed_tables.append(f"{table_name}({index_match.group(1)})")
            
            # Check for index searches
            elif "SEARCH TABLE" in detail and "USING INDEX" in detail:
                table_match = re.search(r'SEARCH TABLE (\w+)', detail)
                index_match = re.search(r'USING INDEX (\w+)', detail)
                if table_match and index_match:
                    table_name = table_match.group(1)
                    index_name = index_match.group(1)
                    indexed_tables.append(f"{table_name}({index_name})")
                    complexity_score += 2
                    
            # Check for automatic index creation
            elif "AUTOMATIC COVERING INDEX" in detail or "AUTOMATIC INDEX" in detail:
                warnings.append(
                    "SQLite created an automatic index. Consider creating permanent indexes "
                    "for better performance on repeated queries."
                )
                complexity_score += 5
                
            # Check for nested loops (multiple scans)
            elif "NESTED LOOP" in detail:
                has_nested_loop = True
                complexity_score += 5
                
            # Check for sorting operations
            elif "ORDER BY" in detail or "SORT" in detail:
                warnings.append(
                    "Query involves sorting. Consider adding indexes on ORDER BY columns "
                    "for better performance."
                )
                complexity_score += 3
                
            # Check for temporary B-trees
            elif "TEMP B-TREE" in detail:
                warnings.append(
                    "Query uses temporary B-tree structures. This may indicate missing indexes "
                    "or complex operations that could be optimized."
                )
                complexity_score += 4

        # Determine overall cost estimation
        if complexity_score == 0:
            estimated_cost = "Very Low"
        elif complexity_score <= 5:
            estimated_cost = "Low"
        elif complexity_score <= 15:
            estimated_cost = "Medium"
        elif complexity_score <= 30:
            estimated_cost = "High"
        else:
            estimated_cost = "Very High"

        return QueryPlanAnalysis(
            steps=steps,
            has_table_scan=len(table_scans) > 0,
            table_scans=table_scans,
            has_nested_loop=has_nested_loop,
            uses_index=len(indexed_tables) > 0,
            indexed_tables=indexed_tables,
            warnings=warnings,
            estimated_cost=estimated_cost,
            complexity_score=complexity_score
        )

    def _get_table_sizes(self) -> Dict[str, int]:
        """Get row counts for all tables to help with performance analysis."""
        table_sizes = {}
        
        for alias, conn in self.connections.items():
            try:
                cursor = conn.cursor()
                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                for (table_name,) in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        table_sizes[table_name] = count
                    except sqlite3.Error:
                        # Skip if table access fails
                        continue
                        
            except sqlite3.Error as e:
                logging.warning(f"Could not get table sizes for {alias}: {e}")
                
        return table_sizes

    def _add_table_size_warnings(self, analysis: QueryPlanAnalysis, table_sizes: Dict[str, int]) -> QueryPlanAnalysis:
        """Add warnings based on table sizes and query plan."""
        large_table_threshold = 100000  # Rows
        medium_table_threshold = 10000   # Rows
        
        for table_name in analysis.table_scans:
            table_size = table_sizes.get(table_name, 0)
            
            if table_size > large_table_threshold:
                analysis.warnings.append(
                    f"⚠️  PERFORMANCE WARNING: Table scan on large table '{table_name}' "
                    f"({table_size:,} rows). This query may be very slow. "
                    f"Consider adding indexes on JOIN/WHERE conditions."
                )
                analysis.complexity_score += 20
                
            elif table_size > medium_table_threshold:
                analysis.warnings.append(
                    f"⚠️  Table scan on medium table '{table_name}' ({table_size:,} rows). "
                    f"Consider optimizing with indexes for better performance."
                )
                analysis.complexity_score += 10
                
        # Update estimated cost based on table sizes
        if analysis.complexity_score > 50:
            analysis.estimated_cost = "Very High"
        elif analysis.complexity_score > 30:
            analysis.estimated_cost = "High"
            
        return analysis

    def analyze_join_performance(self, joins: List[Dict[str, Any]], select_columns: Optional[List[str]] = None) -> QueryPlanAnalysis:
        """Analyze JOIN query performance before execution.
        
        Args:
            joins: List of JOIN configurations
            select_columns: Optional list of columns to select
            
        Returns:
            QueryPlanAnalysis with performance warnings
        """
        # Build the JOIN query (same logic as execute_join_query)
        if select_columns:
            select_clause = ", ".join(select_columns)
        else:
            select_clause = "*"

        # Start with the first join to establish the base query
        first_join = joins[0]
        left_table = self._parse_table_reference(first_join["left_table"])
        right_table = self._parse_table_reference(first_join["right_table"])

        query = f"""
        SELECT {select_clause}
        FROM {left_table['table']} as {left_table['alias']}
        {first_join['type']} JOIN {right_table['table']} as {right_table['alias']}
        ON {first_join['condition']}
        """

        # Add additional joins
        for join in joins[1:]:
            right_table = self._parse_table_reference(join["right_table"])
            query += f"""
            {join['type']} JOIN {right_table['table']} as {right_table['alias']}
            ON {join['condition']}
            """

        # Analyze the complete query
        analysis = self.analyze_query_plan(query.strip())
        
        # Add JOIN-specific warnings
        if len(joins) > 3:
            analysis.warnings.append(
                f"Complex query with {len(joins)} JOINs. Consider breaking into smaller queries "
                f"or ensuring all JOIN conditions use indexed columns."
            )
            
        return analysis

    def execute_join_query(
        self, joins: List[Dict[str, Any]], select_columns: Optional[List[str]] = None,
        analyze_performance: bool = True
    ) -> pd.DataFrame:
        """Execute a query with JOIN operations.
        
        Args:
            joins: List of JOIN configurations
            select_columns: Optional list of columns to select
            analyze_performance: If True, analyze query plan before execution
        """
        if not joins:
            raise ValueError("No joins specified")

        # Analyze query performance before execution
        if analyze_performance:
            try:
                analysis = self.analyze_join_performance(joins, select_columns)
                
                # Log performance warnings
                if analysis.warnings:
                    for warning in analysis.warnings:
                        logging.warning(f"Query Performance: {warning}")
                
                # Log performance summary
                logging.info(
                    f"Query Analysis: Cost={analysis.estimated_cost}, "
                    f"TableScans={len(analysis.table_scans)}, "
                    f"UsesIndex={analysis.uses_index}, "
                    f"ComplexityScore={analysis.complexity_score}"
                )
                
                # Warn if query might be very slow
                if analysis.estimated_cost in ["High", "Very High"]:
                    logging.warning(
                        f"⚠️  Query has {analysis.estimated_cost.lower()} estimated cost. "
                        f"Consider optimizing before executing on large datasets."
                    )
                    
            except Exception as e:
                logging.warning(f"Could not analyze query performance: {e}")

        # Build JOIN query
        if select_columns:
            select_clause = ", ".join(select_columns)
        else:
            select_clause = "*"

        # Start with the first join to establish the base query
        first_join = joins[0]
        left_table = self._parse_table_reference(first_join["left_table"])
        right_table = self._parse_table_reference(first_join["right_table"])

        query = f"""
        SELECT {select_clause}
        FROM {left_table['table']} as {left_table['alias']}
        {first_join['type']} JOIN {right_table['table']} as {right_table['alias']}
        ON {first_join['condition']}
        """

        # Add additional joins
        for join in joins[1:]:
            right_table = self._parse_table_reference(join["right_table"])
            query += f"""
            {join['type']} JOIN {right_table['table']} as {right_table['alias']}
            ON {join['condition']}
            """

        # For multi-database joins, we need to attach databases
        self._attach_databases_for_join(joins)

        # Execute query and get result DataFrame
        result_df = self.execute_query(query.strip())

        # Add column prefixes to prevent collisions (similar to CSV reader approach)
        result_df = self._add_column_prefixes(result_df, joins)

        return result_df

    def _parse_table_reference(self, table_ref: str) -> Dict[str, str]:
        """Parse table reference like 'db_alias.table_name'."""
        if "." not in table_ref:
            raise ValueError(
                f"Table reference must include database alias: {table_ref}"
            )

        parts = table_ref.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid table reference format: {table_ref}")

        db_alias, table_name = parts
        return {
            "db_alias": db_alias,
            "table": table_name,
            "alias": f"{db_alias}_{table_name}",
        }

    def _attach_databases_for_join(self, joins: List[Dict[str, Any]]):
        """Attach databases for multi-database joins."""
        # Get all database aliases used in joins
        db_aliases = set()
        for join in joins:
            left_alias = join["left_table"].split(".")[0]
            right_alias = join["right_table"].split(".")[0]
            db_aliases.add(left_alias)
            db_aliases.add(right_alias)

        # Use the first database as the primary connection
        primary_alias = next(iter(db_aliases))
        primary_conn = self.connections[primary_alias]

        # Attach other databases
        for alias in db_aliases:
            if alias != primary_alias:
                db_path = self.databases[alias]
                try:
                    cursor = primary_conn.cursor()
                    cursor.execute(f"ATTACH DATABASE '{db_path}' AS {alias}")
                    logging.info(f"Attached database {alias} for join operation")
                except sqlite3.Error as e:
                    raise RuntimeError(f"Failed to attach database {alias}: {e}")

    def _add_column_prefixes(
        self, df: pd.DataFrame, joins: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Add column prefixes to prevent name collisions."""
        # Get all table references involved in joins
        table_refs = []
        for join in joins:
            table_refs.append(join["left_table"])
            table_refs.append(join["right_table"])

        # Remove duplicates while preserving order
        seen = set()
        unique_refs = []
        for ref in table_refs:
            if ref not in seen:
                seen.add(ref)
                unique_refs.append(ref)

        # Create prefixed column names
        renamed_columns = {}
        for col in df.columns:
            # For now, use a simple approach - add database alias as prefix
            # This can be enhanced based on specific requirements
            renamed_columns[col] = col

        return df.rename(columns=renamed_columns)

    def list_tables(self, alias: str) -> List[str]:
        """List all tables in a database."""
        if alias not in self.connections:
            raise ValueError(f"Database alias '{alias}' not found")

        conn = self.connections[alias]

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            return tables
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to list tables for {alias}: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.connect_databases()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connections()
