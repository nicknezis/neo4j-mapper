"""SQLite database reader with JOIN support."""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging


class SQLiteReader:
    """Reads data from SQLite databases with support for complex joins."""
    
    def __init__(self, databases: List[Dict[str, str]]):
        """Initialize with database configurations."""
        self.databases = {}
        self.connections = {}
        
        for db_config in databases:
            alias = db_config['alias']
            path = db_config['path']
            
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
                raise ConnectionError(f"Failed to connect to {alias} ({path}): {e}")
    
    def close_connections(self):
        """Close all database connections."""
        for alias, conn in self.connections.items():
            if conn:
                conn.close()
                logging.info(f"Closed connection to: {alias}")
        self.connections.clear()
    
    def get_table_info(self, alias: str, table_name: str) -> List[Dict[str, Any]]:
        """Get column information for a table."""
        if alias not in self.connections:
            raise ValueError(f"Database alias '{alias}' not found")
        
        conn = self.connections[alias]
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            return [
                {
                    'name': col['name'],
                    'type': col['type'],
                    'notnull': bool(col['notnull']),
                    'pk': bool(col['pk'])
                }
                for col in columns
            ]
        except sqlite3.Error as e:
            raise ValueError(f"Error getting table info for {alias}.{table_name}: {e}")
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        # For complex queries involving multiple databases, we need to handle this carefully
        # This is a simplified version - in practice, you might need to use ATTACH DATABASE
        
        # For now, assume single database queries
        if len(self.connections) == 1:
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
    
    def read_table(self, alias: str, table_name: str, 
                   columns: Optional[List[str]] = None,
                   where_clause: Optional[str] = None) -> pd.DataFrame:
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
    
    def execute_join_query(self, joins: List[Dict[str, Any]], 
                          select_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Execute a query with JOIN operations."""
        if not joins:
            raise ValueError("No joins specified")
        
        # Build JOIN query
        if select_columns:
            select_clause = ", ".join(select_columns)
        else:
            select_clause = "*"
        
        # Start with the first join to establish the base query
        first_join = joins[0]
        left_table = self._parse_table_reference(first_join['left_table'])
        right_table = self._parse_table_reference(first_join['right_table'])
        
        query = f"""
        SELECT {select_clause}
        FROM {left_table['table']} as {left_table['alias']}
        {first_join['type']} JOIN {right_table['table']} as {right_table['alias']}
        ON {first_join['on']}
        """
        
        # Add additional joins
        for join in joins[1:]:
            right_table = self._parse_table_reference(join['right_table'])
            query += f"""
            {join['type']} JOIN {right_table['table']} as {right_table['alias']}
            ON {join['on']}
            """
        
        # For multi-database joins, we need to attach databases
        self._attach_databases_for_join(joins)
        
        return self.execute_query(query.strip())
    
    def _parse_table_reference(self, table_ref: str) -> Dict[str, str]:
        """Parse table reference like 'db_alias.table_name'."""
        if '.' not in table_ref:
            raise ValueError(f"Table reference must include database alias: {table_ref}")
        
        parts = table_ref.split('.')
        if len(parts) != 2:
            raise ValueError(f"Invalid table reference format: {table_ref}")
        
        db_alias, table_name = parts
        return {
            'db_alias': db_alias,
            'table': table_name,
            'alias': f"{db_alias}_{table_name}"
        }
    
    def _attach_databases_for_join(self, joins: List[Dict[str, Any]]):
        """Attach databases for multi-database joins."""
        # Get all database aliases used in joins
        db_aliases = set()
        for join in joins:
            left_alias = join['left_table'].split('.')[0]
            right_alias = join['right_table'].split('.')[0]
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
    
    def get_row_count(self, alias: str, table_name: str, 
                      where_clause: Optional[str] = None) -> int:
        """Get row count for a table."""
        if alias not in self.connections:
            raise ValueError(f"Database alias '{alias}' not found")
        
        conn = self.connections[alias]
        query = f"SELECT COUNT(*) FROM {table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            count = cursor.fetchone()[0]
            return count
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to get row count for {alias}.{table_name}: {e}")
    
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