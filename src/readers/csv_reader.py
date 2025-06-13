"""CSV file reader with JOIN support."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging


class CSVReader:
    """Reads data from CSV files with support for complex joins."""

    def __init__(self, csv_sources: List[Dict[str, Any]]):
        """Initialize with CSV source configurations."""
        self.csv_sources = {}
        self.dataframes = {}

        for csv_config in csv_sources:
            alias = csv_config["alias"]
            path = csv_config["path"]
            options = csv_config.get("options", {})

            if not Path(path).exists():
                raise FileNotFoundError(f"CSV file not found: {path}")

            self.csv_sources[alias] = {"path": path, "options": options}

        logging.info(f"Initialized CSVReader with {len(csv_sources)} CSV sources")

    def connect_databases(self):
        """Load all CSV files into DataFrames."""
        for alias, config in self.csv_sources.items():
            try:
                # Set default CSV options
                csv_options = {
                    "delimiter": ",",
                    "encoding": "utf-8",
                    "header": 0,
                    **config["options"],
                }

                # Load CSV file
                df = pd.read_csv(config["path"], **csv_options)
                self.dataframes[alias] = df

                logging.info(
                    f"Loaded CSV: {alias} ({config['path']}) - {len(df)} rows, {len(df.columns)} columns"
                )

            except Exception as e:
                raise ConnectionError(
                    f"Failed to load CSV {alias} ({config['path']}): {e}"
                )

    def close_connections(self):
        """Clear all loaded DataFrames."""
        for alias in list(self.dataframes.keys()):
            del self.dataframes[alias]
            logging.info(f"Cleared DataFrame for: {alias}")
        self.dataframes.clear()

    def get_table_info(
        self, alias: str, table_name: str = None
    ) -> List[Dict[str, Any]]:
        """Get column information for a CSV source (table_name is ignored for CSV)."""
        if alias not in self.dataframes:
            raise ValueError(f"CSV source alias '{alias}' not found")

        df = self.dataframes[alias]

        columns = []
        for col_name in df.columns:
            dtype = df[col_name].dtype

            # Map pandas dtypes to SQL-like types
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = "INTEGER"
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = "FLOAT"
            elif pd.api.types.is_bool_dtype(dtype):
                sql_type = "BOOLEAN"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                sql_type = "DATETIME"
            else:
                sql_type = "TEXT"

            columns.append(
                {
                    "name": col_name,
                    "type": sql_type,
                    "notnull": not df[col_name].isnull().any(),
                    "pk": False,  # CSV files don't have explicit primary keys
                }
            )

        return columns

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """Execute a pandas query on loaded DataFrames."""
        # For CSV, we'll use pandas query functionality
        # This is a simplified implementation - complex SQL queries would need more sophisticated parsing
        raise NotImplementedError(
            "Direct SQL query execution not supported for CSV sources. Use join operations instead."
        )

    def read_table(
        self,
        alias: str,
        table_name: str = None,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read data from a CSV source (table_name is ignored for CSV)."""
        if alias not in self.dataframes:
            raise ValueError(f"CSV source alias '{alias}' not found")

        df = self.dataframes[alias].copy()

        # Apply column selection
        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in {alias}: {missing_cols}")
            df = df[columns]

        # Apply basic where clause using pandas query
        if where_clause:
            try:
                # Convert basic SQL WHERE syntax to pandas query syntax
                pandas_query = self._convert_where_to_pandas_query(where_clause)
                df = df.query(pandas_query)
            except Exception as e:
                logging.warning(f"Could not apply where clause '{where_clause}': {e}")

        logging.info(f"Read {len(df)} rows from CSV source: {alias}")
        return df

    def execute_join_query(
        self, joins: List[Dict[str, Any]], select_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Execute JOIN operations using pandas merge."""
        if not joins:
            raise ValueError("No joins specified")

        # Start with the first join
        first_join = joins[0]
        left_table_ref = self._parse_table_reference(first_join["left_table"])
        right_table_ref = self._parse_table_reference(first_join["right_table"])

        # Get the base DataFrames
        left_df = self.dataframes[left_table_ref["db_alias"]].copy()
        right_df = self.dataframes[right_table_ref["db_alias"]].copy()

        # Add prefixes to avoid column conflicts
        left_df = left_df.add_prefix(f"{left_table_ref['alias']}_")
        right_df = right_df.add_prefix(f"{right_table_ref['alias']}_")

        # Parse the join condition
        left_col, right_col = self._parse_join_condition(
            first_join["on"], left_table_ref["alias"], right_table_ref["alias"]
        )

        # Perform the join
        how = self._convert_join_type(first_join["type"])
        result_df = pd.merge(
            left_df,
            right_df,
            left_on=left_col,
            right_on=right_col,
            how=how,
            suffixes=("", "_right"),
        )

        # Process additional joins
        for join in joins[1:]:
            right_table_ref = self._parse_table_reference(join["right_table"])
            right_df = self.dataframes[right_table_ref["db_alias"]].copy()
            right_df = right_df.add_prefix(f"{right_table_ref['alias']}_")

            left_col, right_col = self._parse_join_condition(
                join["on"],
                result_df.columns.tolist(),  # Use existing columns as left reference
                right_table_ref["alias"],
            )

            how = self._convert_join_type(join["type"])
            result_df = pd.merge(
                result_df,
                right_df,
                left_on=left_col,
                right_on=right_col,
                how=how,
                suffixes=("", "_right"),
            )

        # Apply column selection if specified
        if select_columns:
            available_cols = [col for col in select_columns if col in result_df.columns]
            if available_cols:
                result_df = result_df[available_cols]

        logging.info(f"Join operation completed, returned {len(result_df)} rows")
        return result_df

    def _parse_table_reference(self, table_ref: str) -> Dict[str, str]:
        """Parse table reference like 'csv_alias.table_name' (table_name ignored for CSV)."""
        if "." not in table_ref:
            raise ValueError(f"Table reference must include CSV alias: {table_ref}")

        parts = table_ref.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid table reference format: {table_ref}")

        csv_alias, table_name = parts
        return {
            "db_alias": csv_alias,
            "table": table_name,  # This is ignored for CSV but kept for compatibility
            "alias": csv_alias,
        }

    def _parse_join_condition(
        self, condition: str, left_ref: str, right_ref: str
    ) -> Tuple[str, str]:
        """Parse JOIN condition like 'table1.col1 = table2.col2'."""
        # Simple parsing for basic equality joins
        if "=" not in condition:
            raise ValueError(f"Only equality joins are supported: {condition}")

        left_part, right_part = condition.split("=", 1)
        left_part = left_part.strip()
        right_part = right_part.strip()

        # Convert to column names with prefixes
        left_col = self._resolve_column_reference(left_part, left_ref)
        right_col = self._resolve_column_reference(right_part, right_ref)

        return left_col, right_col

    def _resolve_column_reference(self, col_ref: str, context_alias: str) -> str:
        """Resolve column reference to actual DataFrame column name."""
        if "." in col_ref:
            # Extract table.column format
            parts = col_ref.split(".")
            if len(parts) == 2:
                table_alias, col_name = parts
                return f"{table_alias}_{col_name}"

        # If no table prefix, assume it belongs to the context
        return f"{context_alias}_{col_ref}"

    def _convert_join_type(self, join_type: str) -> str:
        """Convert SQL JOIN type to pandas merge 'how' parameter."""
        join_mapping = {
            "INNER": "inner",
            "LEFT": "left",
            "RIGHT": "right",
            "OUTER": "outer",
            "FULL": "outer",
        }

        return join_mapping.get(join_type.upper(), "inner")

    def _convert_where_to_pandas_query(self, where_clause: str) -> str:
        """Convert basic SQL WHERE clause to pandas query syntax."""
        # This is a very basic implementation
        # A full implementation would need proper SQL parsing
        query = where_clause

        # Basic conversions
        query = query.replace(" AND ", " and ")
        query = query.replace(" OR ", " or ")
        query = query.replace(" = ", " == ")

        return query

    def get_row_count(
        self, alias: str, table_name: str = None, where_clause: Optional[str] = None
    ) -> int:
        """Get row count for a CSV source."""
        if alias not in self.dataframes:
            raise ValueError(f"CSV source alias '{alias}' not found")

        df = self.dataframes[alias]

        if where_clause:
            try:
                pandas_query = self._convert_where_to_pandas_query(where_clause)
                filtered_df = df.query(pandas_query)
                return len(filtered_df)
            except Exception as e:
                logging.warning(f"Could not apply where clause '{where_clause}': {e}")

        return len(df)

    def list_tables(self, alias: str) -> List[str]:
        """List tables for CSV source (returns the alias as single 'table')."""
        if alias not in self.dataframes:
            raise ValueError(f"CSV source alias '{alias}' not found")

        # For CSV, we return the alias itself as the "table" name
        return [alias]

    def __enter__(self):
        """Context manager entry."""
        self.connect_databases()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connections()
