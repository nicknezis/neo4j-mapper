"""Factory for creating appropriate data readers."""

from typing import Dict, Any, List, Union
from .sqlite_reader import SQLiteReader
from .csv_reader import CSVReader


class DataReaderFactory:
    """Factory for creating data readers based on configuration."""

    @staticmethod
    def create_reader(
        config: Dict[str, Any], chunk_size: int = 10000
    ) -> Union[SQLiteReader, CSVReader, "MixedDataReader"]:
        """Create appropriate reader based on configuration."""
        databases = config.get("databases", [])
        csv_sources = config.get("csv_sources", [])

        has_databases = len(databases) > 0
        has_csv_sources = len(csv_sources) > 0

        if has_databases and has_csv_sources:
            # Mixed data sources - use MixedDataReader
            return MixedDataReader(databases, csv_sources)
        elif has_databases:
            # SQLite only
            return SQLiteReader(databases, chunk_size)
        elif has_csv_sources:
            # CSV only
            return CSVReader(csv_sources)
        else:
            raise ValueError("No data sources found in configuration")


class MixedDataReader:
    """Reader that handles both SQLite databases and CSV sources."""

    def __init__(
        self, databases: List[Dict[str, str]], csv_sources: List[Dict[str, Any]]
    ):
        """Initialize with both SQLite and CSV configurations."""
        self.sqlite_reader = SQLiteReader(databases) if databases else None
        self.csv_reader = CSVReader(csv_sources) if csv_sources else None

        # Create a mapping of all aliases to their reader type
        self.alias_to_reader = {}

        if self.sqlite_reader:
            for db in databases:
                self.alias_to_reader[db["alias"]] = "sqlite"

        if self.csv_reader:
            for csv_source in csv_sources:
                self.alias_to_reader[csv_source["alias"]] = "csv"

    def connect_databases(self):
        """Connect to all data sources."""
        if self.sqlite_reader:
            self.sqlite_reader.connect_databases()
        if self.csv_reader:
            self.csv_reader.connect_databases()

    def close_connections(self):
        """Close all connections."""
        if self.sqlite_reader:
            self.sqlite_reader.close_connections()
        if self.csv_reader:
            self.csv_reader.close_connections()

    def get_table_info(
        self, alias: str, table_name: str = None
    ) -> List[Dict[str, Any]]:
        """Get table info from the appropriate reader."""
        reader_type = self.alias_to_reader.get(alias)

        if reader_type == "sqlite":
            return self.sqlite_reader.get_table_info(alias, table_name)
        elif reader_type == "csv":
            return self.csv_reader.get_table_info(alias, table_name)
        else:
            raise ValueError(f"Unknown alias: {alias}")

    def read_table(
        self,
        alias: str,
        table_name: str = None,
        columns: List[str] = None,
        where_clause: str = None,
    ):
        """Read table from the appropriate reader."""
        reader_type = self.alias_to_reader.get(alias)

        if reader_type == "sqlite":
            return self.sqlite_reader.read_table(
                alias, table_name, columns, where_clause
            )
        elif reader_type == "csv":
            return self.csv_reader.read_table(alias, table_name, columns, where_clause)
        else:
            raise ValueError(f"Unknown alias: {alias}")

    def execute_join_query(
        self, joins: List[Dict[str, Any]], select_columns: List[str] = None
    ):
        """Execute join query across different data sources."""
        # Determine which readers are involved in the joins
        involved_readers = set()
        for join in joins:
            left_alias = join["left_table"].split(".")[0]
            right_alias = join["right_table"].split(".")[0]

            left_reader = self.alias_to_reader.get(left_alias)
            right_reader = self.alias_to_reader.get(right_alias)

            involved_readers.add(left_reader)
            involved_readers.add(right_reader)

        # If all joins are within the same reader type, delegate to that reader
        if len(involved_readers) == 1:
            reader_type = next(iter(involved_readers))
            if reader_type == "sqlite":
                return self.sqlite_reader.execute_join_query(joins, select_columns)
            elif reader_type == "csv":
                return self.csv_reader.execute_join_query(joins, select_columns)

        # Mixed reader types - need to handle cross-source joins
        return self._execute_mixed_join_query(joins, select_columns)

    def _execute_mixed_join_query(
        self, joins: List[Dict[str, Any]], select_columns: List[str] = None
    ):
        """Execute joins across different data source types."""
        # This is a complex operation that requires:
        # 1. Loading all required tables into memory as DataFrames
        # 2. Performing pandas-based joins

        import pandas as pd

        # Get all unique table references
        table_refs = set()
        for join in joins:
            table_refs.add(join["left_table"])
            table_refs.add(join["right_table"])

        # Load all required tables
        tables = {}
        for table_ref in table_refs:
            alias, table_name = table_ref.split(".", 1)
            df = self.read_table(alias, table_name)
            # Add prefix to avoid column conflicts
            df = df.add_prefix(f"{alias}_")
            tables[table_ref] = df

        # Execute joins using pandas
        # Start with the first join
        first_join = joins[0]
        left_df = tables[first_join["left_table"]]
        right_df = tables[first_join["right_table"]]

        # Parse join condition
        left_col, right_col = self._parse_mixed_join_condition(
            first_join["on"],
            first_join["left_table"].split(".")[0],
            first_join["right_table"].split(".")[0],
        )

        # Convert SQL join type to pandas
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
            right_df = tables[join["right_table"]]

            # This is simplified - a full implementation would need more sophisticated handling
            left_col, right_col = self._parse_mixed_join_condition(
                join["on"],
                result_df.columns.tolist(),
                join["right_table"].split(".")[0],
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

        return result_df

    def _parse_mixed_join_condition(self, condition: str, left_ref, right_ref):
        """Parse join condition for mixed data sources."""
        if "=" not in condition:
            raise ValueError(f"Only equality joins are supported: {condition}")

        left_part, right_part = condition.split("=", 1)
        left_part = left_part.strip()
        right_part = right_part.strip()

        # Resolve column references
        if isinstance(left_ref, list):
            # left_ref is a column list from existing DataFrame
            left_col = self._find_column_in_list(left_part, left_ref)
        else:
            # left_ref is an alias
            left_col = f"{left_ref}_{left_part.split('.')[-1]}"

        right_col = f"{right_ref}_{right_part.split('.')[-1]}"

        return left_col, right_col

    def _find_column_in_list(self, col_ref: str, column_list: List[str]) -> str:
        """Find matching column in the list."""
        if "." in col_ref:
            # Extract just the column name part
            col_name = col_ref.split(".")[-1]
            # Find column with this suffix
            matches = [col for col in column_list if col.endswith(f"_{col_name}")]
            if matches:
                return matches[0]

        return col_ref

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

    def get_row_count(
        self, alias: str, table_name: str = None, where_clause: str = None
    ) -> int:
        """Get row count from the appropriate reader."""
        reader_type = self.alias_to_reader.get(alias)

        if reader_type == "sqlite":
            return self.sqlite_reader.get_row_count(alias, table_name, where_clause)
        elif reader_type == "csv":
            return self.csv_reader.get_row_count(alias, table_name, where_clause)
        else:
            raise ValueError(f"Unknown alias: {alias}")

    def list_tables(self, alias: str) -> List[str]:
        """List tables from the appropriate reader."""
        reader_type = self.alias_to_reader.get(alias)

        if reader_type == "sqlite":
            return self.sqlite_reader.list_tables(alias)
        elif reader_type == "csv":
            return self.csv_reader.list_tables(alias)
        else:
            raise ValueError(f"Unknown alias: {alias}")

    def __enter__(self):
        """Context manager entry."""
        self.connect_databases()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connections()
