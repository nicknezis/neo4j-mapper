"""Data mapping utilities for Neo4j transformation."""

import pandas as pd
from typing import Dict, Any, List, Optional, Union

# datetime imports used in TYPE_CONVERTERS lambdas
import logging
import re


class DataMapper:
    """Handles mapping of DataFrame data to graph node and relationship properties."""

    # Type conversion mapping
    TYPE_CONVERTERS = {
        "string": str,
        "integer": int,
        "float": float,
        "boolean": bool,
        "datetime": pd.to_datetime,
        "date": lambda x: pd.to_datetime(x).dt.date,
        "time": lambda x: pd.to_datetime(x).dt.time,
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._regex_cache = {}
        
        # Vectorized type conversion methods
        self.VECTORIZED_CONVERTERS = {
            "string": self._convert_to_string_vectorized,
            "integer": self._convert_to_integer_vectorized,
            "float": self._convert_to_float_vectorized,
            "boolean": self._convert_to_boolean_vectorized,
            "datetime": self._convert_to_datetime_vectorized,
            "date": self._convert_to_date_vectorized,
            "time": self._convert_to_time_vectorized,
        }

    def map_node_properties(
        self, df: pd.DataFrame, node_config: Dict[str, Any], inplace: bool = False
    ) -> pd.DataFrame:
        """Map DataFrame columns to node properties according to configuration.
        
        Args:
            df: Input DataFrame
            node_config: Node configuration
            inplace: If True, modify df in-place to save memory (default: False)
        """
        if inplace:
            result_df = df
        else:
            result_df = df.copy()

        # Get source table/alias information
        source = node_config.get("source", "")

        # Extract and transform properties
        property_mappings = {}
        for prop_config in node_config["properties"]:
            field_name = prop_config["field"]
            field_type = prop_config["type"]

            # Resolve column name - use source from node config for better resolution
            actual_column = self._resolve_column_name(df, field_name, source)

            if actual_column is None:
                self.logger.warning(
                    f"Column '{field_name}' not found in DataFrame for node {node_config['label']}"
                )
                continue

            # Update field_name to the resolved column name
            field_name = actual_column

            # Apply regex extractor if specified
            if (
                "extractor" in prop_config
                and prop_config["extractor"].get("type") == "regex"
            ):
                try:
                    extracted_data = self._apply_regex_extractor(
                        result_df[field_name], prop_config["extractor"], field_name
                    )

                    # Handle multiple extractions (groups)
                    if isinstance(extracted_data, dict):
                        for ext_field, ext_values in extracted_data.items():
                            result_df[ext_field] = ext_values
                            # Apply type conversion to extracted fields
                            if field_type in self.TYPE_CONVERTERS:
                                result_df[ext_field] = self._convert_series_vectorized(
                                    result_df[ext_field], field_type
                                )
                            property_mappings[ext_field] = field_type
                    else:
                        # Single extraction - replace original field
                        result_df[field_name] = extracted_data
                        # Apply type conversion
                        if field_type in self.TYPE_CONVERTERS:
                            result_df[field_name] = self._convert_series_vectorized(
                                result_df[field_name], field_type
                            )
                        property_mappings[field_name] = field_type

                except Exception as e:
                    self.logger.error(
                        f"Error applying regex extractor to field '{field_name}': {e}"
                    )
                    continue
            else:
                # Apply type conversion without extraction
                try:
                    if field_type in self.TYPE_CONVERTERS:
                        result_df[field_name] = self._convert_series_vectorized(
                            result_df[field_name], field_type
                        )

                    property_mappings[field_name] = field_type

                except Exception as e:
                    self.logger.error(
                        f"Error converting field '{field_name}' to type '{field_type}': {e}"
                    )
                    continue

        self.logger.info(
            f"Mapped {len(property_mappings)} properties for node {node_config['label']}"
        )
        return result_df

    def map_relationship_properties(
        self, df: pd.DataFrame, rel_config: Dict[str, Any], inplace: bool = False
    ) -> pd.DataFrame:
        """Map DataFrame columns to relationship properties.
        
        Args:
            df: Input DataFrame
            rel_config: Relationship configuration
            inplace: If True, modify df in-place to save memory (default: False)
        """
        if inplace:
            result_df = df
        else:
            result_df = df.copy()

        if "properties" not in rel_config:
            return result_df

        property_mappings = {}
        for prop_config in rel_config["properties"]:
            field_name = prop_config["field"]
            field_type = prop_config["type"]

            # Resolve column name using the same logic as nodes
            actual_column = self._resolve_column_name(df, field_name, "")

            if actual_column is None:
                self.logger.warning(
                    f"Column '{field_name}' not found in DataFrame for relationship"
                )
                continue

            field_name = actual_column

            # Apply regex extractor if specified
            if (
                "extractor" in prop_config
                and prop_config["extractor"].get("type") == "regex"
            ):
                try:
                    extracted_data = self._apply_regex_extractor(
                        result_df[field_name], prop_config["extractor"], field_name
                    )

                    # Handle multiple extractions (groups)
                    if isinstance(extracted_data, dict):
                        for ext_field, ext_values in extracted_data.items():
                            result_df[ext_field] = ext_values
                            # Apply type conversion to extracted fields
                            if field_type in self.TYPE_CONVERTERS:
                                converter = self.TYPE_CONVERTERS[field_type]
                                result_df[ext_field] = result_df[ext_field].apply(
                                    lambda x: self._safe_convert(x, converter)
                                )
                            property_mappings[ext_field] = field_type
                    else:
                        # Single extraction - replace original field
                        result_df[field_name] = extracted_data
                        # Apply type conversion
                        if field_type in self.TYPE_CONVERTERS:
                            converter = self.TYPE_CONVERTERS[field_type]
                            result_df[field_name] = result_df[field_name].apply(
                                lambda x: self._safe_convert(x, converter)
                            )
                        property_mappings[field_name] = field_type

                except Exception as e:
                    self.logger.error(
                        f"Error applying regex extractor to relationship field '{field_name}': {e}"
                    )
                    continue
            else:
                # Apply type conversion without extraction
                try:
                    if field_type in self.TYPE_CONVERTERS:
                        converter = self.TYPE_CONVERTERS[field_type]
                        result_df[field_name] = result_df[field_name].apply(
                            lambda x: self._safe_convert(x, converter)
                        )

                    property_mappings[field_name] = field_type

                except Exception as e:
                    self.logger.error(
                        f"Error converting relationship property '{field_name}': {e}"
                    )
                    continue

        self.logger.info(
            f"Mapped {len(property_mappings)} properties for relationship {rel_config['type']}"
        )
        return result_df

    def extract_node_data(
        self, df: pd.DataFrame, node_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Extract node data with unique IDs and properties."""
        # Get source for column resolution
        source = node_config.get("source", "")

        # Resolve ID field
        id_field = node_config["id_field"]
        resolved_id_field = self._resolve_column_name(df, id_field, source)

        if resolved_id_field is None:
            raise ValueError(f"ID field '{id_field}' not found in DataFrame")

        # Resolve property fields
        property_fields = []
        for prop_config in node_config["properties"]:
            field_name = prop_config["field"]
            resolved_field = self._resolve_column_name(df, field_name, source)
            if resolved_field:
                property_fields.append(resolved_field)

        # Select relevant columns
        columns_to_select = [resolved_id_field] + property_fields

        if not columns_to_select:
            raise ValueError(f"No valid columns found for node {node_config['label']}")

        # Extract unique nodes
        node_df = df[columns_to_select].drop_duplicates(subset=[resolved_id_field])

        # Add node label and ID (copy back original approach to fix assignment)
        node_df = node_df.copy()
        node_df["_label"] = node_config["label"]
        node_df["_id"] = node_df[resolved_id_field]

        self.logger.info(
            f"Extracted {len(node_df)} unique nodes for label {node_config['label']}"
        )
        return node_df

    def extract_relationship_data(
        self,
        df: pd.DataFrame,
        rel_config: Dict[str, Any],
        node_configs: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Extract relationship data between nodes."""
        from_node_config = None
        to_node_config = None

        # Find node configurations
        for node_config in node_configs:
            if node_config["label"] == rel_config["from_node"]:
                from_node_config = node_config
            elif node_config["label"] == rel_config["to_node"]:
                to_node_config = node_config

        if not from_node_config or not to_node_config:
            raise ValueError(
                f"Node configurations not found for relationship {rel_config['type']}"
            )

        # Get sources for column resolution
        from_source = from_node_config.get("source", "")
        to_source = to_node_config.get("source", "")

        # Resolve ID fields
        from_id_field = from_node_config["id_field"]
        to_id_field = to_node_config["id_field"]

        resolved_from_id = self._resolve_column_name(df, from_id_field, from_source)
        resolved_to_id = self._resolve_column_name(df, to_id_field, to_source)

        if resolved_from_id is None:
            raise ValueError(f"From ID field '{from_id_field}' not found in DataFrame")
        if resolved_to_id is None:
            raise ValueError(f"To ID field '{to_id_field}' not found in DataFrame")

        # Collect relationship property columns
        columns_to_select = [resolved_from_id, resolved_to_id]
        if "properties" in rel_config:
            for prop_config in rel_config["properties"]:
                field_name = prop_config["field"]
                resolved_field = self._resolve_column_name(df, field_name, "")
                if resolved_field:
                    columns_to_select.append(resolved_field)

        # Extract relationships (combine operations to avoid extra copy)
        rel_df = df[columns_to_select].dropna(subset=[resolved_from_id, resolved_to_id])

        # Add relationship metadata
        rel_df["_type"] = rel_config["type"]
        rel_df["_from_id"] = rel_df[resolved_from_id]
        rel_df["_to_id"] = rel_df[resolved_to_id]

        self.logger.info(
            f"Extracted {len(rel_df)} relationships of type {rel_config['type']}"
        )
        return rel_df

    def _safe_convert(self, value: Any, converter) -> Any:
        """Safely convert a value using the provided converter."""
        if pd.isna(value) or value is None:
            return None

        try:
            return converter(value)
        except (ValueError, TypeError, AttributeError):
            return None

    def validate_property_types(
        self, df: pd.DataFrame, property_configs: List[Dict[str, Any]]
    ) -> List[str]:
        """Validate that DataFrame columns can be converted to expected types."""
        errors = []

        for prop_config in property_configs:
            field_name = prop_config["field"]
            expected_type = prop_config["type"]

            if field_name not in df.columns:
                continue

            if expected_type not in self.TYPE_CONVERTERS:
                errors.append(
                    f"Unknown type '{expected_type}' for field '{field_name}'"
                )
                continue

            # Sample a few values to check conversion
            sample_values = df[field_name].dropna().head(10).tolist()
            if not sample_values:
                continue

            converter = self.TYPE_CONVERTERS[expected_type]
            type_errors = 0

            for value in sample_values:
                try:
                    converter(value)
                except Exception:
                    type_errors += 1

            if type_errors > len(sample_values) * 0.5:  # More than 50% errors
                errors.append(
                    f"Column '{field_name}' has incompatible data for type '{expected_type}'"
                )

        return errors

    def _apply_regex_extractor(
        self, series: pd.Series, extractor_config: Dict[str, Any], field_name: str
    ) -> Union[pd.Series, Dict[str, pd.Series]]:
        """Apply regex extractor to extract values from a pandas Series."""
        pattern = extractor_config.get("pattern")
        if not pattern:
            raise ValueError(
                f"Regex extractor missing 'pattern' for field '{field_name}'"
            )

        try:
            # Use cached compiled regex pattern for performance
            regex = self._get_compiled_regex(pattern)
        except re.error as e:
            raise ValueError(
                f"Invalid regex pattern '{pattern}' for field '{field_name}': {e}"
            )

        # Determine extraction mode
        if "groups" in extractor_config:
            # Multiple group extraction - extract specific groups into named fields
            return self._extract_multiple_groups(
                series,
                regex,
                extractor_config["groups"],
                extractor_config.get("fallback_strategy", "original"),
            )
        elif "group" in extractor_config:
            # Single group extraction
            group_num = extractor_config["group"]
            return self._extract_single_group(
                series,
                regex,
                group_num,
                extractor_config.get("fallback_strategy", "original"),
            )
        elif extractor_config.get("named_groups", False):
            # Extract all named groups
            return self._extract_named_groups(
                series, regex, extractor_config.get("fallback_strategy", "original")
            )
        else:
            # Default: extract first group if available, otherwise full match
            return self._extract_default(
                series, regex, extractor_config.get("fallback_strategy", "original")
            )

    def _get_compiled_regex(self, pattern: str) -> re.Pattern:
        """Get compiled regex pattern from cache or compile and cache it."""
        if pattern not in self._regex_cache:
            self._regex_cache[pattern] = re.compile(pattern)
        return self._regex_cache[pattern]

    def _extract_multiple_groups(
        self,
        series: pd.Series,
        regex: re.Pattern,
        group_names: List[str],
        fallback_strategy: str,
    ) -> Dict[str, pd.Series]:
        """Extract multiple regex groups using vectorized operations."""
        result = {}

        # Convert series to string, handling NaN values
        str_series = series.astype(str)

        try:
            # Use pandas str.extract for vectorized regex processing
            extracted_df = str_series.str.extract(regex.pattern, expand=True)

            for i, group_name in enumerate(group_names):
                if i < len(extracted_df.columns):
                    extracted_series = extracted_df.iloc[:, i]

                    # Apply fallback strategy for non-matches
                    if fallback_strategy == "original":
                        extracted_series = extracted_series.where(
                            extracted_series.notna(), series
                        )
                    elif fallback_strategy == "null":
                        extracted_series = extracted_series.where(
                            extracted_series.notna(), None
                        )
                    elif fallback_strategy == "empty":
                        extracted_series = extracted_series.fillna("")

                    # Handle original NaN values
                    extracted_series = extracted_series.where(series.notna(), None)
                    result[group_name] = extracted_series
                else:
                    # Group doesn't exist, apply fallback
                    if fallback_strategy == "original":
                        result[group_name] = series  # No need to copy for fallback
                    elif fallback_strategy == "null":
                        result[group_name] = pd.Series(
                            [None] * len(series), index=series.index
                        )
                    elif fallback_strategy == "empty":
                        result[group_name] = pd.Series(
                            [""] * len(series), index=series.index
                        )

        except Exception as e:
            # Fallback to original implementation if vectorized approach fails
            self.logger.warning(
                f"Vectorized regex extraction failed, using iterative approach: {e}"
            )
            return self._extract_multiple_groups_iterative(
                series, regex, group_names, fallback_strategy
            )

        return result

    def _extract_single_group(
        self,
        series: pd.Series,
        regex: re.Pattern,
        group_num: int,
        fallback_strategy: str,
    ) -> pd.Series:
        """Extract a single regex group using vectorized operations."""
        # Convert series to string, handling NaN values
        str_series = series.astype(str)

        try:
            # Use pandas str.extract for vectorized regex processing
            extracted_df = str_series.str.extract(regex.pattern, expand=True)

            if group_num > 0 and group_num <= len(extracted_df.columns):
                extracted_series = extracted_df.iloc[
                    :, group_num - 1
                ]  # group_num is 1-based
            else:
                # Invalid group number, apply fallback
                if fallback_strategy == "original":
                    return series  # No need to copy for fallback
                elif fallback_strategy == "null":
                    return pd.Series([None] * len(series), index=series.index)
                elif fallback_strategy == "empty":
                    return pd.Series([""] * len(series), index=series.index)
                return series  # No need to copy for fallback

            # Apply fallback strategy for non-matches
            if fallback_strategy == "original":
                extracted_series = extracted_series.where(
                    extracted_series.notna(), series
                )
            elif fallback_strategy == "null":
                extracted_series = extracted_series.where(
                    extracted_series.notna(), None
                )
            elif fallback_strategy == "empty":
                extracted_series = extracted_series.fillna("")

            # Handle original NaN values
            extracted_series = extracted_series.where(series.notna(), None)
            return extracted_series

        except Exception as e:
            # Fallback to original implementation if vectorized approach fails
            self.logger.warning(
                f"Vectorized regex extraction failed, using iterative approach: {e}"
            )
            return self._extract_single_group_iterative(
                series, regex, group_num, fallback_strategy
            )

    def _extract_named_groups(
        self, series: pd.Series, regex: re.Pattern, fallback_strategy: str
    ) -> Dict[str, pd.Series]:
        """Extract all named groups from regex using vectorized operations."""
        # Get all named groups from the pattern
        if not regex.groupindex:
            raise ValueError("No named groups found in regex pattern")

        result = {}
        group_names = list(regex.groupindex.keys())

        # Convert series to string, handling NaN values
        str_series = series.astype(str)

        try:
            # Use pandas str.extract for vectorized regex processing
            extracted_df = str_series.str.extract(regex.pattern, expand=True)

            for group_name in group_names:
                # Get the group index (0-based)
                group_idx = regex.groupindex[group_name] - 1

                if group_idx < len(extracted_df.columns):
                    extracted_series = extracted_df.iloc[:, group_idx]

                    # Apply fallback strategy for non-matches
                    if fallback_strategy == "original":
                        extracted_series = extracted_series.where(
                            extracted_series.notna(), series
                        )
                    elif fallback_strategy == "null":
                        extracted_series = extracted_series.where(
                            extracted_series.notna(), None
                        )
                    elif fallback_strategy == "empty":
                        extracted_series = extracted_series.fillna("")

                    # Handle original NaN values
                    extracted_series = extracted_series.where(series.notna(), None)
                    result[group_name] = extracted_series
                else:
                    # Group doesn't exist, apply fallback
                    if fallback_strategy == "original":
                        result[group_name] = series  # No need to copy for fallback
                    elif fallback_strategy == "null":
                        result[group_name] = pd.Series(
                            [None] * len(series), index=series.index
                        )
                    elif fallback_strategy == "empty":
                        result[group_name] = pd.Series(
                            [""] * len(series), index=series.index
                        )

        except Exception as e:
            # Fallback to original implementation if vectorized approach fails
            self.logger.warning(
                f"Vectorized regex extraction failed, using iterative approach: {e}"
            )
            return self._extract_named_groups_iterative(
                series, regex, fallback_strategy
            )

        return result

    def _extract_default(
        self, series: pd.Series, regex: re.Pattern, fallback_strategy: str
    ) -> pd.Series:
        """Extract using default strategy (first group if available, otherwise full match)."""
        # Convert series to string, handling NaN values
        str_series = series.astype(str)

        try:
            # Use pandas str.extract for vectorized regex processing
            extracted_df = str_series.str.extract(regex.pattern, expand=True)

            if len(extracted_df.columns) > 0:
                # Use first group if available
                extracted_series = extracted_df.iloc[:, 0]
            else:
                # Use full match if no groups
                extracted_series = str_series.str.extract(
                    f"({regex.pattern})", expand=False
                )

            # Apply fallback strategy for non-matches
            if fallback_strategy == "original":
                extracted_series = extracted_series.where(
                    extracted_series.notna(), series
                )
            elif fallback_strategy == "null":
                extracted_series = extracted_series.where(
                    extracted_series.notna(), None
                )
            elif fallback_strategy == "empty":
                extracted_series = extracted_series.fillna("")

            # Handle original NaN values
            extracted_series = extracted_series.where(series.notna(), None)
            return extracted_series

        except Exception as e:
            # Fallback to original implementation if vectorized approach fails
            self.logger.warning(
                f"Vectorized regex extraction failed, using iterative approach: {e}"
            )
            return self._extract_default_iterative(series, regex, fallback_strategy)

    def _resolve_column_name(
        self, df: pd.DataFrame, field_name: str, source: str = ""
    ) -> Optional[str]:
        """Resolve field name to actual DataFrame column, prioritizing database alias prefixes."""
        # Create list of possible column names in priority order
        possible_columns = []

        # If source is provided, try various prefix patterns
        if source:
            if "." in source:
                # Source like "hr.employees" - extract database alias
                db_alias = source.split(".")[0]
                possible_columns.extend(
                    [
                        f"{db_alias}_{field_name}",  # Highest priority: db_alias_field
                        f"{source}_{field_name}",  # db_alias.table_field
                        f"{source}.{field_name}",  # db_alias.table.field
                    ]
                )
            else:
                # Simple source name
                possible_columns.extend(
                    [
                        f"{source}_{field_name}",  # source_field
                        f"{source}.{field_name}",  # source.field
                    ]
                )

        # Add database alias prefixed versions by scanning existing columns
        # This handles cases where we don't know the source but columns are prefixed
        for col in df.columns:
            if col.endswith(f"_{field_name}") and col not in possible_columns:
                possible_columns.append(col)

        # Add original field name as fallback
        possible_columns.append(field_name)

        # Find first matching column
        for col in possible_columns:
            if col in df.columns:
                if col != field_name:
                    self.logger.debug(
                        f"Resolved field '{field_name}' to column '{col}'"
                    )
                return col

        return None

    # Iterative fallback methods for regex extraction (used when vectorized approach fails)
    def _extract_multiple_groups_iterative(
        self,
        series: pd.Series,
        regex: re.Pattern,
        group_names: List[str],
        fallback_strategy: str,
    ) -> Dict[str, pd.Series]:
        """Extract multiple regex groups using iterative approach (fallback)."""
        result = {}

        for i, group_name in enumerate(group_names, 1):
            extracted_values = []

            for value in series:
                if pd.isna(value) or value is None:
                    extracted_values.append(None)
                    continue

                match = regex.search(str(value))
                if match and len(match.groups()) >= i:
                    extracted_values.append(match.group(i))
                else:
                    # Apply fallback strategy
                    if fallback_strategy == "original":
                        extracted_values.append(value)
                    elif fallback_strategy == "null":
                        extracted_values.append(None)
                    elif fallback_strategy == "empty":
                        extracted_values.append("")
                    else:
                        extracted_values.append(value)

            result[group_name] = pd.Series(extracted_values, index=series.index)

        return result

    def _extract_single_group_iterative(
        self,
        series: pd.Series,
        regex: re.Pattern,
        group_num: int,
        fallback_strategy: str,
    ) -> pd.Series:
        """Extract a single regex group using iterative approach (fallback)."""
        extracted_values = []

        for value in series:
            if pd.isna(value) or value is None:
                extracted_values.append(None)
                continue

            match = regex.search(str(value))
            if match and len(match.groups()) >= group_num:
                extracted_values.append(match.group(group_num))
            else:
                # Apply fallback strategy
                if fallback_strategy == "original":
                    extracted_values.append(value)
                elif fallback_strategy == "null":
                    extracted_values.append(None)
                elif fallback_strategy == "empty":
                    extracted_values.append("")
                else:
                    extracted_values.append(value)

        return pd.Series(extracted_values, index=series.index)

    def _extract_named_groups_iterative(
        self, series: pd.Series, regex: re.Pattern, fallback_strategy: str
    ) -> Dict[str, pd.Series]:
        """Extract all named groups from regex using iterative approach (fallback)."""
        if not regex.groupindex:
            raise ValueError("No named groups found in regex pattern")

        result = {}
        group_names = list(regex.groupindex.keys())

        for group_name in group_names:
            extracted_values = []

            for value in series:
                if pd.isna(value) or value is None:
                    extracted_values.append(None)
                    continue

                match = regex.search(str(value))
                if match:
                    try:
                        group_value = match.group(group_name)
                        extracted_values.append(group_value)
                    except IndexError:
                        # Apply fallback strategy
                        if fallback_strategy == "original":
                            extracted_values.append(value)
                        elif fallback_strategy == "null":
                            extracted_values.append(None)
                        elif fallback_strategy == "empty":
                            extracted_values.append("")
                        else:
                            extracted_values.append(value)
                else:
                    # Apply fallback strategy
                    if fallback_strategy == "original":
                        extracted_values.append(value)
                    elif fallback_strategy == "null":
                        extracted_values.append(None)
                    elif fallback_strategy == "empty":
                        extracted_values.append("")
                    else:
                        extracted_values.append(value)

            result[group_name] = pd.Series(extracted_values, index=series.index)

        return result

    def _extract_default_iterative(
        self, series: pd.Series, regex: re.Pattern, fallback_strategy: str
    ) -> pd.Series:
        """Extract using default strategy with iterative approach (fallback)."""
        extracted_values = []

        for value in series:
            if pd.isna(value) or value is None:
                extracted_values.append(None)
                continue

            match = regex.search(str(value))
            if match:
                # Use first group if available, otherwise full match
                if match.groups():
                    extracted_values.append(match.group(1))
                else:
                    extracted_values.append(match.group(0))
            else:
                # Apply fallback strategy
                if fallback_strategy == "original":
                    extracted_values.append(value)
                elif fallback_strategy == "null":
                    extracted_values.append(None)
                elif fallback_strategy == "empty":
                    extracted_values.append("")
                else:
                    extracted_values.append(value)

        return pd.Series(extracted_values, index=series.index)
    # Vectorized type conversion methods for performance optimization
    def _convert_to_string_vectorized(self, series: pd.Series) -> pd.Series:
        """Convert series to string using vectorized operations."""
        try:
            # Use pandas astype for fast string conversion
            return series.astype(str).where(series.notna(), None)
        except Exception as e:
            self.logger.warning(f"Vectorized string conversion failed: {e}")
            return series.apply(lambda x: self._safe_convert(x, str))

    def _convert_to_integer_vectorized(self, series: pd.Series) -> pd.Series:
        """Convert series to integer using vectorized operations."""
        try:
            # Use pd.to_numeric with errors='coerce' for fast numeric conversion
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            # Convert to integer, preserving NaN values
            return numeric_series.astype('Int64')  # Nullable integer type
        except Exception as e:
            self.logger.warning(f"Vectorized integer conversion failed: {e}")
            return series.apply(lambda x: self._safe_convert(x, int))

    def _convert_to_float_vectorized(self, series: pd.Series) -> pd.Series:
        """Convert series to float using vectorized operations."""
        try:
        # Use pd.to_numeric for fast float conversion
            return pd.to_numeric(series, errors='coerce')
        except Exception as e:
            self.logger.warning(f"Vectorized float conversion failed: {e}")
            return series.apply(lambda x: self._safe_convert(x, float))

    def _convert_to_boolean_vectorized(self, series: pd.Series) -> pd.Series:
        """Convert series to boolean using vectorized operations."""
        try:
            # Handle common boolean representations
            str_series = series.astype(str).str.lower()
            
            # Define true/false mappings
            true_values = {'true', '1', 'yes', 'y', 't', 'on'}
            false_values = {'false', '0', 'no', 'n', 'f', 'off'}
            
            # Create boolean mask
            is_true = str_series.isin(true_values)
            is_false = str_series.isin(false_values)
            is_valid = is_true | is_false
            
            # Convert to boolean, setting invalid values to None
            result = pd.Series(index=series.index, dtype='boolean')
            result.loc[is_true] = True
            result.loc[is_false] = False
            result.loc[~is_valid] = None
            
            # Preserve original NaN values
            result.loc[series.isna()] = None
            
            return result
        except Exception as e:
            self.logger.warning(f"Vectorized boolean conversion failed: {e}")
            return series.apply(lambda x: self._safe_convert(x, bool))

    def _convert_to_datetime_vectorized(self, series: pd.Series) -> pd.Series:
        """Convert series to datetime using vectorized operations."""
        try:
        # Use pd.to_datetime for fast datetime conversion
            return pd.to_datetime(series, errors='coerce')
        except Exception as e:
            self.logger.warning(f"Vectorized datetime conversion failed: {e}")
            return series.apply(lambda x: self._safe_convert(x, pd.to_datetime))

    def _convert_to_date_vectorized(self, series: pd.Series) -> pd.Series:
        """Convert series to date using vectorized operations."""
        try:
            # Convert to datetime first, then extract date
            datetime_series = pd.to_datetime(series, errors='coerce')
            return datetime_series.dt.date.where(datetime_series.notna(), None)
        except Exception as e:
            self.logger.warning(f"Vectorized date conversion failed: {e}")
            return series.apply(lambda x: self._safe_convert(x, lambda y: pd.to_datetime(y).date()))

    def _convert_to_time_vectorized(self, series: pd.Series) -> pd.Series:
        """Convert series to time using vectorized operations."""
        try:
            # Convert to datetime first, then extract time
            datetime_series = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
            return datetime_series.dt.time.where(datetime_series.notna(), None)
        except Exception as e:
            self.logger.warning(f"Vectorized time conversion failed: {e}")
            return series.apply(lambda x: self._safe_convert(x, lambda y: pd.to_datetime(y).time()))

    def _convert_series_vectorized(self, series: pd.Series, field_type: str) -> pd.Series:
        """Convert series using vectorized operations with fallback to apply()."""
        if field_type in self.VECTORIZED_CONVERTERS:
            try:
                return self.VECTORIZED_CONVERTERS[field_type](series)
            except Exception as e:
                self.logger.warning(
                    f"Vectorized conversion failed for type '{field_type}': {e}. "
                    f"Falling back to apply() method."
                )
        
        # Fallback to original apply() method
        if field_type in self.TYPE_CONVERTERS:
            converter = self.TYPE_CONVERTERS[field_type]
            return series.apply(lambda x: self._safe_convert(x, converter))
        
        return series