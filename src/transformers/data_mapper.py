"""Data mapping utilities for Neo4j transformation."""

import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date, time
import logging
import re


class DataMapper:
    """Maps and transforms data according to configuration specifications."""

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

    def map_node_properties(
        self, df: pd.DataFrame, node_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Map DataFrame columns to node properties according to configuration."""
        result_df = df.copy()

        # Get source table/alias information
        source = node_config.get("source", "")

        # Extract and transform properties
        property_mappings = {}
        for prop_config in node_config["properties"]:
            field_name = prop_config["field"]
            field_type = prop_config["type"]

            # Handle source prefix if needed
            if source and "." in source:
                # For joined tables, might need to handle prefixed columns
                possible_columns = [
                    field_name,
                    f"{source}.{field_name}",
                    f"{source}_{field_name}",
                ]

                actual_column = None
                for col in possible_columns:
                    if col in df.columns:
                        actual_column = col
                        break

                if actual_column is None:
                    self.logger.warning(
                        f"Column '{field_name}' not found in DataFrame for node {node_config['label']}"
                    )
                    continue

                field_name = actual_column

            if field_name not in df.columns:
                self.logger.warning(f"Column '{field_name}' not found in DataFrame")
                continue

            # Apply regex extractor if specified
            if "extractor" in prop_config and prop_config["extractor"].get("type") == "regex":
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
                        f"Error applying regex extractor to field '{field_name}': {e}"
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
                        f"Error converting field '{field_name}' to type '{field_type}': {e}"
                    )
                    continue

        self.logger.info(
            f"Mapped {len(property_mappings)} properties for node {node_config['label']}"
        )
        return result_df

    def map_relationship_properties(
        self, df: pd.DataFrame, rel_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Map DataFrame columns to relationship properties."""
        result_df = df.copy()

        if "properties" not in rel_config:
            return result_df

        property_mappings = {}
        for prop_config in rel_config["properties"]:
            field_name = prop_config["field"]
            field_type = prop_config["type"]

            if field_name not in df.columns:
                self.logger.warning(
                    f"Column '{field_name}' not found in DataFrame for relationship"
                )
                continue

            # Apply regex extractor if specified
            if "extractor" in prop_config and prop_config["extractor"].get("type") == "regex":
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
        # Get ID field
        id_field = node_config["id_field"]

        if id_field not in df.columns:
            raise ValueError(f"ID field '{id_field}' not found in DataFrame")

        # Get property fields
        property_fields = [prop["field"] for prop in node_config["properties"]]

        # Select relevant columns
        columns_to_select = [id_field] + [f for f in property_fields if f in df.columns]

        if not columns_to_select:
            raise ValueError(f"No valid columns found for node {node_config['label']}")

        # Extract unique nodes
        node_df = df[columns_to_select].drop_duplicates(subset=[id_field])

        # Add node label
        node_df = node_df.copy()
        node_df["_label"] = node_config["label"]
        node_df["_id"] = node_df[id_field]

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
                f"Could not find node configurations for relationship {rel_config['type']}"
            )

        from_id_field = from_node_config["id_field"]
        to_id_field = to_node_config["id_field"]

        # Select relationship columns
        columns_to_select = [from_id_field, to_id_field]

        # Add relationship properties if specified
        if "properties" in rel_config:
            rel_properties = [prop["field"] for prop in rel_config["properties"]]
            columns_to_select.extend([f for f in rel_properties if f in df.columns])

        # Extract relationships
        rel_df = df[columns_to_select].copy()
        rel_df = rel_df.dropna(subset=[from_id_field, to_id_field])

        # Add relationship metadata
        rel_df["_type"] = rel_config["type"]
        rel_df["_from_id"] = rel_df[from_id_field]
        rel_df["_to_id"] = rel_df[to_id_field]

        self.logger.info(
            f"Extracted {len(rel_df)} relationships of type {rel_config['type']}"
        )
        return rel_df

    def _safe_convert(self, value: Any, converter) -> Any:
        """Safely convert a value using the provided converter."""
        if pd.isna(value) or value is None:
            return None

        try:
            if converter == bool:
                # Handle various boolean representations
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on", "t", "y")
                return bool(value)

            elif converter == pd.to_datetime:
                return pd.to_datetime(value)

            elif callable(converter):
                return converter(value)

            else:
                return converter(value)

        except (ValueError, TypeError) as e:
            self.logger.warning(
                f"Could not convert value '{value}' using {converter}: {e}"
            )
            return value

    def validate_data_types(
        self, df: pd.DataFrame, config_properties: List[Dict[str, Any]]
    ) -> List[str]:
        """Validate that DataFrame columns match expected types."""
        errors = []

        for prop_config in config_properties:
            field_name = prop_config["field"]
            expected_type = prop_config["type"]

            if field_name not in df.columns:
                errors.append(f"Missing column: {field_name}")
                continue

            # Check for type compatibility
            column_data = df[field_name].dropna()

            if len(column_data) == 0:
                continue

            # Sample a few values to check type compatibility
            sample_values = column_data.head(10)
            type_errors = 0

            for value in sample_values:
                try:
                    if expected_type in self.TYPE_CONVERTERS:
                        converter = self.TYPE_CONVERTERS[expected_type]
                        self._safe_convert(value, converter)
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
            raise ValueError(f"Regex extractor missing 'pattern' for field '{field_name}'")
        
        try:
            # Compile regex pattern
            regex = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}' for field '{field_name}': {e}")
        
        # Determine extraction mode
        if "groups" in extractor_config:
            # Multiple group extraction - extract specific groups into named fields
            return self._extract_multiple_groups(series, regex, extractor_config["groups"], 
                                               extractor_config.get("fallback_strategy", "original"))
        elif "group" in extractor_config:
            # Single group extraction
            group_num = extractor_config["group"]
            return self._extract_single_group(series, regex, group_num, 
                                            extractor_config.get("fallback_strategy", "original"))
        elif extractor_config.get("named_groups", False):
            # Extract all named groups
            return self._extract_named_groups(series, regex, 
                                            extractor_config.get("fallback_strategy", "original"))
        else:
            # Default: extract first group if available, otherwise full match
            return self._extract_default(series, regex, 
                                       extractor_config.get("fallback_strategy", "original"))

    def _extract_multiple_groups(
        self, series: pd.Series, regex: re.Pattern, group_names: List[str], fallback_strategy: str
    ) -> Dict[str, pd.Series]:
        """Extract multiple regex groups into separate fields."""
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

    def _extract_single_group(
        self, series: pd.Series, regex: re.Pattern, group_num: int, fallback_strategy: str
    ) -> pd.Series:
        """Extract a single regex group."""
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

    def _extract_named_groups(
        self, series: pd.Series, regex: re.Pattern, fallback_strategy: str
    ) -> Dict[str, pd.Series]:
        """Extract all named groups from regex."""
        # Get all named groups from the pattern
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

    def _extract_default(
        self, series: pd.Series, regex: re.Pattern, fallback_strategy: str
    ) -> pd.Series:
        """Extract using default strategy (first group if available, otherwise full match)."""
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
