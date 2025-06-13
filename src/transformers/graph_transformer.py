"""Graph transformation engine for Neo4j Mapper."""

import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from .data_mapper import DataMapper


class GraphTransformer:
    """Transforms relational data into graph format for Neo4j."""

    def __init__(self):
        self.data_mapper = DataMapper()
        self.logger = logging.getLogger(__name__)

    def transform_mapping(
        self, df: pd.DataFrame, mapping_config: Dict[str, Any]
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """Transform a complete mapping configuration into nodes and relationships."""
        nodes_data = []
        relationships_data = []

        mapping_name = mapping_config.get("name", "unnamed_mapping")
        self.logger.info(f"Processing mapping: {mapping_name}")

        # Process nodes
        for node_config in mapping_config["nodes"]:
            try:
                node_df = self._transform_node(df, node_config)
                nodes_data.append(node_df)
                self.logger.info(
                    f"Processed node: {node_config['label']} ({len(node_df)} records)"
                )
            except Exception as e:
                self.logger.error(f"Error processing node {node_config['label']}: {e}")
                raise

        # Process relationships if defined
        if "relationships" in mapping_config:
            for rel_config in mapping_config["relationships"]:
                try:
                    rel_df = self._transform_relationship(
                        df, rel_config, mapping_config["nodes"]
                    )
                    relationships_data.append(rel_df)
                    self.logger.info(
                        f"Processed relationship: {rel_config['type']} ({len(rel_df)} records)"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error processing relationship {rel_config['type']}: {e}"
                    )
                    raise

        return nodes_data, relationships_data

    def _transform_node(
        self, df: pd.DataFrame, node_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Transform data for a single node type."""
        # Map properties according to configuration
        mapped_df = self.data_mapper.map_node_properties(df, node_config)

        # Extract unique node data
        node_df = self.data_mapper.extract_node_data(mapped_df, node_config)

        # Add any computed properties
        if "computed_properties" in node_config:
            node_df = self._add_computed_properties(
                node_df, node_config["computed_properties"]
            )

        return node_df

    def _transform_relationship(
        self,
        df: pd.DataFrame,
        rel_config: Dict[str, Any],
        node_configs: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Transform data for a single relationship type."""
        # Map relationship properties
        mapped_df = self.data_mapper.map_relationship_properties(df, rel_config)

        # Extract relationship data
        rel_df = self.data_mapper.extract_relationship_data(
            mapped_df, rel_config, node_configs
        )

        # Add any computed properties
        if "computed_properties" in rel_config:
            rel_df = self._add_computed_properties(
                rel_df, rel_config["computed_properties"]
            )

        return rel_df

    def _add_computed_properties(
        self, df: pd.DataFrame, computed_props: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Add computed properties to the DataFrame."""
        result_df = df.copy()

        for comp_prop in computed_props:
            prop_name = comp_prop["name"]
            prop_type = comp_prop["type"]
            expression = comp_prop["expression"]

            try:
                # Simple expression evaluation (extend as needed)
                if expression.startswith("CONCAT("):
                    # Handle concatenation
                    result_df[prop_name] = self._handle_concat(result_df, expression)
                elif expression.startswith("FORMAT_DATE("):
                    # Handle date formatting
                    result_df[prop_name] = self._handle_date_format(
                        result_df, expression
                    )
                elif expression.startswith("UPPER(") or expression.startswith("LOWER("):
                    # Handle case conversion
                    result_df[prop_name] = self._handle_case_conversion(
                        result_df, expression
                    )
                else:
                    # Simple column reference or constant
                    if expression in result_df.columns:
                        result_df[prop_name] = result_df[expression]
                    else:
                        result_df[prop_name] = expression

                # Apply type conversion
                if prop_type in self.data_mapper.TYPE_CONVERTERS:
                    converter = self.data_mapper.TYPE_CONVERTERS[prop_type]
                    result_df[prop_name] = result_df[prop_name].apply(
                        lambda x: self.data_mapper._safe_convert(x, converter)
                    )

                self.logger.info(f"Added computed property: {prop_name}")

            except Exception as e:
                self.logger.error(f"Error computing property '{prop_name}': {e}")
                continue

        return result_df

    def _handle_concat(self, df: pd.DataFrame, expression: str) -> pd.Series:
        """Handle CONCAT() expressions."""
        # Extract columns from CONCAT(col1, col2, ...)
        import re

        match = re.match(r"CONCAT\((.*)\)", expression)
        if not match:
            return pd.Series([expression] * len(df))

        columns_str = match.group(1)
        columns = [col.strip().strip("\"'") for col in columns_str.split(",")]

        # Concatenate columns
        result = df[columns[0]].astype(str)
        for col in columns[1:]:
            if col in df.columns:
                result = result + df[col].astype(str)
            else:
                result = result + col  # Literal string

        return result

    def _handle_date_format(self, df: pd.DataFrame, expression: str) -> pd.Series:
        """Handle FORMAT_DATE() expressions."""
        import re

        match = re.match(r'FORMAT_DATE\(([^,]+),\s*["\']([^"\']+)["\']\)', expression)
        if not match:
            return pd.Series([expression] * len(df))

        column = match.group(1).strip()
        format_str = match.group(2)

        if column not in df.columns:
            return pd.Series([expression] * len(df))

        # Convert to datetime and format
        try:
            date_series = pd.to_datetime(df[column])
            return date_series.dt.strftime(format_str)
        except Exception:
            return df[column].astype(str)

    def _handle_case_conversion(self, df: pd.DataFrame, expression: str) -> pd.Series:
        """Handle UPPER() and LOWER() expressions."""
        import re

        upper_match = re.match(r"UPPER\(([^)]+)\)", expression)
        lower_match = re.match(r"LOWER\(([^)]+)\)", expression)

        if upper_match:
            column = upper_match.group(1).strip()
            if column in df.columns:
                return df[column].astype(str).str.upper()
        elif lower_match:
            column = lower_match.group(1).strip()
            if column in df.columns:
                return df[column].astype(str).str.lower()

        return pd.Series([expression] * len(df))

    def validate_graph_data(
        self, nodes_data: List[pd.DataFrame], relationships_data: List[pd.DataFrame]
    ) -> List[str]:
        """Validate the transformed graph data."""
        errors = []

        # Validate nodes
        node_ids = {}  # label -> set of IDs

        for node_df in nodes_data:
            if "_label" not in node_df.columns or "_id" not in node_df.columns:
                errors.append("Node DataFrame missing required _label or _id columns")
                continue

            label = node_df.iloc[0]["_label"] if len(node_df) > 0 else "unknown"

            # Check for duplicate IDs within node type
            ids = node_df["_id"].tolist()
            if len(ids) != len(set(ids)):
                errors.append(f"Duplicate IDs found in node type: {label}")

            # Store IDs for relationship validation
            node_ids[label] = set(ids)

        # Validate relationships
        for rel_df in relationships_data:
            if (
                "_type" not in rel_df.columns
                or "_from_id" not in rel_df.columns
                or "_to_id" not in rel_df.columns
            ):
                errors.append("Relationship DataFrame missing required columns")
                continue

            # Check for orphaned relationships (IDs that don't exist in nodes)
            # This is a basic check - in practice, you might want more sophisticated validation

            if len(rel_df) > 0:
                rel_type = rel_df.iloc[0]["_type"]

                # Check for null relationship IDs
                null_from = rel_df["_from_id"].isnull().sum()
                null_to = rel_df["_to_id"].isnull().sum()

                if null_from > 0:
                    errors.append(
                        f"Relationship {rel_type} has {null_from} null from_id values"
                    )
                if null_to > 0:
                    errors.append(
                        f"Relationship {rel_type} has {null_to} null to_id values"
                    )

        return errors

    def get_transformation_stats(
        self, nodes_data: List[pd.DataFrame], relationships_data: List[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Get statistics about the transformed data."""
        stats = {
            "nodes": {},
            "relationships": {},
            "totals": {
                "node_count": 0,
                "relationship_count": 0,
                "node_types": 0,
                "relationship_types": 0,
            },
        }

        # Node statistics
        for node_df in nodes_data:
            if len(node_df) > 0 and "_label" in node_df.columns:
                label = node_df.iloc[0]["_label"]
                stats["nodes"][label] = {
                    "count": len(node_df),
                    "properties": len(
                        [col for col in node_df.columns if not col.startswith("_")]
                    ),
                }
                stats["totals"]["node_count"] += len(node_df)

        stats["totals"]["node_types"] = len(stats["nodes"])

        # Relationship statistics
        for rel_df in relationships_data:
            if len(rel_df) > 0 and "_type" in rel_df.columns:
                rel_type = rel_df.iloc[0]["_type"]
                stats["relationships"][rel_type] = {
                    "count": len(rel_df),
                    "properties": len(
                        [col for col in rel_df.columns if not col.startswith("_")]
                    ),
                }
                stats["totals"]["relationship_count"] += len(rel_df)

        stats["totals"]["relationship_types"] = len(stats["relationships"])

        return stats
