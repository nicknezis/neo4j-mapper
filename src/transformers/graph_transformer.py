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
        # Apply WHERE clause filtering if specified
        filtered_df = df
        if "where" in node_config:
            filtered_df = self._apply_where_filter(df, node_config["where"], node_config.get("source", ""))
            self.logger.info(
                f"Applied WHERE filter to node {node_config['label']}: {len(df)} -> {len(filtered_df)} records"
            )

        # Map properties according to configuration
        mapped_df = self.data_mapper.map_node_properties(filtered_df, node_config)

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
        # Apply WHERE clause filtering if specified
        filtered_df = df
        if "where" in rel_config:
            filtered_df = self._apply_where_filter(df, rel_config["where"], "")
            self.logger.info(
                f"Applied WHERE filter to relationship {rel_config['type']}: {len(df)} -> {len(filtered_df)} records"
            )

        # Map relationship properties
        mapped_df = self.data_mapper.map_relationship_properties(filtered_df, rel_config)

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

    def _apply_where_filter(
        self, df: pd.DataFrame, where_clause: str, source: str = ""
    ) -> pd.DataFrame:
        """Apply WHERE clause filtering to a DataFrame."""
        if not where_clause or not where_clause.strip():
            return df
        
        try:
            # Use pandas query method for filtering
            # Convert SQL-style column references to pandas-compatible ones
            pandas_query = self._convert_where_to_pandas_query(df, where_clause, source)
            
            if pandas_query:
                filtered_df = df.query(pandas_query)
                self.logger.debug(f"Applied WHERE filter: '{where_clause}' -> '{pandas_query}'")
                return filtered_df
            else:
                self.logger.warning(f"Could not convert WHERE clause to pandas query: {where_clause}")
                return df
                
        except Exception as e:
            self.logger.error(f"Error applying WHERE filter '{where_clause}': {e}")
            # Return original DataFrame if filtering fails
            return df

    def _convert_where_to_pandas_query(
        self, df: pd.DataFrame, where_clause: str, source: str = ""
    ) -> Optional[str]:
        """Convert SQL WHERE clause to pandas query syntax."""
        try:
            # Resolve column names in the WHERE clause using the same logic as DataMapper
            resolved_query = where_clause
            
            # Find potential column references and resolve them
            import re
            
            # Match word boundaries to find potential column names
            # This is a simplified approach - more sophisticated parsing could be added
            words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', where_clause)
            
            # Track which words we've already processed to avoid double-wrapping
            processed_words = set()
            
            for word in words:
                # Skip SQL keywords and operators
                if word.upper() in {'AND', 'OR', 'NOT', 'IN', 'LIKE', 'IS', 'NULL', 'TRUE', 'FALSE', 
                                   'BETWEEN', 'EXISTS', 'ANY', 'ALL', 'SOME', 'ASC', 'DESC'}:
                    continue
                
                # Skip if we've already processed this word
                if word in processed_words:
                    continue
                
                # Try to resolve the column name
                resolved_col = self.data_mapper._resolve_column_name(df, word, source)
                if resolved_col and resolved_col != word:
                    # Replace the column reference in the query
                    # Use word boundaries to avoid partial replacements
                    pattern = r'\b' + re.escape(word) + r'\b'
                    resolved_query = re.sub(pattern, f"`{resolved_col}`", resolved_query)
                    processed_words.add(word)
                elif word in df.columns:
                    # Column exists as-is, but wrap in backticks for pandas query
                    pattern = r'\b' + re.escape(word) + r'\b'
                    resolved_query = re.sub(pattern, f"`{word}`", resolved_query)
                    processed_words.add(word)
            
            # Convert SQL operators to pandas query syntax
            # Handle basic SQL to pandas conversions
            resolved_query = resolved_query.replace(' = ', ' == ')
            resolved_query = resolved_query.replace('!=', '!=')  # Already correct
            resolved_query = resolved_query.replace('<>', '!=')  # SQL alternative for !=
            
            # Convert SQL boolean literals to Python boolean literals
            resolved_query = re.sub(r'\btrue\b', 'True', resolved_query, flags=re.IGNORECASE)
            resolved_query = re.sub(r'\bfalse\b', 'False', resolved_query, flags=re.IGNORECASE)
            
            # Convert SQL logical operators to Python logical operators
            resolved_query = re.sub(r'\bAND\b', 'and', resolved_query, flags=re.IGNORECASE)
            resolved_query = re.sub(r'\bOR\b', 'or', resolved_query, flags=re.IGNORECASE)
            resolved_query = re.sub(r'\bNOT\b', 'not', resolved_query, flags=re.IGNORECASE)
            
            # Handle IS NULL / IS NOT NULL
            resolved_query = re.sub(r'`([^`]+)`\s+IS\s+NULL', r'\1.isnull()', resolved_query, flags=re.IGNORECASE)
            resolved_query = re.sub(r'`([^`]+)`\s+IS\s+NOT\s+NULL', r'\1.notnull()', resolved_query, flags=re.IGNORECASE)
            
            # Handle LIKE operator (convert to string contains - simplified)
            def convert_like(match):
                column = match.group(1)
                pattern = match.group(2).strip("'\"")
                if pattern.startswith('%') and pattern.endswith('%'):
                    # Contains
                    search_term = pattern[1:-1]
                    return f"{column}.str.contains('{search_term}', na=False)"
                elif pattern.startswith('%'):
                    # Ends with
                    search_term = pattern[1:]
                    return f"{column}.str.endswith('{search_term}', na=False)"
                elif pattern.endswith('%'):
                    # Starts with
                    search_term = pattern[:-1]
                    return f"{column}.str.startswith('{search_term}', na=False)"
                else:
                    # Exact match
                    return f"{column} == '{pattern}'"
            
            resolved_query = re.sub(
                r'`([^`]+)`\s+LIKE\s+([\'"][^\'\"]*[\'"])', 
                convert_like, 
                resolved_query, 
                flags=re.IGNORECASE
            )
            
            # Handle IN operator
            def convert_in(match):
                column = match.group(1)
                values = match.group(2)
                return f"{column}.isin([{values}])"
            
            resolved_query = re.sub(
                r'`([^`]+)`\s+IN\s*\(([^)]+)\)', 
                convert_in, 
                resolved_query, 
                flags=re.IGNORECASE
            )
            
            return resolved_query
            
        except Exception as e:
            self.logger.error(f"Error converting WHERE clause '{where_clause}': {e}")
            return None
