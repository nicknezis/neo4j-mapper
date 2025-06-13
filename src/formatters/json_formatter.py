"""JSON formatter for Neo4j data."""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union
from datetime import datetime, date, time
import logging


class JSONFormatter:
    """Formats graph data as JSON for flexibility and debugging."""

    def __init__(self, output_directory: str = "output"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def format_nodes(
        self, nodes_data: List[pd.DataFrame], mapping_name: str = "mapping"
    ) -> str:
        """Format nodes data as JSON."""
        nodes_json = []

        for node_df in nodes_data:
            if len(node_df) == 0:
                continue

            # Convert DataFrame to list of dictionaries
            node_records = self._dataframe_to_records(node_df)
            nodes_json.extend(node_records)

        # Write JSON file
        filename = f"{mapping_name}_nodes.json"
        filepath = self.output_directory / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                nodes_json,
                f,
                indent=2,
                default=self._json_serializer,
                ensure_ascii=False,
            )

        self.logger.info(f"Wrote {len(nodes_json)} nodes to {filepath}")
        return str(filepath)

    def format_relationships(
        self, relationships_data: List[pd.DataFrame], mapping_name: str = "mapping"
    ) -> str:
        """Format relationships data as JSON."""
        relationships_json = []

        for rel_df in relationships_data:
            if len(rel_df) == 0:
                continue

            # Convert DataFrame to list of dictionaries
            rel_records = self._dataframe_to_records(rel_df)
            relationships_json.extend(rel_records)

        # Write JSON file
        filename = f"{mapping_name}_relationships.json"
        filepath = self.output_directory / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                relationships_json,
                f,
                indent=2,
                default=self._json_serializer,
                ensure_ascii=False,
            )

        self.logger.info(f"Wrote {len(relationships_json)} relationships to {filepath}")
        return str(filepath)

    def format_graph_structure(
        self,
        nodes_data: List[pd.DataFrame],
        relationships_data: List[pd.DataFrame],
        mapping_name: str = "mapping",
    ) -> str:
        """Format complete graph structure as a single JSON file."""
        graph_data = {
            "metadata": {
                "mapping_name": mapping_name,
                "generated_at": datetime.now().isoformat(),
                "format_version": "1.0",
            },
            "nodes": [],
            "relationships": [],
        }

        # Add nodes
        for node_df in nodes_data:
            if len(node_df) > 0:
                node_records = self._dataframe_to_records(node_df)
                graph_data["nodes"].extend(node_records)

        # Add relationships
        for rel_df in relationships_data:
            if len(rel_df) > 0:
                rel_records = self._dataframe_to_records(rel_df)
                graph_data["relationships"].extend(rel_records)

        # Add summary statistics
        graph_data["metadata"]["statistics"] = {
            "total_nodes": len(graph_data["nodes"]),
            "total_relationships": len(graph_data["relationships"]),
            "node_types": len(
                set(node.get("_label", "Unknown") for node in graph_data["nodes"])
            ),
            "relationship_types": len(
                set(rel.get("_type", "Unknown") for rel in graph_data["relationships"])
            ),
        }

        # Write JSON file
        filename = f"{mapping_name}_graph.json"
        filepath = self.output_directory / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                graph_data,
                f,
                indent=2,
                default=self._json_serializer,
                ensure_ascii=False,
            )

        self.logger.info(f"Wrote complete graph structure to {filepath}")
        return str(filepath)

    def format_cypher_compatible(
        self,
        nodes_data: List[pd.DataFrame],
        relationships_data: List[pd.DataFrame],
        mapping_name: str = "mapping",
    ) -> str:
        """Format data in a structure that can easily generate Cypher statements."""
        cypher_data = {
            "metadata": {
                "mapping_name": mapping_name,
                "generated_at": datetime.now().isoformat(),
            },
            "create_statements": {"nodes": [], "relationships": []},
        }

        # Format nodes for Cypher generation
        for node_df in nodes_data:
            if len(node_df) == 0:
                continue

            label = node_df.iloc[0].get("_label", "UnknownNode")

            for _, row in node_df.iterrows():
                node_data = {"label": label, "id": row.get("_id"), "properties": {}}

                # Add properties (exclude metadata columns)
                for col, value in row.items():
                    if not col.startswith("_") and pd.notna(value):
                        node_data["properties"][col] = self._convert_value_for_cypher(
                            value
                        )

                cypher_data["create_statements"]["nodes"].append(node_data)

        # Format relationships for Cypher generation
        for rel_df in relationships_data:
            if len(rel_df) == 0:
                continue

            rel_type = rel_df.iloc[0].get("_type", "UnknownRelationship")

            for _, row in rel_df.iterrows():
                rel_data = {
                    "type": rel_type,
                    "from_id": row.get("_from_id"),
                    "to_id": row.get("_to_id"),
                    "properties": {},
                }

                # Add properties (exclude metadata columns)
                for col, value in row.items():
                    if not col.startswith("_") and pd.notna(value):
                        rel_data["properties"][col] = self._convert_value_for_cypher(
                            value
                        )

                cypher_data["create_statements"]["relationships"].append(rel_data)

        # Write JSON file
        filename = f"{mapping_name}_cypher_data.json"
        filepath = self.output_directory / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                cypher_data,
                f,
                indent=2,
                default=self._json_serializer,
                ensure_ascii=False,
            )

        self.logger.info(f"Wrote Cypher-compatible data to {filepath}")
        return str(filepath)

    def _dataframe_to_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to list of dictionaries with proper type handling."""
        records = []

        for _, row in df.iterrows():
            record = {}
            for col, value in row.items():
                if pd.notna(value):
                    record[col] = self._convert_pandas_types(value)
                else:
                    record[col] = None
            records.append(record)

        return records

    def _convert_pandas_types(self, value: Any) -> Any:
        """Convert pandas types to JSON-serializable types."""
        # Handle pandas Timestamp
        if hasattr(value, "to_pydatetime"):
            return value.to_pydatetime().isoformat()

        # Handle numpy types
        if hasattr(value, "item"):
            return value.item()

        # Handle pandas NaType
        if pd.isna(value):
            return None

        return value

    def _convert_value_for_cypher(self, value: Any) -> Any:
        """Convert value to Cypher-compatible format."""
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float)):
            return value
        elif isinstance(value, bool):
            return value
        elif isinstance(value, (datetime, date, time)):
            return value.isoformat()
        elif pd.isna(value):
            return None
        else:
            return str(value)

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.to_pydatetime().isoformat()
        elif hasattr(obj, "item"):  # numpy types
            return obj.item()
        elif pd.isna(obj):
            return None

        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def format_complete_mapping(
        self,
        nodes_data: List[pd.DataFrame],
        relationships_data: List[pd.DataFrame],
        mapping_name: str = "mapping",
    ) -> Dict[str, Any]:
        """Format complete mapping data to JSON files."""
        result = {
            "files": {},
            "summary": {
                "total_nodes": sum(len(df) for df in nodes_data),
                "total_relationships": sum(len(df) for df in relationships_data),
                "generated_files": [],
            },
        }

        # Generate different JSON formats
        files_generated = []

        # Individual nodes and relationships files
        nodes_file = self.format_nodes(nodes_data, mapping_name)
        relationships_file = self.format_relationships(relationships_data, mapping_name)
        files_generated.extend([nodes_file, relationships_file])

        # Complete graph structure
        graph_file = self.format_graph_structure(
            nodes_data, relationships_data, mapping_name
        )
        files_generated.append(graph_file)

        # Cypher-compatible format
        cypher_file = self.format_cypher_compatible(
            nodes_data, relationships_data, mapping_name
        )
        files_generated.append(cypher_file)

        result["files"] = {
            "nodes": nodes_file,
            "relationships": relationships_file,
            "graph_structure": graph_file,
            "cypher_compatible": cypher_file,
        }

        result["summary"]["generated_files"] = files_generated

        return result
