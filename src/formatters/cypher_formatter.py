"""Cypher formatter for direct Neo4j execution."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union
import logging
import re


class CypherFormatter:
    """Formats graph data as Cypher statements for direct Neo4j execution."""
    
    def __init__(self, output_directory: str = "output"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def format_nodes(self, nodes_data: List[pd.DataFrame], 
                    mapping_name: str = "mapping") -> str:
        """Format nodes data as Cypher CREATE statements."""
        cypher_statements = [
            f"// Neo4j Cypher statements for nodes - {mapping_name}",
            f"// Generated at: {pd.Timestamp.now()}",
            ""
        ]
        
        for node_df in nodes_data:
            if len(node_df) == 0:
                continue
            
            label = node_df.iloc[0].get('_label', 'UnknownNode')
            cypher_statements.append(f"// Creating {label} nodes")
            
            for _, row in node_df.iterrows():
                statement = self._create_node_statement(row, label)
                cypher_statements.append(statement)
            
            cypher_statements.append("")  # Empty line between node types
        
        # Write Cypher file
        filename = f"{mapping_name}_nodes.cypher"
        filepath = self.output_directory / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cypher_statements))
        
        self.logger.info(f"Generated Cypher statements for nodes: {filepath}")
        return str(filepath)
    
    def format_relationships(self, relationships_data: List[pd.DataFrame],
                           mapping_name: str = "mapping") -> str:
        """Format relationships data as Cypher CREATE statements."""
        cypher_statements = [
            f"// Neo4j Cypher statements for relationships - {mapping_name}",
            f"// Generated at: {pd.Timestamp.now()}",
            ""
        ]
        
        for rel_df in relationships_data:
            if len(rel_df) == 0:
                continue
            
            rel_type = rel_df.iloc[0].get('_type', 'UnknownRelationship')
            cypher_statements.append(f"// Creating {rel_type} relationships")
            
            for _, row in rel_df.iterrows():
                statement = self._create_relationship_statement(row, rel_type)
                cypher_statements.append(statement)
            
            cypher_statements.append("")  # Empty line between relationship types
        
        # Write Cypher file
        filename = f"{mapping_name}_relationships.cypher"
        filepath = self.output_directory / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cypher_statements))
        
        self.logger.info(f"Generated Cypher statements for relationships: {filepath}")
        return str(filepath)
    
    def format_complete_script(self, nodes_data: List[pd.DataFrame], 
                             relationships_data: List[pd.DataFrame],
                             mapping_name: str = "mapping") -> str:
        """Format complete graph as a single Cypher script."""
        cypher_statements = [
            f"// Complete Neo4j Cypher script - {mapping_name}",
            f"// Generated at: {pd.Timestamp.now()}",
            "",
            "// Clear existing data (uncomment if needed)",
            "// MATCH (n) DETACH DELETE n;",
            ""
        ]
        
        # Add nodes first
        cypher_statements.append("// ============= NODES =============")
        for node_df in nodes_data:
            if len(node_df) == 0:
                continue
            
            label = node_df.iloc[0].get('_label', 'UnknownNode')
            cypher_statements.append(f"// Creating {label} nodes")
            
            for _, row in node_df.iterrows():
                statement = self._create_node_statement(row, label)
                cypher_statements.append(statement)
            
            cypher_statements.append("")
        
        # Add relationships
        cypher_statements.append("// ============= RELATIONSHIPS =============")
        for rel_df in relationships_data:
            if len(rel_df) == 0:
                continue
            
            rel_type = rel_df.iloc[0].get('_type', 'UnknownRelationship')
            cypher_statements.append(f"// Creating {rel_type} relationships")
            
            for _, row in rel_df.iterrows():
                statement = self._create_relationship_statement(row, rel_type)
                cypher_statements.append(statement)
            
            cypher_statements.append("")
        
        # Add final statistics query
        cypher_statements.extend([
            "// ============= VERIFICATION =============",
            "// Check created nodes and relationships",
            "MATCH (n) RETURN labels(n) as labels, count(n) as count;",
            "MATCH ()-[r]->() RETURN type(r) as relationship_type, count(r) as count;"
        ])
        
        # Write complete script
        filename = f"{mapping_name}_complete.cypher"
        filepath = self.output_directory / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cypher_statements))
        
        self.logger.info(f"Generated complete Cypher script: {filepath}")
        return str(filepath)
    
    def format_batch_script(self, nodes_data: List[pd.DataFrame], 
                          relationships_data: List[pd.DataFrame],
                          mapping_name: str = "mapping",
                          batch_size: int = 1000) -> str:
        """Format graph data as batched Cypher statements for better performance."""
        cypher_statements = [
            f"// Batched Neo4j Cypher script - {mapping_name}",
            f"// Generated at: {pd.Timestamp.now()}",
            f"// Batch size: {batch_size}",
            ""
        ]
        
        # Process nodes in batches
        cypher_statements.append("// ============= BATCHED NODES =============")
        for node_df in nodes_data:
            if len(node_df) == 0:
                continue
            
            label = node_df.iloc[0].get('_label', 'UnknownNode')
            cypher_statements.append(f"// Creating {label} nodes in batches")
            
            # Process in batches
            for i in range(0, len(node_df), batch_size):
                batch_df = node_df.iloc[i:i+batch_size]
                unwind_statement = self._create_batch_node_statement(batch_df, label)
                cypher_statements.append(unwind_statement)
                cypher_statements.append("")
        
        # Process relationships in batches
        cypher_statements.append("// ============= BATCHED RELATIONSHIPS =============")
        for rel_df in relationships_data:
            if len(rel_df) == 0:
                continue
            
            rel_type = rel_df.iloc[0].get('_type', 'UnknownRelationship')
            cypher_statements.append(f"// Creating {rel_type} relationships in batches")
            
            # Process in batches
            for i in range(0, len(rel_df), batch_size):
                batch_df = rel_df.iloc[i:i+batch_size]
                unwind_statement = self._create_batch_relationship_statement(batch_df, rel_type)
                cypher_statements.append(unwind_statement)
                cypher_statements.append("")
        
        # Write batch script
        filename = f"{mapping_name}_batched.cypher"
        filepath = self.output_directory / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cypher_statements))
        
        self.logger.info(f"Generated batched Cypher script: {filepath}")
        return str(filepath)
    
    def _create_node_statement(self, row: pd.Series, label: str) -> str:
        """Create a single node Cypher statement."""
        node_id = row.get('_id', 'unknown')
        properties = {}
        
        # Extract properties (exclude metadata columns)
        for col, value in row.items():
            if not col.startswith('_') and pd.notna(value):
                properties[col] = self._format_cypher_value(value)
        
        # Build properties string
        props_str = ""
        if properties:
            props_list = [f"{key}: {value}" for key, value in properties.items()]
            props_str = f" {{{', '.join(props_list)}}}"
        
        return f"CREATE (n_{self._sanitize_id(node_id)}:{label}{props_str});"
    
    def _create_relationship_statement(self, row: pd.Series, rel_type: str) -> str:
        """Create a single relationship Cypher statement."""
        from_id = row.get('_from_id', 'unknown')
        to_id = row.get('_to_id', 'unknown')
        properties = {}
        
        # Extract properties (exclude metadata columns)
        for col, value in row.items():
            if not col.startswith('_') and pd.notna(value):
                properties[col] = self._format_cypher_value(value)
        
        # Build properties string
        props_str = ""
        if properties:
            props_list = [f"{key}: {value}" for key, value in properties.items()]
            props_str = f" {{{', '.join(props_list)}}}"
        
        return (
            f"MATCH (a), (b) WHERE a._id = {self._format_cypher_value(from_id)} "
            f"AND b._id = {self._format_cypher_value(to_id)} "
            f"CREATE (a)-[r:{rel_type}{props_str}]->(b);"
        )
    
    def _create_batch_node_statement(self, batch_df: pd.DataFrame, label: str) -> str:
        """Create batched node creation using UNWIND."""
        # Prepare data for UNWIND
        batch_data = []
        for _, row in batch_df.iterrows():
            node_data = {'_id': row.get('_id')}
            
            # Add properties
            for col, value in row.items():
                if not col.startswith('_') and pd.notna(value):
                    node_data[col] = value
            
            batch_data.append(node_data)
        
        # Format as Cypher list
        data_str = self._format_cypher_list(batch_data)
        
        return (
            f"UNWIND {data_str} as row\n"
            f"CREATE (n:{label})\n"
            f"SET n = row;"
        )
    
    def _create_batch_relationship_statement(self, batch_df: pd.DataFrame, rel_type: str) -> str:
        """Create batched relationship creation using UNWIND."""
        # Prepare data for UNWIND
        batch_data = []
        for _, row in batch_df.iterrows():
            rel_data = {
                '_from_id': row.get('_from_id'),
                '_to_id': row.get('_to_id')
            }
            
            # Add properties
            for col, value in row.items():
                if not col.startswith('_') and pd.notna(value):
                    rel_data[col] = value
            
            batch_data.append(rel_data)
        
        # Format as Cypher list
        data_str = self._format_cypher_list(batch_data)
        
        return (
            f"UNWIND {data_str} as row\n"
            f"MATCH (a {{_id: row._from_id}}), (b {{_id: row._to_id}})\n"
            f"CREATE (a)-[r:{rel_type}]->(b)\n"
            f"SET r = apoc.map.removeKeys(row, ['_from_id', '_to_id']);"
        )
    
    def _format_cypher_value(self, value: Any) -> str:
        """Format a value for Cypher syntax."""
        if value is None or pd.isna(value):
            return "null"
        elif isinstance(value, str):
            # Escape quotes and format as string
            escaped = value.replace("'", "\\'").replace('"', '\\"')
            return f"'{escaped}'"
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            # Convert to string and format
            escaped = str(value).replace("'", "\\'").replace('"', '\\"')
            return f"'{escaped}'"
    
    def _format_cypher_list(self, data_list: List[Dict[str, Any]]) -> str:
        """Format a list of dictionaries as Cypher list syntax."""
        formatted_items = []
        
        for item in data_list:
            props = []
            for key, value in item.items():
                cypher_value = self._format_cypher_value(value)
                props.append(f"{key}: {cypher_value}")
            
            formatted_items.append(f"{{{', '.join(props)}}}")
        
        return f"[{', '.join(formatted_items)}]"
    
    def _sanitize_id(self, node_id: Any) -> str:
        """Sanitize node ID for use as Cypher variable name."""
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(node_id))
        # Ensure it starts with a letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = f"id_{sanitized}"
        return sanitized or "unknown"
    
    def format_complete_mapping(self, nodes_data: List[pd.DataFrame], 
                              relationships_data: List[pd.DataFrame],
                              mapping_name: str = "mapping") -> Dict[str, Any]:
        """Format complete mapping data to Cypher files."""
        result = {
            'files': {},
            'summary': {
                'total_nodes': sum(len(df) for df in nodes_data),
                'total_relationships': sum(len(df) for df in relationships_data),
                'generated_files': []
            }
        }
        
        # Generate different Cypher formats
        files_generated = []
        
        # Individual nodes and relationships files
        nodes_file = self.format_nodes(nodes_data, mapping_name)
        relationships_file = self.format_relationships(relationships_data, mapping_name)
        files_generated.extend([nodes_file, relationships_file])
        
        # Complete script
        complete_file = self.format_complete_script(nodes_data, relationships_data, mapping_name)
        files_generated.append(complete_file)
        
        # Batched script for better performance
        batched_file = self.format_batch_script(nodes_data, relationships_data, mapping_name)
        files_generated.append(batched_file)
        
        result['files'] = {
            'nodes': nodes_file,
            'relationships': relationships_file,
            'complete_script': complete_file,
            'batched_script': batched_file
        }
        
        result['summary']['generated_files'] = files_generated
        
        return result