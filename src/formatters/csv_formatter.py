"""CSV formatter for Neo4j import."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging


class CSVFormatter:
    """Formats graph data as CSV files for Neo4j import."""
    
    def __init__(self, output_directory: str = "output"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def format_nodes(self, nodes_data: List[pd.DataFrame], 
                    mapping_name: str = "mapping") -> Dict[str, str]:
        """Format nodes data as CSV files."""
        output_files = {}
        
        for node_df in nodes_data:
            if len(node_df) == 0:
                continue
            
            # Get node label
            label = node_df.iloc[0]['_label'] if '_label' in node_df.columns else 'UnknownNode'
            
            # Prepare CSV data
            csv_df = self._prepare_node_csv(node_df)
            
            # Generate filename
            filename = f"{mapping_name}_nodes_{label.lower()}.csv"
            filepath = self.output_directory / filename
            
            # Write CSV
            csv_df.to_csv(filepath, index=False, encoding='utf-8')
            output_files[label] = str(filepath)
            
            self.logger.info(f"Wrote {len(csv_df)} {label} nodes to {filepath}")
        
        return output_files
    
    def format_relationships(self, relationships_data: List[pd.DataFrame],
                           mapping_name: str = "mapping") -> Dict[str, str]:
        """Format relationships data as CSV files."""
        output_files = {}
        
        for rel_df in relationships_data:
            if len(rel_df) == 0:
                continue
            
            # Get relationship type
            rel_type = rel_df.iloc[0]['_type'] if '_type' in rel_df.columns else 'UnknownRelationship'
            
            # Prepare CSV data
            csv_df = self._prepare_relationship_csv(rel_df)
            
            # Generate filename
            filename = f"{mapping_name}_relationships_{rel_type.lower()}.csv"
            filepath = self.output_directory / filename
            
            # Write CSV
            csv_df.to_csv(filepath, index=False, encoding='utf-8')
            output_files[rel_type] = str(filepath)
            
            self.logger.info(f"Wrote {len(csv_df)} {rel_type} relationships to {filepath}")
        
        return output_files
    
    def _prepare_node_csv(self, node_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare node DataFrame for CSV export."""
        csv_df = node_df.copy()
        
        # Rename ID column for Neo4j import
        if '_id' in csv_df.columns:
            csv_df = csv_df.rename(columns={'_id': ':ID'})
        
        # Add label column for Neo4j import
        if '_label' in csv_df.columns:
            label = csv_df.iloc[0]['_label']
            csv_df[':LABEL'] = label
            csv_df = csv_df.drop(columns=['_label'])
        
        # Handle data types and null values
        csv_df = self._clean_csv_data(csv_df)
        
        # Reorder columns (ID and LABEL first)
        columns = []
        if ':ID' in csv_df.columns:
            columns.append(':ID')
        if ':LABEL' in csv_df.columns:
            columns.append(':LABEL')
        
        # Add remaining columns
        remaining_cols = [col for col in csv_df.columns if col not in columns]
        columns.extend(remaining_cols)
        
        return csv_df[columns]
    
    def _prepare_relationship_csv(self, rel_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare relationship DataFrame for CSV export."""
        csv_df = rel_df.copy()
        
        # Rename columns for Neo4j import
        column_mapping = {
            '_from_id': ':START_ID',
            '_to_id': ':END_ID',
            '_type': ':TYPE'
        }
        
        csv_df = csv_df.rename(columns=column_mapping)
        
        # Handle data types and null values
        csv_df = self._clean_csv_data(csv_df)
        
        # Reorder columns (relationship metadata first)
        columns = []
        for col in [':START_ID', ':END_ID', ':TYPE']:
            if col in csv_df.columns:
                columns.append(col)
        
        # Add remaining columns (properties)
        remaining_cols = [col for col in csv_df.columns if col not in columns]
        columns.extend(remaining_cols)
        
        return csv_df[columns]
    
    def _clean_csv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data for CSV export."""
        cleaned_df = df.copy()
        
        # Handle different data types
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                # Convert to string and handle nulls
                cleaned_df[col] = cleaned_df[col].astype(str)
                cleaned_df[col] = cleaned_df[col].replace('nan', '')
                cleaned_df[col] = cleaned_df[col].replace('None', '')
            
            elif cleaned_df[col].dtype in ['int64', 'float64']:
                # Handle numeric nulls
                cleaned_df[col] = cleaned_df[col].fillna('')
            
            elif 'datetime' in str(cleaned_df[col].dtype):
                # Format datetime columns
                cleaned_df[col] = cleaned_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                cleaned_df[col] = cleaned_df[col].fillna('')
        
        return cleaned_df
    
    def generate_import_script(self, node_files: Dict[str, str], 
                             relationship_files: Dict[str, str],
                             mapping_name: str = "mapping") -> str:
        """Generate Neo4j import script for the CSV files."""
        script_lines = [
            "#!/bin/bash",
            "# Neo4j CSV Import Script",
            f"# Generated for mapping: {mapping_name}",
            "",
            "# Set Neo4j import tool path",
            "NEO4J_IMPORT_TOOL=\"neo4j-admin database import full\"",
            "",
            "# Import command",
            "$NEO4J_IMPORT_TOOL \\"
        ]
        
        # Add node files
        for label, filepath in node_files.items():
            filename = Path(filepath).name
            script_lines.append(f"  --nodes={label}=\"{filename}\" \\")
        
        # Add relationship files
        for rel_type, filepath in relationship_files.items():
            filename = Path(filepath).name
            script_lines.append(f"  --relationships=\"{filename}\" \\")
        
        # Remove last backslash and add database name
        script_lines[-1] = script_lines[-1].rstrip(' \\')
        script_lines.append("  --database=neo4j")
        
        script_content = "\n".join(script_lines)
        
        # Write script file
        script_path = self.output_directory / f"{mapping_name}_import.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Make script executable
        script_path.chmod(0o755)
        
        self.logger.info(f"Generated import script: {script_path}")
        return str(script_path)
    
    def format_complete_mapping(self, nodes_data: List[pd.DataFrame], 
                              relationships_data: List[pd.DataFrame],
                              mapping_name: str = "mapping") -> Dict[str, Any]:
        """Format complete mapping data to CSV files."""
        result = {
            'nodes': {},
            'relationships': {},
            'import_script': None,
            'summary': {
                'total_nodes': 0,
                'total_relationships': 0,
                'node_files': 0,
                'relationship_files': 0
            }
        }
        
        # Format nodes
        node_files = self.format_nodes(nodes_data, mapping_name)
        result['nodes'] = node_files
        result['summary']['node_files'] = len(node_files)
        result['summary']['total_nodes'] = sum(len(df) for df in nodes_data)
        
        # Format relationships
        relationship_files = self.format_relationships(relationships_data, mapping_name)
        result['relationships'] = relationship_files
        result['summary']['relationship_files'] = len(relationship_files)
        result['summary']['total_relationships'] = sum(len(df) for df in relationships_data)
        
        # Generate import script
        if node_files or relationship_files:
            script_path = self.generate_import_script(node_files, relationship_files, mapping_name)
            result['import_script'] = script_path
        
        return result