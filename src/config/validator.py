"""Configuration validator for Neo4j Mapper."""

from typing import Dict, Any, List
import re


class ConfigValidator:
    """Validates configuration files."""
    
    REQUIRED_FIELDS = {
        'databases': list,
        'mappings': list
    }
    
    DATABASE_REQUIRED_FIELDS = {
        'path': str,
        'alias': str
    }
    
    MAPPING_REQUIRED_FIELDS = {
        'name': str,
        'nodes': list
    }
    
    NODE_REQUIRED_FIELDS = {
        'label': str,
        'source': str,
        'id_field': str,
        'properties': list
    }
    
    PROPERTY_REQUIRED_FIELDS = {
        'field': str,
        'type': str
    }
    
    VALID_TYPES = {
        'string', 'integer', 'float', 'boolean', 'datetime', 'date', 'time'
    }
    
    VALID_JOIN_TYPES = {
        'INNER', 'LEFT', 'RIGHT', 'FULL'
    }
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate complete configuration."""
        self._validate_required_fields(config, self.REQUIRED_FIELDS, "root")
        
        # Validate databases
        for i, db in enumerate(config['databases']):
            self._validate_database(db, f"databases[{i}]")
        
        # Validate mappings
        for i, mapping in enumerate(config['mappings']):
            self._validate_mapping(mapping, f"mappings[{i}]")
        
        # Validate output configuration if present
        if 'output' in config:
            self._validate_output(config['output'])
        
        return True
    
    def _validate_required_fields(self, obj: Dict[str, Any], required: Dict[str, type], context: str):
        """Validate required fields exist and have correct types."""
        for field, expected_type in required.items():
            if field not in obj:
                raise ValueError(f"Missing required field '{field}' in {context}")
            
            if not isinstance(obj[field], expected_type):
                raise ValueError(
                    f"Field '{field}' in {context} must be of type {expected_type.__name__}, "
                    f"got {type(obj[field]).__name__}"
                )
    
    def _validate_database(self, db: Dict[str, Any], context: str):
        """Validate database configuration."""
        self._validate_required_fields(db, self.DATABASE_REQUIRED_FIELDS, context)
        
        # Validate alias format (alphanumeric and underscore only)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', db['alias']):
            raise ValueError(
                f"Database alias '{db['alias']}' in {context} must be alphanumeric with underscores"
            )
    
    def _validate_mapping(self, mapping: Dict[str, Any], context: str):
        """Validate mapping configuration."""
        self._validate_required_fields(mapping, self.MAPPING_REQUIRED_FIELDS, context)
        
        # Validate joins if present
        if 'joins' in mapping:
            for i, join in enumerate(mapping['joins']):
                self._validate_join(join, f"{context}.joins[{i}]")
        
        # Validate nodes
        for i, node in enumerate(mapping['nodes']):
            self._validate_node(node, f"{context}.nodes[{i}]")
        
        # Validate relationships if present
        if 'relationships' in mapping:
            for i, rel in enumerate(mapping['relationships']):
                self._validate_relationship(rel, f"{context}.relationships[{i}]")
    
    def _validate_join(self, join: Dict[str, Any], context: str):
        """Validate join configuration."""
        required_fields = {
            'type': str,
            'left_table': str,
            'right_table': str,
            'on': str
        }
        
        self._validate_required_fields(join, required_fields, context)
        
        if join['type'] not in self.VALID_JOIN_TYPES:
            raise ValueError(
                f"Invalid join type '{join['type']}' in {context}. "
                f"Must be one of: {', '.join(self.VALID_JOIN_TYPES)}"
            )
    
    def _validate_node(self, node: Dict[str, Any], context: str):
        """Validate node configuration."""
        self._validate_required_fields(node, self.NODE_REQUIRED_FIELDS, context)
        
        # Validate properties
        for i, prop in enumerate(node['properties']):
            self._validate_property(prop, f"{context}.properties[{i}]")
    
    def _validate_property(self, prop: Dict[str, Any], context: str):
        """Validate property configuration."""
        self._validate_required_fields(prop, self.PROPERTY_REQUIRED_FIELDS, context)
        
        if prop['type'] not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid property type '{prop['type']}' in {context}. "
                f"Must be one of: {', '.join(self.VALID_TYPES)}"
            )
    
    def _validate_relationship(self, rel: Dict[str, Any], context: str):
        """Validate relationship configuration."""
        required_fields = {
            'type': str,
            'from_node': str,
            'to_node': str
        }
        
        self._validate_required_fields(rel, required_fields, context)
        
        # Validate properties if present
        if 'properties' in rel:
            for i, prop in enumerate(rel['properties']):
                self._validate_property(prop, f"{context}.properties[{i}]")
    
    def _validate_output(self, output: Dict[str, Any]):
        """Validate output configuration."""
        valid_formats = {'csv', 'json', 'cypher'}
        
        if 'format' in output and output['format'] not in valid_formats:
            raise ValueError(
                f"Invalid output format '{output['format']}'. "
                f"Must be one of: {', '.join(valid_formats)}"
            )