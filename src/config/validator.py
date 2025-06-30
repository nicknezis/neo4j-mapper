"""Configuration validator for Neo4j Mapper."""

from typing import Dict, Any, List
import re


class ConfigValidator:
    """Validates configuration files."""

    REQUIRED_FIELDS = {"mappings": list}

    DATABASE_REQUIRED_FIELDS = {"path": str, "alias": str}

    CSV_SOURCE_REQUIRED_FIELDS = {"path": str, "alias": str}

    MAPPING_REQUIRED_FIELDS = {"name": str, "nodes": list}

    NODE_REQUIRED_FIELDS = {
        "label": str,
        "source": str,
        "id_field": str,
        "properties": list,
    }

    PROPERTY_REQUIRED_FIELDS = {"field": str, "type": str}

    VALID_TYPES = {"string", "integer", "float", "boolean", "datetime", "date", "time"}

    VALID_JOIN_TYPES = {"INNER", "LEFT", "RIGHT", "FULL"}

    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate complete configuration."""
        self._validate_required_fields(config, self.REQUIRED_FIELDS, "root")

        # Validate that at least one data source is present
        has_databases = "databases" in config and config["databases"]
        has_csv_sources = "csv_sources" in config and config["csv_sources"]

        if not has_databases and not has_csv_sources:
            raise ValueError(
                "Configuration must include either 'databases' or 'csv_sources' (or both)"
            )

        # Validate databases if present
        if has_databases:
            for i, db in enumerate(config["databases"]):
                self._validate_database(db, f"databases[{i}]")

        # Validate CSV sources if present
        if has_csv_sources:
            for i, csv_source in enumerate(config["csv_sources"]):
                self._validate_csv_source(csv_source, f"csv_sources[{i}]")

        # Validate mappings
        for i, mapping in enumerate(config["mappings"]):
            self._validate_mapping(mapping, f"mappings[{i}]")

        # Validate output configuration if present
        if "output" in config:
            self._validate_output(config["output"])

        return True

    def _validate_required_fields(
        self, obj: Dict[str, Any], required: Dict[str, type], context: str
    ):
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
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", db["alias"]):
            raise ValueError(
                f"Database alias '{db['alias']}' in {context} must be alphanumeric with underscores"
            )

    def _validate_csv_source(self, csv_source: Dict[str, Any], context: str):
        """Validate CSV source configuration."""
        self._validate_required_fields(
            csv_source, self.CSV_SOURCE_REQUIRED_FIELDS, context
        )

        # Validate alias format (same as database aliases)
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", csv_source["alias"]):
            raise ValueError(
                f"CSV source alias '{csv_source['alias']}' in {context} must be alphanumeric with underscores"
            )

        # Validate options if present
        if "options" in csv_source:
            self._validate_csv_options(csv_source["options"], f"{context}.options")

    def _validate_csv_options(self, options: Dict[str, Any], context: str):
        """Validate CSV options configuration."""
        valid_options = {
            "delimiter": str,
            "encoding": str,
            "header": (int, bool),
            "skiprows": int,
            "nrows": int,
            "na_values": list,
            "dtype": dict,
        }

        for option, value in options.items():
            if option not in valid_options:
                raise ValueError(f"Unknown CSV option '{option}' in {context}")

            expected_type = valid_options[option]
            if isinstance(expected_type, tuple):
                if not isinstance(value, expected_type):
                    type_names = " or ".join(t.__name__ for t in expected_type)
                    raise ValueError(
                        f"CSV option '{option}' in {context} must be of type {type_names}, "
                        f"got {type(value).__name__}"
                    )
            else:
                if not isinstance(value, expected_type):
                    raise ValueError(
                        f"CSV option '{option}' in {context} must be of type {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )

    def _validate_mapping(self, mapping: Dict[str, Any], context: str):
        """Validate mapping configuration."""
        self._validate_required_fields(mapping, self.MAPPING_REQUIRED_FIELDS, context)

        # Validate joins if present
        if "joins" in mapping:
            for i, join in enumerate(mapping["joins"]):
                self._validate_join(join, f"{context}.joins[{i}]")

        # Validate nodes
        for i, node in enumerate(mapping["nodes"]):
            self._validate_node(node, f"{context}.nodes[{i}]")

        # Validate relationships if present
        if "relationships" in mapping:
            for i, rel in enumerate(mapping["relationships"]):
                self._validate_relationship(rel, f"{context}.relationships[{i}]")

    def _validate_join(self, join: Dict[str, Any], context: str):
        """Validate join configuration."""
        required_fields = {
            "type": str,
            "left_table": str,
            "right_table": str,
            "condition": str,
        }

        self._validate_required_fields(join, required_fields, context)

        if join["type"] not in self.VALID_JOIN_TYPES:
            raise ValueError(
                f"Invalid join type '{join['type']}' in {context}. "
                f"Must be one of: {', '.join(self.VALID_JOIN_TYPES)}"
            )

    def _validate_node(self, node: Dict[str, Any], context: str):
        """Validate node configuration."""
        self._validate_required_fields(node, self.NODE_REQUIRED_FIELDS, context)

        # Validate properties
        for i, prop in enumerate(node["properties"]):
            self._validate_property(prop, f"{context}.properties[{i}]")

        # Validate WHERE clause if present
        if "where" in node:
            self._validate_where_clause(node["where"], f"{context}.where")

    def _validate_property(self, prop: Dict[str, Any], context: str):
        """Validate property configuration."""
        self._validate_required_fields(prop, self.PROPERTY_REQUIRED_FIELDS, context)

        if prop["type"] not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid property type '{prop['type']}' in {context}. "
                f"Must be one of: {', '.join(self.VALID_TYPES)}"
            )

        # Validate name field if present (optional Neo4j property name)
        if "name" in prop:
            self._validate_property_name(prop["name"], f"{context}.name")

        # Validate extractor configuration if present
        if "extractor" in prop:
            self._validate_extractor(prop["extractor"], f"{context}.extractor")

    def _validate_relationship(self, rel: Dict[str, Any], context: str):
        """Validate relationship configuration."""
        # All fields are now required for relationships
        required_fields = {
            "type": str,
            "source": str,
            "from_id_column": str,
            "to_id_column": str,
            "from_node": str,
            "to_node": str,
        }

        self._validate_required_fields(rel, required_fields, context)

        # Validate properties if present
        if "properties" in rel:
            for i, prop in enumerate(rel["properties"]):
                self._validate_property(prop, f"{context}.properties[{i}]")

        # Validate WHERE clause if present
        if "where" in rel:
            self._validate_where_clause(rel["where"], f"{context}.where")

    def _validate_extractor(self, extractor: Dict[str, Any], context: str):
        """Validate extractor configuration."""
        if not isinstance(extractor, dict):
            raise ValueError(f"Extractor in {context} must be a dictionary")

        # Validate extractor type
        if "type" not in extractor:
            raise ValueError(f"Missing required field 'type' in {context}")

        if extractor["type"] != "regex":
            raise ValueError(
                f"Unsupported extractor type '{extractor['type']}' in {context}. Only 'regex' is supported"
            )

        # Validate regex pattern
        if "pattern" not in extractor:
            raise ValueError(f"Missing required field 'pattern' in {context}")

        if not isinstance(extractor["pattern"], str):
            raise ValueError(f"Field 'pattern' in {context} must be a string")

        # Validate regex pattern is compilable
        try:
            re.compile(extractor["pattern"])
        except re.error as e:
            raise ValueError(f"Invalid regex pattern in {context}: {e}")

        # Validate extraction mode configuration
        extraction_modes = ["groups", "group", "named_groups"]
        specified_modes = [mode for mode in extraction_modes if mode in extractor]

        if len(specified_modes) > 1:
            raise ValueError(
                f"Only one extraction mode can be specified in {context}. "
                f"Found: {', '.join(specified_modes)}"
            )

        # Validate specific extraction modes
        if "groups" in extractor:
            if not isinstance(extractor["groups"], list):
                raise ValueError(f"Field 'groups' in {context} must be a list")
            if not extractor["groups"]:
                raise ValueError(f"Field 'groups' in {context} cannot be empty")
            for i, group_name in enumerate(extractor["groups"]):
                if not isinstance(group_name, str):
                    raise ValueError(
                        f"Group name at index {i} in {context} must be a string"
                    )

        if "group" in extractor:
            if not isinstance(extractor["group"], int):
                raise ValueError(f"Field 'group' in {context} must be an integer")
            if extractor["group"] < 1:
                raise ValueError(f"Field 'group' in {context} must be >= 1")

        if "named_groups" in extractor:
            if not isinstance(extractor["named_groups"], bool):
                raise ValueError(f"Field 'named_groups' in {context} must be a boolean")

        # Validate fallback strategy if present
        if "fallback_strategy" in extractor:
            valid_strategies = {"original", "null", "empty"}
            if extractor["fallback_strategy"] not in valid_strategies:
                raise ValueError(
                    f"Invalid fallback_strategy '{extractor['fallback_strategy']}' in {context}. "
                    f"Must be one of: {', '.join(valid_strategies)}"
                )

    def _validate_where_clause(self, where_clause: str, context: str):
        """Validate WHERE clause configuration."""
        if not isinstance(where_clause, str):
            raise ValueError(f"WHERE clause in {context} must be a string")

        if not where_clause.strip():
            raise ValueError(f"WHERE clause in {context} cannot be empty")

        # Basic syntax validation - check for balanced parentheses
        paren_count = 0
        for char in where_clause:
            if char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
                if paren_count < 0:
                    raise ValueError(
                        f"Unmatched closing parenthesis in WHERE clause: {context}"
                    )

        if paren_count != 0:
            raise ValueError(f"Unmatched parentheses in WHERE clause: {context}")

        # Check for potentially dangerous SQL keywords
        dangerous_keywords = [
            "drop",
            "delete",
            "insert",
            "update",
            "create",
            "alter",
            "truncate",
        ]
        where_lower = where_clause.lower()
        for keyword in dangerous_keywords:
            if f" {keyword} " in f" {where_lower} ":
                raise ValueError(
                    f"Potentially dangerous SQL keyword '{keyword}' found in WHERE clause: {context}"
                )

    def _validate_property_name(self, name: str, context: str):
        """Validate Neo4j property name field."""
        if not isinstance(name, str):
            raise ValueError(f"Property name in {context} must be a string")

        if not name.strip():
            raise ValueError(f"Property name in {context} cannot be empty")

        # Check for valid Neo4j property name format
        # Allow alphanumeric, underscore, and camelCase
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(
                f"Invalid property name '{name}' in {context}. "
                f"Property names must start with a letter or underscore and contain only "
                f"alphanumeric characters and underscores"
            )

        # Check for Neo4j reserved words (common ones)
        neo4j_reserved = {
            "id",
            "type",
            "start",
            "end",
            "length",
            "rank",
            "nodes",
            "relationships",
            "path",
            "shortestpath",
            "allshortestpaths",
            "extract",
            "filter",
            "reduce",
            "any",
            "all",
            "none",
            "single",
            "exists",
            "size",
            "head",
            "tail",
            "last",
            "labels",
            "keys",
            "properties",
            "distinct",
            "count",
            "sum",
            "avg",
            "min",
            "max",
        }

        if name.lower() in neo4j_reserved:
            raise ValueError(
                f"Property name '{name}' in {context} conflicts with Neo4j reserved word. "
                f"Consider using a different name like '{name}_value' or '{name}Property'"
            )

    def _validate_output(self, output: Dict[str, Any]):
        """Validate output configuration."""
        valid_formats = {"csv", "json", "cypher"}

        if "format" in output and output["format"] not in valid_formats:
            raise ValueError(
                f"Invalid output format '{output['format']}'. "
                f"Must be one of: {', '.join(valid_formats)}"
            )
