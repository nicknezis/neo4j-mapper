"""Configuration loader for Neo4j Mapper."""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from .validator import ConfigValidator


class ConfigLoader:
    """Loads and validates configuration files."""

    def __init__(self):
        self.validator = ConfigValidator()

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if not config_file.suffix.lower() in [".yaml", ".yml"]:
            raise ValueError(f"Configuration file must be YAML format: {config_path}")

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")

        if not config:
            raise ValueError(f"Empty configuration file: {config_path}")

        # Validate configuration
        self.validator.validate(config)

        return config

    def get_databases(self, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract database configurations."""
        return config.get("databases", [])

    def get_csv_sources(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract CSV source configurations."""
        return config.get("csv_sources", [])

    def get_mappings(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract mapping configurations."""
        return config.get("mappings", [])

    def get_output_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract output configuration."""
        return config.get("output", {"format": "csv", "directory": "output"})
