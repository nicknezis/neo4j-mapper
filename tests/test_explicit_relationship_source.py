"""Tests for explicit relationship source configuration."""

import pytest
import pandas as pd
from unittest.mock import Mock
from src.transformers.data_mapper import DataMapper
from src.config.validator import ConfigValidator


class TestExplicitRelationshipSource:
    """Test explicit source configuration for relationships."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_mapper = DataMapper()
        self.validator = ConfigValidator()

    def test_validate_explicit_relationship_config(self):
        """Test validation of explicit relationship source configuration."""
        # Valid configuration with explicit source
        valid_config = {
            "type": "REPORTS_TO",
            "source": "employees",
            "from_id_column": "employee_id",
            "to_id_column": "manager_id",
            "from_node": "Employee",
            "to_node": "Manager",
        }
        
        # Should not raise any errors
        self.validator._validate_relationship(valid_config, "test_relationship")

    def test_validate_partial_explicit_config_fails(self):
        """Test that partial explicit configuration fails validation."""
        # Missing required fields
        invalid_configs = [
            {
                "type": "REPORTS_TO",
                "source": "employees",  # Missing from_id_column and to_id_column
                "from_node": "Employee",
                "to_node": "Manager",
            },
            {
                "type": "REPORTS_TO",
                "from_id_column": "employee_id",  # Missing source and to_id_column
                "from_node": "Employee",
                "to_node": "Manager",
            },
            {
                "type": "REPORTS_TO",
                "to_id_column": "manager_id",  # Missing source and from_id_column
                "from_node": "Employee",
                "to_node": "Manager",
            },
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError, match="Missing required field"):
                self.validator._validate_relationship(config, "test_relationship")

    def test_validate_missing_explicit_source_fails(self):
        """Test that configuration without explicit source fails validation."""
        invalid_config = {
            "type": "REPORTS_TO",
            "from_node": "Employee",
            "to_node": "Manager",
            "properties": [
                {"field": "start_date", "type": "date"}
            ]
        }
        
        # Should raise error due to missing required fields
        with pytest.raises(ValueError, match="Missing required field"):
            self.validator._validate_relationship(invalid_config, "test_relationship")

    def test_extract_relationship_with_explicit_source(self):
        """Test relationship extraction using explicit source configuration."""
        # Create test DataFrame
        df = pd.DataFrame({
            "employee_id": [1, 2, 3, 4],
            "manager_id": [None, 1, 1, 2],
            "employee_name": ["Alice", "Bob", "Charlie", "David"],
            "start_date": ["2020-01-01", "2021-01-01", "2021-06-01", "2022-01-01"],
        })
        
        # Explicit source configuration
        rel_config = {
            "type": "REPORTS_TO",
            "source": "employees",
            "from_id_column": "employee_id",
            "to_id_column": "manager_id",
            "from_node": "Employee",
            "to_node": "Manager",
            "properties": [
                {"field": "start_date", "type": "date"}
            ]
        }
        
        # Mock node configs (not needed with explicit source)
        node_configs = []
        
        # Extract relationships
        rel_df = self.data_mapper.extract_relationship_data(df, rel_config, node_configs)
        
        # Verify results
        assert len(rel_df) == 3  # Only 3 valid relationships (Alice has no manager)
        assert "_type" in rel_df.columns
        assert "_from_id" in rel_df.columns
        assert "_to_id" in rel_df.columns
        assert "start_date" in rel_df.columns
        
        # Check relationship values
        assert rel_df["_type"].iloc[0] == "REPORTS_TO"
        assert rel_df["_from_id"].iloc[0] == 2  # Bob reports to Alice
        assert rel_df["_to_id"].iloc[0] == 1

    def test_extract_relationship_with_prefixed_columns(self):
        """Test relationship extraction with database-prefixed columns."""
        # Create test DataFrame with prefixed columns (simulating JOIN result)
        df = pd.DataFrame({
            "hr_employee_id": [1, 2, 3, 4],
            "hr_manager_id": [None, 1, 1, 2],
            "hr_employee_name": ["Alice", "Bob", "Charlie", "David"],
            "hr_start_date": ["2020-01-01", "2021-01-01", "2021-06-01", "2022-01-01"],
        })
        
        # Explicit source configuration with database alias
        rel_config = {
            "type": "REPORTS_TO",
            "source": "hr.employees",
            "from_id_column": "employee_id",
            "to_id_column": "manager_id",
            "from_node": "Employee",
            "to_node": "Manager",
            "properties": [
                {"field": "start_date", "name": "since", "type": "date"}
            ]
        }
        
        # Extract relationships
        rel_df = self.data_mapper.extract_relationship_data(df, rel_config, [])
        
        # Verify column resolution worked correctly
        assert len(rel_df) == 3
        assert "since" in rel_df.columns  # Property name mapping
        assert rel_df["_from_id"].iloc[0] == 2
        assert rel_df["_to_id"].iloc[0] == 1

    def test_junction_table_relationships(self):
        """Test relationships from junction/lookup tables."""
        # Create junction table DataFrame
        df = pd.DataFrame({
            "assignment_id": [1, 2, 3, 4, 5],
            "employee_id": [1, 1, 2, 2, 3],
            "project_id": [101, 102, 101, 103, 102],
            "role": ["Lead", "Contributor", "Contributor", "Lead", "Reviewer"],
            "allocation": [100, 50, 50, 100, 25],
        })
        
        # Junction table configuration
        rel_config = {
            "type": "ASSIGNED_TO",
            "source": "assignments",
            "from_id_column": "employee_id",
            "to_id_column": "project_id",
            "from_node": "Employee",
            "to_node": "Project",
            "properties": [
                {"field": "role", "type": "string"},
                {"field": "allocation", "type": "float"}
            ]
        }
        
        # Extract relationships
        rel_df = self.data_mapper.extract_relationship_data(df, rel_config, [])
        
        # Verify all relationships are extracted
        assert len(rel_df) == 5
        assert "role" in rel_df.columns
        assert "allocation" in rel_df.columns
        assert rel_df["_type"].iloc[0] == "ASSIGNED_TO"
        
        # Check specific relationship
        first_rel = rel_df.iloc[0]
        assert first_rel["_from_id"] == 1
        assert first_rel["_to_id"] == 101
        assert first_rel["role"] == "Lead"
        assert first_rel["allocation"] == 100

    def test_self_referencing_relationships(self):
        """Test that self-referencing relationships work with explicit source."""
        # Create test DataFrame
        df = pd.DataFrame({
            "employee_id": [1, 2, 3, 4],
            "manager_id": [None, 1, 1, 2],
            "employee_name": ["Alice", "Bob", "Charlie", "David"],
        })
        
        # Self-referencing with explicit source - straightforward
        explicit_config = {
            "type": "REPORTS_TO",
            "source": "employees",
            "from_id_column": "employee_id",
            "to_id_column": "manager_id",
            "from_node": "Employee",
            "to_node": "Employee",  # Same node type, different role
        }
        
        explicit_rel_df = self.data_mapper.extract_relationship_data(
            df, explicit_config, []
        )
        
        # Verify explicit approach works correctly
        assert len(explicit_rel_df) == 3  # 3 valid manager relationships
        assert explicit_rel_df["_from_id"].tolist() == [2, 3, 4]
        assert explicit_rel_df["_to_id"].tolist() == [1, 1, 2]