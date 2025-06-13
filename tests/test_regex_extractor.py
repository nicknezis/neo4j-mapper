"""Tests for regex extractor functionality."""

import pytest
import pandas as pd
from src.transformers.data_mapper import DataMapper
from src.config.validator import ConfigValidator


class TestRegexExtractor:
    """Test regex extractor functionality in DataMapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_mapper = DataMapper()
        self.validator = ConfigValidator()

    def test_single_group_extraction(self):
        """Test extracting a single group from regex."""
        # Test data with email addresses
        df = pd.DataFrame({
            'email': ['john.doe@example.com', 'jane.smith@test.org', 'invalid-email']
        })
        
        extractor_config = {
            'type': 'regex',
            'pattern': r'([^@]+)@[^@]+',
            'group': 1,
            'fallback_strategy': 'null'
        }
        
        result = self.data_mapper._apply_regex_extractor(
            df['email'], extractor_config, 'email'
        )
        
        expected = pd.Series(['john.doe', 'jane.smith', None])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_multiple_groups_extraction(self):
        """Test extracting multiple groups into separate fields."""
        # Test data with full names
        df = pd.DataFrame({
            'full_name': ['John Doe', 'Jane Smith', 'Single', 'Not-A-Name-Format']
        })
        
        extractor_config = {
            'type': 'regex',
            'pattern': r'^([A-Za-z]+)\s+([A-Za-z]+)$',
            'groups': ['first_name', 'last_name'],
            'fallback_strategy': 'null'
        }
        
        result = self.data_mapper._apply_regex_extractor(
            df['full_name'], extractor_config, 'full_name'
        )
        
        assert isinstance(result, dict)
        assert 'first_name' in result
        assert 'last_name' in result
        
        pd.testing.assert_series_equal(
            result['first_name'], 
            pd.Series(['John', 'Jane', None, None]),
            check_names=False
        )
        pd.testing.assert_series_equal(
            result['last_name'], 
            pd.Series(['Doe', 'Smith', None, None]),
            check_names=False
        )

    def test_named_groups_extraction(self):
        """Test extracting named groups from regex."""
        # Test data with phone numbers
        df = pd.DataFrame({
            'phone': ['(555) 123-4567', '555-123-4567', 'invalid-phone']
        })
        
        extractor_config = {
            'type': 'regex',
            'pattern': r'(?:\((?P<area>\d{3})\)\s*|(?P<area2>\d{3})-)?(?P<exchange>\d{3})-(?P<number>\d{4})',
            'named_groups': True,
            'fallback_strategy': 'null'
        }
        
        result = self.data_mapper._apply_regex_extractor(
            df['phone'], extractor_config, 'phone'
        )
        
        assert isinstance(result, dict)
        # Should have all named groups from the pattern
        expected_groups = ['area', 'area2', 'exchange', 'number']
        for group in expected_groups:
            assert group in result

    def test_fallback_strategies(self):
        """Test different fallback strategies when regex doesn't match."""
        df = pd.DataFrame({
            'data': ['match123', 'nomatch', 'match456']
        })
        
        # Test original fallback
        extractor_config = {
            'type': 'regex',
            'pattern': r'match(\d+)',
            'group': 1,
            'fallback_strategy': 'original'
        }
        
        result = self.data_mapper._apply_regex_extractor(
            df['data'], extractor_config, 'data'
        )
        
        expected = pd.Series(['123', 'nomatch', '456'])
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test null fallback
        extractor_config['fallback_strategy'] = 'null'
        result = self.data_mapper._apply_regex_extractor(
            df['data'], extractor_config, 'data'
        )
        
        expected = pd.Series(['123', None, '456'])
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # Test empty fallback
        extractor_config['fallback_strategy'] = 'empty'
        result = self.data_mapper._apply_regex_extractor(
            df['data'], extractor_config, 'data'
        )
        
        expected = pd.Series(['123', '', '456'])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_default_extraction_with_groups(self):
        """Test default extraction behavior when no extraction mode is specified."""
        df = pd.DataFrame({
            'data': ['prefix-content-suffix', 'no-match', 'another-test-case']
        })
        
        extractor_config = {
            'type': 'regex',
            'pattern': r'(\w+)-(\w+)-(\w+)',
            'fallback_strategy': 'null'
        }
        
        result = self.data_mapper._apply_regex_extractor(
            df['data'], extractor_config, 'data'
        )
        
        # Should extract first group by default
        expected = pd.Series(['prefix', None, 'another'])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_default_extraction_no_groups(self):
        """Test default extraction when no groups in pattern."""
        df = pd.DataFrame({
            'data': ['test123', 'nomatch', 'test456']
        })
        
        extractor_config = {
            'type': 'regex',
            'pattern': r'test\d+',
            'fallback_strategy': 'null'
        }
        
        result = self.data_mapper._apply_regex_extractor(
            df['data'], extractor_config, 'data'
        )
        
        # Should extract full match
        expected = pd.Series(['test123', None, 'test456'])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_null_and_na_handling(self):
        """Test handling of null and NA values."""
        df = pd.DataFrame({
            'data': ['match123', None, pd.NA, 'match456']
        })
        
        extractor_config = {
            'type': 'regex',
            'pattern': r'match(\d+)',
            'group': 1
        }
        
        result = self.data_mapper._apply_regex_extractor(
            df['data'], extractor_config, 'data'
        )
        
        expected = pd.Series(['123', None, None, '456'])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_integration_with_node_properties(self):
        """Test regex extractor integration with node property mapping."""
        # Create test DataFrame
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'full_name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'email': ['john@example.com', 'jane@test.org', 'bob@company.net']
        })
        
        # Node configuration with regex extractors
        node_config = {
            'label': 'Person',
            'source': '',
            'id_field': 'id',
            'properties': [
                {
                    'field': 'full_name',
                    'type': 'string',
                    'extractor': {
                        'type': 'regex',
                        'pattern': r'^([A-Za-z]+)\s+([A-Za-z]+)$',
                        'groups': ['first_name', 'last_name']
                    }
                },
                {
                    'field': 'email',
                    'type': 'string',
                    'extractor': {
                        'type': 'regex',
                        'pattern': r'([^@]+)@[^@]+',
                        'group': 1
                    }
                }
            ]
        }
        
        result_df = self.data_mapper.map_node_properties(df, node_config)
        
        # Check that new fields were created
        assert 'first_name' in result_df.columns
        assert 'last_name' in result_df.columns
        # Email field should be replaced with extracted username
        assert result_df['email'].tolist() == ['john', 'jane', 'bob']
        assert result_df['first_name'].tolist() == ['John', 'Jane', 'Bob']
        assert result_df['last_name'].tolist() == ['Doe', 'Smith', 'Johnson']


class TestRegexExtractorValidation:
    """Test configuration validation for regex extractors."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()

    def test_valid_extractor_configs(self):
        """Test validation of valid extractor configurations."""
        valid_configs = [
            {
                'type': 'regex',
                'pattern': r'(\d+)',
                'group': 1
            },
            {
                'type': 'regex',
                'pattern': r'([A-Za-z]+)\s+([A-Za-z]+)',
                'groups': ['first', 'last']
            },
            {
                'type': 'regex',
                'pattern': r'(?P<name>[A-Za-z]+)',
                'named_groups': True
            },
            {
                'type': 'regex',
                'pattern': r'test(\d+)',
                'group': 1,
                'fallback_strategy': 'null'
            }
        ]
        
        for i, config in enumerate(valid_configs):
            try:
                self.validator._validate_extractor(config, f"test[{i}]")
            except ValueError as e:
                pytest.fail(f"Valid config {i} failed validation: {e}")

    def test_invalid_extractor_configs(self):
        """Test validation of invalid extractor configurations."""
        invalid_configs = [
            # Missing type
            {'pattern': r'(\d+)', 'group': 1},
            # Invalid type
            {'type': 'invalid', 'pattern': r'(\d+)', 'group': 1},
            # Missing pattern
            {'type': 'regex', 'group': 1},
            # Invalid regex pattern
            {'type': 'regex', 'pattern': r'[unclosed', 'group': 1},
            # Multiple extraction modes
            {'type': 'regex', 'pattern': r'(\d+)', 'group': 1, 'groups': ['num']},
            # Invalid group number
            {'type': 'regex', 'pattern': r'(\d+)', 'group': 0},
            # Empty groups list
            {'type': 'regex', 'pattern': r'(\d+)', 'groups': []},
            # Invalid fallback strategy
            {'type': 'regex', 'pattern': r'(\d+)', 'group': 1, 'fallback_strategy': 'invalid'}
        ]
        
        for i, config in enumerate(invalid_configs):
            with pytest.raises(ValueError):
                self.validator._validate_extractor(config, f"test[{i}]")

    def test_property_with_extractor_validation(self):
        """Test validation of properties with extractors."""
        valid_property = {
            'field': 'test_field',
            'type': 'string',
            'extractor': {
                'type': 'regex',
                'pattern': r'(\d+)',
                'group': 1
            }
        }
        
        try:
            self.validator._validate_property(valid_property, "test_property")
        except ValueError as e:
            pytest.fail(f"Valid property with extractor failed validation: {e}")


if __name__ == '__main__':
    pytest.main([__file__])