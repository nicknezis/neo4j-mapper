# Neo4j Mapper

A configurable Python tool for mapping SQLite data to Neo4j graph format. Transform relational data into nodes and relationships with support for complex JOINs and multiple output formats.

## Features

- 🗃️ **Multiple SQLite Database Support**: Connect to and query multiple SQLite databases
- 🔗 **Complex JOIN Operations**: Support for INNER, LEFT, RIGHT, and FULL OUTER joins
- 📊 **Flexible Configuration**: YAML-based configuration for easy customization
- 🎯 **Multiple Output Formats**: Generate CSV, JSON, and Cypher files
- ⚡ **High Performance**: Efficient data processing for large datasets
- 🔧 **CLI Interface**: Command-line tool for easy automation
- ✅ **Data Validation**: Comprehensive validation and error handling

## Dependencies
- PyYAML: Configuration file parsing
- sqlite3: SQLite database connectivity (built-in)
- pandas: Data manipulation and analysis
- click: CLI framework
- jsonschema: Configuration validation

## Installation

### From Source

```bash
git clone <repository-url>
cd neo4j_mapper
pip install -r requirements.txt
pip install -e .
```

### Using pip (when published)

```bash
pip install neo4j-mapper
```

## Quick Start

1. **Create a configuration file**:

```bash
neo4j-mapper create-config my_config.yaml
```

2. **Edit the configuration** to match your SQLite database structure

3. **Transform your data**:

```bash
neo4j-mapper transform my_config.yaml --format csv --output-dir output
```

## Configuration

The tool uses YAML configuration files to define:

- Database connections
- Table mappings
- JOIN operations
- Node and relationship definitions
- Output formats

### Basic Example

```yaml
databases:
  - path: "data/users.db"
    alias: "users_db"

mappings:
  - name: "user_mapping"
    nodes:
      - label: "User"
        source: "users"
        id_field: "id"
        properties:
          - field: "name"
            type: "string"
          - field: "email"
            type: "string"
```

### Advanced Example with JOINs

```yaml
databases:
  - path: "data/users.db"
    alias: "users_db"
  - path: "data/orders.db"
    alias: "orders_db"

mappings:
  - name: "user_order_mapping"
    joins:
      - type: "INNER"
        left_table: "users_db.users"
        right_table: "orders_db.orders"
        condition: "users.id = orders.user_id"
    
    nodes:
      - label: "User"
        source: "users"
        id_field: "id"
        properties:
          - field: "name"
            type: "string"
      
      - label: "Order"
        source: "orders"
        id_field: "id"
        properties:
          - field: "total"
            type: "float"
    
    relationships:
      - type: "PLACED_ORDER"
        from_node: "User"
        to_node: "Order"
```

## CLI Commands

### Transform Data

```bash
# Transform using CSV format (default)
neo4j-mapper transform config.yaml

# Generate all formats (CSV, JSON, Cypher)
neo4j-mapper transform config.yaml --format all

# Process specific mapping only
neo4j-mapper transform config.yaml --mapping user_mapping

# Validate configuration without processing
neo4j-mapper transform config.yaml --dry-run
```

### Validate Configuration

```bash
neo4j-mapper validate config.yaml
```

### Inspect Database

```bash
# List all tables in database
neo4j-mapper inspect data/users.db

# Inspect specific table
neo4j-mapper inspect data/users.db --table users
```

### Create Sample Configuration

```bash
neo4j-mapper create-config sample_config.yaml
```

## Output Formats

### CSV Format

Generates CSV files compatible with Neo4j's import tools:
- `mapping_name_nodes_label.csv` - Node data
- `mapping_name_relationships_type.csv` - Relationship data
- `mapping_name_import.sh` - Neo4j import script

### JSON Format

Generates JSON files for flexibility:
- `mapping_name_nodes.json` - All nodes
- `mapping_name_relationships.json` - All relationships
- `mapping_name_graph.json` - Complete graph structure
- `mapping_name_cypher_data.json` - Cypher-compatible format

### Cypher Format

Generates Cypher scripts for direct execution:
- `mapping_name_nodes.cypher` - Node creation statements
- `mapping_name_relationships.cypher` - Relationship creation statements
- `mapping_name_complete.cypher` - Complete script
- `mapping_name_batched.cypher` - Batched statements for performance

## Supported Data Types

- `string` - Text data
- `integer` - Whole numbers
- `float` - Decimal numbers
- `boolean` - True/false values
- `datetime` - Date and time
- `date` - Date only
- `time` - Time only

## Configuration Schema

### Database Configuration

```yaml
databases:
  - path: "path/to/database.db"    # SQLite file path
    alias: "db_alias"              # Unique identifier
```

### Node Configuration

```yaml
nodes:
  - label: "NodeLabel"             # Neo4j node label
    source: "table_name"           # Source table
    id_field: "id"                 # Unique identifier field
    properties:                    # Property mappings
      - field: "column_name"       # Source column
        type: "string"             # Target data type
```

### Relationship Configuration

```yaml
relationships:
  - type: "RELATIONSHIP_TYPE"      # Neo4j relationship type
    from_node: "SourceNodeLabel"   # Source node label
    to_node: "TargetNodeLabel"     # Target node label
    properties:                    # Optional properties
      - field: "column_name"
        type: "string"
```

### JOIN Configuration

```yaml
joins:
  - type: "INNER"                  # JOIN type (INNER, LEFT, RIGHT, FULL)
    left_table: "alias.table"      # Left table (alias.table_name)
    right_table: "alias.table"     # Right table (alias.table_name)
    condition: "left.id = right.id"       # JOIN condition
```

## Examples

See the `config/examples/` directory for complete examples:

- `simple_mapping.yaml` - Basic single-table mapping
- `complex_join_mapping.yaml` - Multi-table JOINs with relationships
- `multi_database_mapping.yaml` - Cross-database mapping

## Development

### Setup Development Environment

```bash
git clone <repository-url>
cd neo4j_mapper
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Code Formatting

```bash
black src/
flake8 src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- 📖 Documentation: See configuration examples
- 🐛 Bug Reports: Submit issues on GitHub
- 💡 Feature Requests: Open GitHub discussions
- ❓ Questions: Check existing issues or create new ones