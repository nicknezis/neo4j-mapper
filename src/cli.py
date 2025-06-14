"""Command-line interface for Neo4j Mapper."""

import click
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from .config import ConfigLoader
from .readers import DataReaderFactory
from .transformers import GraphTransformer
from .formatters import CSVFormatter, JSONFormatter, CypherFormatter


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--quiet", "-q", is_flag=True, help="Enable quiet mode (warnings and errors only)"
)
def cli(verbose: bool, quiet: bool):
    """Neo4j Mapper - Transform SQLite data to Neo4j graph format."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.WARNING)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o", default=None, help="Output directory for generated files (overrides config file setting)"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["csv", "json", "cypher", "all"]),
    default=None,
    help="Output format (overrides config file setting)",
)
@click.option(
    "--mapping", "-m", help="Specific mapping to process (default: all mappings)"
)
@click.option(
    "--dry-run", is_flag=True, help="Validate configuration without processing data"
)
def transform(
    config_file: str, output_dir: str, format: str, mapping: str, dry_run: bool
):
    """Transform SQLite data according to configuration file."""
    logger = logging.getLogger(__name__)

    try:
        # Load and validate configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_file)
        
        # Get output configuration from config file
        output_config = config_loader.get_output_config(config)
        
        # Get output format from config if not specified via CLI
        if format is None:
            format = output_config.get("format", "csv")
            logger.info(f"Using format from config file: {format}")
        else:
            logger.info(f"Using format from CLI: {format}")
            
        # Get output directory from config if not specified via CLI
        if output_dir is None:
            output_dir = output_config.get("directory", "output")
            logger.info(f"Using output directory from config file: {output_dir}")
        else:
            logger.info(f"Using output directory from CLI: {output_dir}")

        logger.info(f"Loaded configuration from: {config_file}")

        if dry_run:
            click.echo("✓ Configuration validation successful")
            _print_config_summary(config)
            return

        # Get mappings
        mappings = config_loader.get_mappings(config)

        # Filter mappings if specified
        if mapping:
            mappings = [m for m in mappings if m["name"] == mapping]
            if not mappings:
                click.echo(
                    f"Error: Mapping '{mapping}' not found in configuration", err=True
                )
                sys.exit(1)

        # Process each mapping
        for mapping_config in mappings:
            _process_mapping(mapping_config, config, output_dir, format)

        click.echo(f"✓ Transformation completed. Output written to: {output_dir}")

    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def validate(config_file: str):
    """Validate configuration file."""
    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_file)

        click.echo("✓ Configuration is valid")
        _print_config_summary(config)

    except Exception as e:
        click.echo(f"✗ Configuration validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--table", "-t", help="Specific table to inspect")
@click.option(
    "--type",
    "-T",
    type=click.Choice(["auto", "sqlite", "csv"]),
    default="auto",
    help="Data source type (auto-detected by default)",
)
def inspect(data_path: str, table: str, type: str):
    """Inspect data source structure (SQLite database or CSV file)."""
    try:
        # Determine data source type
        path_obj = Path(data_path)
        if type == "auto":
            if path_obj.suffix.lower() == ".csv":
                source_type = "csv"
            elif path_obj.suffix.lower() in [".db", ".sqlite", ".sqlite3"]:
                source_type = "sqlite"
            else:
                # Try to determine by attempting to read as SQLite first
                try:
                    import sqlite3

                    conn = sqlite3.connect(data_path)
                    conn.close()
                    source_type = "sqlite"
                except Exception:
                    source_type = "csv"
        else:
            source_type = type

        # Create appropriate configuration
        if source_type == "sqlite":
            config = {"databases": [{"path": data_path, "alias": "temp_db"}]}
        else:  # csv
            config = {"csv_sources": [{"path": data_path, "alias": "temp_csv"}]}

        # Create reader and inspect
        reader = DataReaderFactory.create_reader(config)
        alias = "temp_db" if source_type == "sqlite" else "temp_csv"

        with reader:
            if table:
                # Inspect specific table
                _inspect_table(reader, alias, table)
            else:
                # List all tables
                _inspect_data_source(reader, alias, data_path, source_type)

    except Exception as e:
        click.echo(f"Error inspecting data source: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("output_file", type=click.Path())
def create_config(output_file: str):
    """Create a sample configuration file."""
    sample_config = """# Neo4j Mapper Configuration File
# This is a sample configuration - modify according to your needs

# SQLite databases (choose one or both data source types)
databases:
  - path: "data/users.db"
    alias: "users_db"
  - path: "data/orders.db"
    alias: "orders_db"

# CSV sources (alternative or additional to databases)
csv_sources:
  - path: "data/users.csv"
    alias: "users_csv"
    options:
      delimiter: ","
      encoding: "utf-8"
      header: true
  - path: "data/products.csv"
    alias: "products_csv"
    options:
      delimiter: ","
      encoding: "utf-8"
      header: true

mappings:
  - name: "user_order_mapping"
    # Optional joins between tables/sources (mix of SQLite and CSV)
    joins:
      - type: "INNER"
        left_table: "users_csv.users"
        right_table: "orders_db.orders"
        on: "users.user_id = orders.user_id"
      - type: "LEFT"
        left_table: "orders_db.orders"
        right_table: "products_csv.products"
        on: "orders.product_id = products.product_id"

    # Node definitions
    nodes:
      - label: "User"
        source: "users"
        id_field: "id"
        properties:
          - field: "name"
            type: "string"
          - field: "email"
            type: "string"
          - field: "created_at"
            type: "datetime"

      - label: "Order"
        source: "orders"
        id_field: "id"
        properties:
          - field: "order_number"
            type: "string"
          - field: "total_amount"
            type: "float"
          - field: "order_date"
            type: "datetime"

    # Relationship definitions
    relationships:
      - type: "PLACED_ORDER"
        from_node: "User"
        to_node: "Order"
        properties:
          - field: "order_date"
            type: "datetime"

# Output configuration (optional)
output:
  format: "csv"
  directory: "output"
"""

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(sample_config)

        click.echo(f"✓ Sample configuration created: {output_file}")
        click.echo(
            "Edit this file according to your data sources (SQLite databases and/or CSV files)."
        )

    except Exception as e:
        click.echo(f"Error creating configuration: {e}", err=True)
        sys.exit(1)


def _process_mapping(
    mapping_config: Dict[str, Any], config: Dict[str, Any], output_dir: str, format: str
):
    """Process a single mapping configuration."""
    logger = logging.getLogger(__name__)
    mapping_name = mapping_config["name"]

    logger.info(f"Processing mapping: {mapping_name}")

    # Create appropriate reader using factory
    reader = DataReaderFactory.create_reader(config)

    with reader:
        # Execute joins if specified
        if "joins" in mapping_config and mapping_config["joins"]:
            df = reader.execute_join_query(mapping_config["joins"])
        else:
            # Read from single table (use first node's source)
            first_node = mapping_config["nodes"][0]
            source_parts = first_node["source"].split(".")
            if len(source_parts) == 2:
                alias, table_name = source_parts
            else:
                # Try to find the first available alias
                databases = config.get("databases", [])
                csv_sources = config.get("csv_sources", [])

                if databases:
                    alias = databases[0]["alias"]
                elif csv_sources:
                    alias = csv_sources[0]["alias"]
                else:
                    raise ValueError("No data sources found")

                table_name = first_node["source"]

            df = reader.read_table(alias, table_name)

    logger.info(f"Read {len(df)} rows from database")

    # Transform data
    transformer = GraphTransformer()
    nodes_data, relationships_data = transformer.transform_mapping(df, mapping_config)

    # Validate transformed data
    errors = transformer.validate_graph_data(nodes_data, relationships_data)
    if errors:
        for error in errors:
            logger.warning(f"Validation warning: {error}")

    # Get transformation statistics
    stats = transformer.get_transformation_stats(nodes_data, relationships_data)
    _print_transformation_stats(stats)

    # Format output
    if format == "csv" or format == "all":
        formatter = CSVFormatter(output_dir)
        result = formatter.format_complete_mapping(
            nodes_data, relationships_data, mapping_name
        )
        logger.info(
            f"Generated CSV files: {len(result['nodes']) + len(result['relationships'])} files"
        )

    if format == "json" or format == "all":
        formatter = JSONFormatter(output_dir)
        result = formatter.format_complete_mapping(
            nodes_data, relationships_data, mapping_name
        )
        logger.info(f"Generated JSON files: {len(result['files'])} files")

    if format == "cypher" or format == "all":
        formatter = CypherFormatter(output_dir)
        result = formatter.format_complete_mapping(
            nodes_data, relationships_data, mapping_name
        )
        logger.info(f"Generated Cypher files: {len(result['files'])} files")


def _print_config_summary(config: Dict[str, Any]):
    """Print a summary of the configuration."""
    click.echo("\nConfiguration Summary:")
    click.echo("=" * 50)

    # Data sources summary
    databases = config.get("databases", [])
    csv_sources = config.get("csv_sources", [])

    if databases:
        click.echo(f"Databases: {len(databases)}")
        for db in databases:
            click.echo(f"  - {db['alias']}: {db['path']}")

    if csv_sources:
        click.echo(f"CSV Sources: {len(csv_sources)}")
        for csv_src in csv_sources:
            click.echo(f"  - {csv_src['alias']}: {csv_src['path']}")
            if "options" in csv_src and csv_src["options"]:
                for key, value in csv_src["options"].items():
                    click.echo(f"    {key}: {value}")

    # Mappings summary
    mappings = config.get("mappings", [])
    click.echo(f"\nMappings: {len(mappings)}")
    for mapping in mappings:
        click.echo(f"  - {mapping['name']}")
        click.echo(f"    Nodes: {len(mapping.get('nodes', []))}")
        click.echo(f"    Relationships: {len(mapping.get('relationships', []))}")
        click.echo(f"    Joins: {len(mapping.get('joins', []))}")


def _print_transformation_stats(stats: Dict[str, Any]):
    """Print transformation statistics."""
    click.echo("\nTransformation Statistics:")
    click.echo("=" * 50)

    totals = stats["totals"]
    click.echo(f"Total Nodes: {totals['node_count']} ({totals['node_types']} types)")
    click.echo(
        f"Total Relationships: {totals['relationship_count']} ({totals['relationship_types']} types)"
    )

    # Node details
    if stats["nodes"]:
        click.echo("\nNode Types:")
        for label, info in stats["nodes"].items():
            click.echo(
                f"  - {label}: {info['count']} nodes, {info['properties']} properties"
            )

    # Relationship details
    if stats["relationships"]:
        click.echo("\nRelationship Types:")
        for rel_type, info in stats["relationships"].items():
            click.echo(
                f"  - {rel_type}: {info['count']} relationships, {info['properties']} properties"
            )


def _inspect_data_source(reader, alias: str, data_path: str, source_type: str):
    """Inspect data source structure."""
    source_label = "Database" if source_type == "sqlite" else "CSV File"
    click.echo(f"\n{source_label}: {data_path}")
    click.echo("=" * 50)

    tables = reader.list_tables(alias)
    table_label = "Tables" if source_type == "sqlite" else "Data"
    click.echo(f"{table_label}: {len(tables)}")

    for table in tables:
        row_count = reader.get_row_count(alias, table)
        if source_type == "sqlite":
            click.echo(f"  - {table}: {row_count} rows")
        else:
            click.echo(f"  - {row_count} rows")


def _inspect_table(reader, alias: str, table: str):
    """Inspect specific table structure."""
    click.echo(f"\nTable: {table}")
    click.echo("=" * 50)

    # Get table info
    columns = reader.get_table_info(alias, table)
    row_count = reader.get_row_count(alias, table)

    click.echo(f"Rows: {row_count}")
    click.echo(f"Columns: {len(columns)}")

    click.echo("\nColumn Details:")
    for col in columns:
        pk_indicator = " (PK)" if col["pk"] else ""
        null_indicator = " NOT NULL" if col["notnull"] else ""
        click.echo(f"  - {col['name']}: {col['type']}{pk_indicator}{null_indicator}")


if __name__ == "__main__":
    cli()
