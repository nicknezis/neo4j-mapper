"""Setup script for Neo4j Mapper."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt", "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="neo4j-mapper",
    version="0.1.0",
    author="Neo4j Mapper Team",
    author_email="contact@example.com",
    description="A configurable tool for mapping SQLite data to Neo4j graph format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/neo4j-mapper",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Engineers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
    },
    entry_points={
        "console_scripts": [
            "neo4j-mapper=src.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.md"],
    },
    keywords="neo4j sqlite graph database etl data-transformation",
    project_urls={
        "Bug Reports": "https://github.com/example/neo4j-mapper/issues",
        "Source": "https://github.com/example/neo4j-mapper",
        "Documentation": "https://github.com/example/neo4j-mapper/wiki",
    },
)