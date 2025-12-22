"""
Math tool package.

Re-exports tool interface: name, description, schema, run, TOOLS_SCHEMA.
"""

from pathlib import Path

from .tool import (
    name,
    description,
    display_description,
    schema,
    run,
    TOOLS_SCHEMA,
)

# Path to configuration file
CONFIG_PATH = Path(__file__).parent / "config.yaml"

__all__ = [
    "name",
    "description",
    "display_description",
    "schema",
    "run",
    "TOOLS_SCHEMA",
    "CONFIG_PATH",
]

