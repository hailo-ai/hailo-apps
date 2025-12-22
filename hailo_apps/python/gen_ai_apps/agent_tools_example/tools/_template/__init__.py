"""
Template tool package.

Copy this entire directory to create a new tool:
  cp -r tools/_template tools/my_new_tool

Then:
1. Rename and update tool.py with your logic
2. Update config.yaml with prompts and examples
3. Remove this docstring and update __init__.py exports
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

