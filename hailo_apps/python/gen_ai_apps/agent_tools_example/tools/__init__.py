"""
Tools package.

Provides automatic discovery of tools from subdirectories.
Each tool is a self-contained package in tools/<tool_name>/.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def discover_tool_packages(tools_dir: Optional[Path] = None) -> List[ModuleType]:
    """
    Discover tool packages from subdirectories in the tools folder.

    Each tool is expected to be a package (folder with __init__.py) that
    exports: name, description, schema, run, TOOLS_SCHEMA.

    Args:
        tools_dir: Directory containing tool packages. Defaults to this directory.

    Returns:
        List of imported tool modules.
    """
    modules: List[ModuleType] = []

    if tools_dir is None:
        tools_dir = Path(__file__).parent

    logger.debug("Scanning tools directory: %s", tools_dir)

    for item in sorted(tools_dir.iterdir()):
        # Skip non-directories, hidden dirs, and special dirs
        if not item.is_dir():
            continue
        if item.name.startswith("_") or item.name.startswith("."):
            continue
        if item.name == "__pycache__":
            continue

        # Check if it's a valid Python package (has __init__.py or tool.py)
        init_file = item / "__init__.py"
        tool_file = item / "tool.py"

        if not init_file.exists() and not tool_file.exists():
            logger.debug("Skipping non-package directory: %s", item.name)
            continue

        # Try to import the package
        try:
            # Determine import path based on how we're running
            # Use relative import from this package
            package_name = f".{item.name}"
            module = importlib.import_module(package_name, package=__package__)
            modules.append(module)
            logger.debug("Loaded tool package: %s", item.name)
        except ImportError as e:
            logger.warning("Failed to import tool package %s: %s", item.name, e)
            continue
        except Exception as e:
            logger.error("Error loading tool package %s: %s", item.name, e)
            continue

    return modules


def collect_tools_from_packages(modules: List[ModuleType]) -> List[Dict[str, Any]]:
    """
    Collect tool metadata from tool package modules.

    Args:
        modules: List of tool package modules.

    Returns:
        List of tool metadata dictionaries.
    """
    tools: List[Dict[str, Any]] = []
    seen_names: set[str] = set()

    for module in modules:
        module_name = getattr(module, "__name__", "unknown")

        # Check for required attributes
        run_fn = getattr(module, "run", None)
        if not callable(run_fn):
            logger.warning("Tool module %s missing 'run' function", module_name)
            continue

        tool_schemas = getattr(module, "TOOLS_SCHEMA", None)
        if not tool_schemas or not isinstance(tool_schemas, list):
            logger.warning("Tool module %s missing valid TOOLS_SCHEMA", module_name)
            continue

        # Get optional attributes
        display_description = getattr(module, "display_description", None)
        config_path = getattr(module, "CONFIG_PATH", None)

        # Parse TOOLS_SCHEMA
        for entry in tool_schemas:
            if not isinstance(entry, dict) or entry.get("type") != "function":
                continue

            function_def = entry.get("function", {})
            name = function_def.get("name")
            description = function_def.get("description", "")

            if not name:
                logger.warning("Tool in %s has no name", module_name)
                continue

            if name in seen_names:
                logger.warning("Duplicate tool name: %s", name)
                continue

            seen_names.add(name)

            tools.append({
                "name": str(name),
                "display_description": str(display_description or description or name),
                "llm_description": str(description),
                "tool_def": entry,
                "runner": run_fn,
                "module": module,
                "config_path": config_path,
            })

    tools.sort(key=lambda t: t["name"])
    return tools

