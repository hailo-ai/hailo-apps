"""
Tool discovery module.

Handles automatic discovery and collection of tool modules.
Tools are expected to be packages in tools/<name>/ with either:
- __init__.py that exports the tool interface, or
- tool.py with the tool implementation
"""

import importlib
import importlib.util
import logging
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, List, Dict, Optional

# Setup logger
logger = logging.getLogger(__name__)


def discover_tool_modules(tool_dir: Optional[Path] = None) -> List[ModuleType]:
    """
    Discover tool packages from the tools/ subdirectory.

    Each tool is expected to be a package (folder) in tools/<tool_name>/
    with either __init__.py or tool.py that exports the tool interface.

    Args:
        tool_dir: Directory containing the tools/ subdirectory.
                  If None, searches the current directory.

    Returns:
        List of imported tool modules.
    """
    modules: List[ModuleType] = []

    if tool_dir is None:
        target_dir = Path(__file__).parent
    else:
        target_dir = tool_dir

    # Ensure target directory is in sys.path
    if str(target_dir) not in sys.path:
        sys.path.insert(0, str(target_dir))

    logger.debug("Scanning for tools in: %s", target_dir)

    # Scan for tool packages in tools/ subdirectory
    tools_subdir = target_dir / "tools"
    if tools_subdir.is_dir():
        modules.extend(_discover_tool_packages(tools_subdir))
    else:
        logger.warning("No tools/ directory found in: %s", target_dir)

    return modules


def _discover_tool_packages(tools_dir: Path) -> List[ModuleType]:
    """
    Discover tool packages from subdirectories in tools/.

    Each tool package should have either:
    - __init__.py that exports the tool interface
    - tool.py with the tool implementation

    Args:
        tools_dir: The tools/ directory path.

    Returns:
        List of imported tool modules.
    """
    modules: List[ModuleType] = []

    # Ensure tools_dir is in sys.path for imports
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))

    # Also ensure parent is in path for package imports
    parent_dir = tools_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    for item in sorted(tools_dir.iterdir()):
        # Skip non-directories, hidden dirs, and special dirs
        if not item.is_dir():
            continue
        if item.name.startswith("_") or item.name.startswith("."):
            continue
        if item.name == "__pycache__":
            continue

        # Check for valid tool package
        init_file = item / "__init__.py"
        tool_file = item / "tool.py"

        if not init_file.exists() and not tool_file.exists():
            logger.debug("Skipping non-package: %s", item.name)
            continue

        try:
            # Try importing as a subpackage of tools
            module_name = f"tools.{item.name}"
            module = importlib.import_module(module_name)
            logger.debug("Imported tool package: %s", module_name)
            modules.append(module)
        except ImportError:
            # Fallback: try direct import if tools isn't a proper package
            try:
                if init_file.exists():
                    spec = importlib.util.spec_from_file_location(item.name, init_file)
                else:
                    spec = importlib.util.spec_from_file_location(item.name, tool_file)

                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[item.name] = module
                    spec.loader.exec_module(module)
                    logger.debug("Imported tool package (direct): %s", item.name)
                    modules.append(module)
            except Exception as e:
                logger.error("Import failed for package %s: %s", item.name, e)
                logger.debug("Traceback: %s", traceback.format_exc())
        except Exception as e:
            logger.error("Import failed for package %s: %s", item.name, e)
            logger.debug("Traceback: %s", traceback.format_exc())

    return modules


def collect_tools(modules: List[ModuleType]) -> List[Dict[str, Any]]:
    """
    Collect tool metadata and schemas from tool modules.

    Args:
        modules: List of tool modules to process.

    Returns:
        List of dictionaries with keys:
            - name: Tool name (string)
            - display_description: User-facing description for CLI (string)
            - llm_description: Description for LLM/tool schema (string)
            - tool_def: Full tool definition dict following the TOOL_SCHEMA format
            - runner: Callable that executes the tool (usually module.run)
            - module: The originating module (for debugging/logging)
            - config_path: Optional path to YAML config file
    """
    tools: List[Dict[str, Any]] = []
    seen_names: set[str] = set()

    for m in modules:
        module_filename = getattr(m, "__file__", "unknown")

        # Check for run function
        run_fn = getattr(m, "run", None)
        if not callable(run_fn):
            logger.warning("Missing 'run': %s", module_filename)
            continue

        # Check for template or example tools that shouldn't be loaded
        module_tool_name = getattr(m, "name", None)
        if module_tool_name == "template_tool" or module_tool_name == "mytool":
            logger.debug("Skipping template: %s", module_filename)
            continue

        # Get metadata attributes
        tool_schemas = getattr(m, "TOOLS_SCHEMA", None)
        display_description = getattr(m, "display_description", None)
        llm_description_attr = getattr(m, "description", None)
        config_path = getattr(m, "CONFIG_PATH", None)

        # Parse TOOLS_SCHEMA
        if tool_schemas and isinstance(tool_schemas, list):
            for entry in tool_schemas:
                if not isinstance(entry, dict):
                    logger.warning("Invalid schema: %s - %s", module_filename, type(entry).__name__)
                    continue

                if entry.get("type") != "function":
                    continue

                function_def = entry.get("function", {})
                name = function_def.get("name")
                description = function_def.get("description", llm_description_attr or "")

                if not name:
                    logger.warning("Unnamed tool: %s", module_filename)
                    continue

                if name in seen_names:
                    logger.warning("Duplicate tool: %s in %s", name, module_filename)
                    continue

                seen_names.add(name)
                display_desc = display_description if display_description else description or name

                tools.append(
                    {
                        "name": str(name),
                        "display_description": str(display_desc),
                        "llm_description": str(description),
                        "tool_def": entry,
                        "runner": run_fn,
                        "module": m,
                        "config_path": config_path,
                    }
                )
        else:
            logger.warning("Missing TOOLS_SCHEMA: %s", module_filename)

    tools.sort(key=lambda t: t["name"])
    return tools

