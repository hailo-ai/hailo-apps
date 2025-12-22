"""
Tool execution module.

Handles tool initialization and execution.
"""

import json
import logging
import traceback
from typing import Any, Dict

# Setup logger
logger = logging.getLogger(__name__)


def initialize_tool_if_needed(tool: Dict[str, Any]) -> None:
    """
    Initialize tool if it has an initialize_tool function.

    Args:
        tool: Tool dictionary containing a 'module' key.
    """
    tool_module = tool.get("module")
    if tool_module and hasattr(tool_module, "initialize_tool"):
        try:
            tool_module.initialize_tool()
        except Exception as e:
            logger.warning("Tool initialization failed: %s", e)
            logger.debug("Traceback: %s", traceback.format_exc())


def execute_tool_call(
    tool_call: Dict[str, Any], tools_lookup: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Execute a tool call and return the result.

    Args:
        tool_call: Parsed tool call dictionary with 'name' and 'arguments' keys.
        tools_lookup: Dictionary mapping tool names to tool metadata.

    Returns:
        Tool execution result dictionary with 'ok' key and either 'result' or 'error'.
    """
    if not isinstance(tool_call, dict):
        return {"ok": False, "error": "Invalid tool call format: expected dictionary"}

    tool_name = str(tool_call.get("name", "")).strip()
    if not tool_name:
         return {"ok": False, "error": "Tool call missing 'name' field"}

    args = tool_call.get("arguments", {})
    if not isinstance(args, dict):
        return {"ok": False, "error": f"Invalid arguments format for tool '{tool_name}': expected dictionary, got {type(args).__name__}"}

    logger.debug("Tool call: %s", tool_name)
    logger.debug("Args: %s", json.dumps(args, ensure_ascii=False))

    selected = tools_lookup.get(tool_name)
    if not selected:
        available = ", ".join(sorted(tools_lookup.keys()))
        error_msg = f"Unknown tool '{tool_name}'. Available: {available}"
        logger.error("%s", error_msg)
        return {"ok": False, "error": error_msg}

    runner = selected.get("runner")
    if not callable(runner):
        error_msg = f"Tool '{tool_name}' is missing an executable runner."
        logger.error("%s", error_msg)
        return {"ok": False, "error": error_msg}

    try:
        result = runner(args)  # type: ignore[misc]

        if not isinstance(result, dict):
            error_msg = f"Tool '{tool_name}' returned invalid format: expected dict, got {type(result).__name__}"
            logger.error("%s", error_msg)
            return {"ok": False, "error": error_msg}

        logger.debug("Result: %s", json.dumps(result, ensure_ascii=False))
        return result
    except Exception as exc:
        logger.error("%s: %s", tool_name, exc)
        logger.debug("Traceback: %s", traceback.format_exc())

        error_msg = f"Tool '{tool_name}' execution failed: {str(exc)}"
        result = {"ok": False, "error": error_msg}
        return result


def print_tool_result(result: Dict[str, Any]) -> None:
    """
    Print tool execution result to the user.

    Distinguishes between:
    - Agent errors (ok=False): Tool was used incorrectly
    - User errors (ok=True with error): Default option was correctly used

    Args:
        result: Tool execution result dictionary with 'ok' key.
    """
    if not isinstance(result, dict):
        print(f"\n[Tool Error] Invalid result format: {result}\n")
        return

    if result.get("ok"):
        # Success or user error (default option)
        if result.get("error") and not result.get("result"):
            # User error: default option was used, show error message
            error_msg = result.get("error", "Unknown error")
            print(f"\n[Tool] {error_msg}\n")
        else:
            # Success: show result
            tool_result_text = result.get("result", "")
            if tool_result_text:
                print(f"\n[Tool] {tool_result_text}\n")
    else:
        # Agent error: tool was used incorrectly
        # Use standard message (don't expose internal errors to user)
        logger.warning("Agent error: %s", result.get("error", "Unknown error"))
        print(f"\n[Tool Error] There was an error executing the tool. Please try rephrasing your request.\n")

