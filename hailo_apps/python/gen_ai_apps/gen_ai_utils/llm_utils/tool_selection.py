"""
Tool selection module.

Handles interactive tool selection in a separate thread.
"""

import logging
import threading
from typing import Any, Dict, List, Optional, cast

# Setup logger
logger = logging.getLogger(__name__)


def select_tool_interactive(tools: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
    """
    Handle user tool selection in a background thread.

    Args:
        tools: List of available tools.
        result: Shared dictionary to store selection result with keys:
            - selected_tool: The selected tool dict or None
            - should_exit: Boolean flag to indicate user wants to quit
            - lock: Threading lock for thread-safe access
    """
    print("\nAvailable tools:")
    for idx, tool_info in enumerate(tools, start=1):
        print(f"  {idx}. {tool_info['name']}: {tool_info['display_description']}")

    logger.info("Tool selection started: %d tools available", len(tools))

    while True:
        choice = input("\nSelect a tool by number (or 'q' to quit): ").strip()
        if choice.lower() in {"q", "quit", "exit"}:
            logger.info("User chose to exit tool selection")
            print("Bye.")
            with result["lock"]:
                result["should_exit"] = True
            return
        try:
            tool_idx = int(choice) - 1
            if 0 <= tool_idx < len(tools):
                selected_tool = tools[tool_idx]
                logger.info("Tool selected: %s", selected_tool.get("name", "unknown"))
                with result["lock"]:
                    result["selected_tool"] = selected_tool
                return
            logger.warning("Invalid tool selection: %s (valid range: 1-%d)", choice, len(tools))
            print(f"Invalid selection. Please choose 1-{len(tools)}.")
        except ValueError:
            logger.warning("Invalid input in tool selection: %s", choice)
            print("Invalid input. Please enter a number or 'q' to quit.")


def start_tool_selection_thread(
    all_tools: List[Dict[str, Any]],
) -> tuple[threading.Thread, Dict[str, Any]]:
    """
    Start tool selection in a background thread.

    Args:
        all_tools: List of available tools to choose from.

    Returns:
        Tuple of (thread, result_dict) where result_dict contains:
            - selected_tool: The selected tool dict or None
            - should_exit: Boolean flag to indicate user wants to quit
            - lock: Threading lock for thread-safe access
    """
    # Shared result structure for tool selection
    tool_result: Dict[str, Any] = {
        "selected_tool": None,
        "should_exit": False,
        "lock": threading.Lock(),
    }

    # Start tool selection in background thread
    tool_thread = threading.Thread(
        target=select_tool_interactive, args=(all_tools, tool_result), daemon=False
    )
    tool_thread.start()

    return tool_thread, tool_result


def get_tool_selection_result(
    tool_thread: threading.Thread, tool_result: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Wait for tool selection thread to complete and return the selected tool.

    Args:
        tool_thread: The tool selection thread.
        tool_result: Shared result dictionary from start_tool_selection_thread.

    Returns:
        Selected tool dictionary, or None if user chose to exit or no tool was selected.
    """
    # Wait for tool selection to complete
    tool_thread.join()

    # Check tool selection result
    with tool_result["lock"]:
        if tool_result["should_exit"]:
            return None
        selected_tool = tool_result["selected_tool"]

    if selected_tool is None:
        logger.error("No tool selected")
        print("[Error] No tool selected.")
        return None

    # Type cast: selected_tool is guaranteed to be non-None after the check above
    selected_tool = cast(Dict[str, Any], selected_tool)
    selected_tool_name = selected_tool.get("name", "")
    if not selected_tool_name:
        logger.error("Selected tool missing 'name' field")
        print("[Error] Selected tool missing 'name' field.")
        return None

    logger.info("Tool selection completed: %s", selected_tool_name)
    print(f"\nSelected tool: {selected_tool_name}")
    return selected_tool


def wait_for_tool_selection(
    all_tools: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Start tool selection in a background thread and wait for user selection.

    Convenience function that combines start_tool_selection_thread and get_tool_selection_result.

    Args:
        all_tools: List of available tools to choose from.

    Returns:
        Selected tool dictionary, or None if user chose to exit or no tool was selected.
    """
    tool_thread, tool_result = start_tool_selection_thread(all_tools)
    return get_tool_selection_result(tool_thread, tool_result)

