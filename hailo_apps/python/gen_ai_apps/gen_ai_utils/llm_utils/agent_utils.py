"""
Shared utilities for agent applications.

Contains common logic for context updates, resource cleanup, and other shared functionality.
"""

import json
import logging
from typing import Any, Dict, Optional

from hailo_platform import VDevice
from hailo_platform.genai import LLM

from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import (
    message_formatter,
    context_manager,
)

logger = logging.getLogger(__name__)


def update_context_with_tool_result(
    llm: LLM,
    result: Dict[str, Any],
    logger_instance: Optional[logging.Logger] = None,
) -> None:
    """
    Update LLM context with the result of a tool execution.

    Args:
        llm: The LLM instance.
        result: The tool execution result dictionary.
        logger_instance: Optional logger.
    """
    log = logger_instance or logger

    tool_result_text = json.dumps(result, ensure_ascii=False)
    tool_response_message = f"<tool_response>{tool_result_text}</tool_response>"
    log.debug("Tool result: %s", tool_result_text)

    # LLM has context, just add the tool result
    prompt = [message_formatter.messages_tool(tool_response_message)]

    # Add to context by making a minimal generation (just to update context)
    log.debug("Updating context")
    context_manager.add_to_context(llm, prompt, logger_instance=log)


def cleanup_resources(
    llm: Optional[LLM],
    vdevice: Optional[VDevice],
    tool_module: Optional[Any] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> None:
    """
    Clean up agent resources safely.

    Args:
        llm: LLM instance to release.
        vdevice: VDevice instance to release.
        tool_module: Tool module to call cleanup_tool() on.
        logger_instance: Optional logger.
    """
    log = logger_instance or logger

    # Cleanup tool resources
    if tool_module and hasattr(tool_module, "cleanup_tool"):
        try:
            tool_module.cleanup_tool()
        except Exception as e:
            log.debug("Cleanup failed: %s", e)

    if llm:
        try:
            llm.clear_context()
        except Exception as e:
            log.debug("Context clear failed: %s", e)
        try:
            llm.release()
        except Exception as e:
            log.debug("LLM release failed: %s", e)

    if vdevice:
        try:
            vdevice.release()
        except Exception as e:
            log.debug("VDevice release failed: %s", e)
