"""
Message formatting utilities for LLM interactions.

Provides helper functions to create formatted messages for system, user, assistant, and tool roles.

The Hailo LLM API expects messages in the format:
    [{"role": "user", "content": "..."}]

Where content is a string for text-only LLM models. For VLM models, content can be a list
of content objects like [{"type": "text", "text": "..."}, {"type": "image"}].

This module provides the LLM (text-only) format where content is a simple string.
"""

from typing import Any, Dict


def messages_system(system_text: str) -> Dict[str, Any]:
    """
    Create a system message in the format expected by Hailo LLM.

    According to the Hailo LLM API documentation, the format is:
        {"role": "system", "content": "..."}

    Args:
        system_text (str): System prompt text.

    Returns:
        Dict[str, Any]: Formatted message dictionary with content as a string.
    """
    return {"role": "system", "content": system_text}


def messages_user(text: str) -> Dict[str, Any]:
    """
    Create a user message in the format expected by Hailo LLM.

    According to the Hailo LLM API documentation, the format is:
        {"role": "user", "content": "..."}

    Args:
        text (str): User message text.

    Returns:
        Dict[str, Any]: Formatted message dictionary with content as a string.
    """
    return {"role": "user", "content": text}


def messages_assistant(text: str) -> Dict[str, Any]:
    """
    Create an assistant message in the format expected by Hailo LLM.

    According to the Hailo LLM API documentation, the format is:
        {"role": "assistant", "content": "..."}

    Args:
        text (str): Assistant message text.

    Returns:
        Dict[str, Any]: Formatted message dictionary with content as a string.
    """
    return {"role": "assistant", "content": text}


def messages_tool(text: str) -> Dict[str, Any]:
    """
    Create a tool message in the format expected by Hailo LLM.

    According to the Hailo LLM API documentation, the format is:
        {"role": "tool", "content": "..."}

    Args:
        text (str): Tool message text.

    Returns:
        Dict[str, Any]: Formatted message dictionary with content as a string.
    """
    return {"role": "tool", "content": text}
