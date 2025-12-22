"""
Template tool implementation.

Copy and modify this file for your tool. Required exports:
- name: str - Unique tool identifier
- description: str - LLM-facing description
- display_description: str - User-facing description for CLI
- schema: dict - JSON schema for parameters
- TOOLS_SCHEMA: list - OpenAI function calling format
- run(input_data: dict) -> dict - Execution function
"""

from __future__ import annotations

from typing import Any

# Unique tool identifier (used in tool calls)
name: str = "template_tool"

# User-facing description (shown in CLI tool list)
display_description: str = "Template tool - replace with your description."

# LLM instruction description
# Include CRITICAL warnings and exact usage instructions
description: str = (
    "Template tool description for the LLM. "
    "Include specific instructions on when and how to use this tool.\n\n"
    "DEFAULT OPTION: If the user's request is not supported by this tool or if you cannot understand "
    "how to translate the request, set 'default' to true. The tool will automatically generate an appropriate error message."
)

# JSON schema for parameters
# Follow OpenAI function calling format
# DO NOT use: default, minimum, maximum, minItems, maxItems, additionalProperties
schema: dict[str, Any] = {
    "type": "object",
    "properties": {
        "example_param": {
            "type": "string",
            "description": "An example string parameter. Required unless 'default' is used.",
        },
        "default": {
            "type": "boolean",
            "description": (
                "Set to true when the user's request is not supported or if you cannot understand "
                "how to translate the request. The tool will automatically generate an appropriate error message."
            ),
        },
    },
    "required": [],
}

# OpenAI function calling format schema
TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }
]


def initialize_tool() -> None:
    """
    Initialize the tool (optional).

    Called once when the tool is first loaded.
    Use for setup like connecting to hardware, loading models, etc.
    """
    pass


def cleanup_tool() -> None:
    """
    Clean up tool resources (optional).

    Called when the agent shuts down.
    Use for cleanup like disconnecting hardware, releasing resources.
    """
    pass


def run(input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Execute the tool logic.

    Args:
        input_data: Dictionary with tool parameters or default.

    Returns:
        Dictionary with:
        - ok: bool - Success status
        - result: Any - Result value (if ok=True)
        - error: str - Error message (if ok=False)
    """
    # Check for default option first (user error - agent correctly used default)
    if input_data.get("default") is True:
        return {
            "ok": True,  # Agent used tool correctly (default option)
            "error": "Unsupported request. This tool supports processing text parameters.",
        }

    # Extract parameters
    example_param = str(input_data.get("example_param", "")).strip()

    # Validate inputs
    if not example_param:
        return {"ok": False, "error": "Either 'example_param' or 'default' must be provided."}

    # Tool logic here
    # Replace with your implementation

    result = f"Processed: {example_param}"
    return {"ok": True, "result": result}

