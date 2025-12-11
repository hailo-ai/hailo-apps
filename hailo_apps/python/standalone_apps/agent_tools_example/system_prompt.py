"""
System prompt generation module.

Generates the system prompt for the LLM with tool definitions and usage instructions.
Supports YAML-based configuration for persona, capabilities, and tool instructions.
Also provides functions to add few-shot examples to context for priming.
"""

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hailo_platform.genai import LLM

try:
    from .yaml_config import ToolYamlConfig, FewShotExample
except ImportError:
    from yaml_config import ToolYamlConfig, FewShotExample


def create_system_prompt(
    tools: List[Dict[str, Any]],
    yaml_config: Optional[ToolYamlConfig] = None,
) -> str:
    """
    Create system prompt with tool definitions.

    Uses YAML configuration if available for persona, capabilities, and tool instructions.
    Falls back to generic prompt if YAML config is not provided.

    Args:
        tools: List of tool metadata dictionaries containing a ready-to-use tool definition
        yaml_config: Optional ToolYamlConfig with persona, capabilities, etc.

    Returns:
        System prompt string for the LLM
    """
    # Each tool already supplies a full function definition via TOOLS_SCHEMA
    tool_defs = [t["tool_def"] for t in tools]
    tools_json = json.dumps(tool_defs, separators=(",", ":"))

    # Extract tool names for explicit listing
    tool_names = [t["name"] for t in tools]
    tool_names_list = ", ".join(f'"{name}"' for name in tool_names)

    # Build persona section from YAML if available
    persona_section = ""
    if yaml_config and yaml_config.persona:
        components = yaml_config.get_system_prompt_components()
        if components["persona"]:
            persona_section = f"\n# Persona\n{components['persona']}\n"

    # Build capabilities section
    capabilities_section = ""
    if yaml_config and yaml_config.capabilities:
        components = yaml_config.get_system_prompt_components()
        if components["capabilities"]:
            capabilities_section = f"\n# Capabilities\n{components['capabilities']}\n"

    # Build tool instructions section (from YAML or tool description)
    tool_instructions_section = ""
    if yaml_config and yaml_config.tool_instructions:
        components = yaml_config.get_system_prompt_components()
        tool_instructions_section = f"\n# Tool Instructions\n{components['tool_instructions']}\n"
    else:
        # Fallback: use tool description from first tool
        if tools and tools[0].get("description"):
            tool_instructions_section = f"\n# Tool Instructions\n{tools[0]['description']}\n"

    return f"""You are Qwen, created by Alibaba Cloud. You are a helpful assistant.{persona_section}

# Available Tools
<tools>
{tools_json}
</tools>

Available tools: {tool_names_list}{capabilities_section}

# CRITICAL: Your Role vs Tool Role
- YOU are the ASSISTANT - you CALL tools, you do NOT respond as tools
- YOU output <tool_call> tags to REQUEST tool execution
- The SYSTEM executes the tool and sends you <tool_response> tags
- YOU then respond to the user based on the tool result
- NEVER output <tool_response> tags yourself - that's what the system sends TO you
- ONLY output <tool_call> tags when you want to use a tool{tool_instructions_section}

# Tool Usage Rules
- DEFAULT: If a tool can handle the request, CALL IT using <tool_call>
- ONLY these tools exist: {tool_names_list}. NEVER invent or call tools with different names
- When unsure, CALL THE TOOL (better to use it than skip it)
- Skip tools ONLY for: greetings, small talk, meta questions about capabilities, or clearly conversational requests with no tool match

# How to Call a Tool
When you need to use a tool, output ONLY this format:
<tool_call>
{{"name": "<function-name>", "arguments": <args-json-object>}}
</tool_call>

Rules:
- Use double quotes (") in JSON, not single quotes
- Arguments must be a JSON object, not a string
- Wrap JSON in <tool_call></tool_call> tags
- Use only these tool names: {tool_names_list}
- After calling, wait for the system to send you <tool_response>

# Tool Results
- The system will present tool results directly to the user
- Tool results are already formatted and ready for display
- Your role is to use tools when appropriate, the system handles showing results

# Decision Process - Think Before Responding
BEFORE each response, think about whether to use a tool:
1. Analyze the user's request carefully
2. Check if any available tool ({tool_names_list}) can handle it
3. Determine if tool execution is needed or you can answer directly
4. If no tool needed: respond directly with text

"""


def add_few_shot_examples_to_context(
    llm: "LLM",
    examples: List[FewShotExample],
    logger: Optional[Any] = None,
) -> None:
    """
    Add few-shot examples to LLM context for priming.

    Converts YAML few-shot examples into message sequences and adds them to context.
    This helps the model learn the expected interaction pattern.

    Args:
        llm: LLM instance to add context to.
        examples: List of FewShotExample objects from YAML config.
        logger: Optional logger instance.
    """
    if not examples:
        return

    try:
        from hailo_apps.python.core.gen_ai_utils.llm_utils import (
            context_manager,
            message_formatter,
        )
    except ImportError:
        # Fallback if imports fail
        if logger:
            logger.warning("Could not import context_manager, skipping few-shot examples")
        return

    messages = []

    for example in examples:
        # User message
        messages.append(message_formatter.messages_user(example.user))

        # Tool call (if present)
        if example.tool_call:
            tool_call_json = json.dumps(
                {
                    "name": example.tool_call.get("name", ""),
                    "arguments": example.tool_call.get("arguments", {}),
                },
                separators=(",", ":"),
            )
            tool_call_xml = f"<tool_call>\n{tool_call_json}\n</tool_call>"
            messages.append(message_formatter.messages_assistant(tool_call_xml))

            # Tool response (if present)
            if example.tool_response:
                messages.append(message_formatter.messages_tool(example.tool_response))

        # Final assistant response
        if example.final_response:
            messages.append(message_formatter.messages_assistant(example.final_response))

    # Add all messages to context
    if messages:
        context_manager.add_to_context(llm, messages, logger)
        if logger:
            logger.debug("Added %d few-shot examples to context", len(examples))

