"""
System prompt generation module.

Generates the system prompt for the LLM with tool definitions and usage instructions.
Supports YAML-based configuration for persona, capabilities, and tool instructions.
Also provides functions to add few-shot examples to context for priming.

Message Format:
    Messages follow the Hailo LLM API format where content is a string:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    This format is used by message_formatter functions and is compatible with
    LLM.generate() which expects a list of message dictionaries.
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
    # Extract tool definitions and names
    tool_defs = [t["tool_def"] for t in tools]
    tool_names = [t["name"] for t in tools]
    tool_names_list = ", ".join(f'"{name}"' for name in tool_names)
    tools_json = json.dumps(tool_defs, separators=(",", ":"))

    # Build persona section from YAML if available
    persona_section = ""
    if yaml_config and yaml_config.persona:
        components = yaml_config.get_system_prompt_components()
        if components["persona"]:
            persona_section = f"# Persona\n{components['persona']}\n"

    # Build available tools section
    available_tools_section = f"""# Available Tools
<tools>
{tools_json}
</tools>

Available tools: {tool_names_list}"""

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

    # Build tool usage rules section
    tool_usage_rules_section = f"""# Tool Usage Rules
- DEFAULT: If a tool can handle the request, CALL IT using <tool_call>
- ONLY these tools exist: {tool_names_list}. NEVER invent or call tools with different names
- When unsure, CALL THE TOOL (better to use it than skip it)
- Skip tools ONLY for: greetings, small talk, meta questions about capabilities, or clearly conversational requests with no tool match"""

    # Build how to call a tool section
    how_to_call_section = f"""# How to Call a Tool
When you need to use a tool, output ONLY this format:
<tool_call>
{{"name": "<function-name>", "arguments": <args-json-object>}}
</tool_call>

Rules:
- Use double quotes (") in JSON, not single quotes
- Arguments must be a JSON object, not a string
- Wrap JSON in <tool_call></tool_call> tags
- Use only these tool names: {tool_names_list}
- After calling, wait for the system to send you <tool_response>"""

    # Build tool results section
    tool_results_section = """# Tool Results
- The system will present tool results directly to the user
- Tool results are already formatted and ready for display
- Your role is to use tools when appropriate, the system handles showing results"""

    # Build decision process section
    decision_process_section = f"""# Decision Process - Think Before Responding
BEFORE each response, think about whether to use a tool:
1. Analyze the user's request carefully
2. Check if any available tool ({tool_names_list}) can handle it
3. Determine if tool execution is needed or you can answer directly
4. If no tool needed: respond directly with text"""

    # Combine all sections
    prompt_parts = [
        persona_section,
        available_tools_section,
        capabilities_section,
        tool_instructions_section,
        tool_usage_rules_section,
        how_to_call_section,
        tool_results_section,
        decision_process_section,
    ]

    # Join all parts with double newlines for readability, then add final newline
    system_prompt = "\n\n".join(part for part in prompt_parts if part) + "\n"

    return system_prompt


def prepare_few_shot_examples_messages(
    examples: List[FewShotExample],
) -> List[Dict[str, Any]]:
    """
    Prepare few-shot example messages for context.

    Converts YAML few-shot examples into message sequences.
    This helps the model learn the expected interaction pattern.

    Args:
        examples: List of FewShotExample objects from YAML config.

    Returns:
        List of formatted message dictionaries ready to be added to context.
    """
    if not examples:
        return []

    try:
        from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import message_formatter
    except ImportError:
        # Fallback if imports fail
        return []

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

    return messages


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
        from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import context_manager
    except ImportError:
        # Fallback if imports fail
        if logger:
            logger.warning("Could not import context_manager, skipping few-shot examples")
        return

    messages = prepare_few_shot_examples_messages(examples)

    # Add all messages to context
    if messages:
        context_manager.add_to_context(llm, messages, logger)
        if logger:
            logger.debug("Added %d few-shot examples to context", len(examples))

