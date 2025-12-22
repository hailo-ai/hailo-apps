# LLM Utilities Module

This module provides utilities for building LLM (Large Language Model) applications using Hailo's AI platform. It handles context management, message formatting, response streaming, tool discovery and execution, and terminal UI interactions.

## Overview

The `llm_utils` module is a collection of utilities designed to simplify building LLM-powered applications. It provides abstractions for common tasks like managing conversation context, formatting messages for the Hailo LLM API, streaming responses, and handling tool-based function calling.

## Prerequisites

- **Hardware**: Hailo AI accelerator device (H10 or compatible)
- **Python**: Python 3.10 or higher
- **Hailo Platform SDK**: Must be installed and configured

## Features

- **Context Management**: Token-based context window tracking, trimming, and caching
- **Message Formatting**: Helper functions to format messages for the Hailo LLM API
- **Streaming**: Real-time token streaming with XML tag filtering and special token handling
- **Tool Discovery**: Automatic discovery and collection of tool modules
- **Tool Execution**: Robust tool call parsing, validation, and execution framework
- **Terminal UI**: Terminal interaction helpers for interactive applications
- **Agent Utilities**: Shared utilities for agent applications (context updates, resource cleanup)

## Components

### Context Manager (`context_manager.py`)

Handles LLM context lifecycle management including usage tracking, trimming, and caching.

**Key Functions:**
- `is_context_full(llm, context_threshold=0.95)` - Check if context usage exceeds threshold
- `print_context_usage(llm, show_always=False)` - Display context usage statistics with visual progress bar
- `save_context_to_cache(llm, tool_name, cache_dir)` - Save context to cache file with atomic writes
- `load_context_from_cache(llm, tool_name, cache_dir)` - Load context from cache file with validation
- `add_to_context(llm, prompt)` - Add content to context by generating minimal response
- `get_context_cache_path(tool_name, cache_dir)` - Get the path to the context cache file for a given tool

### Message Formatter (`message_formatter.py`)

Provides helper functions to create formatted messages for the Hailo LLM API.

**Key Functions:**
- `messages_system(system_text)` - Create system message
- `messages_user(text)` - Create user message
- `messages_assistant(text)` - Create assistant message
- `messages_tool(text)` - Create tool message

All functions return dictionaries in the format: `{"role": "...", "content": "..."}`

### Streaming (`streaming.py`)

Handles streaming LLM responses with real-time filtering of XML tags and special tokens.

**Key Classes:**
- `StreamingTextFilter(debug_mode=False)` - Filters streaming tokens to remove XML tags and special tokens

**Key Functions:**
- `generate_and_stream_response(llm, prompt, temperature=0.1, max_tokens=200, prefix="Assistant: ", token_callback=None, abort_callback=None, show_raw_stream=True)` - Generate and stream response with filtering
- `clean_response(response)` - Clean LLM response by removing special tokens and extracting text

**Features:**
- Filters `<tool_call>`, `<tool_response>`, and `<text>` XML tags
- Handles partial tags across token boundaries
- Supports raw JSON tool call format (`>{...}>`)
- Removes special tokens like `<|im_end|>`
- Automatically filters recovery sequence tokens from LLM API
- Optional token callback for TTS integration (receives filtered output)
- Optional abort callback for user interruption
- Supports both raw and filtered streaming modes

### Tool Discovery (`tool_discovery.py`)

Automatic discovery and collection of tool modules from a `tools/` subdirectory.

**Key Functions:**
- `discover_tool_modules(tool_dir=None)` - Discover tool packages from tools/ subdirectory
- `collect_tools(modules)` - Collect tool metadata and schemas from modules

Returns list of dictionaries with keys: `name`, `display_description`, `llm_description`, `tool_def`, `runner`, `module`, `config_path`

**Tool Module Requirements:**
- Must be a package in `tools/<tool_name>/` with either:
  - `__init__.py` that exports the tool interface, or
  - `tool.py` with the tool implementation
- Must have a `run()` function
- Must have a `TOOLS_SCHEMA` list with tool definitions
- Optional: `display_description`, `description`, `CONFIG_PATH`

### Tool Parsing (`tool_parsing.py`)

Parses and validates tool calls from LLM responses.

**Key Functions:**
- `parse_function_call(response)` - Parse function call from LLM response, returns dict with `name` and `arguments` keys or None
- `validate_and_fix_call(call)` - Validate and fix tool call format, returns fixed dict or None

**Supported Formats:**
- XML-wrapped: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- Handles missing closing tags with robust brace matching
- Fixes common JSON errors (single quotes, trailing commas)
- Handles stringified Python format edge cases (e.g., `[{'type': 'text', 'text': '<tool_call>...</tool_call>'}]`)
- Extracts tool calls from message format wrappers
- Validates and fixes nested JSON in arguments field

### Tool Execution (`tool_execution.py`)

Handles tool initialization and execution with error handling.

**Key Functions:**
- `initialize_tool_if_needed(tool)` - Initialize tool if it has `initialize_tool` function
- `execute_tool_call(tool_call, tools_lookup)` - Execute a tool call and return result dict
- `print_tool_result(result)` - Print tool execution result to user

**Result Format:**
- Success: `{"ok": True, "result": "..."}`
- Error: `{"ok": False, "error": "..."}`
- User error (default option): `{"ok": True, "error": "..."}`

### Tool Selection (`tool_selection.py`)

Handles interactive tool selection in a separate thread.

**Key Functions:**
- `wait_for_tool_selection(all_tools)` - Start tool selection and wait for user selection (convenience function)
- `start_tool_selection_thread(all_tools)` - Start tool selection in background thread, returns (thread, result_dict)
- `get_tool_selection_result(tool_thread, tool_result)` - Wait for selection and return selected tool dict or None

### Terminal UI (`terminal_ui.py`)

Terminal interaction helpers for interactive applications.

**Key Classes:**
- `TerminalUI` - Handles terminal user interface interactions

**Key Methods:**
- `show_banner(title="Terminal Voice Assistant", controls=None)` - Display application banner with instructions
- `get_char()` - Read a single character from stdin (preserves Ctrl+C handling)

**Features:**
- Uses `tty.setcbreak()` to preserve Ctrl+C handling
- Falls back gracefully when stdin is not a TTY (e.g., redirected input)
- Returns control characters for special keys (e.g., `\x03` for Ctrl+C)

### Agent Utilities (`agent_utils.py`)

Shared utilities for agent applications.

**Key Functions:**
- `update_context_with_tool_result(llm, result)` - Update LLM context with tool execution result (wraps in `<tool_response>` tags)
- `cleanup_resources(llm, vdevice, tool_module=None)` - Clean up agent resources safely (handles errors gracefully)

**Features:**
- Automatically wraps tool results in `<tool_response>` XML tags
- Uses `add_to_context()` to update LLM context without generating full response
- Safe resource cleanup with error handling (continues even if individual cleanup steps fail)
- Supports optional tool module cleanup via `cleanup_tool()` function

## Files

- `__init__.py` - Module exports
- `context_manager.py` - Context management utilities
- `message_formatter.py` - Message formatting helpers
- `streaming.py` - Streaming and response filtering
- `tool_discovery.py` - Tool discovery and collection
- `tool_parsing.py` - Tool call parsing and validation
- `tool_execution.py` - Tool execution and error handling
- `tool_selection.py` - Interactive tool selection
- `terminal_ui.py` - Terminal UI helpers
- `agent_utils.py` - Agent application utilities

## Usage Examples

### Basic LLM Chat Loop

This example demonstrates a complete chat loop with context management, message formatting, and streaming:

```python
from hailo_platform import VDevice
from hailo_platform.genai import LLM
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import (
    message_formatter,
    streaming,
    context_manager,
    agent_utils,
)

# Initialize hardware and LLM
vdevice = VDevice()
llm = LLM(vdevice=vdevice)

# Create system message
messages = [message_formatter.messages_system("You are a helpful assistant.")]

try:
    while True:
        # Get user input
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            break

        # Add user message to conversation
        messages.append(message_formatter.messages_user(user_input))

        # Check if context is full and clear if needed
        if context_manager.is_context_full(llm, context_threshold=0.95):
            print("[Info] Context full, clearing...")
            llm.clear_context()
            messages = [message_formatter.messages_system("You are a helpful assistant.")]

        # Display context usage
        context_manager.print_context_usage(llm, show_always=False)

        # Generate and stream response
        response = streaming.generate_and_stream_response(
            llm=llm,
            prompt=messages,
            temperature=0.1,
            max_tokens=200,
            prefix="Assistant: ",
        )

        # Add assistant response to conversation history
        cleaned_response = streaming.clean_response(response)
        messages.append(message_formatter.messages_assistant(cleaned_response))

finally:
    # Cleanup resources
    agent_utils.cleanup_resources(llm, vdevice)
```

### Tool-Based Agent with Context Caching

This example demonstrates a complete agent with tool discovery, execution, context management, and caching:

```python
from pathlib import Path
from hailo_platform import VDevice
from hailo_platform.genai import LLM
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import (
    message_formatter,
    streaming,
    tool_discovery,
    tool_parsing,
    tool_execution,
    agent_utils,
    context_manager,
)

# Initialize hardware and LLM
vdevice = VDevice()
llm = LLM(vdevice=vdevice)

# Discover tools from tools/ subdirectory
# Note: In your actual code, replace this with the path to your tools directory
# For example: tool_dir = Path("path/to/your/tools/directory")
tool_dir = Path("tools")  # Adjust this path to your tools directory
modules = tool_discovery.discover_tool_modules(tool_dir)
tools = tool_discovery.collect_tools(modules)

# Build tools lookup dictionary
tools_lookup = {tool["name"]: tool for tool in tools}

# Initialize all tools
for tool in tools:
    tool_execution.initialize_tool_if_needed(tool)

# Create system prompt with tool information
tool_names = ", ".join(tools_lookup.keys())
system_prompt = f"You are a helpful assistant with access to these tools: {tool_names}"
messages = [message_formatter.messages_system(system_prompt)]

# Setup cache directory
cache_dir = Path("cache")
tool_name = "my_agent"

# Try to load context from cache
if context_manager.load_context_from_cache(llm, tool_name, cache_dir):
    print(f"[Info] Loaded context from cache: {tool_name}")

try:
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            break

        # Add user message
        messages.append(message_formatter.messages_user(user_input))

        # Check context usage
        if context_manager.is_context_full(llm, context_threshold=0.95):
            print("[Info] Context full, saving cache and clearing...")
            context_manager.save_context_to_cache(llm, tool_name, cache_dir)
            llm.clear_context()
            messages = [message_formatter.messages_system(system_prompt)]

        # Generate response
        response = streaming.generate_and_stream_response(
            llm=llm,
            prompt=messages,
            temperature=0.1,
            max_tokens=200,
            prefix="Assistant: ",
        )

        # Check for tool call in response
        tool_call = tool_parsing.parse_function_call(response)
        if tool_call:
            # Execute tool
            result = tool_execution.execute_tool_call(tool_call, tools_lookup)
            tool_execution.print_tool_result(result)

            # Update context with tool result (adds <tool_response> to LLM context)
            agent_utils.update_context_with_tool_result(llm, result)

            # Generate response using the tool result in context
            # The LLM will see the tool result and can respond naturally
            final_response = streaming.generate_and_stream_response(
                llm=llm,
                prompt=[message_formatter.messages_user("Please respond to the user's request.")],
                temperature=0.1,
                max_tokens=200,
                prefix="Assistant: ",
            )
            cleaned = streaming.clean_response(final_response)
            messages.append(message_formatter.messages_assistant(cleaned))
        else:
            # No tool call, add regular response to conversation history
            cleaned = streaming.clean_response(response)
            messages.append(message_formatter.messages_assistant(cleaned))

finally:
    # Save context to cache before cleanup
    context_manager.save_context_to_cache(llm, tool_name, cache_dir)
    # Cleanup resources
    agent_utils.cleanup_resources(llm, vdevice)
```

### Streaming with Token Callback (TTS Integration)

This example shows how to integrate streaming with text-to-speech:

```python
from hailo_platform import VDevice
from hailo_platform.genai import LLM
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import (
    message_formatter,
    streaming,
    agent_utils,
)

vdevice = VDevice()
llm = LLM(vdevice=vdevice)

# Example token callback for TTS
def on_token(token: str):
    """Called with each filtered token chunk for TTS."""
    # Note: token_callback receives filtered output (tool calls are removed)
    print(f"[TTS] Speaking: {token}", end="", flush=True)

# Example abort callback
should_abort = False

def check_abort() -> bool:
    """Check if generation should be aborted."""
    return should_abort

messages = [
    message_formatter.messages_system("You are a helpful assistant."),
    message_formatter.messages_user("Tell me a short story."),
]

# Stream with TTS callback
response = streaming.generate_and_stream_response(
    llm=llm,
    prompt=messages,
    temperature=0.1,
    max_tokens=200,
    prefix="Assistant: ",
    token_callback=on_token,  # Each filtered token chunk sent to TTS
    abort_callback=check_abort,  # Check for user interruption
    show_raw_stream=False,  # Only show filtered output
)

# Cleanup resources
agent_utils.cleanup_resources(llm, vdevice)
```

## Integration with Other Modules

The `llm_utils` module is designed to work seamlessly with other GenAI utilities:

- **Voice Processing**: Use `streaming.generate_and_stream_response()` with `token_callback` to feed filtered tokens to TTS. The callback automatically receives cleaned output (tool calls are filtered out).
- **Agent Applications**: Use `agent_utils` for context updates and resource cleanup. The `cleanup_resources()` function handles all cleanup tasks safely.
- **Tool-Based Apps**: Use the complete tool discovery and execution pipeline. Tools are automatically discovered from `tools/` subdirectories and can be executed with proper error handling.

## API Reference

For detailed API documentation, see the docstrings in each module:
- `context_manager.py` - Context management functions
- `message_formatter.py` - Message formatting helpers
- `streaming.py` - Streaming and response filtering classes and functions
- `tool_discovery.py` - Tool discovery and collection functions
- `tool_parsing.py` - Tool call parsing and validation functions
- `tool_execution.py` - Tool execution and error handling functions
- `tool_selection.py` - Interactive tool selection functions
- `terminal_ui.py` - TerminalUI class
- `agent_utils.py` - Agent utility functions

## Additional Resources

- [GenAI Applications README](../../README.md) - Overview of GenAI applications
- [Agent Tools Example README](../../agent_tools_example/README.md) - Example application using this module
- [Voice Assistant README](../../voice_assistant/README.md) - Another example using LLM utilities

