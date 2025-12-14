"""
Text processing utilities for tool parsing.

Handles parsing and validation of tool calls from LLM responses.
"""

import ast
import json
import logging
import re
import traceback
from typing import Any, Dict, Optional

# Setup logger
logger = logging.getLogger(__name__)


def validate_and_fix_call(call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate that call has required fields and fix nested JSON."""
    if not isinstance(call, dict):
        logger.debug("Invalid format: %s", type(call).__name__)
        return None

    # Must have 'name' field
    if "name" not in call or not call.get("name"):
        logger.debug("Tool call missing 'name' field")
        return None

    # Must have 'arguments' field
    if "arguments" not in call:
        logger.debug("Tool call missing 'arguments' field")
        return None

    # Fix nested JSON in arguments
    # Reason: Some models stringify the arguments object
    if isinstance(call.get("arguments"), str):
            try:
                # Try to parse stringified JSON arguments
                # Replace single quotes with double quotes if needed
                args_str = call["arguments"]
                if "'" in args_str and '"' not in args_str:
                    args_str = args_str.replace("'", '"')
                call["arguments"] = json.loads(args_str)
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug("JSON parse failed: %s", e)

    # Ensure arguments is a dict
    if not isinstance(call.get("arguments"), dict):
        logger.debug("Invalid args format: %s", type(call.get('arguments')).__name__)
        return None

    return call


def parse_function_call(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse function call from LLM response.

    ONLY supports XML-wrapped format:
    <tool_call>
    {"name": "...", "arguments": {...}}
    </tool_call>

    Also handles edge case where response might be a stringified Python list/dict format:
    [{'type': 'text', 'text': '<tool_call>...</tool_call>'}]

    Args:
        response: Raw response string from LLM

    Returns:
        Parsed function call dict with 'name' and 'arguments' keys, or None if not found
    """
    # Handle edge case: response might be stringified Python format
    # Extract actual text content if wrapped in message format
    original_response = response
    if response.strip().startswith("[") and ("'text'" in response or '"text"' in response):
        try:
            # Try to parse as Python literal
            parsed = ast.literal_eval(response)
            if isinstance(parsed, list) and len(parsed) > 0:
                # Extract text from first message
                msg = parsed[0]
                if isinstance(msg, dict):
                    # Check for content field (message format)
                    if "content" in msg:
                        content = msg["content"]
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    response = item.get("text", "")
                                    break
                        elif isinstance(content, str):
                            response = content
                    # Check for direct text field
                    elif "text" in msg:
                        response = msg["text"]
            logger.debug("Extracted text from message format: %s", response[:100] if len(response) > 100 else response)
        except (ValueError, SyntaxError, AttributeError, TypeError):
            # If parsing fails, try regex extraction
            # Look for text field in the string (handles both single and double quotes)
            # Match: 'text': '...' or "text": "..."
            text_match = re.search(r"['\"]text['\"]\s*:\s*['\"](.*?)['\"]", response, re.DOTALL)
            if text_match:
                # Unescape the matched text
                extracted = text_match.group(1)
                response = extracted.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
                logger.debug("Extracted text via regex: %s", response[:100] if len(response) > 100 else response)
            else:
                logger.warning("Could not extract text from message format: %s", original_response[:200])
                response = original_response  # Fall back to original

    # Check for XML tag
    if "<tool_call>" not in response:
        return None

    try:
        # Extract content between tags
        start = response.find("<tool_call>") + len("<tool_call>")

        # Find closing tag, or use brace matching if missing
        end = response.find("</tool_call>", start)

        if end == -1:
            # No closing tag, use robust brace matching
            # Reason: Streaming response might be truncated or model forgot closing tag
            json_str = response[start:].strip()

            # Find the complete JSON object by matching braces
            brace_count = 0
            json_end = -1
            in_string = False
            escape = False

            for i, char in enumerate(json_str):
                if escape:
                    escape = False
                    continue

                if char == '\\':
                    escape = True
                    continue

                if char == '"':
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break

            if json_end > 0:
                json_str = json_str[:json_end]
            else:
                logger.debug("Could not find complete JSON object in partial response")
                return None
        else:
            json_str = response[start:end].strip()

        # Clean up JSON string
        # 1. Handle single quotes (common error)
        # Only replace if it looks like property names or string values
        # Simple heuristic: if no double quotes, assume single quotes usage
        if "'" in json_str and '"' not in json_str:
            json_str = json_str.replace("'", '"')

        # 2. Handle trailing commas (invalid JSON but common)
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        try:
            call = json.loads(json_str)
            return validate_and_fix_call(call)
        except json.JSONDecodeError as e:
            logger.debug("JSON decode failed: %s", e)
            # Last resort: try partial fix for unquoted keys?
            # (Maybe too risky for general tool calling)
            return None

    except Exception as e:
        logger.error("Parse error: %s", e)
        logger.debug("Traceback: %s", traceback.format_exc())
        return None
