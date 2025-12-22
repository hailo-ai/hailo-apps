"""
Streaming utilities for LLM generation.

Handles streaming tokens, filtering XML tags, and cleaning up response content.
"""

import logging
import re
from typing import Callable, List, Optional

from hailo_platform.genai import LLM

# Setup logger
logger = logging.getLogger(__name__)


class StreamingTextFilter:
    """
    Filter streaming tokens on-the-fly to remove XML tags and special tokens.

    Maintains state to handle partial tags that arrive across token boundaries.
    Also handles raw JSON tool calls with > delimiters (LLM format inconsistency).
    """

    def __init__(self, debug_mode: bool = False):
        """
        Initialize the streaming text filter.

        Args:
            debug_mode (bool): If True, disables all filtering logic and returns
                tokens unchanged. Used for debugging the filter itself.
        """
        self.buffer = ""
        self.inside_text_tag = False
        self.inside_tool_call_tag = False
        self.inside_tool_response_tag = False
        self.inside_raw_json_tool_call = False  # For >...> format
        self.debug_mode = debug_mode

    def process_token(self, token: str) -> str:
        """
        Process a single token and return cleaned text ready for display.

        Args:
            token (str): Raw token from LLM.

        Returns:
            str: Cleaned text to print (may be empty if token should be suppressed).
                In debug_mode, returns token unchanged (no filtering).
        """
        # In debug mode, don't filter anything - show raw output
        if self.debug_mode:
            return token

        self.buffer += token

        # Remove <|im_end|> tokens immediately
        if "<|im_end|>" in self.buffer:
            self.buffer = self.buffer.replace("<|im_end|>", "")

        output = ""

        # Process buffer until no more complete tags are found
        changed = True
        while changed:
            changed = False

            # Check for <text> tag start
            text_start = self.buffer.find("<text>")
            if text_start != -1 and not self.inside_text_tag:
                # Extract anything before <text> tag (should be empty, but just in case)
                if text_start > 0:
                    output += self.buffer[:text_start]
                self.buffer = self.buffer[text_start + 6:]  # Remove "<text>"
                self.inside_text_tag = True
                changed = True
                continue

            # Check for </text> tag end
            text_end = self.buffer.find("</text>")
            if text_end != -1 and self.inside_text_tag:
                # Extract text content before </text>
                output += self.buffer[:text_end]
                self.buffer = self.buffer[text_end + 7:]  # Remove "</text>"
                self.inside_text_tag = False
                changed = True
                continue

            # Check for <tool_call> tag start
            tool_call_start = self.buffer.find("<tool_call>")
            if tool_call_start != -1 and not self.inside_tool_call_tag:
                # If we're inside <text>, output content before <tool_call>
                if self.inside_text_tag and tool_call_start > 0:
                    output += self.buffer[:tool_call_start]
                self.buffer = self.buffer[tool_call_start + 10:]  # Remove "<tool_call>"
                self.inside_tool_call_tag = True
                changed = True
                continue

            # Check for </tool_call> tag end
            tool_call_end = self.buffer.find("</tool_call>")
            if tool_call_end != -1 and self.inside_tool_call_tag:
                # Suppress everything inside <tool_call>
                self.buffer = self.buffer[tool_call_end + 12:]  # Remove "</tool_call>"
                self.inside_tool_call_tag = False
                changed = True
                continue

            # Check for <tool_response> tag start
            tool_response_start = self.buffer.find("<tool_response>")
            if tool_response_start != -1 and not self.inside_tool_response_tag:
                # If we're inside <text>, output content before <tool_response>
                if self.inside_text_tag and tool_response_start > 0:
                    output += self.buffer[:tool_response_start]
                self.buffer = self.buffer[tool_response_start + 15:]  # Remove "<tool_response>"
                self.inside_tool_response_tag = True
                changed = True
                continue

            # Check for </tool_response> tag end
            tool_response_end = self.buffer.find("</tool_response>")
            if tool_response_end != -1 and self.inside_tool_response_tag:
                # Suppress everything inside <tool_response>
                self.buffer = self.buffer[tool_response_end + 16:]  # Remove "</tool_response>"
                self.inside_tool_response_tag = False
                changed = True
                continue

            # Check for raw JSON tool call format: >\n{...}\n>
            # This handles LLM inconsistency where it outputs > delimited JSON instead of <tool_call>
            if not self.inside_raw_json_tool_call:
                # Look for pattern: > followed by whitespace and { with "name"
                raw_start_match = re.search(r'>\s*\{\s*"name"\s*:', self.buffer)
                if raw_start_match:
                    # Output anything before the >
                    start_pos = raw_start_match.start()
                    if start_pos > 0:
                        output += self.buffer[:start_pos]
                    self.buffer = self.buffer[start_pos:]  # Keep from > onwards for now
                    self.inside_raw_json_tool_call = True
                    changed = True
                    continue

            # Check for end of raw JSON tool call (closing >)
            if self.inside_raw_json_tool_call:
                # Look for closing pattern: }\n> or }\s*>
                raw_end_match = re.search(r'\}\s*>', self.buffer)
                if raw_end_match:
                    # Suppress everything up to and including the closing >
                    end_pos = raw_end_match.end()
                    self.buffer = self.buffer[end_pos:]
                    self.inside_raw_json_tool_call = False
                    changed = True
                    continue

        # If we're inside <text> tag and not inside any suppressed section, output remaining buffer
        if self.inside_text_tag and not self.inside_tool_call_tag and not self.inside_tool_response_tag and not self.inside_raw_json_tool_call and self.buffer:
            output += self.buffer
            self.buffer = ""
        elif not self.inside_text_tag and not self.inside_tool_call_tag and not self.inside_tool_response_tag and not self.inside_raw_json_tool_call:
            # If not inside any tag, the text is still valid for streaming.
            # To avoid printing partial tags, we find the last complete chunk of text.
            # A simple heuristic: find the start of the next potential tag.
            next_tag_start = self.buffer.find('<')
            if next_tag_start != -1:
                # Output text up to the potential start of a tag
                output += self.buffer[:next_tag_start]
                self.buffer = self.buffer[next_tag_start:]
            else:
                # No partial tag found, output the whole buffer
                output += self.buffer
                self.buffer = ""

        return output

    def get_remaining(self) -> str:
        """
        Get any remaining buffered content after streaming completes.

        Returns:
            str: Remaining buffered text, cleaned of partial tags.
        """
        # In debug mode, return empty (everything was printed already)
        if self.debug_mode:
            return ""

        # If we're inside a raw JSON tool call, suppress the remaining buffer
        if self.inside_raw_json_tool_call:
            return ""

        # Clean up any remaining partial tags or buffer content
        if self.inside_text_tag and not self.inside_tool_call_tag and not self.inside_tool_response_tag:
            # If we're still inside text tag, return the buffer (might have partial closing tag)
            remaining = self.buffer
            # Remove any partial closing tags like "</text" or "text>"
            remaining = remaining.replace("</text", "").replace("text>", "")
            return remaining
        # Also clean up any partial tags that might remain in buffer
        cleaned = self.buffer.replace("</text", "").replace("text>", "").replace("<text", "")
        cleaned = cleaned.replace("</tool_response", "").replace("tool_response>", "").replace("<tool_response", "")

        # Remove raw JSON tool call patterns (>{"name":...}>)
        cleaned = re.sub(r'>\s*\{[^}]*"name"[^}]*\}\s*>', '', cleaned)

        # Remove orphan '<' or '>' which might be left over from incomplete tags
        if cleaned.strip() in ("<", ">"):
            return ""

        return cleaned


def generate_and_stream_response(
    llm: LLM,
    prompt: List[dict],
    temperature: float = 0.1,
    seed: int = 42,
    max_tokens: int = 200,
    prefix: str = "Assistant: ",
    token_callback: Optional[Callable[[str], None]] = None,
    abort_callback: Optional[Callable[[], bool]] = None,
    show_raw_stream: bool = True,
) -> str:
    """
    Generate response from LLM and stream it to stdout with filtering.

    Handles streaming tokens, filtering XML tags, and cleaning up remaining content.

    Args:
        llm (LLM): The LLM instance to use for generation.
        prompt (List[dict]): List of message dictionaries to send to the LLM.
        temperature (float): Sampling temperature. Defaults to 0.1.
        seed (int): Random seed. Defaults to 42.
        max_tokens (int): Maximum tokens to generate. Defaults to 200.
        prefix (str): Prefix to print before streaming. Defaults to "Assistant: ".
        token_callback (Optional[Callable[[str], None]]): Optional callback function
            called with each cleaned token/chunk. Useful for TTS integration.
            Callbacks always receive filtered output (tool calls are filtered out).
        abort_callback (Optional[Callable[[], bool]]): Optional callback function
            that returns True if generation should be aborted.
        show_raw_stream (bool): If True (default), prints raw tokens in real-time
            including tool calls for visibility. Filtering still runs for callbacks.
            If False, prints only filtered/cleaned tokens.

    Returns:
        str: Raw response string (before filtering, for tool call parsing).
    """
    print(prefix, end="", flush=True)
    response_parts: List[str] = []
    token_filter = StreamingTextFilter()  # Always filter for callbacks

    # Get recovery sequence to filter it out
    recovery_seq = None
    try:
        recovery_seq = llm.get_generation_recovery_sequence()
        logger.debug("Recovery sequence configured")
    except AttributeError:
        logger.debug("Recovery sequence not available")

    with llm.generate(
        prompt=prompt,
        temperature=temperature,
        seed=seed,
        max_generated_tokens=max_tokens,
    ) as gen:
        for token in gen:
            # Check for abort signal
            if abort_callback and abort_callback():
                logger.info("Generation aborted by user")
                print("\n[Aborted]", end="", flush=True)
                break

            # Filter recovery sequence if present
            if recovery_seq and token == recovery_seq:
                continue

            response_parts.append(token)

            # Filter token to get cleaned chunk (for callbacks)
            cleaned_chunk = token_filter.process_token(token)

            # Print output: raw tokens if show_raw_stream, otherwise filtered tokens
            if show_raw_stream:
                print(token, end="", flush=True)
            else:
                if cleaned_chunk:
                    print(cleaned_chunk, end="", flush=True)

            # Callbacks always receive filtered output (ensures TTS doesn't speak tool calls)
            if cleaned_chunk and token_callback:
                token_callback(cleaned_chunk)

    # Print any remaining filtered content after streaming completes
    # Skip if showing raw stream (already printed as raw tokens)
    if not show_raw_stream:
        remaining = token_filter.get_remaining()
        if remaining:
            # Final cleanup: remove any remaining XML tags and partial tags
            remaining = re.sub(r"</?text>?", "", remaining)  # </text>, </text, <text>, text>
            remaining = re.sub(r"</?tool_call>?", "", remaining)  # </tool_call>, <tool_call>, etc.
            remaining = re.sub(r"<\|im_end\|>", "", remaining)  # Special tokens
            print(remaining, end="", flush=True)
            if token_callback:
                token_callback(remaining)

    print()  # New line after streaming completes

    raw_response = "".join(response_parts)
    logger.debug("Generated %d tokens", len(response_parts))
    return raw_response


def clean_response(response: str) -> str:
    """
    Clean LLM response by removing special tokens and extracting text from XML tags.

    Removes:
    - <|im_end|> tokens
    - <text>...</text> wrapper tags (extracts content)
    - <tool_call>...</tool_call> tags (tool calls are parsed separately)

    Args:
        response (str): Raw response string from LLM.

    Returns:
        str: Cleaned response text ready for display.
    """
    # Remove special tokens
    cleaned = response.replace("<|im_end|>", "")

    # Extract text from <text>...</text> tags
    text_match = re.search(r"<text>(.*?)</text>", cleaned, re.DOTALL)
    if text_match:
        cleaned = text_match.group(1).strip()

    # Remove <tool_call>...</tool_call> tags if present (we parse these separately)
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL)

    # Remove <tool_response>...</tool_response> tags if present
    cleaned = re.sub(r"<tool_response>.*?</tool_response>", "", cleaned, flags=re.DOTALL)

    # Clean up any remaining whitespace
    return cleaned.strip()
