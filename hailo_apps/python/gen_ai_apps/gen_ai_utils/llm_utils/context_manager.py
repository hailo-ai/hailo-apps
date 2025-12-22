"""
Context management utilities for LLM interactions.

Handles checking context usage, trimming context, and caching context state.
Provides robust file operations with atomic writes.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from hailo_platform.genai import LLM

# Setup logger
logger = logging.getLogger(__name__)


def is_context_full(
    llm: LLM,
    context_threshold: float = 0.95,
    logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Check if context usage exceeds the threshold.

    Uses actual token usage from the LLM API to determine when to clear context.

    Args:
        llm (LLM): The LLM instance to check.
        context_threshold (float): Threshold percentage (0.0-1.0) to trigger clear.
        logger_instance (logging.Logger): Logger to use. Defaults to module logger.

    Returns:
        bool: True if context usage exceeds threshold, False otherwise.
    """
    log = logger_instance or logger
    try:
        if log.isEnabledFor(logging.DEBUG):
            print_context_usage(llm, logger_instance=log)
        max_capacity = llm.max_context_capacity()
        current_usage = llm.get_context_usage_size()

        # Check if we reach threshold
        threshold = int(max_capacity * context_threshold)

        if current_usage < threshold:
            return False

        log.debug("Context: %d/%d tokens (%d%%)", current_usage, max_capacity, current_usage*100//max_capacity)
        return True

    except Exception as e:
        log.warning("Context check failed: %s", e)
        return False


def print_context_usage(
    llm: LLM, show_always: bool = False, logger_instance: Optional[logging.Logger] = None
) -> None:
    """
    Display context usage statistics.

    Args:
        llm (LLM): The LLM instance.
        show_always (bool): If True, print to user. If False, only log at DEBUG level.
        logger_instance (logging.Logger): Logger to use. Defaults to module logger.
    """
    log = logger_instance or logger
    try:
        max_capacity = llm.max_context_capacity()
        current_usage = llm.get_context_usage_size()
        percentage = (current_usage * 100) // max_capacity if max_capacity > 0 else 0

        # Create visual progress bar
        bar_length = 30
        filled = (current_usage * bar_length) // max_capacity if max_capacity > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)

        usage_str = f"[{bar}] {current_usage}/{max_capacity} ({percentage}%)"

        if show_always:
            print(f"[Info] Context: {usage_str}")
        else:
            log.debug("Context: %s", usage_str)

    except Exception as e:
        log.debug("Context usage unavailable: %s", e)


def get_context_cache_path(tool_name: str, cache_dir: Path) -> Path:
    """
    Get the path to the context cache file for a given tool.

    Args:
        tool_name (str): Name of the tool.
        cache_dir (Path): Directory to store cache files.

    Returns:
        Path: Path to the context cache file.
    """
    cache_filename = f"context_{tool_name}.cache"
    return cache_dir / cache_filename


def save_context_to_cache(
    llm: LLM,
    tool_name: str,
    cache_dir: Path,
    logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Save LLM context to a cache file for faster future loading.

    Uses atomic writes (write to temp then rename) to prevent corruption.

    Args:
        llm (LLM): The LLM instance with context to save.
        tool_name (str): Name of the tool.
        cache_dir (Path): Directory to store cache files.
        logger_instance (logging.Logger): Logger to use.

    Returns:
        bool: True if context was saved successfully, False otherwise.
    """
    log = logger_instance or logger

    try:
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)

        cache_path = get_context_cache_path(tool_name, cache_dir)
        log.debug("Saving cache: %s", cache_path)

        # Get context data from LLM
        try:
            context_data = llm.save_context()
        except Exception as e:
            log.warning("Failed to save context: %s", e)
            return False

        if not context_data:
            log.warning("LLM returned empty context data, skipping save")
            return False

        # Atomic write: write to .tmp then rename
        temp_path = cache_path.with_suffix(".tmp")
        with open(temp_path, 'wb') as f:
            f.write(context_data)

        # Rename is atomic on POSIX
        shutil.move(str(temp_path), str(cache_path))

        log.info("Cache saved: %s", tool_name)
        return True
    except Exception as e:
        log.warning("Cache save failed: %s - %s", tool_name, e)
        # Clean up temp file
        try:
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        return False


def load_context_from_cache(
    llm: LLM, tool_name: str, cache_dir: Path, logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Load LLM context from a cache file with validation.
    No need to clear context before loading from cache.

    Args:
        llm (LLM): The LLM instance to load context into.
        tool_name (str): Name of the tool.
        cache_dir (Path): Directory to read cache files from.
        logger_instance (logging.Logger): Logger to use.

    Returns:
        bool: True if context was loaded successfully, False if file doesn't exist or load failed.
    """
    log = logger_instance or logger
    try:
        cache_path = get_context_cache_path(tool_name, cache_dir)

        if not cache_path.exists():
            log.debug("No cache: %s", tool_name)
            return False

        if cache_path.stat().st_size == 0:
            log.warning("Cache empty: %s", tool_name)
            return False

        log.debug("Loading cache: %s", cache_path)

        try:
            with open(cache_path, 'rb') as f:
                context_data = f.read()
        except Exception as e:
            log.warning("Failed to read cache file: %s", e)
            return False

        if not context_data:
            return False

        try:
            llm.load_context(context_data)
        except Exception as e:
            log.warning("Failed to load context: %s", e)
            log.debug("Cache may be corrupted: %s", cache_path)
            return False

        log.info("Cache loaded: %s", tool_name)
        return True
    except Exception as e:
        log.warning("Cache load failed: %s - %s", tool_name, e)
        return False


def add_to_context(
    llm: LLM, prompt: list, logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Add content to the LLM context by generating a minimal response.

    This is a placeholder mechanism until official API support is available.
    It works by sending the prompt and generating a single token.
    The API will automatically append an recovery sequence token to the context.

    Args:
        llm (LLM): The LLM instance.
        prompt (list): The prompt messages to add to context.
        logger_instance (logging.Logger): Logger to use.

    Returns:
        bool: True if successful, False otherwise.
    """
    log = logger_instance or logger
    try:
        # Generate a single token to add the prompt to context
        for token in llm.generate(prompt=prompt, max_generated_tokens=1):
            # Consume token to add prompt to context
            pass

        # Verify context was updated
        try:
            new_usage = llm.get_context_usage_size()
            max_capacity = llm.max_context_capacity()
            if new_usage > max_capacity:
                log.error(
                    "Context exceeds capacity after update: %d/%d tokens. "
                    "This may cause errors on next operation.",
                    new_usage, max_capacity
                )
                return False
        except Exception:
            pass  # Verification failed, but operation may have succeeded

        log.debug("Context updated")
        return True

    except Exception as e:
        log.warning("Context update failed: %s", e)
        return False

