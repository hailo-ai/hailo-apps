"""
Hailo GenAI Utilities Package.

This package contains shared utilities for Generative AI applications,
including voice processing and LLM interaction helpers.
"""

# Re-export TerminalUI from llm_utils for convenience
from .llm_utils.terminal_ui import TerminalUI

__all__ = ["TerminalUI"]
