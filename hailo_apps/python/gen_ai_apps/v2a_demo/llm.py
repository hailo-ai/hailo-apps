"""
LLM Engine — Extracts tool parameters from user text.

Interface:
    - __init__(vdevice): Initialize the engine
    - run(text, tool_name) -> dict: Extract parameters as a JSON dict
    - close(): Clean up resources

To customize:
Replace the implementation in `__init__`, `run`, and `close` methods.
"""

import json_repair
import logging
import sys
from pathlib import Path
from hailo_platform import VDevice
from hailo_platform.genai import LLM
from tools import TOOL_PROMPTS, NO_PARAM_TOOLS

try:
    from hailo_apps.python.core.common.core import resolve_hef_path
    from hailo_apps.python.core.common.defines import HAILO10H_ARCH, V2A_DEMO_APP
except ImportError:
    repo_root = None
    for p in Path(__file__).resolve().parents:
        if (p / "hailo_apps" / "config" / "config_manager.py").exists():
            repo_root = p
            break
    if repo_root is not None:
        sys.path.insert(0, str(repo_root))
    from hailo_apps.python.core.common.core import resolve_hef_path
    from hailo_apps.python.core.common.defines import HAILO10H_ARCH, V2A_DEMO_APP

logger = logging.getLogger("v2a_demo")

class LLMEngine:
    """LLM engine for tool parameter extraction."""

    def __init__(self, vdevice: VDevice):
        model_path = resolve_hef_path(
            hef_path="Qwen2.5-Coder-1.5B-Instruct",
            app_name=V2A_DEMO_APP,
            arch=HAILO10H_ARCH,
        )
        if model_path is None:
            raise RuntimeError("Failed to resolve HEF path for LLM model 'Qwen2.5-Coder-1.5B-Instruct'")
        self.llm = LLM(vdevice, str(model_path))

    def run(self, text: str, tool_name: str) -> dict:
        """Extract parameters for the given tool from user text.

        Args:
            text: The user's transcribed speech
            tool_name: The tool selected by the tool selector

        Returns:
            dict of parsed parameters for the tool
        """
        if tool_name in NO_PARAM_TOOLS:
            logger.debug(f"Tool '{tool_name}' requires no params, skipping LLM")
            return {}

        if tool_name not in TOOL_PROMPTS:
            logger.warning(f"No prompt defined for tool '{tool_name}', returning empty params")
            return {}

        self.llm.clear_context()
        raw_response = self.llm.generate_all(
            prompt=[
                {"role": "system", "content": TOOL_PROMPTS[tool_name]},
                {"role": "user", "content": text.strip()},
            ]
        )
        logger.debug(f"LLM raw response for {tool_name}: {raw_response}")

        return self._parse_params(raw_response)

    def _parse_params(self, raw_response: str) -> dict:
        """Parse parameter JSON from LLM response."""
        parsed = json_repair.loads(raw_response)
        if isinstance(parsed, dict):
            return parsed
        logger.warning(f"LLM response parsed to non-dict: {type(parsed)}")
        return {}

    def close(self):
        if self.llm:
            self.llm.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
