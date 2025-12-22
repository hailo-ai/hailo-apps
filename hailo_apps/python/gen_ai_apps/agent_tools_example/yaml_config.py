"""
YAML configuration loader for tools.

Loads and parses tool configuration from YAML files, including:
- Persona and capabilities
- Tool instructions
- Few-shot examples
- Test cases
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import yaml, provide fallback if not available
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed. YAML configuration will not be available.")


@dataclass
class FewShotExample:
    """
    A single few-shot example for context priming.

    Attributes:
        user: User query text.
        tool_call: Tool call dict (name, arguments) if tool should be called.
        tool_response: Expected tool response string.
        final_response: Expected assistant response after tool execution.
    """

    user: str
    tool_call: Optional[Dict[str, Any]] = None
    tool_response: Optional[str] = None
    final_response: str = ""


@dataclass
class TestCase:
    """
    A test case for benchmarking.

    Attributes:
        id: Unique test case identifier.
        input: User input query.
        expected: Expected results dict.
        tags: List of tags for filtering.
        difficulty: Difficulty level (easy, medium, hard).
    """

    id: str
    input: str
    expected: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"


@dataclass
class ToolYamlConfig:
    """
    Complete tool configuration loaded from YAML.

    Attributes:
        version: Config file version.
        tool_name: Name of the tool (must match tool.py).
        persona: Persona configuration.
        capabilities: List of tool capabilities.
        output_format: Output formatting rules.
        tool_instructions: Detailed LLM instructions.
        few_shot_examples: List of few-shot examples.
        test_cases: List of test cases.
        metadata: Additional metadata.
        raw_config: Original raw YAML dict.
    """

    version: str = "1.0"
    tool_name: str = ""
    persona: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    output_format: Dict[str, str] = field(default_factory=dict)
    tool_instructions: str = ""
    few_shot_examples: List[FewShotExample] = field(default_factory=list)
    test_cases: List[TestCase] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def get_system_prompt_components(self) -> Dict[str, str]:
        """
        Get components for building system prompt.

        Returns:
            Dict with keys: persona, capabilities, output_format, tool_instructions.
        """
        # Build persona text
        persona_parts = []
        if self.persona.get("role"):
            persona_parts.append(f"Role: {self.persona['role']}")
        if self.persona.get("style"):
            persona_parts.append(f"Style: {self.persona['style']}")
        if self.persona.get("constraints"):
            constraints = self.persona["constraints"]
            if isinstance(constraints, list):
                persona_parts.append("Constraints:")
                for c in constraints:
                    persona_parts.append(f"  - {c}")

        # Build capabilities text
        capabilities_text = ""
        if self.capabilities:
            capabilities_text = "\n".join(f"- {c}" for c in self.capabilities)

        # Build output format text
        output_parts = []
        for key, value in self.output_format.items():
            output_parts.append(f"- {key}: {value}")
        output_format_text = "\n".join(output_parts)

        return {
            "persona": "\n".join(persona_parts),
            "capabilities": capabilities_text,
            "output_format": output_format_text,
            "tool_instructions": self.tool_instructions,
        }


def load_yaml_config(config_path: Path) -> Optional[ToolYamlConfig]:
    """
    Load tool configuration from a YAML file.

    Args:
        config_path: Path to the config.yaml file.

    Returns:
        ToolYamlConfig if successful, None if file doesn't exist or parsing fails.
    """
    if not YAML_AVAILABLE:
        logger.warning("YAML not available, cannot load config: %s", config_path)
        return None

    if not config_path.exists():
        logger.debug("Config file not found: %s", config_path)
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except Exception as e:
        logger.error("Failed to parse YAML config %s: %s", config_path, e)
        return None

    if not isinstance(raw_config, dict):
        logger.error("Invalid YAML config format (expected dict): %s", config_path)
        return None

    # Parse few-shot examples
    few_shot_examples = []
    for ex in raw_config.get("few_shot_examples", []):
        if not isinstance(ex, dict):
            continue
        few_shot_examples.append(
            FewShotExample(
                user=ex.get("user", ""),
                tool_call=ex.get("tool_call"),
                tool_response=ex.get("tool_response"),
                final_response=ex.get("final_response", ""),
            )
        )

    # Parse test cases
    test_cases = []
    for tc in raw_config.get("test_cases", []):
        if not isinstance(tc, dict):
            continue
        test_cases.append(
            TestCase(
                id=tc.get("id", ""),
                input=tc.get("input", ""),
                expected=tc.get("expected", {}),
                tags=tc.get("tags", []),
                difficulty=tc.get("difficulty", "medium"),
            )
        )

    return ToolYamlConfig(
        version=raw_config.get("version", "1.0"),
        tool_name=raw_config.get("tool_name", ""),
        persona=raw_config.get("persona", {}),
        capabilities=raw_config.get("capabilities", []),
        output_format=raw_config.get("output_format", {}),
        tool_instructions=raw_config.get("tool_instructions", ""),
        few_shot_examples=few_shot_examples,
        test_cases=test_cases,
        metadata=raw_config.get("metadata", {}),
        raw_config=raw_config,
    )


def find_tool_config(tool_dir: Path) -> Optional[Path]:
    """
    Find the config.yaml file for a tool.

    Args:
        tool_dir: Path to the tool directory.

    Returns:
        Path to config.yaml if found, None otherwise.
    """
    config_path = tool_dir / "config.yaml"
    if config_path.exists():
        return config_path

    # Try alternative names
    for alt_name in ["config.yml", "tool_config.yaml", "tool_config.yml"]:
        alt_path = tool_dir / alt_name
        if alt_path.exists():
            return alt_path

    return None

