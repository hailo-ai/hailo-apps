"""
Base classes for tools.

Provides optional abstract base class and type definitions for tools.
Tools can inherit from BaseTool for a structured approach, or simply
define the required module-level attributes (name, description, schema, run, TOOLS_SCHEMA).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """
    Standardized result from tool execution.

    Attributes:
        ok: Whether the tool execution succeeded.
        result: The result value (if ok=True).
        error: Error message (if ok=False).
    """

    ok: bool
    result: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format expected by the agent.

        Returns:
            Dictionary with 'ok' and either 'result' or 'error'.
        """
        if self.ok:
            return {"ok": True, "result": self.result}
        return {"ok": False, "error": self.error or "Unknown error"}

    @classmethod
    def success(cls, result: Any) -> "ToolResult":
        """
        Create a successful result.

        Args:
            result: The result value.

        Returns:
            ToolResult with ok=True.
        """
        return cls(ok=True, result=result)

    @classmethod
    def failure(cls, error: str) -> "ToolResult":
        """
        Create a failure result.

        Args:
            error: The error message.

        Returns:
            ToolResult with ok=False.
        """
        return cls(ok=False, error=error)


@dataclass
class ToolConfig:
    """
    Tool configuration loaded from YAML.

    Attributes:
        version: Config version string.
        tool_name: Name of the tool (must match tool.py name).
        persona: Persona configuration dict.
        capabilities: List of capability descriptions.
        output_format: Output format rules dict.
        tool_instructions: Detailed instructions for the LLM.
        few_shot_examples: List of few-shot example dicts.
        test_cases: List of test case dicts.
        metadata: Additional metadata dict.
    """

    version: str = "1.0"
    tool_name: str = ""
    persona: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    output_format: Dict[str, str] = field(default_factory=dict)
    tool_instructions: str = ""
    few_shot_examples: List[Dict[str, Any]] = field(default_factory=list)
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """
    Abstract base class for tools.

    Provides a structured approach to creating tools. Subclasses must implement
    the abstract methods and properties.

    Usage:
        class MyTool(BaseTool):
            @property
            def name(self) -> str:
                return "my_tool"

            @property
            def description(self) -> str:
                return "My tool description"

            @property
            def schema(self) -> Dict[str, Any]:
                return {"type": "object", "properties": {...}}

            def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                # Tool logic here
                return {"ok": True, "result": "..."}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this tool.

        Returns:
            Tool name string.
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Description of this tool for the LLM.

        Returns:
            Description string.
        """
        ...

    @property
    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """
        JSON schema for tool parameters.

        Returns:
            Schema dictionary following OpenAI function calling format.
        """
        ...

    @property
    def display_description(self) -> str:
        """
        User-facing description for CLI display.

        Defaults to the LLM description if not overridden.

        Returns:
            Display description string.
        """
        return self.description

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool logic.

        Args:
            input_data: Tool input parameters.

        Returns:
            Dictionary with 'ok' and 'result' (success) or 'error' (failure).
        """
        ...

    def initialize(self) -> None:
        """
        Initialize the tool (optional).

        Called once when the tool is first loaded. Override for setup logic.
        """
        pass

    def cleanup(self) -> None:
        """
        Clean up tool resources (optional).

        Called when the agent shuts down. Override for cleanup logic.
        """
        pass

    @property
    def tools_schema(self) -> List[Dict[str, Any]]:
        """
        Generate TOOLS_SCHEMA from tool properties.

        Returns:
            List containing the tool definition in OpenAI function calling format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.schema,
                },
            }
        ]

