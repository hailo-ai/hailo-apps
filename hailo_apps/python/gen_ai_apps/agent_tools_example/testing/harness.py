"""
Agent Test Harness.

Provides a programmatic interface for testing agents without UI.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hailo_platform.genai import LLM

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """
    Response from agent query processing.

    Attributes:
        tool_called: Whether a tool was called.
        tool_name: Name of the tool called (if any).
        tool_args: Arguments passed to the tool.
        tool_result: Result from tool execution.
        text: Final text response.
        raw_response: Raw LLM response before processing.
        latency_ms: Response latency in milliseconds.
    """

    tool_called: bool = False
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[Dict[str, Any]] = None
    text: str = ""
    raw_response: str = ""
    latency_ms: float = 0.0


@dataclass
class TestResult:
    """
    Result of a single test case evaluation.

    Attributes:
        test_id: Test case identifier.
        passed: Whether the test passed.
        input: Test input query.
        expected: Expected results.
        actual: Actual response.
        errors: List of error messages.
        latency_ms: Response latency.
    """

    test_id: str
    passed: bool
    input: str
    expected: Dict[str, Any]
    actual: AgentResponse
    errors: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


class AgentTestHarness:
    """
    Programmatic interface for testing agents.

    Allows sending queries and evaluating responses without
    interactive UI, suitable for automated testing and benchmarks.
    """

    def __init__(
        self,
        tool_name: str,
        state_name: str = "default",
        hef_path: Optional[str] = None,
        headless: bool = True,
    ):
        """
        Initialize test harness for a tool.

        Args:
            tool_name: Name of the tool to test.
            state_name: Context state to load.
            hef_path: Optional HEF path override.
            headless: Run without UI output.
        """
        self.tool_name = tool_name
        self.state_name = state_name
        self.headless = headless
        self._initialized = False

        # Lazy imports to avoid loading Hailo during import
        self._llm: Optional["LLM"] = None
        self._vdevice = None
        self._tool = None
        self._tools_lookup: Dict[str, Any] = {}

        # Store hef_path for lazy init
        self._hef_path = hef_path

    def _ensure_initialized(self) -> None:
        """Ensure LLM and tool are initialized."""
        if self._initialized:
            return

        from hailo_platform import VDevice
        from hailo_platform.genai import LLM

        from hailo_apps.python.core.common.core import resolve_hef_path
        from hailo_apps.python.core.common.defines import AGENT_APP, HAILO10H_ARCH
        from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import (
            context_manager,
            message_formatter,
            tool_discovery,
            tool_execution,
        )

        # Import local modules
        import sys

        agent_dir = Path(__file__).parent.parent
        if str(agent_dir) not in sys.path:
            sys.path.insert(0, str(agent_dir))

        from state_manager import StateManager
        import system_prompt

        # Resolve HEF path
        hef_path = self._hef_path
        if not hef_path:
            hef_path = resolve_hef_path(
                hef_path="Qwen2.5-Coder-1.5B-Instruct",
                app_name=AGENT_APP,
                arch=HAILO10H_ARCH,
            )

        if not hef_path:
            raise RuntimeError("Failed to resolve HEF path")

        # Initialize VDevice and LLM
        self._vdevice = VDevice()
        self._llm = LLM(self._vdevice, str(hef_path))

        # Discover and find the tool
        modules = tool_discovery.discover_tool_modules(tool_dir=agent_dir)
        all_tools = tool_discovery.collect_tools(modules)

        for tool in all_tools:
            if tool["name"] == self.tool_name:
                self._tool = tool
                break

        if not self._tool:
            available = [t["name"] for t in all_tools]
            raise ValueError(f"Tool '{self.tool_name}' not found. Available: {available}")

        # Initialize tool
        tool_execution.initialize_tool_if_needed(self._tool)
        self._tools_lookup = {self.tool_name: self._tool}

        # Initialize context
        self._state_manager = StateManager(
            tool_name=self.tool_name,
            contexts_dir=agent_dir / "tools" / self.tool_name / "contexts",
        )

        # Try to load state
        if not self._state_manager.load_state(self.state_name, self._llm):
            # Initialize fresh context
            system_text = system_prompt.create_system_prompt([self._tool])
            prompt = [message_formatter.messages_system(system_text)]
            context_manager.add_to_context(self._llm, prompt, logger)

        self._initialized = True

    def send_query(self, text: str) -> AgentResponse:
        """
        Send a query and get structured response.

        Args:
            text: User query text.

        Returns:
            AgentResponse with results.
        """
        self._ensure_initialized()

        from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import (
            message_formatter,
            streaming,
            tool_parsing,
            tool_execution,
        )

        import config

        start_time = time.perf_counter()

        # Build prompt using gen_ai_utils message_formatter
        prompt = [message_formatter.messages_user(text)]

        # Generate response using gen_ai_utils streaming function (same as main agent)
        # Use headless mode: suppress output by redirecting print
        import io
        import sys
        from contextlib import redirect_stdout

        try:
            # Suppress stdout for headless testing
            f = io.StringIO()
            with redirect_stdout(f):
                raw_response = streaming.generate_and_stream_response(
                    llm=self._llm,
                    prompt=prompt,  # Use same approach as main agent
                    temperature=config.TEMPERATURE,
                    seed=config.SEED,
                    max_tokens=config.MAX_GENERATED_TOKENS,
                    prefix="",  # No prefix in headless mode
                    show_raw_stream=True,  # Show raw stream for debugging (output redirected anyway)
                    token_callback=None,  # No TTS callback in headless mode
                )
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return AgentResponse(
                text=f"Error: {e}",
                raw_response="",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Parse tool call
        tool_call = tool_parsing.parse_function_call(raw_response)

        if tool_call is None:
            # Log raw response for debugging if it looks like it might contain a tool call
            if "<tool_call>" in raw_response or '{"name"' in raw_response:
                logger.warning(
                    "Tool call XML found in response but parsing failed. "
                    "Raw response (first 500 chars): %s",
                    raw_response[:500]
                )
            return AgentResponse(
                tool_called=False,
                text=raw_response,
                raw_response=raw_response,
                latency_ms=latency_ms,
            )

        # Log parsed tool call for debugging
        logger.debug("Parsed tool call: name=%s, args=%s", tool_call.get("name"), tool_call.get("arguments"))

        # Execute tool
        result = tool_execution.execute_tool_call(tool_call, self._tools_lookup)

        if not result.get("ok"):
            logger.error("Tool execution failed: %s", result.get("error", "Unknown error"))

        # Update context with tool result (like the main agent does)
        from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import agent_utils
        agent_utils.update_context_with_tool_result(self._llm, result, logger)

        return AgentResponse(
            tool_called=True,
            tool_name=tool_call.get("name", ""),
            tool_args=tool_call.get("arguments", {}),
            tool_result=result,
            text=str(result.get("result", "")),
            raw_response=raw_response,
            latency_ms=latency_ms,
        )

    def run_test_case(self, test_case: Dict[str, Any]) -> TestResult:
        """
        Run a single test case and check expectations.

        Args:
            test_case: Test case dict with 'id', 'input', 'expected'.

        Returns:
            TestResult with pass/fail and details.
        """
        test_id = test_case.get("id", "unknown")
        input_text = test_case.get("input", "")
        expected = test_case.get("expected", {})

        # Send query
        response = self.send_query(input_text)

        # Evaluate
        errors: List[str] = []

        # Check tool_called
        if "tool_called" in expected:
            if expected["tool_called"] != response.tool_called:
                errors.append(
                    f"tool_called: expected {expected['tool_called']}, got {response.tool_called}"
                )

        # Check tool_name
        if "tool_name" in expected and response.tool_called:
            if expected["tool_name"] != response.tool_name:
                errors.append(
                    f"tool_name: expected {expected['tool_name']}, got {response.tool_name}"
                )

        # Check result_contains
        if "result_contains" in expected and response.tool_result:
            result_str = str(response.tool_result.get("result", ""))
            if expected["result_contains"] not in result_str:
                errors.append(
                    f"result_contains: '{expected['result_contains']}' not in '{result_str}'"
                )

        passed = len(errors) == 0

        return TestResult(
            test_id=test_id,
            passed=passed,
            input=input_text,
            expected=expected,
            actual=response,
            errors=errors,
            latency_ms=response.latency_ms,
        )

    def reset_context(self) -> None:
        """Reset to initial state (for isolated tests)."""
        if self._initialized and self._state_manager:
            self._state_manager.reload_state(self._llm)

    def close(self) -> None:
        """Clean up resources."""
        if self._tool and hasattr(self._tool.get("module"), "cleanup_tool"):
            try:
                self._tool["module"].cleanup_tool()
            except Exception:
                pass

        if self._llm:
            try:
                self._llm.release()
            except Exception:
                pass

        if self._vdevice:
            try:
                self._vdevice.release()
            except Exception:
                pass

        self._initialized = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

