"""
Tests for the Agent Tools Testing Framework.

Tests the testing infrastructure (harness, benchmark, metrics)
without requiring Hailo hardware.
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch


# --- Test Fixtures ---

@dataclass
class MockAgentResponse:
    """Mock response for testing."""
    tool_called: bool = False
    tool_name: str = ""
    tool_args: Dict[str, Any] = None
    tool_result: Dict[str, Any] = None
    text: str = ""
    raw_response: str = ""
    latency_ms: float = 100.0

    def __post_init__(self):
        if self.tool_args is None:
            self.tool_args = {}
        if self.tool_result is None:
            self.tool_result = {}


@dataclass
class MockTestResult:
    """Mock test result for metrics testing."""
    test_id: str
    passed: bool
    input: str
    expected: Dict[str, Any]
    actual: MockAgentResponse
    errors: List[str] = None
    latency_ms: float = 100.0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


# --- Test: Metrics ---

class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_empty_results(self):
        """Empty results returns empty dict."""
        from hailo_apps.python.gen_ai_apps.agent_tools_example.testing.metrics import (
            calculate_metrics,
        )

        result = calculate_metrics([])
        assert result == {}

    def test_all_passing(self):
        """All passing tests should have 100% accuracy."""
        from hailo_apps.python.gen_ai_apps.agent_tools_example.testing.metrics import (
            calculate_metrics,
        )

        results = [
            MockTestResult(
                test_id="test_1",
                passed=True,
                input="test input",
                expected={"tool_called": True},
                actual=MockAgentResponse(tool_called=True),
                latency_ms=100,
            ),
            MockTestResult(
                test_id="test_2",
                passed=True,
                input="test input 2",
                expected={"tool_called": True},
                actual=MockAgentResponse(tool_called=True),
                latency_ms=150,
            ),
        ]

        metrics = calculate_metrics(results)

        assert metrics["e2e_accuracy"] == 100.0
        assert metrics["tool_call_accuracy"] == 100.0
        assert metrics["test_cases_passed"] == 2
        assert metrics["test_cases_total"] == 2

    def test_mixed_results(self):
        """Mixed results calculates correct percentages."""
        from hailo_apps.python.gen_ai_apps.agent_tools_example.testing.metrics import (
            calculate_metrics,
        )

        results = [
            MockTestResult(
                test_id="test_1",
                passed=True,
                input="test",
                expected={"tool_called": True},
                actual=MockAgentResponse(tool_called=True),
                latency_ms=100,
            ),
            MockTestResult(
                test_id="test_2",
                passed=False,
                input="test",
                expected={"tool_called": True},
                actual=MockAgentResponse(tool_called=False),
                latency_ms=100,
            ),
        ]

        metrics = calculate_metrics(results)

        assert metrics["e2e_accuracy"] == 50.0
        assert metrics["tool_call_accuracy"] == 50.0

    def test_latency_calculations(self):
        """Latency metrics are calculated correctly."""
        from hailo_apps.python.gen_ai_apps.agent_tools_example.testing.metrics import (
            calculate_metrics,
        )

        results = [
            MockTestResult(
                test_id="test_1",
                passed=True,
                input="test",
                expected={},
                actual=MockAgentResponse(),
                latency_ms=100,
            ),
            MockTestResult(
                test_id="test_2",
                passed=True,
                input="test",
                expected={},
                actual=MockAgentResponse(),
                latency_ms=200,
            ),
            MockTestResult(
                test_id="test_3",
                passed=True,
                input="test",
                expected={},
                actual=MockAgentResponse(),
                latency_ms=300,
            ),
        ]

        metrics = calculate_metrics(results)

        assert metrics["avg_latency_ms"] == 200.0
        assert metrics["p50_latency_ms"] == 200.0


class TestEvaluateResponse:
    """Tests for evaluate_response function."""

    def test_tool_called_check(self):
        """Check tool_called expectation."""
        from hailo_apps.python.gen_ai_apps.agent_tools_example.testing.metrics import (
            evaluate_response,
        )

        response = MockAgentResponse(tool_called=True)
        result = evaluate_response(response, {"tool_called": True})
        assert result["passed"] is True

        result = evaluate_response(response, {"tool_called": False})
        assert result["passed"] is False
        assert len(result["errors"]) == 1

    def test_tool_name_check(self):
        """Check tool_name expectation."""
        from hailo_apps.python.gen_ai_apps.agent_tools_example.testing.metrics import (
            evaluate_response,
        )

        response = MockAgentResponse(tool_called=True, tool_name="math")
        result = evaluate_response(response, {"tool_name": "math"})
        assert result["passed"] is True

        result = evaluate_response(response, {"tool_name": "weather"})
        assert result["passed"] is False


class TestScoreState:
    """Tests for score_state function."""

    def test_perfect_score(self):
        """Perfect metrics should give high score."""
        from hailo_apps.python.gen_ai_apps.agent_tools_example.testing.metrics import (
            score_state,
        )

        metrics = {
            "tool_call_accuracy": 100.0,
            "e2e_accuracy": 100.0,
            "no_tool_precision": 100.0,
            "avg_latency_ms": 100.0,
        }

        score = score_state(metrics)
        assert score > 90

    def test_empty_metrics(self):
        """Empty metrics returns 0."""
        from hailo_apps.python.gen_ai_apps.agent_tools_example.testing.metrics import (
            score_state,
        )

        assert score_state({}) == 0.0


# --- Test: Benchmark Result ---

class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_pass_rate_calculation(self):
        """Pass rate is calculated correctly."""
        from hailo_apps.python.gen_ai_apps.agent_tools_example.testing.benchmark import (
            BenchmarkResult,
        )

        result = BenchmarkResult(
            tool_name="math",
            state_name="default",
            total_tests=10,
            passed=7,
            failed=3,
        )

        assert result.pass_rate == 70.0

    def test_pass_rate_zero_tests(self):
        """Zero tests returns 0% pass rate."""
        from hailo_apps.python.gen_ai_apps.agent_tools_example.testing.benchmark import (
            BenchmarkResult,
        )

        result = BenchmarkResult(
            tool_name="math",
            state_name="default",
            total_tests=0,
        )

        assert result.pass_rate == 0.0

    def test_to_dict(self):
        """to_dict returns serializable dict."""
        from hailo_apps.python.gen_ai_apps.agent_tools_example.testing.benchmark import (
            BenchmarkResult,
        )

        result = BenchmarkResult(
            tool_name="math",
            state_name="default",
            total_tests=5,
            passed=4,
            failed=1,
            metrics={"e2e_accuracy": 80.0},
        )

        d = result.to_dict()
        assert d["tool_name"] == "math"
        assert d["pass_rate"] == 80.0
        assert d["metrics"]["e2e_accuracy"] == 80.0

