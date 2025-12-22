"""
Evaluation Metrics.

Calculates accuracy, latency, and other metrics from test results.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .harness import TestResult


def calculate_metrics(results: List["TestResult"]) -> Dict[str, float]:
    """
    Calculate metrics from test results.

    Args:
        results: List of TestResult objects.

    Returns:
        Dictionary with metrics:
        - tool_call_accuracy: % of correct tool calls
        - e2e_accuracy: % of fully passing tests
        - avg_latency_ms: Average latency
        - p50_latency_ms: Median latency
        - p95_latency_ms: 95th percentile latency
        - no_tool_precision: % of correctly NOT calling tools
    """
    if not results:
        return {}

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    latencies = [r.latency_ms for r in results]

    # Calculate tool call accuracy
    # How many times the tool was called/not called correctly
    tool_call_correct = 0
    tool_call_total = 0
    no_tool_correct = 0
    no_tool_total = 0

    for r in results:
        expected_tool_called = r.expected.get("tool_called")
        if expected_tool_called is True:
            tool_call_total += 1
            if r.actual.tool_called:
                tool_call_correct += 1
        elif expected_tool_called is False:
            no_tool_total += 1
            if not r.actual.tool_called:
                no_tool_correct += 1

    # Accuracy calculations
    tool_call_accuracy = (
        (tool_call_correct / tool_call_total * 100)
        if tool_call_total > 0
        else 100.0
    )
    no_tool_precision = (
        (no_tool_correct / no_tool_total * 100)
        if no_tool_total > 0
        else 100.0
    )
    e2e_accuracy = (passed / total * 100) if total > 0 else 0.0

    # Latency calculations
    sorted_latencies = sorted(latencies)
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p50_latency = (
        sorted_latencies[len(sorted_latencies) // 2]
        if latencies
        else 0.0
    )
    p95_idx = int(len(sorted_latencies) * 0.95)
    p95_latency = sorted_latencies[p95_idx] if latencies else 0.0

    return {
        "tool_call_accuracy": round(tool_call_accuracy, 2),
        "e2e_accuracy": round(e2e_accuracy, 2),
        "no_tool_precision": round(no_tool_precision, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "test_cases_passed": passed,
        "test_cases_total": total,
    }


def evaluate_response(
    response: Any,
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate a single response against expectations.

    Args:
        response: AgentResponse object.
        expected: Expected results dict.

    Returns:
        Dictionary with evaluation results:
        - passed: bool
        - errors: list of error messages
        - checks: dict of individual check results
    """
    errors: List[str] = []
    checks: Dict[str, bool] = {}

    # Check tool_called
    if "tool_called" in expected:
        check_passed = expected["tool_called"] == response.tool_called
        checks["tool_called"] = check_passed
        if not check_passed:
            errors.append(
                f"tool_called: expected {expected['tool_called']}, "
                f"got {response.tool_called}"
            )

    # Check tool_name
    if "tool_name" in expected and response.tool_called:
        check_passed = expected["tool_name"] == response.tool_name
        checks["tool_name"] = check_passed
        if not check_passed:
            errors.append(
                f"tool_name: expected {expected['tool_name']}, "
                f"got {response.tool_name}"
            )

    # Check result_contains
    if "result_contains" in expected and response.tool_result:
        result_str = str(response.tool_result.get("result", ""))
        check_passed = expected["result_contains"] in result_str
        checks["result_contains"] = check_passed
        if not check_passed:
            errors.append(
                f"result_contains: '{expected['result_contains']}' "
                f"not in '{result_str[:100]}...'"
            )

    # Check response_contains (for non-tool responses)
    if "response_contains" in expected and not response.tool_called:
        check_passed = expected["response_contains"] in response.text
        checks["response_contains"] = check_passed
        if not check_passed:
            errors.append(
                f"response_contains: '{expected['response_contains']}' "
                f"not in response"
            )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "checks": checks,
    }


def score_state(metrics: Dict[str, float]) -> float:
    """
    Calculate weighted score for state ranking.

    Args:
        metrics: Metrics dictionary from calculate_metrics.

    Returns:
        Weighted score (0-100).
    """
    if not metrics:
        return 0.0

    # Weights for different metrics
    weights = {
        "tool_call_accuracy": 0.40,
        "e2e_accuracy": 0.30,
        "no_tool_precision": 0.10,
        "latency_score": 0.20,
    }

    # Calculate latency score (inverse - lower is better)
    # Target: 200ms = 100%, 500ms = 0%
    avg_latency = metrics.get("avg_latency_ms", 500)
    latency_score = max(0, min(100, (500 - avg_latency) / 3))

    score = (
        weights["tool_call_accuracy"] * metrics.get("tool_call_accuracy", 0)
        + weights["e2e_accuracy"] * metrics.get("e2e_accuracy", 0)
        + weights["no_tool_precision"] * metrics.get("no_tool_precision", 0)
        + weights["latency_score"] * latency_score
    )

    return round(score, 2)

