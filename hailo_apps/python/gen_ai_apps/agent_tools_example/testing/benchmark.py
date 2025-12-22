"""
Benchmark Runner.

Runs test cases from YAML configs and collects metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .harness import AgentTestHarness, TestResult
from .metrics import calculate_metrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """
    Results from a benchmark run.

    Attributes:
        tool_name: Name of the tool tested.
        state_name: Context state used.
        total_tests: Total number of tests run.
        passed: Number of tests passed.
        failed: Number of tests failed.
        results: Individual test results.
        metrics: Calculated metrics dict.
        duration_ms: Total benchmark duration.
    """

    tool_name: str
    state_name: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    results: List[TestResult] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tool_name": self.tool_name,
            "state_name": self.state_name,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "metrics": self.metrics,
            "duration_ms": self.duration_ms,
            "results": [
                {
                    "test_id": r.test_id,
                    "passed": r.passed,
                    "input": r.input,
                    "errors": r.errors,
                    "latency_ms": r.latency_ms,
                }
                for r in self.results
            ],
        }


class BenchmarkRunner:
    """
    Runs benchmarks using test cases from YAML configs.
    """

    def __init__(
        self,
        tool_name: str,
        state_name: str = "default",
        hef_path: Optional[str] = None,
        reset_between_tests: bool = True,
    ):
        """
        Initialize benchmark runner.

        Args:
            tool_name: Name of the tool to benchmark.
            state_name: Context state to use.
            hef_path: Optional HEF path override.
            reset_between_tests: Reset context between tests.
        """
        self.tool_name = tool_name
        self.state_name = state_name
        self.hef_path = hef_path
        self.reset_between_tests = reset_between_tests
        self._harness: Optional[AgentTestHarness] = None

    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """
        Load test cases from tool's config.yaml.

        Returns:
            List of test case dicts.
        """
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML not installed")
            return []

        # Find config.yaml
        tool_dir = (
            Path(__file__).parent.parent / "tools" / self.tool_name
        )
        config_path = tool_dir / "config.yaml"

        if not config_path.exists():
            logger.warning("No config.yaml found for tool: %s", self.tool_name)
            return []

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            return []

        return config.get("test_cases", [])

    def run(
        self,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Run benchmark with test cases.

        Args:
            test_cases: Optional list of test cases. If None, loads from config.yaml.
            tags: Optional list of tags to filter tests.
            difficulty: Optional difficulty filter ('easy', 'medium', 'hard').

        Returns:
            BenchmarkResult with all results and metrics.
        """
        start_time = time.perf_counter()

        # Load test cases
        if test_cases is None:
            test_cases = self._load_test_cases()

        if not test_cases:
            logger.warning("No test cases found")
            return BenchmarkResult(
                tool_name=self.tool_name,
                state_name=self.state_name,
            )

        # Filter by tags
        if tags:
            test_cases = [
                tc for tc in test_cases
                if any(tag in tc.get("tags", []) for tag in tags)
            ]

        # Filter by difficulty
        if difficulty:
            test_cases = [
                tc for tc in test_cases
                if tc.get("difficulty") == difficulty
            ]

        logger.info("Running %d test cases for %s", len(test_cases), self.tool_name)

        # Initialize harness
        self._harness = AgentTestHarness(
            tool_name=self.tool_name,
            state_name=self.state_name,
            hef_path=self.hef_path,
            headless=True,
        )

        results: List[TestResult] = []
        passed = 0
        failed = 0

        try:
            for i, test_case in enumerate(test_cases, 1):
                test_id = test_case.get("id", f"test_{i}")
                logger.info("Running test %d/%d: %s", i, len(test_cases), test_id)

                # Reset context if needed
                if self.reset_between_tests and i > 1:
                    self._harness.reset_context()

                # Run test
                result = self._harness.run_test_case(test_case)
                results.append(result)

                if result.passed:
                    passed += 1
                    logger.info("  ✓ PASSED (%.1fms)", result.latency_ms)
                else:
                    failed += 1
                    logger.info("  ✗ FAILED: %s", "; ".join(result.errors))

        finally:
            self._harness.close()
            self._harness = None

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Calculate metrics
        metrics = calculate_metrics(results)

        return BenchmarkResult(
            tool_name=self.tool_name,
            state_name=self.state_name,
            total_tests=len(test_cases),
            passed=passed,
            failed=failed,
            results=results,
            metrics=metrics,
            duration_ms=duration_ms,
        )

    def run_quick(self, num_tests: int = 3) -> BenchmarkResult:
        """
        Run a quick benchmark with limited tests.

        Args:
            num_tests: Maximum number of tests to run.

        Returns:
            BenchmarkResult with results.
        """
        test_cases = self._load_test_cases()[:num_tests]
        return self.run(test_cases=test_cases)

