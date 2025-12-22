"""
CLI Test Runner.

Command-line interface for running benchmarks and tests.

Usage:
    # Run all tests for a tool
    python -m hailo_apps.python.gen_ai_apps.agent_tools_example.testing.runner math

    # Run quick test (3 tests)
    python -m hailo_apps.python.gen_ai_apps.agent_tools_example.testing.runner math --quick

    # Filter by tags
    python -m hailo_apps.python.gen_ai_apps.agent_tools_example.testing.runner math --tags basic

    # Save results
    python -m hailo_apps.python.gen_ai_apps.agent_tools_example.testing.runner math --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add parent to path for imports
script_dir = Path(__file__).parent.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(name)s | %(message)s",
    )


def list_tools() -> List[str]:
    """Get list of available tools."""
    tools_dir = Path(__file__).parent.parent / "tools"
    if not tools_dir.exists():
        return []

    tools = []
    for item in tools_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            if (item / "config.yaml").exists():
                tools.append(item.name)
    return sorted(tools)


def print_banner() -> None:
    """Print CLI banner."""
    print("\n" + "=" * 60)
    print("   Agent Tools Test Runner")
    print("=" * 60 + "\n")


def print_results(result, verbose: bool = False) -> None:
    """Print benchmark results to console."""
    print("\n" + "-" * 50)
    print(f"Tool: {result.tool_name}")
    print(f"State: {result.state_name}")
    print("-" * 50)

    # Summary
    print(f"\nResults: {result.passed}/{result.total_tests} passed ({result.pass_rate:.1f}%)")
    print(f"Duration: {result.duration_ms:.0f}ms")

    # Metrics
    if result.metrics:
        print("\nMetrics:")
        for key, value in result.metrics.items():
            if "latency" in key:
                print(f"  {key}: {value:.1f}ms")
            elif "accuracy" in key or "precision" in key:
                print(f"  {key}: {value:.1f}%")
            else:
                print(f"  {key}: {value}")

    # Failed tests
    failed = [r for r in result.results if not r.passed]
    if failed:
        print(f"\nFailed Tests ({len(failed)}):")
        for r in failed:
            print(f"  - {r.test_id}: {r.input[:50]}...")
            for err in r.errors:
                print(f"      {err}")

    # Detailed results in verbose mode
    if verbose:
        print("\nAll Test Results:")
        for r in result.results:
            status = "✓" if r.passed else "✗"
            print(f"  {status} {r.test_id} ({r.latency_ms:.1f}ms)")

    print("-" * 50 + "\n")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks for agent tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s math              Run all tests for math tool
  %(prog)s math --quick      Run quick test (3 tests)
  %(prog)s --list            List available tools
  %(prog)s math --tags basic Filter by tags
        """,
    )

    parser.add_argument(
        "tool",
        nargs="?",
        help="Tool name to benchmark",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tools",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (3 tests)",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Filter tests by tags",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        help="Filter tests by difficulty",
    )
    parser.add_argument(
        "--state",
        default="default",
        help="Context state to use (default: 'default')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Don't reset context between tests",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    print_banner()

    # List tools
    if args.list:
        tools = list_tools()
        if tools:
            print("Available tools:")
            for tool in tools:
                print(f"  - {tool}")
        else:
            print("No tools found.")
        return 0

    # Check tool specified
    if not args.tool:
        parser.print_help()
        return 1

    # Check tool exists
    available = list_tools()
    if args.tool not in available:
        print(f"Error: Tool '{args.tool}' not found.")
        print(f"Available: {', '.join(available)}")
        return 1

    # Run benchmark
    print(f"Running benchmark for: {args.tool}")
    print(f"State: {args.state}")
    if args.tags:
        print(f"Tags filter: {args.tags}")
    if args.difficulty:
        print(f"Difficulty filter: {args.difficulty}")
    print()

    try:
        from .benchmark import BenchmarkRunner

        runner = BenchmarkRunner(
            tool_name=args.tool,
            state_name=args.state,
            reset_between_tests=not args.no_reset,
        )

        if args.quick:
            result = runner.run_quick()
        else:
            result = runner.run(
                tags=args.tags,
                difficulty=args.difficulty,
            )

        print_results(result, verbose=args.verbose)

        # Save to file
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"Results saved to: {output_path}")

        # Return exit code based on pass rate
        return 0 if result.pass_rate == 100 else 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        logging.error("Benchmark failed: %s", e)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

