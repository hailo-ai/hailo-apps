#!/usr/bin/env python3
"""
State Management CLI Tool.

Manage context states for agent tools:
- List saved states
- Delete states
- View state details
- Get best performing state

Usage:
    python -m hailo_apps.python.gen_ai_apps.agent_tools_example.cli_state math list
    python -m hailo_apps.python.gen_ai_apps.agent_tools_example.cli_state math delete default
    python -m hailo_apps.python.gen_ai_apps.agent_tools_example.cli_state math info optimized_v2
    python -m hailo_apps.python.gen_ai_apps.agent_tools_example.cli_state math best
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent to path
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from state_manager import StateManager


def cmd_list(tool_name: str) -> int:
    """List all saved states for a tool."""
    manager = StateManager(tool_name)
    states = manager.list_states()

    if not states:
        print(f"No saved states found for tool: {tool_name}")
        return 0

    print(f"\nSaved states for '{tool_name}':")
    print("-" * 80)
    print(f"{'State Name':<20} {'Tokens':<10} {'Accuracy':<12} {'Latency':<12} {'Created'}")
    print("-" * 80)

    for state in states:
        perf = state.performance
        accuracy = f"{perf.e2e_accuracy:.1f}%" if perf.e2e_accuracy > 0 else "N/A"
        latency = f"{perf.avg_latency_ms:.0f}ms" if perf.avg_latency_ms > 0 else "N/A"
        created = state.created[:10] if state.created else "N/A"

        current = " *" if state.state_name == manager.current_state else ""
        print(
            f"{state.state_name:<20} {state.context_tokens:<10} "
            f"{accuracy:<12} {latency:<12} {created}{current}"
        )

    print("-" * 80)
    print(f"Total: {len(states)} states")
    return 0


def cmd_info(tool_name: str, state_name: str) -> int:
    """Show detailed information about a state."""
    manager = StateManager(tool_name)
    info = manager.get_state_info(state_name)

    if not info:
        print(f"State '{state_name}' not found for tool '{tool_name}'")
        return 1

    print(f"\nState: {info.state_name}")
    print("-" * 60)
    print(f"Created:     {info.created}")
    print(f"Context:     {info.context_tokens} tokens")
    print(f"Few-shots:   {info.few_shot_count}")
    print(f"YAML Hash:   {info.yaml_hash}")
    if info.parent_state:
        print(f"Parent:      {info.parent_state}")
    if info.notes:
        print(f"Notes:       {info.notes}")

    perf = info.performance
    if perf.e2e_accuracy > 0 or perf.avg_latency_ms > 0:
        print("\nPerformance Metrics:")
        print(f"  E2E Accuracy:      {perf.e2e_accuracy:.2f}%")
        print(f"  Tool Call Acc:     {perf.tool_call_accuracy:.2f}%")
        print(f"  No-Tool Precision: {perf.no_tool_precision:.2f}%")
        print(f"  Avg Latency:       {perf.avg_latency_ms:.2f}ms")
        print(f"  P95 Latency:       {perf.p95_latency_ms:.2f}ms")
        print(f"  Tests Passed:      {perf.test_cases_passed}/{perf.test_cases_total}")

    # Check if files exist
    state_path = manager._get_state_path(state_name)
    yaml_path = manager._get_yaml_path(state_name)
    meta_path = manager._get_meta_path(state_name)

    print("\nFiles:")
    print(f"  State:  {'✓' if state_path.exists() else '✗'} {state_path}")
    print(f"  YAML:   {'✓' if yaml_path.exists() else '✗'} {yaml_path}")
    print(f"  Meta:   {'✓' if meta_path.exists() else '✗'} {meta_path}")

    return 0


def cmd_delete(tool_name: str, state_name: str, force: bool = False) -> int:
    """Delete a saved state."""
    manager = StateManager(tool_name)

    if not manager._get_state_path(state_name).exists():
        print(f"State '{state_name}' not found for tool '{tool_name}'")
        return 1

    if not force:
        response = input(f"Delete state '{state_name}'? [y/N]: ").strip().lower()
        if response != "y":
            print("Cancelled.")
            return 0

    if manager.delete_state(state_name):
        print(f"Deleted state: {state_name}")
        return 0
    else:
        print(f"Failed to delete state: {state_name}")
        return 1


def cmd_best(tool_name: str, metric: str = "tool_call_accuracy") -> int:
    """Show the best performing state."""
    manager = StateManager(tool_name)
    best_name = manager.get_best_state(metric)

    if not best_name:
        print(f"No states found for tool: {tool_name}")
        return 1

    print(f"Best state by {metric}: {best_name}")
    return cmd_info(tool_name, best_name)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manage context states for agent tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s math list                    List all states for math tool
  %(prog)s math info default            Show details for 'default' state
  %(prog)s math delete old_state        Delete a state (with confirmation)
  %(prog)s math delete old_state --force Delete without confirmation
  %(prog)s math best                    Show best state by accuracy
  %(prog)s math best --metric e2e_accuracy  Show best by E2E accuracy
        """,
    )

    parser.add_argument(
        "tool",
        help="Tool name (e.g., 'math', 'weather', 'rgb_led')",
    )
    parser.add_argument(
        "command",
        choices=["list", "info", "delete", "best"],
        help="Command to execute",
    )
    parser.add_argument(
        "state_name",
        nargs="?",
        help="State name (required for 'info' and 'delete' commands)",
    )
    parser.add_argument(
        "--metric",
        default="tool_call_accuracy",
        choices=["tool_call_accuracy", "e2e_accuracy", "no_tool_precision"],
        help="Metric for 'best' command (default: tool_call_accuracy)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation for delete command",
    )

    args = parser.parse_args()

    # Validate state_name for commands that need it
    if args.command in ("info", "delete") and not args.state_name:
        parser.error(f"Command '{args.command}' requires a state name")

    # Execute command
    if args.command == "list":
        return cmd_list(args.tool)
    elif args.command == "info":
        return cmd_info(args.tool, args.state_name)
    elif args.command == "delete":
        return cmd_delete(args.tool, args.state_name, force=args.force)
    elif args.command == "best":
        return cmd_best(args.tool, metric=args.metric)
    else:
        parser.error(f"Unknown command: {args.command}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

