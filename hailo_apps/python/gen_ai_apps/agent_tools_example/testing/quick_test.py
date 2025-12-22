#!/usr/bin/env python3
"""
Quick integration test for agent tools.

This script tests tool discovery and basic functionality without
requiring the LLM/Hailo hardware. Use the full benchmark runner
for tests that require the LLM.

Usage:
    python -m hailo_apps.python.gen_ai_apps.agent_tools_example.testing.quick_test
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent to path
script_dir = Path(__file__).parent.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))


def test_tool_discovery():
    """Test that all tools are discovered correctly."""
    print("=" * 60)
    print("Test: Tool Discovery")
    print("=" * 60)

    from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import tool_discovery

    modules = tool_discovery.discover_tool_modules(tool_dir=script_dir)
    tools = tool_discovery.collect_tools(modules)

    print(f"  Discovered {len(modules)} modules, {len(tools)} tools")

    expected_tools = {"math", "weather", "rgb_led", "servo", "elevator"}
    found_tools = {t["name"] for t in tools}

    for name in expected_tools:
        if name in found_tools:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} MISSING")

    missing = expected_tools - found_tools
    if missing:
        print(f"\n  FAILED: Missing tools: {missing}")
        return False

    print("\n  PASSED: All tools discovered")
    return True


def test_yaml_configs():
    """Test that YAML configs load correctly."""
    print("\n" + "=" * 60)
    print("Test: YAML Configurations")
    print("=" * 60)

    from yaml_config import load_yaml_config

    tools_dir = script_dir / "tools"
    passed = True

    for tool_dir in tools_dir.iterdir():
        if not tool_dir.is_dir() or tool_dir.name.startswith("_"):
            continue

        config_path = tool_dir / "config.yaml"
        if not config_path.exists():
            print(f"  ✗ {tool_dir.name}: No config.yaml")
            passed = False
            continue

        try:
            config = load_yaml_config(config_path)
            if config:
                tests = len(config.test_cases)
                print(f"  ✓ {tool_dir.name}: {tests} test cases")
            else:
                print(f"  ? {tool_dir.name}: Empty config")
        except Exception as e:
            print(f"  ✗ {tool_dir.name}: Error - {e}")
            passed = False

    if passed:
        print("\n  PASSED: All YAML configs valid")
    else:
        print("\n  FAILED: Some YAML configs have issues")

    return passed


def test_tool_execution():
    """Test direct tool execution (no LLM)."""
    print("\n" + "=" * 60)
    print("Test: Tool Execution (without LLM)")
    print("=" * 60)

    passed = True

    # Test math tool
    try:
        from tools.math import tool as math_tool
        result = math_tool.run({"expression": "2 + 2"})
        # Result is a string like "2 + 2 = 4"
        if result.get("ok") and "4" in str(result.get("result", "")):
            print(f"  ✓ math: 2 + 2 = 4")
        else:
            print(f"  ✗ math: Unexpected result: {result}")
            passed = False
    except Exception as e:
        print(f"  ✗ math: Error - {e}")
        passed = False

    # Test weather tool (API call)
    try:
        from tools.weather import tool as weather_tool
        result = weather_tool.run({"location": "London"})
        if result.get("ok"):
            print(f"  ✓ weather: Got forecast for London")
        else:
            # API might fail, that's okay
            print(f"  ? weather: API returned error (expected if offline)")
    except Exception as e:
        print(f"  ? weather: {e}")

    # Test RGB LED tool (simulator)
    try:
        from tools.rgb_led import tool as led_tool
        result = led_tool.run({"action": "on", "color": "blue"})
        if result.get("ok"):
            print(f"  ✓ rgb_led: LED on (blue)")
        else:
            print(f"  ✗ rgb_led: {result.get('error')}")
            passed = False
    except Exception as e:
        print(f"  ✗ rgb_led: Error - {e}")
        passed = False

    # Test servo tool (simulator)
    try:
        from tools.servo import tool as servo_tool
        result = servo_tool.run({"mode": "absolute", "angle": 45})
        if result.get("ok"):
            print(f"  ✓ servo: Moved to 45°")
        else:
            print(f"  ✗ servo: {result.get('error')}")
            passed = False
    except Exception as e:
        print(f"  ✗ servo: Error - {e}")
        passed = False

    # Test elevator tool (simulator)
    try:
        from tools.elevator import tool as elevator_tool
        result = elevator_tool.run({"floor": 3})
        if result.get("ok"):
            print(f"  ✓ elevator: Moved to floor 3")
        else:
            print(f"  ✗ elevator: {result.get('error')}")
            passed = False
    except Exception as e:
        print(f"  ✗ elevator: Error - {e}")
        passed = False

    if passed:
        print("\n  PASSED: All tool executions successful")
    else:
        print("\n  FAILED: Some tool executions failed")

    return passed


def test_metrics():
    """Test metrics calculation."""
    print("\n" + "=" * 60)
    print("Test: Metrics Module")
    print("=" * 60)

    from testing.metrics import calculate_metrics, score_state
    from dataclasses import dataclass
    from typing import Any, Dict

    @dataclass
    class MockResult:
        test_id: str
        passed: bool
        input: str
        expected: Dict[str, Any]
        actual: Any
        errors: list
        latency_ms: float

    @dataclass
    class MockResponse:
        tool_called: bool = True

    # Create mock results
    results = [
        MockResult("t1", True, "test", {"tool_called": True}, MockResponse(True), [], 100),
        MockResult("t2", True, "test", {"tool_called": True}, MockResponse(True), [], 150),
        MockResult("t3", False, "test", {"tool_called": True}, MockResponse(False), ["error"], 200),
    ]

    metrics = calculate_metrics(results)

    print(f"  Calculated metrics:")
    print(f"    - e2e_accuracy: {metrics.get('e2e_accuracy')}%")
    print(f"    - avg_latency: {metrics.get('avg_latency_ms')}ms")

    score = score_state(metrics)
    print(f"    - state_score: {score}")

    if metrics.get("e2e_accuracy") == 66.67 and metrics.get("avg_latency_ms") == 150.0:
        print("\n  PASSED: Metrics calculated correctly")
        return True
    else:
        print("\n  FAILED: Unexpected metrics values")
        return False


def main():
    """Run all quick tests."""
    print("\n" + "=" * 60)
    print("    AGENT TOOLS QUICK TEST")
    print("=" * 60)

    results = []

    results.append(("Tool Discovery", test_tool_discovery()))
    results.append(("YAML Configs", test_yaml_configs()))
    results.append(("Tool Execution", test_tool_execution()))
    results.append(("Metrics", test_metrics()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n  Total: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

