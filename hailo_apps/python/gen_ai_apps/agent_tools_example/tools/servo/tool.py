"""
Servo control tool.

Supports moving servo to absolute angle or by relative angle.
"""

from __future__ import annotations

import logging
from typing import Any

from .interface import (
    create_servo_controller,
)
from hailo_apps.python.gen_ai_apps.agent_tools_example import config

logger = logging.getLogger(__name__)

name: str = "servo"

# User-facing description
display_description: str = (
    "Control servo: move to absolute angle or by relative angle."
)

# LLM instruction description
description: str = (
    "CRITICAL: You MUST use this tool when the user asks to control, move, or do anything with a servo. "
    "ALWAYS call this tool if the user mentions: servo, move servo, set angle, rotate, turn, position, home, center. "
    "NEVER respond about servo control without calling this tool. "
    "The function name is 'servo' (use this exact name in tool calls). "
    "Modes: 'absolute' (set to specific angle), 'relative' (move by delta). "
    "Angles are in degrees.\n\n"
    "DIRECTION MAPPING (CRITICAL):\n"
    "- 'right' or 'to the right' = POSITIVE angle (e.g., 'move right 70 degrees' → angle: 70)\n"
    "- 'left' or 'to the left' = NEGATIVE angle (e.g., 'move left 70 degrees' → angle: -70)\n\n"
    "When user says 'home', 'center', 'zero', or 'reset position', use mode='absolute' with angle=0. "
    "\n"
    "CRITICAL - DEFAULT OPTION: You MUST use default=true in these cases:\n"
    "1. If the request mentions servo but you cannot extract a valid angle or direction (e.g., 'move servo to Japan')\n"
    "2. If the request is unclear, ambiguous, or doesn't make sense for servo control\n"
    "3. If the request doesn't contain a number or clear direction (left/right) that can be converted to an angle\n"
    "When default=true, the tool will automatically generate an appropriate error message. "
    "DO NOT guess angles or use angle=0 when the request is unclear - always use default=true instead."
)

# Initialize servo controller only when tool is selected
_servo_controller = None
_initialized = False


def initialize_tool() -> None:
    """Initialize servo controller when tool is selected."""
    global _servo_controller, _initialized
    if not _initialized:
        try:
            _servo_controller = create_servo_controller()
            _servo_controller.set_angle(0.0)
            _initialized = True

            if config.HARDWARE_MODE.lower() == "simulator":
                simulator_url = f"http://127.0.0.1:{config.SERVO_SIMULATOR_PORT}"
                print(f"\n[Servo Simulator] Open your browser: {simulator_url}\n", flush=True)

        except Exception as e:
            logger.error("Failed to initialize servo controller: %s", e)
            print(f"[Servo] Warning: Servo controller initialization failed: {e}", flush=True)
            _initialized = True


def _get_servo_controller() -> Any:
    """Get servo controller instance, initializing if needed."""
    if not _initialized:
        initialize_tool()
    return _servo_controller


def cleanup_tool() -> None:
    """Clean up servo controller resources."""
    global _servo_controller
    if _servo_controller is not None and hasattr(_servo_controller, "cleanup"):
        try:
            _servo_controller.cleanup()
        except Exception as e:
            logger.debug("Error during servo controller cleanup: %s", e)


schema: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["absolute", "relative"],
            "description": "'absolute' to set angle, 'relative' to move by delta. Required unless 'default' is used.",
        },
        "angle": {
            "type": "number",
            "description": "Angle in degrees. For absolute: -90 to 90. For relative: delta. Required unless 'default' is used.",
        },
        "default": {
            "type": "boolean",
            "description": (
                "Set to true when the user requests an angle outside -90 to 90 degrees, or if you cannot understand "
                "the servo control request. The tool will automatically generate an appropriate error message."
            ),
        },
    },
    "required": [],
}

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }
]


def _validate_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate input parameters."""
    # If default is provided, mode and angle are not required
    if payload.get("default") is True:
        return {"ok": True, "data": {"mode": None, "angle": None}}

    mode = str(payload.get("mode", "")).strip().lower()
    if mode not in {"absolute", "relative"}:
        return {"ok": False, "error": "Either 'mode' and 'angle' or 'default' must be provided. Mode must be 'absolute' or 'relative'."}

    angle = payload.get("angle")
    if angle is None:
        return {"ok": False, "error": "angle is required when mode is provided"}
    try:
        angle = float(angle)
    except (ValueError, TypeError):
        return {"ok": False, "error": "angle must be a number"}

    return {"ok": True, "data": {"mode": mode, "angle": angle}}


def run(input_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Execute servo control operation.

    Args:
        input_dict: Dictionary with mode and angle, or default.

    Returns:
        Dictionary with 'ok' and 'result' or 'error'.
    """
    # Check for default option first (user error - agent correctly used default)
    if input_dict.get("default") is True:
        return {
            "ok": True,  # Agent used tool correctly (default option)
            "error": "I couldn't understand your request. Please try again.",
        }

    validated = _validate_input(input_dict)
    if not validated.get("ok"):
        return validated

    data = validated["data"]
    mode = data["mode"]
    angle = data["angle"]

    try:
        servo = _get_servo_controller()
    except Exception as e:
        return {"ok": False, "error": f"Servo controller unavailable: {e}"}

    state = servo.get_state()
    min_angle = state["min_angle"]
    max_angle = state["max_angle"]

    if mode == "absolute":
        clamped_angle = max(min_angle, min(max_angle, angle))
        servo.set_angle(clamped_angle)
        final_state = servo.get_state()
        current_angle = final_state["angle"]

        if angle != clamped_angle:
            result = f"Servo moved to {current_angle:.1f}° (requested {angle:.1f}° was clamped)"
        else:
            result = f"Servo moved to {current_angle:.1f}°"

    else:  # relative
        current_angle = state["angle"]
        target_angle = current_angle + angle
        clamped_angle = max(min_angle, min(max_angle, target_angle))
        servo.move_relative(angle)
        final_state = servo.get_state()
        final_angle = final_state["angle"]

        if target_angle != clamped_angle:
            result = f"Servo move by {angle:.1f}° was clamped to {final_angle:.1f}°"
        else:
            result = f"Servo moved by {angle:.1f}° to {final_angle:.1f}°"

    return {"ok": True, "result": result}

