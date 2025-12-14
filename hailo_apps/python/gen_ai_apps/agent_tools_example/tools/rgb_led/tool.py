"""
RGB LED control tool.

Supports turning LED on/off, changing color by name, and adjusting intensity.
"""

from __future__ import annotations

import logging
from typing import Any

# Import from local interface module
from .interface import (
    create_led_controller,
)
from hailo_apps.python.gen_ai_apps.agent_tools_example import config

logger = logging.getLogger(__name__)

name: str = "rgb_led"

# User-facing description (shown in CLI tool list)
display_description: str = (
    "Control RGB LED: turn on/off, change color by name, and adjust intensity (0-100%)."
)

# LLM instruction description
description: str = (
    "CRITICAL: You MUST use this tool when the user asks to control, change, or do anything with an LED or lights. "
    "ALWAYS call this tool if the user mentions: LED, light, lights, turn on, turn off, change color, set color, brightness, intensity, dim, brighten, make it red/blue/green/etc. "
    "NEVER respond about LED control without calling this tool - ALWAYS use this tool for ANY LED or light-related request. "
    "The function name is 'rgb_led' (use this exact name in tool calls). "
    "\n\n"
    "PARAMETER RULES:\n"
    "- action: REQUIRED - always 'on' or 'off'\n"
    "- color: OPTIONAL - ONLY include if the user EXPLICITLY requests a color change (e.g., 'make it red', 'change to blue', 'set color to green'). "
    "  DO NOT include color if the user only asks to change intensity/brightness - the current color will be preserved.\n"
    "- intensity: OPTIONAL - only include if user specifies brightness/intensity (0-100). "
    "  If user only asks to change intensity, do NOT include color parameter - the LED will keep its current color.\n"
    "\n"
    "IMPORTANT: If the user asks to change ONLY intensity/brightness (e.g., 'set intensity to 50%', 'make it dimmer', 'brightness 75%'), "
    "include ONLY action='on' and intensity. DO NOT include color - the current color will be preserved.\n"
    "\n"
    "Color names: red, blue, green, yellow, purple, cyan, white, orange, pink, magenta, lime, teal, navy\n\n"
    "DEFAULT OPTION: If the user requests an unsupported color or operation, or if you cannot understand the LED control request, "
    "set 'default' to true. The tool will automatically generate an appropriate error message."
)

# Color name to RGB mapping
COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255),
    "white": (255, 255, 255),
    "orange": (255, 165, 0),
    "pink": (255, 192, 203),
    "magenta": (255, 0, 255),
    "lime": (0, 255, 0),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "aqua": (0, 255, 255),
    "black": (0, 0, 0),
}

# Initialize LED controller only when tool is selected
_led_controller = None
_initialized = False


def initialize_tool() -> None:
    """Initialize LED controller when tool is selected."""
    global _led_controller, _initialized
    if not _initialized:
        try:
            _led_controller = create_led_controller()
            _led_controller.set_color(255, 255, 255, color_name="white")
            _led_controller.set_intensity(100.0)
            _led_controller.on()
            _initialized = True

            if config.HARDWARE_MODE.lower() == "simulator":
                simulator_url = f"http://127.0.0.1:{config.FLASK_PORT}"
                print(f"\n[LED Simulator] Open your browser: {simulator_url}\n", flush=True)

        except Exception as e:
            logger.error("Failed to initialize LED controller: %s", e)
            print(f"[LED] Warning: LED controller initialization failed: {e}", flush=True)
            _initialized = True


def _get_led_controller() -> Any:
    """Get LED controller instance, initializing if needed."""
    if not _initialized:
        initialize_tool()
    return _led_controller


def cleanup_tool() -> None:
    """Clean up LED controller resources."""
    global _led_controller
    if _led_controller is not None and hasattr(_led_controller, "cleanup"):
        try:
            _led_controller.cleanup()
        except Exception as e:
            logger.debug("Error during LED controller cleanup: %s", e)


schema: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["on", "off"],
            "description": "Action: 'on' or 'off'. Required unless 'default' is used.",
        },
        "color": {
            "type": "string",
            "description": "Color name (case-insensitive). OPTIONAL. ONLY include if user explicitly requests a color change. Do NOT include if user only asks to change intensity - current color will be preserved.",
        },
        "intensity": {
            "type": "number",
            "description": "Brightness 0-100. OPTIONAL.",
        },
        "default": {
            "type": "boolean",
            "description": (
                "Set to true when the user requests an unsupported color or operation, or if you cannot understand "
                "the LED control request. The tool will automatically generate an appropriate error message."
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


def _color_name_to_rgb(color_name: str) -> tuple[int, int, int] | None:
    """Convert color name to RGB values."""
    return COLOR_MAP.get(color_name.strip().lower())


def _validate_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate input parameters."""
    # If default is provided, action is not required
    if payload.get("default") is True:
        return {"ok": True, "data": {"action": None, "color": None, "intensity": None}}

    action = str(payload.get("action", "")).strip().lower()
    if action not in {"on", "off"}:
        return {"ok": False, "error": "Either 'action' or 'default' must be provided. Action must be 'on' or 'off'."}

    color = payload.get("color")
    if color is not None:
        color_str = str(color).strip().lower()
        if color_str not in COLOR_MAP:
            valid_colors = ", ".join(sorted(COLOR_MAP.keys()))
            return {"ok": False, "error": f"Unknown color '{color}'. Valid: {valid_colors}"}
        color = color_str

    intensity = payload.get("intensity")
    if intensity is not None:
        try:
            intensity = float(intensity)
            if intensity < 0 or intensity > 100:
                return {"ok": False, "error": "intensity must be 0-100"}
        except (ValueError, TypeError):
            return {"ok": False, "error": "intensity must be a number 0-100"}

    return {"ok": True, "data": {"action": action, "color": color, "intensity": intensity}}


def run(input_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Execute RGB LED control operation.

    Args:
        input_dict: Dictionary with action, color (optional), intensity (optional), or default.

    Returns:
        Dictionary with 'ok' and 'result' or 'error'.
    """
    # Check for default option first (user error - agent correctly used default)
    if input_dict.get("default") is True:
        return {
            "ok": True,  # Agent used tool correctly (default option)
            "error": "I couldn't understand your request.",
        }

    validated = _validate_input(input_dict)
    if not validated.get("ok"):
        return validated

    data = validated["data"]
    action = data["action"]
    color = data.get("color")
    intensity = data.get("intensity")

    try:
        led = _get_led_controller()
    except Exception as e:
        return {"ok": False, "error": f"LED controller unavailable: {e}"}

    if action == "off":
        led.off()
        return {"ok": True, "result": "LED turned off"}

    # Handle "on" action
    if color is not None:
        color_rgb = _color_name_to_rgb(color)
        if color_rgb:
            led.set_color(color_rgb[0], color_rgb[1], color_rgb[2], color_name=color)

    if intensity is not None:
        led.set_intensity(float(intensity))

    led.on()

    state = led.get_state()
    color_name = state["color"]
    intensity_val = state["intensity"]

    if color is not None and intensity is not None:
        result = f"LED is now on, showing {color} at {intensity_val:.0f}% brightness"
    elif color is not None:
        result = f"LED is now on, showing {color}"
    elif intensity is not None:
        result = f"LED is now on at {intensity_val:.0f}% brightness"
    else:
        result = "LED turned on"

    return {"ok": True, "result": result}

