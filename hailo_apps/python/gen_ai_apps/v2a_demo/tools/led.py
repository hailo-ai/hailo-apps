"""LED tool — controls the Raspberry Pi 5 ACT LED (inverted logic: '0' = ON)."""

import logging
import time
import threading
from typing import Optional

logger = logging.getLogger("v2a_demo")

LED_PATH = "/sys/class/leds/ACT"  # Raspberry Pi 5 ACT LED

TOOL_PROMPT = (
    "Extract parameters from the user's board LED control request as a JSON object.\n"
    "You MUST output ALL 6 fields in every response.\n"
    "\n"
    "Parameters:\n"
    '- "target": Always "board_led".\n'
    '- "state": "on" or "off". Map: turn on/enable/activate -> "on", turn off/disable/deactivate -> "off".\n'
    '- "mode": "steady" or "blink". Map: solid/constant/continuous/stay on -> "steady", flash/flashing/pulse -> "blink". CRITICAL: If the user mentions a count (e.g., "5 times", "twice", "once"), mode MUST be "blink".\n'
    '- "blink_on_ms": For steady mode: null. For blink mode: integer milliseconds. Map: fast/quickly/rapid -> 100, slow/slowly -> 1000, default -> 300.\n'
    '- "blink_off_ms": For steady mode: null. For blink mode: same value as blink_on_ms.\n'
    '- "blink_count": For steady mode: null. For blink mode: integer count. Map: once -> 1, twice -> 2, three times -> 3, few times -> 3, several times -> 5, many times -> 10, default -> 5.\n'
    "\n"
    "IMPORTANT: For steady mode, blink_on_ms, blink_off_ms, and blink_count MUST be null.\n"
    "IMPORTANT: For blink mode, blink_on_ms, blink_off_ms, and blink_count MUST be integers.\n"
    "\n"
    "Examples:\n"
    '"Turn on the board LED" -> {"target": "board_led", "state": "on", "mode": "steady", "blink_on_ms": null, "blink_off_ms": null, "blink_count": null}\n'
    '"Turn off the LED on the board." -> {"target": "board_led", "state": "off", "mode": "steady", "blink_on_ms": null, "blink_off_ms": null, "blink_count": null}\n'
    '"Turn on the LED 5 times" -> {"target": "board_led", "state": "on", "mode": "blink", "blink_on_ms": 300, "blink_off_ms": 300, "blink_count": 5}\n'
    '"Flash the Raspberry Pi LED 5 times." -> {"target": "board_led", "state": "on", "mode": "blink", "blink_on_ms": 300, "blink_off_ms": 300, "blink_count": 5}\n'
    '"Blink the Pi LED 3 times" -> {"target": "board_led", "state": "on", "mode": "blink", "blink_on_ms": 300, "blink_off_ms": 300, "blink_count": 3}\n'
    '"Flash the status light quickly" -> {"target": "board_led", "state": "on", "mode": "blink", "blink_on_ms": 100, "blink_off_ms": 100, "blink_count": 5}\n'
    '"Slowly blink the RPI LED twice" -> {"target": "board_led", "state": "on", "mode": "blink", "blink_on_ms": 1000, "blink_off_ms": 1000, "blink_count": 2}\n'
    '"Turn off the Raspberry Pi LED" -> {"target": "board_led", "state": "off", "mode": "steady", "blink_on_ms": null, "blink_off_ms": null, "blink_count": null}\n'
    "\n"
    "Output ONLY the JSON object, nothing else."
)

TOOL_DESCRIPTIONS = [
    "Turn the Raspberry Pi board LED on or off",
    "Control the Pi status or indicator LED",
    "Blink or flash the board LED",
    "Make the Raspberry Pi LED blink a number of times",
    "Turn on the board LED steadily",
    "Turn off the board LED",
    "Flash the board LED quickly or slowly",
    "Control a single built-in Raspberry Pi LED",
    "Handle simple board LED on, off, or blinking actions",
]


class LEDController:
    """
    Internal controller for Raspberry Pi LED.
    Handles steady and blink modes with proper inversion for Pi 5 ACT LED.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._blink_thread: Optional[threading.Thread] = None
        self._blink_stop = threading.Event()

    def _reset(self) -> None:
        """Disable default triggers and stop any ongoing blink."""
        with open(f"{LED_PATH}/trigger", "w") as f:
            f.write("none")
        if self._blink_thread and self._blink_thread.is_alive():
            self._blink_stop.set()
            self._blink_thread.join()

    def _set_brightness(self, state: str) -> None:
        """Set LED brightness. Inverted for Pi 5 ACT LED: '0' = ON, '1' = OFF."""
        value = "0" if state == "on" else "1"
        with open(f"{LED_PATH}/brightness", "w") as f:
            f.write(value)

    def steady(self, state: str) -> None:
        with self._lock:
            self._reset()
            self._set_brightness(state)

    def blink(self, on_ms: int = 500, off_ms: int = 500,
              count: int = 1) -> None:
        with self._lock:
            self._reset()
            self._blink_stop.clear()

            def blink_func():
                for _ in range(count):
                    if self._blink_stop.is_set():
                        break
                    self._set_brightness("on")
                    time.sleep(on_ms / 1000)
                    self._set_brightness("off")
                    time.sleep(off_ms / 1000)

            self._blink_thread = threading.Thread(target=blink_func, daemon=True)
            self._blink_thread.start()


# Singleton controller instance
_controller = LEDController()


def control_led(target: str, state: str, mode: str = "steady",
                blink_on_ms: int = 300, blink_off_ms: int = 300,
                blink_count: int = 5) -> str:
    """
    Tool for controlling Raspberry Pi LED.
    Returns a human-readable string describing what was done.
    """
    if target != "board_led":
        return f"Unknown LED target: {target}."

    try:
        if mode == "steady":
            _controller.steady(state)
            return f"The board LED has been turned {state}."
        elif mode == "blink":
            # The LLM sometimes leaves blink fields null; a None would crash the blink thread.
            blink_on_ms = blink_on_ms or 300
            blink_off_ms = blink_off_ms or blink_on_ms
            blink_count = blink_count or 5
            _controller.blink(on_ms=blink_on_ms, off_ms=blink_off_ms, count=blink_count)
            if blink_count == 1:
                return "The board LED blinked once."
            return f"The board LED blinked {blink_count} times."
        else:
            return f"Unknown mode: {mode}."
    except (OSError, PermissionError) as e:
        logger.error(f"LED control failed: {e}")
        return "I couldn't control the LED. Make sure this is running on a Raspberry Pi with the right permissions."
