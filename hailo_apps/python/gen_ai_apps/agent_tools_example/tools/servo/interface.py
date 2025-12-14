"""
Hardware interface for servo control.

Supports real hardware (hardware PWM servo via rpi-hardware-pwm) and simulator (Flask browser visualization).
"""

from __future__ import annotations

import logging
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# Import config from parent package
try:
    from hailo_apps.python.gen_ai_apps.agent_tools_example import config
except ImportError:
    # Fallback for direct execution
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from hailo_apps.python.gen_ai_apps.agent_tools_example import config

logger = logging.getLogger(__name__)


class ServoInterface(ABC):
    """Abstract base class for servo control."""

    @abstractmethod
    def set_angle(self, angle: float) -> None:
        """Set servo to absolute angle."""
        pass

    @abstractmethod
    def move_relative(self, delta: float) -> None:
        """Move servo by relative angle."""
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get current servo state."""
        pass


class HardwarePWMServo(ServoInterface):
    """Real hardware implementation using rpi-hardware-pwm for hardware PWM control."""

    def __init__(self, pwm_channel: int = 0, min_angle: float = -90.0, max_angle: float = 90.0) -> None:
        """
        Initialize servo using hardware PWM via rpi-hardware-pwm.

        Args:
            pwm_channel: PWM channel number (0 or 1). Default: 0 (GPIO 18).
                         Channel 0 maps to GPIO 18 (or GPIO 12 if configured).
                         Channel 1 maps to GPIO 19 (or GPIO 13 if configured).
            min_angle: Minimum angle in degrees (default: -90)
            max_angle: Maximum angle in degrees (default: 90)
        """
        try:
            from rpi_hardware_pwm import HardwarePWM
        except ImportError:
            logger.error("rpi-hardware-pwm library not available. Install with: pip install rpi-hardware-pwm")
            raise

        if pwm_channel not in (0, 1):
            raise ValueError(f"PWM channel must be 0 or 1, got {pwm_channel}")

        self.pwm_channel = pwm_channel
        self.min_angle = min_angle
        self.max_angle = max_angle
        self._current_angle = 0.0  # Default to center position
        self._pwm: HardwarePWM | None = None
        self._pwm_sysfs_path: Path | None = None

        try:
            # Standard servo frequency is 50 Hz
            SERVO_FREQUENCY = 50
            # Map logical channel to actual hardware PWM channel
            # Channel 0 (GPIO 18) -> PWM0_CHAN2 (channel 2)
            # Channel 1 (GPIO 19) -> PWM0_CHAN1 (channel 1)
            hardware_pwm_channel = self._get_hardware_pwm_channel()
            self._pwm = HardwarePWM(pwm_channel=hardware_pwm_channel, chip=0, hz=SERVO_FREQUENCY)

            # Start PWM with center position (7.5% duty cycle for 0 degrees)
            center_duty = self._angle_to_duty_cycle(0.0)
            self._pwm.start(center_duty)

            # Verify PWM is actually enabled (workaround for library issues)
            self._verify_and_enable_pwm()

            # Map channel to GPIO pin for logging
            gpio_pin = 18 if pwm_channel == 0 else 19
            logger.info(
                "Servo initialized on PWM channel %d (GPIO %d) with angle range %.1f to %.1f degrees",
                pwm_channel, gpio_pin, min_angle, max_angle
            )
            # Set default position to center (0 degrees)
            self._update_servo(0.0)
        except Exception as e:
            logger.error("Failed to initialize servo: %s", e)
            if self._pwm is not None:
                try:
                    self._pwm.stop()
                except Exception:
                    pass
            raise

    def _get_hardware_pwm_channel(self) -> int:
        """
        Map logical PWM channel to actual hardware PWM channel.

        Returns:
            Hardware PWM channel number (1 or 2)
        """
        # Channel 0 (GPIO 18) -> PWM0_CHAN2 (channel 2)
        # Channel 1 (GPIO 19) -> PWM0_CHAN1 (channel 1)
        return 2 if self.pwm_channel == 0 else 1

    def _verify_and_enable_pwm(self) -> None:
        """
        Verify PWM is enabled in sysfs and enable it if needed.

        This is a workaround for cases where rpi-hardware-pwm doesn't
        properly enable the PWM channel.
        """
        try:
            hardware_pwm_channel = self._get_hardware_pwm_channel()
            pwm_sysfs = Path(f"/sys/class/pwm/pwmchip0/pwm{hardware_pwm_channel}")
            self._pwm_sysfs_path = pwm_sysfs

            if not pwm_sysfs.exists():
                logger.warning("PWM channel %d not exported in sysfs, library should handle this", hardware_pwm_channel)
                return

            enable_path = pwm_sysfs / "enable"
            if enable_path.exists():
                current_enable = enable_path.read_text().strip()
                if current_enable == "0":
                    logger.warning("PWM was not enabled, enabling manually...")
                    try:
                        enable_path.write_text("1")
                        logger.info("PWM enabled successfully")
                    except (PermissionError, OSError) as e:
                        logger.warning("Could not enable PWM manually (may need root): %s", e)
                else:
                    logger.debug("PWM is already enabled")
        except Exception as e:
            logger.debug("Could not verify PWM enable state: %s", e)

    def _angle_to_duty_cycle(self, angle: float) -> float:
        """
        Convert angle in degrees to PWM duty cycle percentage.

        Standard servos expect:
        - 2.5% duty cycle = 0 degrees (or minimum angle)
        - 7.5% duty cycle = 90 degrees (center/neutral)
        - 12.5% duty cycle = 180 degrees (or maximum angle)

        For angle range -90 to 90, we map to 2.5% to 12.5% duty cycle.

        Args:
            angle: Angle in degrees

        Returns:
            Duty cycle percentage (2.5 to 12.5)
        """
        # Clamp angle to valid range
        clamped_angle = max(self.min_angle, min(self.max_angle, angle))

        # Map angle range to duty cycle range (2.5% to 12.5%)
        # Standard mapping: -90° = 2.5%, 0° = 7.5%, +90° = 12.5%
        if self.max_angle == self.min_angle:
            return 7.5  # Center position

        # Linear interpolation: duty = 2.5 + (angle - min) / (max - min) * 10.0
        normalized = (clamped_angle - self.min_angle) / (self.max_angle - self.min_angle)
        duty_cycle = 2.5 + normalized * 10.0

        return duty_cycle

    def _update_servo(self, angle: float) -> None:
        """
        Update servo position to given angle.

        Args:
            angle: Target angle in degrees
        """
        if self._pwm is None:
            raise RuntimeError("PWM not initialized")

        duty_cycle = self._angle_to_duty_cycle(angle)
        self._pwm.change_duty_cycle(duty_cycle)

        # Ensure PWM stays enabled after duty cycle change
        if self._pwm_sysfs_path is not None:
            enable_path = self._pwm_sysfs_path / "enable"
            if enable_path.exists():
                try:
                    current = enable_path.read_text().strip()
                    if current == "0":
                        enable_path.write_text("1")
                        logger.debug("Re-enabled PWM after duty cycle change")
                except (PermissionError, OSError):
                    pass  # Ignore if we can't write

        self._current_angle = max(self.min_angle, min(self.max_angle, angle))

    def set_angle(self, angle: float) -> None:
        """Set servo to absolute angle."""
        self._update_servo(angle)

    def move_relative(self, delta: float) -> None:
        """Move servo by relative angle."""
        new_angle = self._current_angle + delta
        self._update_servo(new_angle)

    def get_state(self) -> dict[str, Any]:
        """Get current servo state."""
        return {
            "angle": self._current_angle,
            "min_angle": self.min_angle,
            "max_angle": self.max_angle,
        }

    def cleanup(self) -> None:
        """Clean up hardware PWM resources."""
        if self._pwm is not None:
            try:
                self._pwm.stop()
                logger.debug("Hardware PWM stopped")
            except Exception as e:
                logger.debug("Error stopping PWM: %s", e)


class SimulatedServo(ServoInterface):
    """Simulator implementation using Flask web server with browser visualization."""

    def __init__(self, port: int = 5001, min_angle: float = -90.0, max_angle: float = 90.0) -> None:
        """
        Initialize simulated servo with Flask web server.

        Args:
            port: Port for Flask web server (default: 5001)
            min_angle: Minimum angle in degrees (default: -90)
            max_angle: Maximum angle in degrees (default: 90)
        """
        try:
            from flask import Flask, jsonify, render_template_string  # noqa: F401
        except ImportError as e:
            logger.error("Flask not available. Install with: pip install flask")
            raise ImportError("Flask is required for simulator mode") from e

        self.port = port
        self.min_angle = min_angle
        self.max_angle = max_angle
        self._current_angle = 0.0  # Default to center position

        try:
            # Suppress Werkzeug logging BEFORE creating Flask app
            werkzeug_logger = logging.getLogger("werkzeug")
            werkzeug_logger.setLevel(logging.CRITICAL)
            werkzeug_logger.disabled = True

            self._app = Flask(__name__)
            self._server_thread: threading.Thread | None = None

            # Disable Flask's app logger to avoid request messages in terminal
            self._app.logger.setLevel(logging.ERROR)
            self._app.logger.disabled = True
        except Exception as e:
            logger.error("Failed to create Flask app: %s", e)
            raise

        # HTML template for servo visualization
        self._html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Servo Simulator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            background: #1a1a1a;
            color: #fff;
        }
        .servo-container {
            text-align: center;
            position: relative;
        }
        .scale-container {
            width: 400px;
            height: 200px;
            position: relative;
            margin: 0 auto 20px;
            overflow: visible;
        }
        .scale-arc {
            width: 400px;
            height: 200px;
            position: relative;
        }
        .scale-mark {
            position: absolute;
            width: 2px;
            height: 15px;
            background: #888;
            top: 0;
            left: 50%;
            transform-origin: 50% 200px;
        }
        .scale-mark.major {
            height: 20px;
            background: #aaa;
            width: 3px;
        }
        .scale-label {
            position: absolute;
            top: 35px;
            left: 50%;
            transform-origin: 50% 200px;
            font-size: 12px;
            color: #aaa;
            text-align: center;
            width: 30px;
            margin-left: -15px;
        }
        .servo-base {
            width: 100px;
            height: 100px;
            background: #444;
            border-radius: 10px;
            position: absolute;
            border: 2px solid #666;
            left: 50%;
            top: 200px;
            margin-left: -50px;
            margin-top: -50px;
        }
        .servo-arm-container {
            position: absolute;
            left: 50%;
            top: 50%;
            margin-top: -8px;
            transform-origin: 0 50%;
            width: 150px;
            height: 16px;
        }
        .servo-arm {
            width: 120px;
            height: 8px;
            background: #888;
            position: absolute;
            left: 0;
            top: 4px;
            border-radius: 0 4px 4px 0;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
        }
        .servo-arm::before {
            content: '';
            position: absolute;
            left: -10px;
            top: -2px;
            width: 10px;
            height: 10px;
            background: #666;
            border-radius: 50%;
        }
        .servo-pivot {
            width: 20px;
            height: 20px;
            background: #666;
            border-radius: 50%;
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            border: 2px solid #888;
            z-index: 10;
        }
        .status {
            margin-top: 100px;
            font-size: 18px;
        }
        .info {
            margin-top: 10px;
            font-size: 14px;
            color: #aaa;
        }
    </style>
</head>
<body>
    <div class="servo-container">
        <div class="scale-container">
            <div class="scale-arc" id="scale-arc"></div>
            <div class="servo-base">
                <div class="servo-pivot"></div>
                <div class="servo-arm-container" id="servo-arm-container">
                    <div class="servo-arm" id="servo-arm"></div>
                </div>
            </div>
        </div>
        <div class="status" id="status">Servo Position: 0°</div>
        <div class="info" id="info">Angle Range: -90° to 90°</div>
    </div>
    <script>
        function createScale(minAngle, maxAngle) {
            const scaleArc = document.getElementById('scale-arc');
            scaleArc.innerHTML = '';

            // Create major marks every 30 degrees
            // Reverse the angle display so visual left shows negative, right shows positive
            for (let angle = minAngle; angle <= maxAngle; angle += 30) {
                const mark = document.createElement('div');
                mark.className = 'scale-mark major';
                // rotation needs to account for the -90° offset we apply to the arm
                const rotation = -angle;
                mark.style.transform = `rotate(${rotation}deg)`;
                scaleArc.appendChild(mark);

                // Add label with counter-rotated text
                const label = document.createElement('div');
                label.className = 'scale-label';
                const span = document.createElement('span');
                // Display the reversed angle: visual rotation matches servo angle
                span.textContent = (-angle) + '°';
                label.appendChild(span);
                label.style.transform = `rotate(${rotation}deg)`;
                scaleArc.appendChild(label);
            }

            // Create minor marks every 15 degrees
            for (let angle = minAngle; angle <= maxAngle; angle += 15) {
                if (angle % 30 !== 0) {
                    const mark = document.createElement('div');
                    mark.className = 'scale-mark';
                    const rotation = -angle;
                    mark.style.transform = `rotate(${rotation}deg)`;
                    scaleArc.appendChild(mark);
                }
            }
        }

        function updateServo() {
            fetch('/state', {
                cache: 'no-store',
                headers: {
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache'
                }
            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    const armContainer = document.getElementById('servo-arm-container');
                    const status = document.getElementById('status');
                    const info = document.getElementById('info');

                    if (!armContainer || !status || !info) {
                        console.error('Required DOM elements not found');
                        return;
                    }

                    const angle = data.angle;
                    // Rotation mapping: -90° = LEFT, 0° = UP, +90° = RIGHT
                    // CSS rotates clockwise from 0° (pointing right)
                    // We need to subtract 90° to shift zero point from RIGHT to UP
                    // So: angle=-90 → rotation=-180 (LEFT), angle=0 → rotation=-90 (UP), angle=+90 → rotation=0 (RIGHT)
                    const rotation = angle - 90;
                    armContainer.style.transform = `rotate(${rotation}deg)`;
                    status.textContent = `Servo Position: ${angle.toFixed(1)}°`;
                    info.textContent = `Angle Range: ${data.min_angle}° to ${data.max_angle}°`;

                    // Initialize scale on first update
                    if (document.getElementById('scale-arc').children.length === 0) {
                        createScale(data.min_angle, data.max_angle);
                    }
                })
                .catch(error => {
                    console.error('Error updating servo:', error);
                });
        }

        // Update every 100ms
        const updateInterval = setInterval(updateServo, 100);
        updateServo(); // Initial update

        // Ensure interval is set up
        if (!updateInterval) {
            console.error('Failed to set up update interval');
        }
    </script>
</body>
</html>
"""

        @self._app.route("/")
        def index() -> str:
            """Serve the servo visualization page."""
            return render_template_string(self._html_template)

        @self._app.route("/state")
        def state() -> Any:
            """Return current servo state as JSON."""
            return jsonify(self.get_state())

        # Start Flask server in background thread
        self._start_server()

    def _start_server(self) -> None:
        """Start Flask server in background thread."""
        def run_server() -> None:
            try:
                # Run Flask server (logging already suppressed in __init__)
                self._app.run(host="127.0.0.1", port=self.port, debug=False, use_reloader=False, threaded=True)
            except OSError as e:
                if "Address already in use" in str(e):
                    logger.warning("Port %d already in use. Simulator may not start properly.", self.port)
                else:
                    logger.error("Flask server error: %s", e)
            except Exception as e:
                logger.error("Unexpected error in Flask server thread: %s", e)

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        # Give server a moment to start
        import time
        time.sleep(0.1)

    def set_angle(self, angle: float) -> None:
        """Set servo to absolute angle."""
        self._current_angle = max(self.min_angle, min(self.max_angle, angle))

    def move_relative(self, delta: float) -> None:
        """Move servo by relative angle."""
        new_angle = self._current_angle + delta
        self._current_angle = max(self.min_angle, min(self.max_angle, new_angle))

    def get_state(self) -> dict[str, Any]:
        """Get current servo state."""
        return {
            "angle": self._current_angle,
            "min_angle": self.min_angle,
            "max_angle": self.max_angle,
        }

    def cleanup(self) -> None:
        """Clean up resources (shutdown Flask server)."""
        # Flask server runs in daemon thread, so it will terminate automatically
        # But we can mark it for cleanup if needed
        if self._server_thread is not None and self._server_thread.is_alive():
            # Flask in daemon mode will terminate with main process
            # No explicit shutdown needed, but we can log it
            logger.debug("Flask servo simulator server will terminate with main process")


# Global singleton instance for servo controller
_servo_controller_instance: ServoInterface | None = None


def create_servo_controller() -> ServoInterface:
    """
    Factory function to create appropriate servo controller based on configuration.
    Uses singleton pattern to ensure only one instance exists.

    Returns:
        ServoInterface instance (HardwarePWMServo or SimulatedServo)

    Raises:
        ImportError: If required libraries are not available
        RuntimeError: If hardware initialization fails
        SystemExit: If hardware mode is requested but fails
    """
    global _servo_controller_instance

    # Return existing instance if available
    if _servo_controller_instance is not None:
        return _servo_controller_instance

    hardware_mode = config.HARDWARE_MODE.lower()

    if hardware_mode == "real":
        # Create real hardware controller - exit on failure
        try:
            _servo_controller_instance = HardwarePWMServo(
                pwm_channel=config.SERVO_PWM_CHANNEL,
                min_angle=config.SERVO_MIN_ANGLE,
                max_angle=config.SERVO_MAX_ANGLE,
            )
        except ImportError as e:
            logger.error("Hardware mode requested but library not available: %s", e)
            logger.error("Install with: pip install rpi-hardware-pwm")
            logger.error("Also ensure hardware PWM is enabled in /boot/firmware/config.txt:")
            logger.error("  Add: dtoverlay=pwm-2chan")
            logger.error("  Then reboot the Raspberry Pi")
            sys.exit(1)
        except Exception as e:
            logger.error("Hardware initialization failed: %s", e)
            logger.error("Ensure hardware PWM is enabled in /boot/firmware/config.txt:")
            logger.error("  Add: dtoverlay=pwm-2chan")
            logger.error("  Then reboot the Raspberry Pi")
            logger.error("Hardware mode is required. Exiting.")
            sys.exit(1)
    else:
        # Simulator mode
        _servo_controller_instance = SimulatedServo(
            port=config.SERVO_SIMULATOR_PORT,
            min_angle=config.SERVO_MIN_ANGLE,
            max_angle=config.SERVO_MAX_ANGLE,
        )

    return _servo_controller_instance

