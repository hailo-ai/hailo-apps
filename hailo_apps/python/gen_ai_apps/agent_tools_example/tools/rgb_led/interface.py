"""
Hardware interface for RGB LED control.

Supports real hardware (SPI-based NeoPixel via rpi5-ws2812) and simulator (Flask browser visualization).
"""

from __future__ import annotations

import logging
import sys
import threading
from abc import ABC, abstractmethod
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


class RGBLEDInterface(ABC):
    """Abstract base class for RGB LED control."""

    @abstractmethod
    def set_color(self, r: int, g: int, b: int) -> None:
        """Set LED color using RGB values (0-255)."""
        pass

    @abstractmethod
    def set_intensity(self, percentage: float) -> None:
        """Set LED intensity/brightness (0-100%)."""
        pass

    @abstractmethod
    def on(self) -> None:
        """Turn LED on."""
        pass

    @abstractmethod
    def off(self) -> None:
        """Turn LED off."""
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get current LED state."""
        pass


class NeoPixelLED(RGBLEDInterface):
    """Real hardware implementation using SPI interface (rpi5-ws2812) for Raspberry Pi 5."""

    def __init__(self, spi_bus: int = 0, spi_device: int = 0, num_pixels: int = 1) -> None:
        """
        Initialize NeoPixel LED using SPI interface via rpi5-ws2812.

        Args:
            spi_bus: SPI bus number (default: 0, corresponds to /dev/spidev0.x)
            spi_device: SPI device number (default: 0, corresponds to /dev/spidev0.0)
            num_pixels: Number of LEDs in strip (default: 1)

        Note:
            SPI uses the MOSI pin (GPIO 10 on Raspberry Pi 5) automatically.
            Ensure SPI is enabled via: sudo raspi-config -> Interfacing Options -> SPI
        """
        try:
            from rpi5_ws2812.ws2812 import WS2812SpiDriver
        except ImportError:
            logger.error("rpi5-ws2812 library not available. Install with: pip install rpi5-ws2812")
            raise ImportError("rpi5-ws2812 library is required for SPI-based NeoPixel control")

        self.spi_bus = spi_bus
        self.spi_device = spi_device
        self.num_pixels = num_pixels
        # Default state: on with white color
        self._power = True
        self._color_rgb = (255, 255, 255)
        self._color_name = "white"
        self._intensity = 100.0

        # Initialize SPI driver
        try:
            driver = WS2812SpiDriver(
                spi_bus=spi_bus,
                spi_device=spi_device,
                led_count=num_pixels
            )
            self.strip = driver.get_strip()
            logger.info(
                "NeoPixel initialized on SPI bus %d, device %d (/dev/spidev%d.%d) with %d LEDs",
                spi_bus, spi_device, spi_bus, spi_device, num_pixels
            )
            # Set default state: on with white color
            self._update_pixels()
        except Exception as e:
            error_str = str(e)
            logger.error("Failed to initialize NeoPixel via SPI: %s", error_str)
            logger.error(
                "Ensure SPI is enabled: sudo raspi-config -> Interfacing Options -> SPI -> Enable"
            )
            raise RuntimeError(
                f"NeoPixel SPI initialization failed: {error_str}. "
                "Please ensure SPI is enabled and the rpi5-ws2812 library is installed correctly."
            ) from e

    def set_color(self, r: int, g: int, b: int, color_name: str | None = None) -> None:
        """
        Set LED color using RGB values (0-255).

        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            color_name: Optional color name (e.g., "red", "blue")
        """
        self._color_rgb = (r, g, b)
        if color_name is not None:
            self._color_name = color_name
        else:
            # Try to find color name from RGB
            self._color_name = self._find_color_name_from_rgb(r, g, b)
        if self._power:
            self._update_pixels()

    def _find_color_name_from_rgb(self, r: int, g: int, b: int) -> str:
        """Find color name from RGB values."""
        # Common color mappings (avoid circular import)
        common_colors = {
            (255, 0, 0): "red",
            (0, 255, 0): "green",
            (0, 0, 255): "blue",
            (255, 255, 0): "yellow",
            (255, 0, 255): "magenta",
            (0, 255, 255): "cyan",
            (255, 255, 255): "white",
            (0, 0, 0): "black",
            (255, 165, 0): "orange",
            (255, 192, 203): "pink",
            (128, 0, 128): "purple",
            (0, 128, 0): "lime",
            (0, 128, 128): "teal",
            (0, 0, 128): "navy",
        }
        rgb_tuple = (r, g, b)
        return common_colors.get(rgb_tuple, "custom")

    def set_intensity(self, percentage: float) -> None:
        """Set LED intensity/brightness (0-100%)."""
        self._intensity = max(0.0, min(100.0, percentage))
        if self._power:
            self._update_pixels()

    def on(self) -> None:
        """Turn LED on."""
        self._power = True
        self._update_pixels()

    def off(self) -> None:
        """Turn LED off."""
        self._power = False
        # Turn off all pixels (set to black)
        try:
            from rpi5_ws2812.ws2812 import Color
            self.strip.set_all_pixels(Color(0, 0, 0))
            self.strip.show()
        except ImportError:
            logger.error("rpi5-ws2812 library not available")

    def _update_pixels(self) -> None:
        """Update pixel colors based on current state."""
        try:
            from rpi5_ws2812.ws2812 import Color
        except ImportError:
            logger.error("rpi5-ws2812 library not available")
            return

        if not self._power:
            # Turn off all pixels (set to black)
            self.strip.set_all_pixels(Color(0, 0, 0))
        else:
            # Apply intensity to color
            brightness = self._intensity / 100.0
            r = int(self._color_rgb[0] * brightness)
            g = int(self._color_rgb[1] * brightness)
            b = int(self._color_rgb[2] * brightness)
            # Set all pixels to the same color
            self.strip.set_all_pixels(Color(r, g, b))
        self.strip.show()

    def get_state(self) -> dict[str, Any]:
        """Get current LED state."""
        return {
            "power": self._power,
            "color": self._color_name,
            "color_rgb": self._color_rgb,
            "intensity": self._intensity,
        }

    def cleanup(self) -> None:
        """Clean up NeoPixel resources."""
        # Turn off LED before cleanup
        try:
            self.off()
        except Exception as e:
            logger.debug("Error during NeoPixel cleanup: %s", e)
        # rpi5-ws2812 library handles cleanup automatically


class SimulatedLED(RGBLEDInterface):
    """Simulator implementation using Flask web server with browser visualization."""

    def __init__(self, port: int = 5000) -> None:
        """
        Initialize simulated LED with Flask web server.

        Args:
            port: Port for Flask web server (default: 5000)
        """
        try:
            from flask import Flask, jsonify, render_template_string  # noqa: F401
        except ImportError as e:
            logger.error("Flask not available. Install with: pip install flask")
            raise ImportError("Flask is required for simulator mode") from e

        self.port = port
        # Default state: on with white color
        self._power = True
        self._color_rgb = (255, 255, 255)
        self._color_name = "white"
        self._intensity = 100.0

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

        # HTML template for LED visualization
        self._html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>LED Simulator</title>
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
        .led-container {
            text-align: center;
        }
        .led {
            width: 200px;
            height: 200px;
            border-radius: 50%;
            border: 3px solid #333;
            margin: 20px;
            box-shadow: 0 0 30px rgba(255,255,255,0.3);
            transition: all 0.3s ease;
        }
        .led.off {
            background: #000;
            box-shadow: none;
        }
        .status {
            margin-top: 20px;
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
    <div class="led-container">
        <div class="led" id="led"></div>
        <div class="status" id="status">LED Off</div>
        <div class="info" id="info">Color: RGB(0, 0, 0) | Intensity: 0%</div>
    </div>
    <script>
        function updateLED() {
            fetch('/state')
                .then(response => response.json())
                .then(data => {
                    const led = document.getElementById('led');
                    const status = document.getElementById('status');
                    const info = document.getElementById('info');

                    if (data.power) {
                        const r = Math.round(data.color_rgb[0] * data.intensity / 100);
                        const g = Math.round(data.color_rgb[1] * data.intensity / 100);
                        const b = Math.round(data.color_rgb[2] * data.intensity / 100);
                        led.style.background = `rgb(${r}, ${g}, ${b})`;
                        led.style.boxShadow = `0 0 30px rgba(${r}, ${g}, ${b}, 0.5)`;
                        led.classList.remove('off');
                        status.textContent = 'LED On';
                        info.textContent = `Color: RGB(${data.color_rgb[0]}, ${data.color_rgb[1]}, ${data.color_rgb[2]}) | Intensity: ${data.intensity.toFixed(0)}%`;
                    } else {
                        led.style.background = '#000';
                        led.style.boxShadow = 'none';
                        led.classList.add('off');
                        status.textContent = 'LED Off';
                        info.textContent = 'Color: RGB(0, 0, 0) | Intensity: 0%';
                    }
                })
                .catch(error => console.error('Error:', error));
        }

        // Update every 100ms
        setInterval(updateLED, 100);
        updateLED(); // Initial update
    </script>
</body>
</html>
"""

        @self._app.route("/")
        def index() -> str:
            """Serve the LED visualization page."""
            return render_template_string(self._html_template)

        @self._app.route("/state")
        def state() -> Any:
            """Return current LED state as JSON."""
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

    def set_color(self, r: int, g: int, b: int, color_name: str | None = None) -> None:
        """
        Set LED color using RGB values (0-255).

        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            color_name: Optional color name (e.g., "red", "blue")
        """
        self._color_rgb = (r, g, b)
        if color_name is not None:
            self._color_name = color_name
        else:
            # Try to find color name from RGB
            self._color_name = self._find_color_name_from_rgb(r, g, b)

    def _find_color_name_from_rgb(self, r: int, g: int, b: int) -> str:
        """Find color name from RGB values."""
        # Common color mappings (avoid circular import)
        common_colors = {
            (255, 0, 0): "red",
            (0, 255, 0): "green",
            (0, 0, 255): "blue",
            (255, 255, 0): "yellow",
            (255, 0, 255): "magenta",
            (0, 255, 255): "cyan",
            (255, 255, 255): "white",
            (0, 0, 0): "black",
            (255, 165, 0): "orange",
            (255, 192, 203): "pink",
            (128, 0, 128): "purple",
            (0, 128, 0): "lime",
            (0, 128, 128): "teal",
            (0, 0, 128): "navy",
        }
        rgb_tuple = (r, g, b)
        return common_colors.get(rgb_tuple, "custom")

    def set_intensity(self, percentage: float) -> None:
        """Set LED intensity/brightness (0-100%)."""
        self._intensity = max(0.0, min(100.0, percentage))

    def on(self) -> None:
        """Turn LED on."""
        self._power = True

    def off(self) -> None:
        """Turn LED off."""
        self._power = False

    def get_state(self) -> dict[str, Any]:
        """Get current LED state."""
        return {
            "power": self._power,
            "color": self._color_name,
            "color_rgb": list(self._color_rgb),
            "intensity": self._intensity,
        }

    def cleanup(self) -> None:
        """Clean up resources (shutdown Flask server)."""
        # Flask server runs in daemon thread, so it will terminate automatically
        # But we can mark it for cleanup if needed
        if self._server_thread is not None and self._server_thread.is_alive():
            # Flask in daemon mode will terminate with main process
            # No explicit shutdown needed, but we can log it
            logger.debug("Flask LED simulator server will terminate with main process")


def create_led_controller() -> RGBLEDInterface:
    """
    Factory function to create appropriate LED controller based on configuration.

    Returns:
        RGBLEDInterface instance (NeoPixelLED or SimulatedLED)

    Raises:
        ImportError: If required libraries are not available
        RuntimeError: If hardware initialization fails
        SystemExit: If hardware mode is requested but fails
    """
    hardware_mode = config.HARDWARE_MODE.lower()

    if hardware_mode == "real":
        # Create real hardware controller - exit on failure
        try:
            return NeoPixelLED(
                spi_bus=config.NEOPIXEL_SPI_BUS,
                spi_device=config.NEOPIXEL_SPI_DEVICE,
                num_pixels=config.NEOPIXEL_COUNT,
            )
        except ImportError as e:
            logger.error("Hardware mode requested but library not available: %s", e)
            logger.error("Install with: pip install rpi5-ws2812")
            logger.error("Also ensure SPI is enabled: sudo raspi-config -> Interfacing Options -> SPI")
            sys.exit(1)
        except Exception as e:
            error_str = str(e)
            logger.error("Hardware initialization failed: %s", error_str)
            logger.error("Ensure SPI is enabled: sudo raspi-config -> Interfacing Options -> SPI -> Enable")
            logger.error("Hardware mode is required. Exiting.")
            sys.exit(1)
    else:
        # Simulator mode
        return SimulatedLED(port=config.FLASK_PORT)

