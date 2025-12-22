"""
Elevator control interface and simulator implementation.

This module provides an abstract interface for elevator control and a
Flask-based web simulator for visualization.
"""

import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

try:
    from hailo_apps.python.gen_ai_apps.agent_tools_example import config
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from hailo_apps.python.gen_ai_apps.agent_tools_example import config

logger = logging.getLogger(__name__)


def _load_floors_from_config(config_path: Path | None = None) -> dict[int, dict[str, Any]]:
    """
    Load floors configuration from config.yaml.

    Args:
        config_path: Optional path to config.yaml. If None, uses default location.

    Returns:
        Dictionary mapping floor numbers to floor data (name, description, keywords).

    Raises:
        FileNotFoundError: If config.yaml is not found.
        ValueError: If floors section is missing or invalid.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed. Cannot load floors from config.")
        raise ImportError("PyYAML is required to load floors from config.yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except Exception as e:
        logger.error("Failed to parse YAML config %s: %s", config_path, e)
        raise ValueError(f"Failed to parse config.yaml: {e}") from e

    if not isinstance(raw_config, dict):
        raise ValueError("Invalid YAML config format (expected dict)")

    floors_config = raw_config.get("floors")
    if not floors_config:
        raise ValueError("'floors' section not found in config.yaml")

    # Convert YAML floors to internal format
    floors: dict[int, dict[str, Any]] = {}
    for floor_str, floor_data in floors_config.items():
        try:
            floor_num = int(floor_str)
        except (ValueError, TypeError):
            logger.warning("Invalid floor number '%s', skipping", floor_str)
            continue

        if not isinstance(floor_data, dict):
            logger.warning("Invalid floor data for floor %d, skipping", floor_num)
            continue

        floors[floor_num] = {
            "name": floor_data.get("name", f"Floor {floor_num}"),
            "description": floor_data.get("description", ""),
            "keywords": floor_data.get("keywords", []),
        }

    if not floors:
        raise ValueError("No valid floors found in config.yaml")

    return floors


# Floor data loaded from config.yaml (single source of truth)
# Initialize FLOORS at module load time
try:
    FLOORS = _load_floors_from_config()
except Exception as e:
    logger.error("Failed to load floors from config.yaml: %s", e)
    logger.error("Falling back to empty floors dictionary. Tool may not work correctly.")
    FLOORS = {}


class ElevatorInterface(ABC):
    """Abstract base class for elevator control."""

    @abstractmethod
    def move_to_floor(self, floor: int) -> None:
        """Move elevator to specified floor (0-5)."""
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get current elevator state."""
        pass

    @abstractmethod
    def get_floor_info(self, floor: int) -> dict[str, Any] | None:
        """Get information about a specific floor."""
        pass


class SimulatedElevator(ElevatorInterface):
    """Simulator implementation using Flask web server with browser visualization."""

    def __init__(self, port: int = 5002, floors_data: dict[int, dict[str, Any]] | None = None) -> None:
        """
        Initialize simulated elevator with Flask web server.

        Args:
            port: Port for Flask web server (default: 5002)
            floors_data: Dictionary of floor data (if None, will use module-level FLOORS from config.yaml)
        """
        try:
            from flask import Flask, jsonify, render_template_string  # noqa: F401
        except ImportError as e:
            logger.error("Flask not available. Install with: pip install flask")
            raise ImportError("Flask is required for simulator mode") from e

        # Use provided floors_data or default to module-level FLOORS (loaded from config.yaml)
        self.FLOORS = floors_data if floors_data is not None else FLOORS

        self.port = port
        self._current_floor = 1

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

        # HTML template for elevator visualization
        self._html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Wonkavator Elevator Simulator</title>
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
        .container {
            text-align: center;
            max-width: 1200px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .title {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #ffd700;
            width: 100%;
        }
        .subtitle {
            font-size: 18px;
            margin-bottom: 30px;
            color: #aaa;
            width: 100%;
        }
        .building-container {
            display: flex;
            justify-content: center;
            align-items: flex-start;
            margin: 0;
            position: relative;
        }
        .building {
            display: flex;
            flex-direction: column-reverse;
            border: 3px solid #444;
            border-radius: 5px;
            overflow: hidden;
            background: #222;
            position: relative;
            margin-left: 80px;
            min-width: 500px;
        }
        .floor {
            display: flex;
            flex-direction: row;
            align-items: center;
            padding: 15px 20px;
            border-top: 2px solid #333;
            min-height: 80px;
            transition: all 0.3s ease;
            background: #2a2a2a;
            position: relative;
        }
        .floor:first-child {
            border-top: none;
        }
        .floor.current {
            background: linear-gradient(90deg, #4a2c2a, #6a3c3a);
            border-left: 5px solid #ffd700;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
        }
        .floor-number {
            font-size: 36px;
            font-weight: bold;
            width: 60px;
            text-align: center;
            color: #ffd700;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .floor.current .floor-number {
            color: #fff;
        }
        .floor-content {
            display: flex;
            flex-direction: column;
            flex: 1;
            padding-left: 20px;
        }
        .floor-name {
            font-size: 20px;
            text-align: left;
            color: #ccc;
            margin-bottom: 4px;
        }
        .floor.current .floor-name {
            color: #fff;
            font-weight: bold;
        }
        .floor-description {
            font-size: 14px;
            color: #aaa;
            text-align: left;
            line-height: 1.4;
        }
        .floor.current .floor-description {
            color: #ddd;
        }
        .content-wrapper {
            display: flex;
            flex-direction: row;
            align-items: flex-start;
            gap: 40px;
            width: 100%;
            justify-content: center;
        }
        .elevator-indicator {
            position: absolute;
            left: 0;
            width: 60px;
            height: 60px;
            background: #ffd700;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: bold;
            color: #1a1a1a;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
            transition: all 0.5s ease;
            top: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="title">🍫 Wonkavator Elevator 🍫</div>
        <div class="subtitle">The Great Glass Elevator Control System</div>

        <div class="content-wrapper">
            <div class="building-container">
                <div class="building" id="building">
                    <!-- Floors will be populated here -->
                </div>
                <div class="elevator-indicator" id="elevator">🛗</div>
            </div>
        </div>
    </div>
    <script>
        function updateElevator() {
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
                    const building = document.getElementById('building');
                    const elevator = document.getElementById('elevator');

                    // Clear and rebuild floors
                    building.innerHTML = '';

                    // Create floors dynamically based on available floors
                    // With column-reverse CSS, first DOM child appears at bottom, last at top
                    const floorNumbers = Object.keys(data.floors).map(Number).sort((a, b) => a - b);
                    for (const i of floorNumbers) {
                        const floorDiv = document.createElement('div');
                        floorDiv.className = 'floor';
                        if (i === data.current_floor) {
                            floorDiv.className += ' current';
                        }

                        // Add floor number (vertically centered)
                        const floorNumber = document.createElement('div');
                        floorNumber.className = 'floor-number';
                        floorNumber.textContent = i;
                        floorDiv.appendChild(floorNumber);

                        // Create floor content container (name + description)
                        const floorContent = document.createElement('div');
                        floorContent.className = 'floor-content';

                        const floorName = document.createElement('div');
                        floorName.className = 'floor-name';
                        floorName.textContent = data.floors[i].name;
                        floorContent.appendChild(floorName);

                        const floorDescription = document.createElement('div');
                        floorDescription.className = 'floor-description';
                        floorDescription.textContent = data.floors[i].description || '';
                        floorContent.appendChild(floorDescription);

                        floorDiv.appendChild(floorContent);
                        building.appendChild(floorDiv);
                    }

                    // Update elevator position (visual indicator)
                    // Calculate position based on current floor
                    // Need to wait for DOM to update to get accurate floor heights
                    setTimeout(() => {
                        const floors = building.querySelectorAll('.floor');
                        const floorNumbers = Object.keys(data.floors).map(Number).sort((a, b) => a - b);
                        const floorIndex = floorNumbers.indexOf(data.current_floor);
                        if (floors.length > 0 && floorIndex >= 0 && floorIndex < floors.length) {
                            const targetFloor = floors[floorIndex];
                            const buildingRect = building.getBoundingClientRect();
                            const floorRect = targetFloor.getBoundingClientRect();
                            const relativeTop = floorRect.top - buildingRect.top;
                            // Center elevator vertically on the floor
                            const elevatorOffset = (floorRect.height - 60) / 2;
                            elevator.style.transform = `translateY(${relativeTop + elevatorOffset}px)`;
                        }
                    }, 10);
                })
                .catch(error => {
                    console.error('Error updating elevator:', error);
                });
        }

        // Update every 100ms
        const updateInterval = setInterval(updateElevator, 100);
        updateElevator(); // Initial update

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
            """Serve the elevator visualization page."""
            return render_template_string(self._html_template)

        @self._app.route("/state")
        def state() -> Any:
            """Return current elevator state as JSON."""
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

    def move_to_floor(self, floor: int) -> None:
        """Move elevator to specified floor."""
        if floor in self.FLOORS:
            self._current_floor = floor

    def get_state(self) -> dict[str, Any]:
        """Get current elevator state."""
        return {
            "current_floor": self._current_floor,
            "current_floor_info": self.FLOORS[self._current_floor],
            "floors": self.FLOORS,
        }

    def get_floor_info(self, floor: int) -> dict[str, Any] | None:
        """Get information about a specific floor."""
        return self.FLOORS.get(floor)

    def cleanup(self) -> None:
        """Clean up resources (shutdown Flask server)."""
        # Flask server runs in daemon thread, so it will terminate automatically
        if self._server_thread is not None and self._server_thread.is_alive():
            logger.debug("Flask elevator simulator server will terminate with main process")


# Global singleton instance for elevator controller
_elevator_controller_instance: ElevatorInterface | None = None


def create_elevator_controller() -> ElevatorInterface:
    """
    Factory function to create elevator controller (simulator only for now).
    Uses singleton pattern to ensure only one instance exists.

    Returns:
        ElevatorInterface instance (SimulatedElevator)

    Raises:
        ImportError: If required libraries are not available
        RuntimeError: If initialization fails
    """
    global _elevator_controller_instance

    # Return existing instance if available
    if _elevator_controller_instance is not None:
        return _elevator_controller_instance

    # For elevator, we only have simulator mode (no real hardware)
    _elevator_controller_instance = SimulatedElevator(port=config.ELEVATOR_SIMULATOR_PORT)

    return _elevator_controller_instance

