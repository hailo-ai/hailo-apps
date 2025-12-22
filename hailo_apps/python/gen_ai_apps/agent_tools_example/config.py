"""
Configuration constants for the chat agent.

Users can modify these values to customize LLM behavior, context management, and model paths.
"""

from hailo_apps.python.core.common.hailo_logger import get_logger

LOGGER = get_logger(__name__)

# LLM Generation Parameters
TEMPERATURE: float = 0.1
SEED: int = 42
MAX_GENERATED_TOKENS: int = 200

# Context Management
CONTEXT_THRESHOLD: float = 0.95  # Clear context when usage reaches this percentage

# Hardware Configuration
HARDWARE_MODE: str = "simulator"  # "real" or "simulator"
# SPI configuration for NeoPixel (Raspberry Pi 5)
# SPI uses MOSI pin (GPIO 10) automatically - no pin configuration needed
NEOPIXEL_SPI_BUS: int = 0  # SPI bus number (0 = /dev/spidev0.x)
NEOPIXEL_SPI_DEVICE: int = 0  # SPI device number (0 = /dev/spidev0.0)
NEOPIXEL_COUNT: int = 8  # Number of LEDs in strip
# Simulator Ports (using 8xxx range to avoid conflicts with Cursor IDE which uses 5xxx)
FLASK_PORT: int = 8500  # Port for LED simulator web server
SERVO_SIMULATOR_PORT: int = 8501  # Port for servo simulator web server
ELEVATOR_SIMULATOR_PORT: int = 8502  # Port for elevator simulator web server

# Servo Hardware Configuration
SERVO_PWM_CHANNEL: int = 0  # Hardware PWM channel (0 or 1). Channel 0 = GPIO 18, Channel 1 = GPIO 19
SERVO_MIN_ANGLE: float = -90.0  # Minimum servo angle in degrees
SERVO_MAX_ANGLE: float = 90.0  # Maximum servo angle in degrees

# Voice Configuration
ENABLE_VOICE: bool = True
ENABLE_TTS: bool = True


def validate_config() -> None:
    """
    Validate configuration values on startup.

    Raises:
        ValueError: If any configuration value is invalid.
    """
    # Validate Hardware Mode
    if HARDWARE_MODE.lower() not in ("real", "simulator"):
        raise ValueError(f"Invalid HARDWARE_MODE '{HARDWARE_MODE}'. Must be 'real' or 'simulator'.")

    # Validate Ports
    ports = {
        "FLASK_PORT": FLASK_PORT,
        "SERVO_SIMULATOR_PORT": SERVO_SIMULATOR_PORT,
        "ELEVATOR_SIMULATOR_PORT": ELEVATOR_SIMULATOR_PORT
    }

    for name, port in ports.items():
        if not (1024 <= port <= 65535):
            raise ValueError(f"Invalid {name} {port}. Must be between 1024 and 65535.")

    # Check for port conflicts
    if len(set(ports.values())) != len(ports):
        raise ValueError(f"Port conflict detected in configuration: {ports}")

    # Validate Servo Angles
    if SERVO_MIN_ANGLE >= SERVO_MAX_ANGLE:
        raise ValueError(f"Invalid servo angles: MIN ({SERVO_MIN_ANGLE}) must be less than MAX ({SERVO_MAX_ANGLE})")

    # Validate Context Threshold
    if not (0.1 <= CONTEXT_THRESHOLD <= 1.0):
        raise ValueError(f"Invalid CONTEXT_THRESHOLD {CONTEXT_THRESHOLD}. Must be between 0.1 and 1.0.")
