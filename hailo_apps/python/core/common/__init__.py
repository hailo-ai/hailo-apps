"""
Common utilities for Hailo applications.

This module provides core utilities used across all Hailo applications, including:
- Buffer utilities for GStreamer buffer handling
- Camera utilities for device detection
- Core argument parsers and configuration loading
- Hailo architecture detection and resource path resolution
- Logging infrastructure

Note: GStreamer-dependent imports (buffer_utils) are lazy-loaded to allow
standalone apps and installation scripts to work without GStreamer/gi.
"""

# Core utilities that don't require GStreamer
from .core import FIFODropQueue, get_model_name, get_resource_path, load_environment
from .hailo_logger import (
    add_logging_cli_args,
    get_logger,
    get_run_id,
    init_logging,
    level_from_args,
)
from .installation_utils import (
    detect_hailo_arch,
    detect_host_arch,
)
from .parser import (
    get_base_parser,
    get_default_parser,
    get_pipeline_parser,
    get_standalone_parser,
)
from .camera_utils import (
    get_usb_video_devices,
    is_rpi_camera_available,
)


# Lazy imports for GStreamer-dependent modules
def __getattr__(name):
    """Lazy-load GStreamer-dependent functions to avoid import errors when gi is not available."""
    if name in ("get_caps_from_pad", "get_numpy_from_buffer", "get_numpy_from_buffer_efficient"):
        from .buffer_utils import (
            get_caps_from_pad,
            get_numpy_from_buffer,
            get_numpy_from_buffer_efficient,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Buffer utilities
    "get_caps_from_pad",
    "get_numpy_from_buffer",
    "get_numpy_from_buffer_efficient",
    # Camera utilities
    "get_usb_video_devices",
    "is_rpi_camera_available",
    # Core utilities
    "FIFODropQueue",
    "get_base_parser",
    "get_default_parser",
    "get_model_name",
    "get_pipeline_parser",
    "get_resource_path",
    "get_standalone_parser",
    "handle_list_models_flag",
    "list_models_for_app",
    "load_environment",
    "resolve_hef_path",
    # HEF utilities
    "get_hef_input_size",
    "get_hef_input_shape",
    "get_hef_labels_json",
    # Logger
    "add_logging_cli_args",
    "get_logger",
    "get_run_id",
    "init_logging",
    "level_from_args",
    # Installation utilities
    "detect_hailo_arch",
    "detect_host_arch",
]
