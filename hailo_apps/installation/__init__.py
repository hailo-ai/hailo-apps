"""
Installation utilities for Hailo applications.

This module provides utilities for:
- Compiling C++ post-processing code
- Downloading resources (models, videos, etc.)
- Post-installation setup
- Environment configuration

Note: Imports are lazy to avoid triggering GStreamer dependencies when
running in no-tappas mode.
"""

# Re-export config_manager functions for convenience (no GStreamer dependency)
from hailo_apps.config.config_manager import (
    get_main_config,
    get_resources_config,
    get_available_apps,
    get_model_names,
    get_default_model_name,
    is_gen_ai_app,
)


# Lazy imports for installation functions
def __getattr__(name):
    """Lazy-load installation functions to avoid import side effects."""
    if name == "compile_cpp":
        from .compile_cpp import main
        return main
    if name == "download_resources":
        from .download_resources import main
        return main
    if name == "post_install":
        from .post_install import main
        return main
    if name == "set_env":
        from .set_env import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "compile_cpp",
    "download_resources",
    "get_available_apps",
    "get_main_config",
    "get_resources_config",
    "get_model_names",
    "get_default_model_name",
    "is_gen_ai_app",
    "post_install",
    "set_env",
]
