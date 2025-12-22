"""
Configuration package for Hailo applications.

This package provides centralized configuration management for:
- config.yaml: Main configuration (environment, venv, resources, system packages)
- resources_config.yaml: Resource download definitions (models, videos, images, JSON)
- test_definition_config.yaml: Test framework definitions
- test_control.yaml: Test execution control

Usage:
    from hailo_apps.config import config_manager

    # Get available apps
    apps = config_manager.get_available_apps()

    # Get models for an app
    models = config_manager.get_default_models("detection", "hailo8")

    # Validate all configs
    # python -m hailo_apps.config.config_manager --dry-run
"""

from pathlib import Path

# Import the unified config manager
from . import config_manager

# Expose config directory path for backward compatibility
CONFIG_DIR = Path(__file__).parent

# Re-export commonly used functions at package level for convenience
get_main_config = config_manager.get_main_config
get_resources_config = config_manager.get_resources_config
get_test_definition_config = config_manager.get_test_definition_config
get_test_control_config = config_manager.get_test_control_config

# Resource config functions
get_available_apps = config_manager.get_available_apps
get_supported_architectures = config_manager.get_supported_architectures
get_default_models = config_manager.get_default_models
get_extra_models = config_manager.get_extra_models
get_all_models = config_manager.get_all_models
get_model_names = config_manager.get_model_names
get_default_model_name = config_manager.get_default_model_name
get_model_info = config_manager.get_model_info
get_videos = config_manager.get_videos
get_images = config_manager.get_images
get_json_files = config_manager.get_json_files
is_gen_ai_app = config_manager.is_gen_ai_app

# Test definition functions
get_app_definition = config_manager.get_app_definition
get_defined_apps = config_manager.get_defined_apps
get_test_suite = config_manager.get_test_suite
get_all_test_suites = config_manager.get_all_test_suites
get_test_suites_for_app = config_manager.get_test_suites_for_app
get_test_run_combination = config_manager.get_test_run_combination

# Test control functions
get_control_parameter = config_manager.get_control_parameter
get_logging_config = config_manager.get_logging_config
get_enabled_run_methods = config_manager.get_enabled_run_methods
get_custom_test_apps = config_manager.get_custom_test_apps
is_special_test_enabled = config_manager.is_special_test_enabled

# Cache management
clear_cache = config_manager.clear_cache
reload_all = config_manager.reload_all

# Paths and exceptions
ConfigPaths = config_manager.ConfigPaths
ConfigError = config_manager.ConfigError

# Data classes
ModelEntry = config_manager.ModelEntry
AppDefinition = config_manager.AppDefinition
TestSuite = config_manager.TestSuite

__all__ = [
    # Module
    "config_manager",
    "CONFIG_DIR",
    # Config loaders
    "get_main_config",
    "get_resources_config",
    "get_test_definition_config",
    "get_test_control_config",
    # Resource config
    "get_available_apps",
    "get_supported_architectures",
    "get_default_models",
    "get_extra_models",
    "get_all_models",
    "get_model_names",
    "get_default_model_name",
    "get_model_info",
    "get_videos",
    "get_images",
    "get_json_files",
    "is_gen_ai_app",
    # Test definition
    "get_app_definition",
    "get_defined_apps",
    "get_test_suite",
    "get_all_test_suites",
    "get_test_suites_for_app",
    "get_test_run_combination",
    # Test control
    "get_control_parameter",
    "get_logging_config",
    "get_enabled_run_methods",
    "get_custom_test_apps",
    "is_special_test_enabled",
    # Cache
    "clear_cache",
    "reload_all",
    # Types
    "ConfigPaths",
    "ConfigError",
    "ModelEntry",
    "AppDefinition",
    "TestSuite",
]
