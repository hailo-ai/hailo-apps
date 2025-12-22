#!/usr/bin/env python3
"""
Unified Configuration Manager for Hailo Apps Infrastructure.

This module provides a single, consistent interface for loading and
querying all configuration files used throughout the project.

Configuration Files Managed:
    - config.yaml: Main installation/runtime settings
    - resources_config.yaml: Model, video, image resource definitions
    - test_definition_config.yaml: Test framework structure and suites
    - test_control.yaml: Test execution control settings

Usage:
    from hailo_apps.config import config_manager

    # Load main config
    main_config = config_manager.get_main_config()

    # Query resources
    models = config_manager.get_models_for_app("detection", "hailo8")

    # Query test definitions
    app_def = config_manager.get_app_definition("detection")
    test_suites = config_manager.get_test_suites_for_app("detection", "default")

    # Query test control
    run_time = config_manager.get_control_parameter("default_run_time")

CLI Usage:
    python -m hailo_apps.config.config_manager --dry-run
    python -m hailo_apps.config.config_manager --list-apps
    python -m hailo_apps.config.config_manager --show-models detection hailo8
"""

from __future__ import annotations

import argparse
import sys
import yaml
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional


# Standalone app naming convention
STANDALONE_SUFFIX = "_standalone"


# =============================================================================
# Exceptions
# =============================================================================

class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


# =============================================================================
# Type Definitions (Dataclasses)
# =============================================================================

@dataclass(frozen=True)
class ModelEntry:
    """Represents a model entry from resources config.
    
    Attributes:
        name: Model name (e.g., "yolov8m")
        source: Source type - "mz" (Model Zoo), "s3", or "gen-ai-mz"
        url: Optional explicit URL for download
    """
    name: str
    source: str  # "mz" | "s3" | "gen-ai-mz"
    url: Optional[str] = None


@dataclass(frozen=True)
class AppDefinition:
    """Represents an application definition from test definition config.
    
    Attributes:
        name: Display name of the application
        description: Brief description
        module: Python module path (e.g., "hailo_apps.python.pipeline_apps.detection.detection")
        script: Script path relative to repo root
        cli: CLI command name (e.g., "hailo-detect")
        default_test_suites: Test suites to run by default
        extra_test_suites: Additional test suites for extended testing
    """
    name: str
    description: str
    module: str
    script: str
    cli: str
    default_test_suites: tuple[str, ...]
    extra_test_suites: tuple[str, ...]


@dataclass(frozen=True)
class TestSuite:
    """Represents a test suite definition.
    
    Attributes:
        name: Suite identifier
        description: Brief description of what the suite tests
        flags: Command-line flags to use for this suite
    """
    name: str
    description: str
    flags: tuple[str, ...]


# =============================================================================
# Path Resolution
# =============================================================================

class ConfigPaths:
    """Centralized path resolution for all configuration files.
    
    This class provides a single source of truth for locating configuration
    files, handling both installed package and development scenarios.
    """

    _repo_root: Optional[Path] = None

    @classmethod
    def _get_repo_root(cls) -> Path:
        """Get the repository root directory."""
        if cls._repo_root is None:
            # This file is at: hailo_apps/config/config_manager.py
            # Repo root is 2 levels up
            cls._repo_root = Path(__file__).resolve().parents[2]
        return cls._repo_root

    @classmethod
    def _get_config_dir(cls) -> Path:
        """Get the configuration directory.
        
        First tries package location (when installed via pip),
        then falls back to repo location (development mode).
        """
        # First try package location
        try:
            import hailo_apps.config
            package_dir = Path(hailo_apps.config.__file__).parent
            if package_dir.exists():
                return package_dir
        except (ImportError, AttributeError):
            pass
        # Fallback to repo location
        return cls._get_repo_root() / "hailo_apps" / "config"

    @classmethod
    def repo_root(cls) -> Path:
        """Get the repository root path."""
        return cls._get_repo_root()

    @classmethod
    def main_config(cls) -> Path:
        """Get path to config.yaml."""
        return cls._get_config_dir() / "config.yaml"

    @classmethod
    def resources_config(cls) -> Path:
        """Get path to resources_config.yaml."""
        return cls._get_config_dir() / "resources_config.yaml"

    @classmethod
    def test_definition_config(cls) -> Path:
        """Get path to test_definition_config.yaml."""
        return cls._get_config_dir() / "test_definition_config.yaml"

    @classmethod
    def test_control_config(cls) -> Path:
        """Get path to test_control.yaml (in tests directory)."""
        return cls._get_repo_root() / "tests" / "test_control.yaml"


# =============================================================================
# Base Loading Utilities
# =============================================================================

def _is_none_value(value: Any) -> bool:
    """Check if a value represents None (handles YAML None parsing).
    
    YAML files may contain None as:
    - Python None (null in YAML)
    - String "None" or "none"
    
    Args:
        value: The value to check
        
    Returns:
        True if the value represents None
    """
    if value is None:
        return True
    if isinstance(value, str) and value.lower() == "none":
        return True
    return False


@lru_cache(maxsize=8)
def _load_yaml_cached(path: str) -> dict:
    """Load YAML file with caching.
    
    Args:
        path: Path to YAML file (string for hashability with lru_cache)
        
    Returns:
        Parsed YAML as dictionary
        
    Raises:
        ConfigError: If file not found or invalid YAML
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path_obj, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")

    return data if data else {}


def _load_yaml(path: Path, use_cache: bool = True) -> dict:
    """Load YAML file, optionally with caching.
    
    Args:
        path: Path to YAML file
        use_cache: Whether to use cached result if available
        
    Returns:
        Parsed YAML as dictionary
    """
    if use_cache:
        return _load_yaml_cached(str(path))

    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")

    return data if data else {}


# =============================================================================
# Main Config API (config.yaml)
# =============================================================================

def get_main_config(use_cache: bool = True) -> dict:
    """Get the main configuration (config.yaml).
    
    Args:
        use_cache: Whether to use cached result
        
    Returns:
        Complete configuration dictionary
    """
    return _load_yaml(ConfigPaths.main_config(), use_cache)


def get_valid_versions(key: str) -> list[str]:
    """Get valid versions for a configuration key.
    
    Args:
        key: Version key (e.g., "hailort", "tappas", "hailo_arch")
        
    Returns:
        List of valid version strings
    """
    config = get_main_config()
    valid_versions = config.get("valid_versions", {})
    return valid_versions.get(key, [])


def get_model_zoo_version_for_arch(arch: str) -> str:
    """Get the Model Zoo version for a given architecture.
    
    Args:
        arch: Hailo architecture ("hailo8", "hailo8l", "hailo10h")
        
    Returns:
        Model Zoo version string (e.g., "v2.17.0")
    """
    config = get_main_config()
    mapping = config.get("model_zoo_mapping", {})
    return mapping.get(arch, "v2.17.0")


def get_venv_config() -> dict:
    """Get virtual environment configuration.
    
    Returns:
        Dictionary with 'name' and 'use_system_site_packages' keys
    """
    config = get_main_config()
    return config.get("venv", {"name": "venv_hailo_apps", "use_system_site_packages": True})


def get_resources_path_config() -> dict:
    """Get resources path configuration.
    
    Returns:
        Dictionary with 'path', 'root', 'env_file', 'download_group', 'dirs' keys
    """
    config = get_main_config()
    return config.get("resources", {})


def get_model_zoo_mapping() -> dict:
    """Get the model zoo version mapping for architectures.
    
    Returns:
        Dictionary mapping architecture names to Model Zoo versions
        e.g., {"hailo8": "v2.17.0", "hailo8l": "v2.17.0", "hailo10h": "v2.17.0"}
    """
    config = get_main_config()
    return config.get("model_zoo_mapping", {})


# =============================================================================
# Resources Config API (resources_config.yaml)
# =============================================================================

def get_resources_config(use_cache: bool = True) -> dict:
    """Get the resources configuration (resources_config.yaml).
    
    Args:
        use_cache: Whether to use cached result
        
    Returns:
        Complete resources configuration dictionary
    """
    return _load_yaml(ConfigPaths.resources_config(), use_cache)


def get_available_apps() -> list[str]:
    """Get list of all available application names from resources config.
    
    Returns:
        Sorted list of application names (excludes 'videos' and 'images' sections)
    """
    config = get_resources_config()
    shared_keys = {"videos", "images"}
    return sorted(
        k for k, v in config.items() if isinstance(v, dict) and k not in shared_keys
    )


def get_supported_architectures(app_name: str) -> list[str]:
    """Get architectures that have valid models for an app.
    
    Args:
        app_name: Application name (e.g., "detection")
        
    Returns:
        Sorted list of supported architecture names
    """
    config = get_resources_config()
    app_config = config.get(app_name, {})
    models_config = app_config.get("models", {})

    supported = []
    for arch, arch_models in models_config.items():
        if isinstance(arch_models, dict):
            default = arch_models.get("default")
            extra = arch_models.get("extra", [])
            if not _is_none_value(default) or extra:
                supported.append(arch)

    return sorted(supported)


def _extract_model_entries(entries: Any) -> list[ModelEntry]:
    """Extract ModelEntry objects from config entries.
    
    Handles various config formats:
    - Single model dict: {"name": "model", "source": "mz"}
    - List of models: [{"name": "model1"}, {"name": "model2"}]
    - None/null values
    
    Args:
        entries: Raw entries from config
        
    Returns:
        List of ModelEntry objects
    """
    if _is_none_value(entries):
        return []

    entries_list = entries if isinstance(entries, list) else [entries]
    models = []

    for entry in entries_list:
        if _is_none_value(entry):
            continue
        if isinstance(entry, dict):
            name = entry.get("name")
            if name and not _is_none_value(name):
                models.append(
                    ModelEntry(
                        name=name,
                        source=entry.get("source", "mz"),
                        url=entry.get("url"),
                    )
                )
        elif isinstance(entry, str) and not _is_none_value(entry):
            models.append(ModelEntry(name=entry, source="mz"))

    return models


def get_default_models(app_name: str, arch: str) -> list[ModelEntry]:
    """Get default model entries for an app and architecture.
    
    Args:
        app_name: Application name (e.g., "detection")
        arch: Hailo architecture (e.g., "hailo8")
        
    Returns:
        List of default ModelEntry objects
    """
    config = get_resources_config()
    app_config = config.get(app_name, {})
    arch_models = app_config.get("models", {}).get(arch, {})
    return _extract_model_entries(arch_models.get("default"))


def get_extra_models(app_name: str, arch: str) -> list[ModelEntry]:
    """Get extra model entries for an app and architecture.
    
    Args:
        app_name: Application name (e.g., "detection")
        arch: Hailo architecture (e.g., "hailo8")
        
    Returns:
        List of extra ModelEntry objects
    """
    config = get_resources_config()
    app_config = config.get(app_name, {})
    arch_models = app_config.get("models", {}).get(arch, {})
    return _extract_model_entries(arch_models.get("extra"))


def get_all_models(app_name: str, arch: str) -> list[ModelEntry]:
    """Get all model entries (default + extra) for an app and architecture.
    
    Args:
        app_name: Application name
        arch: Hailo architecture
        
    Returns:
        Combined list of default and extra ModelEntry objects
    """
    return get_default_models(app_name, arch) + get_extra_models(app_name, arch)


def get_model_names(app_name: str, arch: str, tier: str = "all") -> list[str]:
    """Get model names for an app and architecture.
    
    Args:
        app_name: Application name
        arch: Hailo architecture
        tier: "default", "extra", or "all"
        
    Returns:
        List of model name strings
    """
    if tier == "default":
        models = get_default_models(app_name, arch)
    elif tier == "extra":
        models = get_extra_models(app_name, arch)
    else:
        models = get_all_models(app_name, arch)

    return [m.name for m in models]


def get_default_model_name(app_name: str, arch: str) -> Optional[str]:
    """Get the first default model name for an app and architecture.
    
    Args:
        app_name: Application name
        arch: Hailo architecture
        
    Returns:
        Model name string or None if no default model
    """
    models = get_default_models(app_name, arch)
    return models[0].name if models else None


def get_model_info(app_name: str, arch: str, model_name: str) -> Optional[ModelEntry]:
    """Get full model info for a specific model.
    
    Args:
        app_name: Application name
        arch: Hailo architecture
        model_name: Name of the model to find
        
    Returns:
        ModelEntry if found, None otherwise
    """
    for model in get_all_models(app_name, arch):
        if model.name == model_name:
            return model
    return None


def get_videos() -> list[str]:
    """Get list of video filenames from resources config.
    
    Returns:
        List of video filename strings
    """
    config = get_resources_config()
    videos = []
    for entry in config.get("videos", []):
        if isinstance(entry, dict) and entry.get("name"):
            videos.append(entry["name"])
        elif isinstance(entry, str):
            videos.append(entry)
    return videos


def get_images() -> list[str]:
    """Get list of image filenames from resources config.
    
    Returns:
        List of image filename strings
    """
    config = get_resources_config()
    images = []
    for entry in config.get("images", []):
        if isinstance(entry, dict) and entry.get("name"):
            images.append(entry["name"])
        elif isinstance(entry, str):
            images.append(entry)
    return images


def get_json_files(app_name: str = None) -> list[str]:
    """Get JSON config filenames from the shared json section.
    
    Args:
        app_name: Ignored (kept for backward compatibility). All JSON files are shared.
        
    Returns:
        List of JSON filename strings
    """
    config = get_resources_config()
    json_files = []
    for entry in config.get("json", []):
        if isinstance(entry, dict) and entry.get("name"):
            json_files.append(entry["name"])
        elif isinstance(entry, str):
            json_files.append(entry)
    return json_files


def get_all_json_files() -> list[str]:
    """Get all JSON files from the shared json section.
    
    Returns:
        List of JSON filename strings
    """
    return get_json_files()


def is_gen_ai_app(app_name: str) -> bool:
    """Check if an app uses gen-ai-mz source models.
    
    Args:
        app_name: Application name
        
    Returns:
        True if any model has source "gen-ai-mz"
    """
    for arch in get_supported_architectures(app_name):
        for model in get_all_models(app_name, arch):
            if model.source == "gen-ai-mz":
                return True
    return False


# =============================================================================
# Inputs Config API (images/videos per app based on tags)
# =============================================================================

def _get_resources_by_tag(section: str, app_name: str) -> list[dict]:
    """Get resources from a section that are tagged for an app.
    
    Args:
        section: Section name ("videos", "images", "json")
        app_name: Application name to match in tags
        
    Returns:
        List of resource entry dicts that match the app
    """
    config = get_resources_config()
    all_resources = config.get(section, [])
    
    matching = []
    for resource in all_resources:
        if not isinstance(resource, dict):
            continue
        tags = resource.get("tag", [])
        if isinstance(tags, str):
            tags = [tags]
        if app_name in tags:
            matching.append(resource)
    
    return matching


def get_videos_for_app(app_name: str) -> list[dict]:
    """Get videos that are tagged for a specific application.
    
    Args:
        app_name: Application name (e.g., "detection", "pose_estimation")
        
    Returns:
        List of video entry dicts that match the app
    """
    return _get_resources_by_tag("videos", app_name)


def get_images_for_app(app_name: str) -> list[dict]:
    """Get images that are tagged for a specific application.
    
    Args:
        app_name: Application name (e.g., "detection", "pose_estimation")
        
    Returns:
        List of image entry dicts that match the app
    """
    return _get_resources_by_tag("images", app_name)


def get_json_for_app(app_name: str) -> list[dict]:
    """Get JSON config files that are tagged for a specific application.
    
    Args:
        app_name: Application name (e.g., "detection", "tiling")
        
    Returns:
        List of JSON entry dicts that match the app
    """
    return _get_resources_by_tag("json", app_name)


def get_inputs_for_app(app_name: str, is_standalone: bool = False) -> dict:
    """Get input resources for an application based on app type.
    
    - Standalone apps (is_standalone=True): Returns both videos AND images
    - Pipeline apps (is_standalone=False): Returns only videos
    
    Args:
        app_name: Application name (e.g., "detection", "pose_estimation")
        is_standalone: If True, include images; if False, only videos
        
    Returns:
        Dictionary with 'videos' and optionally 'images' lists for the app
    """
    # Standalone apps use the base app tag names (without _standalone suffix)
    app_name = base_app_name(app_name) if is_standalone or is_standalone_app_name(app_name) else app_name
    result = {
        "videos": get_videos_for_app(app_name),
    }
    
    if is_standalone:
        result["images"] = get_images_for_app(app_name)
    
    return result


def get_all_tags() -> set[str]:
    """Get all unique tags used across videos, images, and json.
    
    Returns:
        Set of all tag strings found in resources config
    """
    config = get_resources_config()
    tags = set()
    
    for section in ["videos", "images", "json"]:
        for entry in config.get(section, []):
            if isinstance(entry, dict):
                entry_tags = entry.get("tag", [])
                if isinstance(entry_tags, str):
                    entry_tags = [entry_tags]
                tags.update(entry_tags)
    
    return tags


def get_apps_with_inputs() -> set[str]:
    """Get all app names that have tagged inputs (videos/images/json).
    
    Returns:
        Set of app names that have at least one tagged resource
    """
    return get_all_tags()


# =============================================================================
# Standalone helpers
# =============================================================================

def is_standalone_app_name(app_name: str) -> bool:
    """Return True if the app name uses the _standalone suffix."""
    return app_name.endswith(STANDALONE_SUFFIX)


def base_app_name(app_name: str) -> str:
    """Strip the _standalone suffix if present."""
    return app_name[: -len(STANDALONE_SUFFIX)] if is_standalone_app_name(app_name) else app_name


def get_defined_standalone_apps() -> list[str]:
    """List app definitions that are marked as standalone (_standalone suffix)."""
    return [name for name in get_defined_apps() if is_standalone_app_name(name)]


def get_standalone_app_definition(app_name: str) -> Optional[AppDefinition]:
    """Get standalone app definition (expects _standalone suffix)."""
    return get_app_definition(app_name)


def get_standalone_test_suites_for_app(app_name: str, mode: str = "default") -> list[str]:
    """Get test suites for a standalone app."""
    return get_test_suites_for_app(app_name, mode)


def get_standalone_model_names(app_name: str, arch: str, tier: str = "default") -> list[str]:
    """Get model names for a standalone app, mapped to its base app resources."""
    base = base_app_name(app_name)
    return get_model_names(base, arch, tier)


def get_standalone_default_model_name(app_name: str, arch: str) -> Optional[str]:
    """Get default model name for a standalone app."""
    base = base_app_name(app_name)
    return get_default_model_name(base, arch)


# =============================================================================
# Test Definition Config API (test_definition_config.yaml)
# =============================================================================

def get_test_definition_config(use_cache: bool = True) -> dict:
    """Get the test definition configuration.
    
    Args:
        use_cache: Whether to use cached result
        
    Returns:
        Complete test definition configuration dictionary
    """
    return _load_yaml(ConfigPaths.test_definition_config(), use_cache)


def get_app_definition(app_name: str) -> Optional[AppDefinition]:
    """Get application definition by name.
    
    Args:
        app_name: Application name (e.g., "detection")
        
    Returns:
        AppDefinition dataclass or None if not found
    """
    config = get_test_definition_config()
    apps = config.get("apps", {})
    app_data = apps.get(app_name)

    if not app_data:
        return None

    return AppDefinition(
        name=app_data.get("name", app_name),
        description=app_data.get("description", ""),
        module=app_data.get("module", ""),
        script=app_data.get("script", ""),
        cli=app_data.get("cli", ""),
        default_test_suites=tuple(app_data.get("default_test_suites", [])),
        extra_test_suites=tuple(app_data.get("extra_test_suites", [])),
    )


def get_defined_apps() -> list[str]:
    """Get list of all defined application names from test definition config.
    
    Returns:
        List of application name strings
    """
    config = get_test_definition_config()
    return list(config.get("apps", {}).keys())


def get_test_suite(suite_name: str) -> Optional[TestSuite]:
    """Get test suite definition by name.
    
    Args:
        suite_name: Test suite name (e.g., "basic_show_fps")
        
    Returns:
        TestSuite dataclass or None if not found
    """
    config = get_test_definition_config()
    suites = config.get("test_suites", {})
    suite_data = suites.get(suite_name)

    if not suite_data:
        return None

    return TestSuite(
        name=suite_name,
        description=suite_data.get("description", ""),
        flags=tuple(suite_data.get("flags", [])),
    )


def get_all_test_suites() -> list[str]:
    """Get all test suite names.
    
    Returns:
        List of test suite name strings
    """
    config = get_test_definition_config()
    return list(config.get("test_suites", {}).keys())


def get_test_suites_for_app(app_name: str, mode: str = "default") -> list[str]:
    """Get test suites for an app based on mode.
    
    Args:
        app_name: Application name
        mode: "default", "extra", or "all"
        
    Returns:
        List of test suite name strings
    """
    app_def = get_app_definition(app_name)
    if not app_def:
        return []

    if mode == "default":
        return list(app_def.default_test_suites)
    elif mode == "extra":
        return list(app_def.extra_test_suites)
    else:  # "all"
        return list(app_def.default_test_suites) + list(app_def.extra_test_suites)


def get_test_run_combination(name: str) -> Optional[dict]:
    """Get a test run combination by name.
    
    Args:
        name: Combination name (e.g., "ci_run", "all_default")
        
    Returns:
        Combination configuration dict or None if not found
    """
    config = get_test_definition_config()
    combinations = config.get("test_run_combinations", {})
    return combinations.get(name)


def get_all_test_run_combinations() -> list[str]:
    """Get all test run combination names.
    
    Returns:
        List of combination name strings
    """
    config = get_test_definition_config()
    return list(config.get("test_run_combinations", {}).keys())


def get_test_resources() -> dict:
    """Get test resources configuration.
    
    Returns:
        Dictionary with 'videos', 'photos', 'json_files' keys
    """
    config = get_test_definition_config()
    return config.get("resources", {})


# =============================================================================
# Test Control Config API (test_control.yaml)
# =============================================================================

def get_test_control_config(use_cache: bool = True) -> dict:
    """Get the test control configuration.
    
    Args:
        use_cache: Whether to use cached result
        
    Returns:
        Complete test control configuration dictionary
    """
    try:
        return _load_yaml(ConfigPaths.test_control_config(), use_cache)
    except ConfigError:
        # test_control.yaml may not exist in all environments
        return {}


def get_control_parameter(key: str, default: Any = None) -> Any:
    """Get a control parameter value.
    
    Args:
        key: Parameter name (e.g., "default_run_time", "term_timeout")
        default: Default value if not found
        
    Returns:
        Parameter value
    """
    config = get_test_control_config()
    params = config.get("control_parameters", {})
    return params.get(key, default)


def get_logging_config() -> dict:
    """Get logging configuration from test control.
    
    Returns:
        Dictionary with logging settings
    """
    config = get_test_control_config()
    return config.get("logging", {})


def get_enabled_run_methods() -> list[str]:
    """Get list of enabled run methods.
    
    Returns:
        List of enabled run method names (e.g., ["pythonpath", "cli"])
    """
    config = get_test_control_config()
    run_methods = config.get("run_methods", {})
    return [name for name, cfg in run_methods.items() if cfg.get("enabled", False)]


def get_custom_test_apps() -> dict[str, dict]:
    """Get custom test configuration for all apps.
    
    Returns:
        Dictionary mapping app names to their custom test config
    """
    config = get_test_control_config()
    custom_tests = config.get("custom_tests", {})
    if not custom_tests.get("enabled", False):
        return {}
    return custom_tests.get("apps", {})


def is_special_test_enabled(test_name: str) -> bool:
    """Check if a special test is enabled.
    
    Args:
        test_name: Special test name (e.g., "h8l_on_h8", "sanity_checks")
        
    Returns:
        True if the special test is enabled
    """
    config = get_test_control_config()
    special_tests = config.get("special_tests", {})
    return special_tests.get(test_name, {}).get("enabled", False)


def get_enabled_test_combinations() -> list[str]:
    """Get list of enabled test combinations.
    
    Returns:
        List of enabled combination names
    """
    config = get_test_control_config()
    test_combinations = config.get("test_combinations", {})
    return [name for name, cfg in test_combinations.items() if cfg.get("enabled", False)]


def get_custom_standalone_tests() -> dict[str, dict]:
    """Get standalone test configuration for standalone apps.
    
    Returns:
        Dictionary mapping standalone app names (_standalone suffix) to their config.
    """
    config = get_test_control_config()
    standalone_tests = config.get("standalone_tests", {})
    if not standalone_tests.get("enabled", False):
        return {}
    return standalone_tests.get("apps", {})


# =============================================================================
# Cache Management
# =============================================================================

def clear_cache():
    """Clear all cached configuration data.
    
    Call this when configuration files have been modified and you
    need to reload them.
    """
    _load_yaml_cached.cache_clear()


def reload_all():
    """Clear cache and reload all configurations.
    
    Useful for testing or when configs have been modified.
    """
    clear_cache()
    # Trigger reloads
    get_main_config()
    get_resources_config()
    get_test_definition_config()
    get_test_control_config()


# =============================================================================
# CLI Interface
# =============================================================================

def _print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _dry_run():
    """Execute dry run to validate all configurations."""
    print("\n🔍 Configuration Manager Dry Run")
    print("=" * 70)

    errors = []
    warnings = []

    # Check all config paths
    _print_header("Configuration Files")
    config_files = [
        ("Main Config", ConfigPaths.main_config()),
        ("Resources Config", ConfigPaths.resources_config()),
        ("Test Definition Config", ConfigPaths.test_definition_config()),
        ("Test Control Config", ConfigPaths.test_control_config()),
    ]

    for name, path in config_files:
        if path.exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} (NOT FOUND)")
            if name != "Test Control Config":  # test_control is optional
                errors.append(f"{name} not found at {path}")

    # Load and validate main config
    _print_header("Main Configuration (config.yaml)")
    try:
        main_config = get_main_config(use_cache=False)
        print(f"  • HailoRT versions: {get_valid_versions('hailort')}")
        print(f"  • TAPPAS versions: {get_valid_versions('tappas')}")
        print(f"  • Hailo architectures: {get_valid_versions('hailo_arch')}")
        print(f"  • Virtual env: {get_venv_config().get('name', 'N/A')}")
    except ConfigError as e:
        errors.append(f"Main config error: {e}")
        print(f"  ❌ Error: {e}")

    # Load and validate resources config
    _print_header("Resources Configuration (resources_config.yaml)")
    try:
        apps = get_available_apps()
        print(f"  • Total apps: {len(apps)}")
        print(f"  • Apps: {', '.join(apps[:5])}{'...' if len(apps) > 5 else ''}")
        print(f"  • Videos: {len(get_videos())}")
        print(f"  • Images: {len(get_images())}")

        # Check each app
        gen_ai_apps = []
        for app in apps:
            archs = get_supported_architectures(app)
            if not archs:
                warnings.append(f"App '{app}' has no supported architectures")
            if is_gen_ai_app(app):
                gen_ai_apps.append(app)

        if gen_ai_apps:
            print(f"  • Gen-AI apps: {', '.join(gen_ai_apps)}")

    except ConfigError as e:
        errors.append(f"Resources config error: {e}")
        print(f"  ❌ Error: {e}")

    # Load and validate test definition config
    _print_header("Test Definition Configuration (test_definition_config.yaml)")
    try:
        defined_apps = get_defined_apps()
        test_suites = get_all_test_suites()
        combinations = get_all_test_run_combinations()

        print(f"  • Defined apps: {len(defined_apps)}")
        print(f"  • Test suites: {len(test_suites)}")
        print(f"  • Test combinations: {', '.join(combinations)}")

        # Verify test suite references
        for app in defined_apps:
            app_def = get_app_definition(app)
            if app_def:
                for suite in app_def.default_test_suites + app_def.extra_test_suites:
                    if suite not in test_suites:
                        errors.append(f"App '{app}' references undefined test suite '{suite}'")

    except ConfigError as e:
        errors.append(f"Test definition config error: {e}")
        print(f"  ❌ Error: {e}")

    # Load and validate test control config
    _print_header("Test Control Configuration (test_control.yaml)")
    try:
        control_config = get_test_control_config(use_cache=False)
        if control_config:
            print(f"  • Default run time: {get_control_parameter('default_run_time', 'N/A')}s")
            print(f"  • Term timeout: {get_control_parameter('term_timeout', 'N/A')}s")
            print(f"  • Enabled run methods: {get_enabled_run_methods()}")
            print(f"  • Enabled combinations: {get_enabled_test_combinations()}")
        else:
            print("  ⚠️  Test control config not found (optional)")

    except ConfigError as e:
        warnings.append(f"Test control config: {e}")
        print(f"  ⚠️  Warning: {e}")

    # Cross-validation
    _print_header("Cross-Validation")
    try:
        resource_apps = set(get_available_apps())
        definition_apps = {app for app in get_defined_apps() if not is_standalone_app_name(app)}

        in_resources_not_definition = resource_apps - definition_apps
        in_definition_not_resources = definition_apps - resource_apps

        if in_resources_not_definition:
            msg = f"Apps in resources but not in test definitions: {in_resources_not_definition}"
            warnings.append(msg)
            print(f"  ⚠️  {msg}")

        if in_definition_not_resources:
            msg = f"Apps in test definitions but not in resources: {in_definition_not_resources}"
            warnings.append(msg)
            print(f"  ⚠️  {msg}")

        if not in_resources_not_definition and not in_definition_not_resources:
            print("  ✅ All apps are consistent between configs")

    except Exception as e:
        warnings.append(f"Cross-validation error: {e}")

    # Summary
    _print_header("Summary")
    if errors:
        print(f"  ❌ Errors: {len(errors)}")
        for error in errors:
            print(f"     • {error}")
    else:
        print("  ✅ No errors found")

    if warnings:
        print(f"  ⚠️  Warnings: {len(warnings)}")
        for warning in warnings:
            print(f"     • {warning}")
    else:
        print("  ✅ No warnings")

    print()
    return len(errors) == 0


def _list_apps():
    """List all available applications."""
    _print_header("Available Applications")

    apps = get_available_apps()
    for app in apps:
        archs = get_supported_architectures(app)
        gen_ai = " [Gen-AI]" if is_gen_ai_app(app) else ""
        print(f"  • {app}{gen_ai}")
        print(f"      Architectures: {', '.join(archs) if archs else 'None'}")


def _show_models(app_name: str, arch: str):
    """Show models for an app and architecture."""
    _print_header(f"Models for {app_name} on {arch}")

    if app_name not in get_available_apps():
        print(f"  ❌ Unknown app: {app_name}")
        print(f"     Available: {', '.join(get_available_apps())}")
        return

    if arch not in get_supported_architectures(app_name):
        print(f"  ❌ Architecture '{arch}' not supported for {app_name}")
        print(f"     Supported: {', '.join(get_supported_architectures(app_name))}")
        return

    print("\n  Default Models:")
    for model in get_default_models(app_name, arch):
        url_info = f" (url: {model.url[:40]}...)" if model.url else ""
        print(f"    • {model.name} [{model.source}]{url_info}")

    print("\n  Extra Models:")
    extra = get_extra_models(app_name, arch)
    if extra:
        for model in extra:
            url_info = f" (url: {model.url[:40]}...)" if model.url else ""
            print(f"    • {model.name} [{model.source}]{url_info}")
    else:
        print("    (none)")


def _test_all_functions():
    """Test all API functions and print their outputs."""
    import json
    
    def _safe_call(func_name: str, func, *args, **kwargs):
        """Safely call a function and print result."""
        try:
            result = func(*args, **kwargs)
            return result, None
        except Exception as e:
            return None, str(e)
    
    def _print_result(name: str, result, error=None, indent=2):
        """Print a result with formatting."""
        prefix = " " * indent
        if error:
            print(f"{prefix}❌ {name}: ERROR - {error}")
            return False
        
        if result is None:
            print(f"{prefix}✅ {name}: None")
        elif isinstance(result, (list, tuple)):
            if len(result) == 0:
                print(f"{prefix}✅ {name}: [] (empty)")
            elif len(result) <= 5:
                print(f"{prefix}✅ {name}: {result}")
            else:
                print(f"{prefix}✅ {name}: [{len(result)} items] {result[:3]}...")
        elif isinstance(result, dict):
            if len(result) == 0:
                print(f"{prefix}✅ {name}: {{}} (empty)")
            elif len(result) <= 3:
                print(f"{prefix}✅ {name}: {result}")
            else:
                keys = list(result.keys())[:5]
                print(f"{prefix}✅ {name}: {{{len(result)} keys}} {keys}...")
        elif hasattr(result, '__dataclass_fields__'):
            # Dataclass
            print(f"{prefix}✅ {name}: {result}")
        else:
            print(f"{prefix}✅ {name}: {result}")
        return True
    
    print("\n" + "=" * 80)
    print("  CONFIG MANAGER - COMPREHENSIVE FUNCTION TEST")
    print("=" * 80)
    
    errors = []
    success_count = 0
    total_count = 0
    
    # =========================================================================
    # Path Resolution Tests
    # =========================================================================
    _print_header("ConfigPaths Class")
    
    for name, func in [
        ("repo_root()", ConfigPaths.repo_root),
        ("main_config()", ConfigPaths.main_config),
        ("resources_config()", ConfigPaths.resources_config),
        ("test_definition_config()", ConfigPaths.test_definition_config),
        ("test_control_config()", ConfigPaths.test_control_config),
    ]:
        total_count += 1
        result, error = _safe_call(name, func)
        if _print_result(name, result, error):
            success_count += 1
        else:
            errors.append(f"ConfigPaths.{name}")
    
    # =========================================================================
    # Main Config API Tests
    # =========================================================================
    _print_header("Main Config API (config.yaml)")
    
    for name, func, args in [
        ("get_main_config()", get_main_config, []),
        ("get_valid_versions('hailort')", get_valid_versions, ["hailort"]),
        ("get_valid_versions('tappas')", get_valid_versions, ["tappas"]),
        ("get_valid_versions('hailo_arch')", get_valid_versions, ["hailo_arch"]),
        ("get_model_zoo_version_for_arch('hailo8')", get_model_zoo_version_for_arch, ["hailo8"]),
        ("get_model_zoo_version_for_arch('hailo8l')", get_model_zoo_version_for_arch, ["hailo8l"]),
        ("get_model_zoo_version_for_arch('hailo10h')", get_model_zoo_version_for_arch, ["hailo10h"]),
        ("get_model_zoo_mapping()", get_model_zoo_mapping, []),
        ("get_venv_config()", get_venv_config, []),
        ("get_resources_path_config()", get_resources_path_config, []),
    ]:
        total_count += 1
        result, error = _safe_call(name, func, *args)
        if _print_result(name, result, error):
            success_count += 1
        else:
            errors.append(name)
    
    # =========================================================================
    # Resources Config API Tests
    # =========================================================================
    _print_header("Resources Config API (resources_config.yaml)")
    
    result, _ = _safe_call("get_resources_config", get_resources_config)
    total_count += 1
    if _print_result("get_resources_config()", result):
        success_count += 1
    
    # Get available apps for further testing
    apps, _ = _safe_call("get_available_apps", get_available_apps)
    total_count += 1
    if _print_result("get_available_apps()", apps):
        success_count += 1
    else:
        apps = []
    
    # Test with first app if available
    test_app = apps[0] if apps else "detection"
    print(f"\n  Using test app: '{test_app}'")
    
    archs, _ = _safe_call("get_supported_architectures", get_supported_architectures, test_app)
    total_count += 1
    if _print_result(f"get_supported_architectures('{test_app}')", archs):
        success_count += 1
    else:
        archs = []
    
    test_arch = archs[0] if archs else "hailo8"
    print(f"  Using test arch: '{test_arch}'")
    
    # Model functions
    for name, func, args in [
        (f"get_default_models('{test_app}', '{test_arch}')", get_default_models, [test_app, test_arch]),
        (f"get_extra_models('{test_app}', '{test_arch}')", get_extra_models, [test_app, test_arch]),
        (f"get_all_models('{test_app}', '{test_arch}')", get_all_models, [test_app, test_arch]),
        (f"get_model_names('{test_app}', '{test_arch}', 'all')", get_model_names, [test_app, test_arch, "all"]),
        (f"get_model_names('{test_app}', '{test_arch}', 'default')", get_model_names, [test_app, test_arch, "default"]),
        (f"get_default_model_name('{test_app}', '{test_arch}')", get_default_model_name, [test_app, test_arch]),
    ]:
        total_count += 1
        result, error = _safe_call(name, func, *args)
        if _print_result(name, result, error):
            success_count += 1
        else:
            errors.append(name)
    
    # Get a model name for model_info test
    models, _ = _safe_call("get_all_models", get_all_models, test_app, test_arch)
    test_model = models[0].name if models else "yolov8m"
    
    total_count += 1
    result, error = _safe_call("get_model_info", get_model_info, test_app, test_arch, test_model)
    if _print_result(f"get_model_info('{test_app}', '{test_arch}', '{test_model}')", result, error):
        success_count += 1
    
    # Shared resources
    for name, func in [
        ("get_videos()", get_videos),
        ("get_images()", get_images),
        ("get_json_files()", get_json_files),
        ("get_all_json_files()", get_all_json_files),
        (f"is_gen_ai_app('{test_app}')", lambda: is_gen_ai_app(test_app)),
    ]:
        total_count += 1
        result, error = _safe_call(name, func)
        if _print_result(name, result, error):
            success_count += 1
        else:
            errors.append(name)
    
    # =========================================================================
    # Inputs Config API Tests (tag-based filtering)
    # =========================================================================
    _print_header("Inputs Config API (tag-based)")
    
    # Use 'detection' for inputs testing since it has many tagged resources
    inputs_test_app = "detection"
    print(f"  Using inputs test app: '{inputs_test_app}'")
    
    for name, func, args in [
        ("get_all_tags()", get_all_tags, []),
        ("get_apps_with_inputs()", get_apps_with_inputs, []),
        (f"get_videos_for_app('{inputs_test_app}')", get_videos_for_app, [inputs_test_app]),
        (f"get_images_for_app('{inputs_test_app}')", get_images_for_app, [inputs_test_app]),
        (f"get_json_for_app('{inputs_test_app}')", get_json_for_app, [inputs_test_app]),
        (f"get_inputs_for_app('{inputs_test_app}', is_standalone=False)", lambda: get_inputs_for_app(inputs_test_app, is_standalone=False), []),
        (f"get_inputs_for_app('{inputs_test_app}', is_standalone=True)", lambda: get_inputs_for_app(inputs_test_app, is_standalone=True), []),
    ]:
        total_count += 1
        result, error = _safe_call(name, func, *args)
        if _print_result(name, result, error):
            success_count += 1
        else:
            errors.append(name)
    
    # =========================================================================
    # Test Definition Config API Tests
    # =========================================================================
    _print_header("Test Definition Config API (test_definition_config.yaml)")
    
    for name, func, args in [
        ("get_test_definition_config()", get_test_definition_config, []),
        ("get_defined_apps()", get_defined_apps, []),
        ("get_all_test_suites()", get_all_test_suites, []),
        ("get_all_test_run_combinations()", get_all_test_run_combinations, []),
        ("get_test_resources()", get_test_resources, []),
    ]:
        total_count += 1
        result, error = _safe_call(name, func, *args)
        if _print_result(name, result, error):
            success_count += 1
        else:
            errors.append(name)
    
    # Get defined apps for further testing
    defined_apps, _ = _safe_call("get_defined_apps", get_defined_apps)
    test_defined_app = defined_apps[0] if defined_apps else "detection"
    
    for name, func, args in [
        (f"get_app_definition('{test_defined_app}')", get_app_definition, [test_defined_app]),
        (f"get_test_suites_for_app('{test_defined_app}', 'default')", get_test_suites_for_app, [test_defined_app, "default"]),
        (f"get_test_suites_for_app('{test_defined_app}', 'extra')", get_test_suites_for_app, [test_defined_app, "extra"]),
        (f"get_test_suites_for_app('{test_defined_app}', 'all')", get_test_suites_for_app, [test_defined_app, "all"]),
    ]:
        total_count += 1
        result, error = _safe_call(name, func, *args)
        if _print_result(name, result, error):
            success_count += 1
        else:
            errors.append(name)
    
    # Test suite details
    suites, _ = _safe_call("get_all_test_suites", get_all_test_suites)
    test_suite = suites[0] if suites else "basic_show_fps"
    
    total_count += 1
    result, error = _safe_call("get_test_suite", get_test_suite, test_suite)
    if _print_result(f"get_test_suite('{test_suite}')", result, error):
        success_count += 1
    
    # Test run combinations
    combinations, _ = _safe_call("get_all_test_run_combinations", get_all_test_run_combinations)
    test_combo = combinations[0] if combinations else "ci_run"
    
    total_count += 1
    result, error = _safe_call("get_test_run_combination", get_test_run_combination, test_combo)
    if _print_result(f"get_test_run_combination('{test_combo}')", result, error):
        success_count += 1
    
    # =========================================================================
    # Test Control Config API Tests
    # =========================================================================
    _print_header("Test Control Config API (test_control.yaml)")
    
    for name, func, args in [
        ("get_test_control_config()", get_test_control_config, []),
        ("get_control_parameter('default_run_time')", get_control_parameter, ["default_run_time"]),
        ("get_control_parameter('term_timeout')", get_control_parameter, ["term_timeout"]),
        ("get_control_parameter('nonexistent', 'default_val')", get_control_parameter, ["nonexistent", "default_val"]),
        ("get_logging_config()", get_logging_config, []),
        ("get_enabled_run_methods()", get_enabled_run_methods, []),
        ("get_custom_test_apps()", get_custom_test_apps, []),
        ("is_special_test_enabled('h8l_on_h8')", is_special_test_enabled, ["h8l_on_h8"]),
        ("is_special_test_enabled('sanity_checks')", is_special_test_enabled, ["sanity_checks"]),
        ("get_enabled_test_combinations()", get_enabled_test_combinations, []),
    ]:
        total_count += 1
        result, error = _safe_call(name, func, *args)
        if _print_result(name, result, error):
            success_count += 1
        else:
            errors.append(name)
    
    # =========================================================================
    # Cache Management Tests
    # =========================================================================
    _print_header("Cache Management")
    
    total_count += 1
    try:
        clear_cache()
        print("  ✅ clear_cache(): OK")
        success_count += 1
    except Exception as e:
        print(f"  ❌ clear_cache(): ERROR - {e}")
        errors.append("clear_cache()")
    
    total_count += 1
    try:
        reload_all()
        print("  ✅ reload_all(): OK")
        success_count += 1
    except Exception as e:
        print(f"  ❌ reload_all(): ERROR - {e}")
        errors.append("reload_all()")
    
    # =========================================================================
    # Summary
    # =========================================================================
    _print_header("TEST SUMMARY")
    
    print(f"  Total functions tested: {total_count}")
    print(f"  ✅ Passed: {success_count}")
    print(f"  ❌ Failed: {len(errors)}")
    
    if errors:
        print("\n  Failed functions:")
        for err in errors:
            print(f"    • {err}")
        return False
    else:
        print("\n  🎉 All functions passed!")
        return True


def main():

    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hailo Apps Configuration Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m hailo_apps.config.config_manager --dry-run
  python -m hailo_apps.config.config_manager --test-all
  python -m hailo_apps.config.config_manager --list-apps
  python -m hailo_apps.config.config_manager --show-models detection hailo8
  python -m hailo_apps.config.config_manager --show-paths
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all configuration files and show summary",
    )
    parser.add_argument(
        "--list-apps", action="store_true", help="List all available applications"
    )
    parser.add_argument(
        "--show-models",
        nargs=2,
        metavar=("APP", "ARCH"),
        help="Show models for an app and architecture",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Show configuration file paths",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test all API functions and print their outputs",
    )

    args = parser.parse_args()

    if args.dry_run:
        success = _dry_run()
        sys.exit(0 if success else 1)
    elif args.test_all:
        success = _test_all_functions()
        sys.exit(0 if success else 1)
    elif args.list_apps:
        _list_apps()
    elif args.show_models:
        _show_models(args.show_models[0], args.show_models[1])
    elif args.show_paths:
        _print_header("Configuration File Paths")
        print(f"  Repo Root: {ConfigPaths.repo_root()}")
        print(f"  Main Config: {ConfigPaths.main_config()}")
        print(f"  Resources Config: {ConfigPaths.resources_config()}")
        print(f"  Test Definition: {ConfigPaths.test_definition_config()}")
        print(f"  Test Control: {ConfigPaths.test_control_config()}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
