"""
Test Runner - Unified test execution based on configuration.

This module:
1. Uses the unified config_manager for all configuration loading
2. Integrates with test_control.yaml, test_definition_config.yaml, and resources_config.yaml
3. Uses existing test framework functions
4. Supports run_mode (default/extra/all) and test_run_combinations
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pytest

# Import from unified config manager
from hailo_apps.config import config_manager
from hailo_apps.config.config_manager import ConfigPaths
from hailo_apps.python.core.common.defines import (
    DEFAULT_DOTENV_PATH,
    HAILO8_ARCH,
    HAILO8L_ARCH,
    HAILO_ARCH_KEY,
    HOST_ARCH_KEY,
    RESOURCES_ROOT_PATH_DEFAULT,
)
from hailo_apps.python.core.common.core import load_environment
from hailo_apps.python.core.common.installation_utils import (
    detect_hailo_arch,
    detect_host_arch,
)

from all_tests import get_pipeline_test_function
from test_utils import build_hef_path, run_pipeline_test

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_runner")


# ============================================================================
# Environment Setup
# ============================================================================


def load_env_file():
    """Load .env file from resources directory or default location."""
    repo_env_file = ConfigPaths.repo_root() / "resources" / ".env"
    if repo_env_file.exists():
        logger.info(f"Loading .env file from: {repo_env_file}")
        load_environment(env_file=str(repo_env_file), required_vars=None)
    else:
        logger.info(f"Loading .env file from default location: {DEFAULT_DOTENV_PATH}")
        load_environment(env_file=DEFAULT_DOTENV_PATH, required_vars=None)


def detect_and_set_environment():
    """Detect host and Hailo architecture and set environment variables."""
    logger.info("=" * 80)
    logger.info("DETECTING SYSTEM ARCHITECTURE")
    logger.info("=" * 80)

    host_arch = detect_host_arch()
    hailo_arch = detect_hailo_arch()

    logger.info(f"Detected host architecture: {host_arch}")
    logger.info(f"Detected Hailo architecture: {hailo_arch or 'None (no device detected)'}")

    # Set in current process environment using the same keys the apps read
    os.environ[HOST_ARCH_KEY] = host_arch
    if hailo_arch:
        os.environ[HAILO_ARCH_KEY] = hailo_arch

    logger.info("=" * 80)
    return host_arch, hailo_arch


# ============================================================================
# Model and Resource Resolution (using config_manager)
# ============================================================================


def get_models_for_app_and_arch(
    app_name: str, architecture: str, model_selection: str = "default"
) -> List[str]:
    """Get models for an app and architecture based on model_selection.

    Args:
        app_name: Application name
        architecture: Architecture (hailo8, hailo8l, hailo10h)
        model_selection: Model selection (default, extra, all)

    Returns:
        List of model names
    """
    return config_manager.get_model_names(app_name, architecture, tier=model_selection)


def is_multi_model_app(app_name: str, architecture: str) -> bool:
    """Check if an app requires multiple models (e.g., face_recognition, paddle_ocr, reid).

    Multi-model apps have more than one default model in resources_config.yaml.
    These apps need all models passed together, not iterated separately.

    Args:
        app_name: Application name
        architecture: Architecture

    Returns:
        True if app requires multiple models
    """
    default_models = config_manager.get_model_names(app_name, architecture, tier="default")
    return len(default_models) > 1


def resolve_test_suite_flags(
    suite_name: str,
    app_name: str,
    architecture: str,
    model: Union[str, Sequence[str]],
) -> List[str]:
    """Resolve test suite flags with placeholders replaced.

    Args:
        suite_name: Test suite name
        app_name: Application name
        architecture: Architecture
        model: Model name or list of model names for multi-model apps

    Returns:
        List of resolved flags
    """
    suite = config_manager.get_test_suite(suite_name)
    if not suite:
        return []

    flags = list(suite.flags)

    # Resolve placeholders
    resources_root = RESOURCES_ROOT_PATH_DEFAULT

    # Handle single or multiple models
    models = [model] if isinstance(model, str) else list(model)
    hef_paths = [build_hef_path(m, architecture, resources_root) for m in models]

    # Get video path
    video_name = "example.mp4"  # default
    video_path = os.path.join(resources_root, "videos", video_name)

    # Get labels JSON path if needed
    labels_json_path = ""
    json_files = config_manager.get_json_files(app_name)
    if json_files:
        labels_json_path = os.path.join(resources_root, "json", json_files[0])

    # Replace placeholders - use first HEF path for ${HEF_PATH} placeholder
    resolved_flags = []
    for flag in flags:
        resolved = flag.replace("${HEF_PATH}", hef_paths[0] if hef_paths else "")
        resolved = resolved.replace("${VIDEO_PATH}", video_path)
        if labels_json_path:
            resolved = resolved.replace("${LABELS_JSON_PATH}", labels_json_path)
        resolved = resolved.replace("${RESOURCES_ROOT}", resources_root)
        resolved_flags.append(resolved)

    # For multi-model apps: normalize HEF flags
    # Remove any --hef-path entries and re-add one per HEF file
    if len(hef_paths) > 1:
        filtered_flags = []
        skip_next = False
        for token in resolved_flags:
            if skip_next:
                skip_next = False
                continue
            if token == "--hef-path":
                skip_next = True
                continue
            filtered_flags.append(token)

        # Add all HEF paths
        for hef in hef_paths:
            filtered_flags.extend(["--hef-path", hef])

        return filtered_flags

    return resolved_flags


def get_test_suites_for_mode(app_name: str, test_suite_mode: str) -> List[str]:
    """Get test suites for an app based on test_suite_mode.

    Args:
        app_name: Application name
        test_suite_mode: Test suite mode (None, default, extra, all)

    Returns:
        List of test suite names (empty list if None)
    """
    if test_suite_mode == "None" or test_suite_mode is None:
        return []

    return config_manager.get_test_suites_for_app(app_name, mode=test_suite_mode)


# ============================================================================
# Test Case Generation
# ============================================================================


def generate_test_cases(
    detected_hailo_arch: Optional[str],
    host_arch: str,
    test_run_combination: Optional[str] = None,
) -> List[Dict]:
    """Generate test cases based on configuration.

    Args:
        detected_hailo_arch: Detected Hailo architecture
        host_arch: Host architecture
        test_run_combination: Optional test run combination name

    Returns:
        List of test case dictionaries
    """
    test_cases = []

    # Get control parameters
    default_run_time = config_manager.get_control_parameter("default_run_time", 24)
    term_timeout = config_manager.get_control_parameter("term_timeout", 10)

    # Get enabled run methods
    enabled_run_methods = config_manager.get_enabled_run_methods()
    if not enabled_run_methods:
        enabled_run_methods = ["pythonpath"]  # Default

    # Determine which apps and modes to test
    enabled_combinations = config_manager.get_enabled_test_combinations()

    if enabled_combinations:
        # Use first enabled test combination
        combination_name = enabled_combinations[0]
        logger.info(f"Using enabled test combination: {combination_name}")
        combo = config_manager.get_test_run_combination(combination_name)
        if not combo:
            logger.warning(f"Test run combination {combination_name} not found")
            return []

        apps_to_test = combo.get("apps", [])
        test_suite_mode = combo.get("test_suite_mode", combo.get("mode", "default"))
        model_selection = combo.get("model_selection", "default")
    elif test_run_combination:
        # Use explicitly provided test run combination
        combo = config_manager.get_test_run_combination(test_run_combination)
        if not combo:
            logger.warning(f"Test run combination {test_run_combination} not found")
            return []

        apps_to_test = combo.get("apps", [])
        test_suite_mode = combo.get("test_suite_mode", combo.get("mode", "default"))
        model_selection = combo.get("model_selection", "default")
    else:
        # Use custom_tests per-app configuration
        custom_apps = config_manager.get_custom_test_apps()
        if custom_apps:
            apps_to_test = []
            for app_name, app_config in custom_apps.items():
                tsm = app_config.get("test_suite_mode", "default")
                ms = app_config.get("model_selection", "default")
                if tsm and tsm != "None":
                    apps_to_test.append((app_name, tsm, ms))
        else:
            logger.warning("No test combination enabled and custom_tests disabled.")
            apps_to_test = []

    # Determine architectures to test
    architectures = []
    if detected_hailo_arch:
        architectures = [detected_hailo_arch]

        # Special case: h8l_on_h8
        if (
            detected_hailo_arch == HAILO8_ARCH
            and config_manager.is_special_test_enabled("h8l_on_h8")
        ):
            logger.info("h8l_on_h8 special test enabled - adding hailo8l architecture")
            architectures.append(HAILO8L_ARCH)
    else:
        logger.warning("No Hailo device detected - skipping architecture-specific tests")
        architectures = []

    # Generate test cases
    if isinstance(apps_to_test, list) and apps_to_test and isinstance(apps_to_test[0], tuple):
        # Use individual app configurations (from custom_tests)
        for app_config_tuple in apps_to_test:
            if len(app_config_tuple) == 3:
                app_name, test_suite_mode, model_selection = app_config_tuple
            else:
                app_name, test_suite_mode = app_config_tuple
                model_selection = "default"
            _generate_cases_for_app(
                test_cases,
                app_name,
                test_suite_mode,
                model_selection,
                architectures,
                enabled_run_methods,
                default_run_time,
                term_timeout,
                host_arch,
            )
    elif isinstance(apps_to_test, list):
        # Use combination mode for all apps
        for app_name in apps_to_test:
            _generate_cases_for_app(
                test_cases,
                app_name,
                test_suite_mode,
                model_selection,
                architectures,
                enabled_run_methods,
                default_run_time,
                term_timeout,
                host_arch,
            )

    logger.info(f"Generated {len(test_cases)} test cases")
    return test_cases


def _generate_cases_for_app(
    test_cases: List[Dict],
    app_name: str,
    test_suite_mode: str,
    model_selection: str,
    architectures: List[str],
    run_methods: List[str],
    default_run_time: int,
    term_timeout: int,
    host_arch: str,
):
    """Generate test cases for a specific app."""
    app_def = config_manager.get_app_definition(app_name)
    if not app_def:
        logger.warning(f"App {app_name} not found in definition config")
        return

    # Get test suites for this mode
    test_suites = get_test_suites_for_mode(app_name, test_suite_mode)

    # Filter out RPI camera tests if host is not rpi
    rpi_suites = ["basic_input_rpi", "input_rpi_with_hef", "input_rpi_with_labels"]
    if host_arch != "rpi":
        test_suites = [ts for ts in test_suites if ts not in rpi_suites]

    for architecture in architectures:
        # Get models for this app and architecture
        models = get_models_for_app_and_arch(app_name, architecture, model_selection)
        if not models:
            logger.info(
                f"No models for {app_name} on {architecture} with {model_selection}"
            )
            continue

        # Check if this is a multi-model app (requires multiple HEFs together)
        # Multi-model apps (e.g., face_recognition, paddle_ocr, reid_multisource)
        # need all their default models passed together, not iterated separately
        multi_model = is_multi_model_app(app_name, architecture)

        # For multi-model apps, treat all models as a single group
        # For single-model apps, iterate each model separately
        model_list: List[Union[str, Sequence[str]]] = []
        if multi_model:
            # Pass all models together as a tuple/list
            model_list.append(tuple(models))
            logger.info(
                f"Multi-model app {app_name}: using models together: {models}"
            )
        else:
            # Iterate each model separately
            model_list.extend(models)

        for model in model_list:
            for run_method in run_methods:
                for test_suite in test_suites:
                    flags = resolve_test_suite_flags(
                        test_suite, app_name, architecture, model
                    )

                    test_cases.append(
                        {
                            "app": app_name,
                            "app_config": {
                                "name": app_def.name,
                                "module": app_def.module,
                                "script": app_def.script,
                                "cli": app_def.cli,
                            },
                            "architecture": architecture,
                            "model": model,
                            "run_method": run_method,
                            "test_suite": test_suite,
                            "flags": flags,
                            "run_time": default_run_time,
                            "term_timeout": term_timeout,
                            "test_suite_mode": test_suite_mode,
                            "model_selection": model_selection,
                        }
                    )


def get_log_file_path_new(
    app_name: str,
    test_suite_mode: str,
    architecture: Optional[str] = None,
    model: Optional[Union[str, Sequence[str]]] = None,
    run_method: Optional[str] = None,
    test_suite: Optional[str] = None,
) -> str:
    """Get log file path using config manager."""
    log_config = config_manager.get_logging_config()
    subdirs = log_config.get("subdirs", {})
    per_app = subdirs.get("per_app", {})

    # Get app-specific log directory
    if app_name in per_app:
        app_log_dirs = per_app[app_name]
        log_mode = "default" if test_suite_mode == "None" else test_suite_mode
        log_dir = app_log_dirs.get(log_mode, app_log_dirs.get("default", "./logs"))
    else:
        log_dir = log_config.get("base_dir", "./logs")

    os.makedirs(log_dir, exist_ok=True)

    # Build filename
    parts = [app_name]
    if architecture:
        parts.append(architecture)
    if model:
        # Handle multi-model case: use first model name for log file
        model_name = model if isinstance(model, str) else model[0]
        parts.append(model_name)
    if run_method:
        parts.append(run_method)
    if test_suite and test_suite != "basic_show_fps":
        suite_name = (
            test_suite.replace("basic_", "")
            .replace("input_", "")
            .replace("pipeline_", "")
            .replace("with_", "")
            .replace("face_", "")
            .replace("tiling_", "")
            .replace("clip_", "")
            .replace("multisource_", "")
        )
        parts.append(suite_name)

    filename = "_".join(parts) + ".log"
    return os.path.join(log_dir, filename)


# ============================================================================
# Module Initialization
# ============================================================================

logger.info("Initializing test runner...")
load_env_file()
_host_arch, _hailo_arch = detect_and_set_environment()

# Generate test cases
_test_cases = generate_test_cases(_hailo_arch, _host_arch)


# ============================================================================
# Test Function
# ============================================================================


def _get_test_id(tc: Dict) -> str:
    """Generate test ID, handling multi-model cases."""
    model = tc['model']
    model_str = model if isinstance(model, str) else "_".join(model)
    return f"{tc['app']}_{tc['architecture']}_{model_str}_{tc['run_method']}_{tc['test_suite']}"


@pytest.mark.parametrize(
    "test_case",
    _test_cases,
    ids=_get_test_id,
)
def test_pipeline(test_case: Dict):
    """Test a pipeline with specific configuration."""
    app_name = test_case["app"]
    app_config = test_case["app_config"]
    architecture = test_case["architecture"]
    model = test_case["model"]
    run_method = test_case["run_method"]
    test_suite = test_case["test_suite"]
    flags = test_case["flags"]
    run_time = test_case["run_time"]
    term_timeout = test_case["term_timeout"]
    test_suite_mode = test_case.get("test_suite_mode", "default")

    # Get test function
    test_func = get_pipeline_test_function(app_name)
    if not test_func:
        pytest.skip(f"No test function for app: {app_name}")

    # Build pipeline config for compatibility
    pipeline_config = {
        "name": app_config.get("name", app_name),
        "module": app_config.get("module", ""),
        "script": app_config.get("script", ""),
        "cli": app_config.get("cli", ""),
    }

    # Get log file path
    log_file = get_log_file_path_new(
        app_name, test_suite_mode, architecture, model, run_method, test_suite
    )

    # Run test
    stdout, stderr, success = run_pipeline_test(
        pipeline_config,
        model,
        architecture,
        run_method,
        flags,
        log_file,
        run_time=run_time,
        term_timeout=term_timeout,
    )

    assert success, (
        f"App {app_name} with model {model} on {architecture} "
        f"using {run_method} with suite {test_suite} failed. "
        f"Check log: {log_file}"
    )


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("TEST RUNNER SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Host architecture: {_host_arch}")
    logger.info(f"Hailo architecture: {_hailo_arch or 'None'}")
    logger.info(f"Total test cases: {len(_test_cases)}")
    logger.info("=" * 80)

    pytest.main(["-v", __file__])
