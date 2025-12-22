"""
Standalone Apps Test Runner

Runs lightweight smoke tests for standalone vision apps using the shared
config-driven harness. Focuses on three suites:
- basic_image_smoke (image in/out, checks output artifact)
- basic_show_fps
- input_video_with_hef

Paddle OCR runs only basic_image_smoke and basic_show_fps.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pytest

from hailo_apps.config import config_manager
from hailo_apps.config.config_manager import ConfigPaths
from hailo_apps.python.core.common.defines import (
    DEFAULT_DOTENV_PATH,
    HAILO8_ARCH,
    HAILO8L_ARCH,
    HAILO_ARCH_KEY,
    HOST_ARCH_KEY,
    RESOURCES_PHOTOS_DIR_NAME,
    RESOURCES_ROOT_PATH_DEFAULT,
    RESOURCES_VIDEOS_DIR_NAME,
)
from hailo_apps.python.core.common.core import load_environment
from hailo_apps.python.core.common.installation_utils import (
    detect_hailo_arch,
    detect_host_arch,
)

from test_utils import build_hef_path, run_pipeline_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_standalone_runner")

# Suites allowed per app override for multi-model apps
# These apps don't support input_video_with_hef since it only passes one HEF
MULTI_MODEL_ALLOWED_SUITES = {"basic_image_smoke", "basic_show_fps"}


def is_multi_model_app(app_name: str, architecture: str) -> bool:
    """Check if an app requires multiple models.

    Multi-model apps have more than one default model in resources_config.yaml.
    These apps need all models passed together, not iterated separately.

    Args:
        app_name: Application name (can be standalone or base app name)
        architecture: Architecture

    Returns:
        True if app requires multiple models
    """
    base_name = config_manager.base_app_name(app_name)
    default_models = config_manager.get_model_names(base_name, architecture, tier="default")
    return len(default_models) > 1


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

    os.environ[HOST_ARCH_KEY] = host_arch
    if hailo_arch:
        os.environ[HAILO_ARCH_KEY] = hailo_arch

    logger.info("=" * 80)
    return host_arch, hailo_arch


# ============================================================================
# Helpers
# ============================================================================


def _select_inputs(app_base_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Pick a default video and image for the app (prefers tagged entries)."""
    resources_root = RESOURCES_ROOT_PATH_DEFAULT
    inputs = config_manager.get_inputs_for_app(app_base_name, is_standalone=True)
    video_path = None
    image_path = None

    videos = inputs.get("videos", []) or []
    if videos:
        name = videos[0]["name"] if isinstance(videos[0], dict) else videos[0]
        video_path = str(Path(resources_root) / RESOURCES_VIDEOS_DIR_NAME / name)
    elif config_manager.get_videos():
        name = config_manager.get_videos()[0]
        video_path = str(Path(resources_root) / RESOURCES_VIDEOS_DIR_NAME / name)

    images = inputs.get("images", []) or []
    if images:
        name = images[0]["name"] if isinstance(images[0], dict) else images[0]
        image_path = str(Path(resources_root) / RESOURCES_PHOTOS_DIR_NAME / name)
    elif config_manager.get_images():
        name = config_manager.get_images()[0]
        image_path = str(Path(resources_root) / RESOURCES_PHOTOS_DIR_NAME / name)

    return video_path, image_path


def _get_labels_json(app_base_name: str) -> str:
    """Return first shared JSON for the app if available."""
    json_files = config_manager.get_json_files(app_base_name)
    if json_files:
        return str(Path(RESOURCES_ROOT_PATH_DEFAULT) / "json" / json_files[0])
    return ""


def _resolve_standalone_flags(
    suite_name: str,
    app_name: str,
    architecture: str,
    model: Union[str, Sequence[str]],
    video_path: Optional[str],
    image_path: Optional[str],
    output_dir: str,
) -> Tuple[List[str], List[str]]:
    """Resolve flags with placeholders for a standalone app.

    Returns resolved flags and the list of HEF paths used.
    """
    suite = config_manager.get_test_suite(suite_name)
    if not suite:
        return [], []

    # Smoke test: no flags, no injected HEF flags
    if not suite.flags and suite_name == "basic_image_smoke":
        return [], []

    resources_root = RESOURCES_ROOT_PATH_DEFAULT
    app_base = config_manager.base_app_name(app_name)

    models = [model] if isinstance(model, str) else list(model)
    hef_paths = [build_hef_path(m, architecture, resources_root) for m in models]

    labels_json_path = _get_labels_json(app_base)

    resolved_flags: List[str] = []
    for flag in suite.flags:
        resolved = flag.replace("${RESOURCES_ROOT}", resources_root)
        resolved = resolved.replace("${OUTPUT_DIR}", output_dir)
        if video_path:
            resolved = resolved.replace("${VIDEO_PATH}", video_path)
        if image_path:
            resolved = resolved.replace("${IMAGE_PATH}", image_path)
        if labels_json_path:
            resolved = resolved.replace("${LABELS_JSON_PATH}", labels_json_path)
        # Temporarily set HEF placeholder to first path; we will normalize later
        if hef_paths:
            resolved = resolved.replace("${HEF_PATH}", hef_paths[0])
        resolved_flags.append(resolved)

    # Normalize HEF flags: remove existing --hef-path entries and re-add one per HEF
    filtered_flags: List[str] = []
    skip_next = False
    for i, token in enumerate(resolved_flags):
        if skip_next:
            skip_next = False
            continue
        if token == "--hef-path":
            skip_next = True
            continue
        filtered_flags.append(token)

    if hef_paths:
        for hef in hef_paths:
            filtered_flags.extend(["--hef-path", hef])

    return filtered_flags, hef_paths


def _get_log_file_path(
    app_name: str,
    test_suite_mode: str,
    architecture: Optional[str],
    model: Union[str, Sequence[str]],
    run_method: str,
    test_suite: str,
) -> str:
    """Build log file path using logging config."""
    log_config = config_manager.get_logging_config()
    subdirs = log_config.get("subdirs", {})
    per_app = subdirs.get("per_app", {})

    log_mode = "default" if test_suite_mode in (None, "None") else test_suite_mode
    if app_name in per_app:
        app_log_dirs = per_app[app_name]
        log_dir = app_log_dirs.get(log_mode, app_log_dirs.get("default", "./logs"))
    else:
        log_dir = log_config.get("base_dir", "./logs")

    os.makedirs(log_dir, exist_ok=True)

    model_part = model if isinstance(model, str) else "_".join(model)
    parts = [app_name]
    if architecture:
        parts.append(architecture)
    if model_part:
        parts.append(model_part)
    if run_method:
        parts.append(run_method)
    if test_suite and test_suite != "basic_show_fps":
        parts.append(test_suite)

    filename = "_".join(parts) + ".log"
    return str(Path(log_dir) / filename)


def _get_output_dir(app_name: str, test_suite: str, architecture: str, model: Union[str, Sequence[str]]) -> str:
    """Create an output directory for a test case."""
    model_part = model if isinstance(model, str) else "_".join(model)
    base = Path("logs") / "standalone_outputs" / app_name / test_suite / architecture / model_part
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def _is_allowed_suite(app_name: str, suite_name: str, architecture: str) -> bool:
    """Enforce per-app allowed suites.

    Multi-model apps don't support suites that pass a single HEF path
    (like input_video_with_hef), so they're limited to smoke tests.
    """
    if is_multi_model_app(app_name, architecture):
        return suite_name in MULTI_MODEL_ALLOWED_SUITES
    return True


# ============================================================================
# Test Case Generation
# ============================================================================


def generate_test_cases(
    detected_hailo_arch: Optional[str],
    host_arch: str,
) -> List[Dict]:
    """Generate standalone test cases based on configuration."""
    test_cases: List[Dict] = []

    default_run_time = config_manager.get_control_parameter("default_run_time", 24)
    term_timeout = config_manager.get_control_parameter("term_timeout", 10)
    run_methods = config_manager.get_enabled_run_methods() or ["pythonpath"]

    custom_apps = config_manager.get_custom_standalone_tests()
    if not custom_apps:
        logger.warning("No standalone tests enabled in test_control.yaml")
        return []

    if not detected_hailo_arch:
        logger.warning("No Hailo device detected - skipping standalone tests")
        return []

    architectures = [detected_hailo_arch]
    if detected_hailo_arch == HAILO8_ARCH and config_manager.is_special_test_enabled("h8l_on_h8"):
        logger.info("h8l_on_h8 enabled - adding hailo8l architecture for standalone tests")
        architectures.append(HAILO8L_ARCH)

    for app_name, cfg in custom_apps.items():
        app_def = config_manager.get_standalone_app_definition(app_name)
        if not app_def:
            logger.warning("Standalone app %s not found in test definitions", app_name)
            continue

        test_suite_mode = cfg.get("test_suite_mode", "default")
        model_selection = cfg.get("model_selection", "default")

        for architecture in architectures:
            models = config_manager.get_standalone_model_names(app_name, architecture, tier=model_selection)
            if not models:
                logger.info("No models for %s on %s", app_name, architecture)
                continue

            # Filter suites based on architecture (multi-model detection needs arch)
            suites = config_manager.get_standalone_test_suites_for_app(app_name, test_suite_mode)
            suites = [s for s in suites if _is_allowed_suite(app_name, s, architecture)]

            if not suites:
                logger.info("No suites for %s with mode %s on %s", app_name, test_suite_mode, architecture)
                continue

            app_base = config_manager.base_app_name(app_name)
            video_path, image_path = _select_inputs(app_base)

            # For multi-model apps, treat the entire list as one test bundle
            # Use dynamic detection instead of hardcoded set
            multi_model = is_multi_model_app(app_name, architecture)
            model_list: List[Union[str, Sequence[str]]] = []
            if multi_model:
                model_list.append(tuple(models))
                logger.info(
                    "Multi-model standalone app %s: using models together: %s",
                    app_name, models
                )
            else:
                model_list.extend(models)

            for model in model_list:
                for run_method in run_methods:
                    for suite in suites:
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
                                "test_suite": suite,
                                "run_time": default_run_time,
                                "term_timeout": term_timeout,
                                "test_suite_mode": test_suite_mode,
                                "model_selection": model_selection,
                                "video_path": video_path,
                                "image_path": image_path,
                            }
                        )

    logger.info("Generated %d standalone test cases", len(test_cases))
    return test_cases


# ============================================================================
# Module Initialization
# ============================================================================


logger.info("Initializing standalone test runner...")
load_env_file()
_host_arch, _hailo_arch = detect_and_set_environment()
_test_cases = generate_test_cases(_hailo_arch, _host_arch)


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "test_case",
    _test_cases,
    ids=lambda tc: f"{tc['app']}_{tc['architecture']}_{tc['model']}_{tc['run_method']}_{tc['test_suite']}",
)
def test_standalone_app(test_case: Dict):
    """Run a standalone app smoke test."""
    app_name = test_case["app"]
    app_config = test_case["app_config"]
    architecture = test_case["architecture"]
    model = test_case["model"]
    run_method = test_case["run_method"]
    test_suite = test_case["test_suite"]
    run_time = test_case["run_time"]
    term_timeout = test_case["term_timeout"]
    test_suite_mode = test_case.get("test_suite_mode", "default")

    video_path = test_case.get("video_path")
    image_path = test_case.get("image_path")

    output_dir = _get_output_dir(app_name, test_suite, architecture, model)
    flags, hef_paths = _resolve_standalone_flags(
        test_suite, app_name, architecture, model, video_path, image_path, output_dir
    )

    # Ensure we always provide an output dir for smoke tests
    if flags and "--output-dir" not in flags:
        flags.extend(["--output-dir", output_dir])

    log_file = _get_log_file_path(
        app_name, test_suite_mode, architecture, model, run_method, test_suite
    )

    # Run test
    stdout, stderr, success = run_pipeline_test(
        app_config,
        model if isinstance(model, str) else "+".join(model),
        architecture,
        run_method,
        flags,
        log_file,
        run_time=run_time,
        term_timeout=term_timeout,
    )

    assert success, (
        f"Standalone app {app_name} with model {model} on {architecture} "
        f"using {run_method} with suite {test_suite} failed. "
        f"Check log: {log_file}"
    )


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("STANDALONE TEST RUNNER SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Host architecture: {_host_arch}")
    logger.info(f"Hailo architecture: {_hailo_arch or 'None'}")
    logger.info(f"Total test cases: {len(_test_cases)}")
    logger.info("=" * 80)

    pytest.main(["-v", __file__])
