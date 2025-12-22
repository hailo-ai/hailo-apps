"""
Test Functions for All Pipeline Types

This module provides a generic test runner that handles all pipeline types,
dynamically loading available pipelines from the unified config_manager.
"""

import logging
from typing import Dict, List, Optional, Tuple

from hailo_apps.config import config_manager

from test_utils import (
    build_test_args,
    get_log_file_path,
    run_pipeline_test,
)

logger = logging.getLogger(__name__)


def _load_supported_pipelines() -> List[str]:
    """Load supported pipeline names from config_manager.

    Returns:
        List of pipeline names from the test definitions.
        Falls back to an empty list if config cannot be loaded.
    """
    try:
        pipelines = config_manager.get_defined_apps()
        logger.debug(f"Loaded {len(pipelines)} pipelines from config: {pipelines}")
        return pipelines
    except Exception as e:
        logger.error(f"Failed to load pipeline config: {e}")
        return []


# Dynamically load supported pipelines from config
SUPPORTED_PIPELINES = _load_supported_pipelines()


def run_pipeline_test_generic(
    config: Dict,
    pipeline_name: str,
    model: str,
    architecture: str,
    run_method: str,
    test_suite: str = "default",
    extra_args: Optional[List[str]] = None,
    run_time: Optional[int] = None,
    term_timeout: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Generic pipeline test runner.

    This function handles testing for any pipeline type, reducing code duplication.

    Args:
        config: Test configuration dictionary containing pipeline definitions
        pipeline_name: Name of the pipeline (e.g., "detection", "pose_estimation")
        model: Model name to test
        architecture: Target architecture (hailo8, hailo8l, hailo10h)
        run_method: How to run the test (module, pythonpath, cli)
        test_suite: Test suite name (default: "default")
        extra_args: Additional command-line arguments
        run_time: Optional run time override in seconds
        term_timeout: Optional termination timeout override in seconds

    Returns:
        Tuple of (success: bool, log_file_path: str)

    Raises:
        KeyError: If the pipeline_name is not found in config["pipelines"]
    """
    # Get pipeline configuration
    pipeline_config = config["pipelines"][pipeline_name]

    # Build arguments
    args = build_test_args(
        config, pipeline_config, model, architecture, test_suite, extra_args
    )

    # Get log file path
    log_file = get_log_file_path(
        config, "pipeline", pipeline_name, architecture, model, run_method, test_suite
    )

    # Run test
    stdout, stderr, success = run_pipeline_test(
        pipeline_config,
        model,
        architecture,
        run_method,
        args,
        log_file,
        run_time=run_time,
        term_timeout=term_timeout,
    )

    # Format pipeline name for display
    display_name = pipeline_name.replace("_", " ").title()

    if success:
        logger.info(
            f"✓ {display_name} test passed: {model} on {architecture} using {run_method}"
        )
    else:
        logger.error(
            f"✗ {display_name} test failed: {model} on {architecture} using {run_method}"
        )
        if stderr:
            logger.error(
                f"Error: {stderr.decode() if isinstance(stderr, bytes) else stderr}"
            )

    return success, log_file


def _create_pipeline_test_func(pipeline_name: str):
    """
    Factory function to create pipeline-specific test functions.

    Args:
        pipeline_name: Name of the pipeline

    Returns:
        A test function for the specific pipeline
    """

    def test_func(
        config: Dict,
        model: str,
        architecture: str,
        run_method: str,
        test_suite: str = "default",
        extra_args: Optional[List[str]] = None,
        run_time: Optional[int] = None,
        term_timeout: Optional[int] = None,
    ) -> Tuple[bool, str]:
        return run_pipeline_test_generic(
            config,
            pipeline_name,
            model,
            architecture,
            run_method,
            test_suite,
            extra_args,
            run_time,
            term_timeout,
        )

    # Set function metadata
    test_func.__doc__ = f"Run {pipeline_name.replace('_', ' ')} pipeline test."
    test_func.__name__ = f"run_{pipeline_name}_test"
    return test_func


# Auto-generate PIPELINE_TEST_FUNCTIONS dictionary from config
PIPELINE_TEST_FUNCTIONS: Dict[str, callable] = {
    name: _create_pipeline_test_func(name) for name in SUPPORTED_PIPELINES
}


def get_pipeline_test_function(pipeline_name: str):
    """
    Get test function for a pipeline.

    For new code, consider using run_pipeline_test_generic() directly
    instead of looking up specific functions.

    Args:
        pipeline_name: Name of the pipeline

    Returns:
        Test function or None if not found
    """
    # If the pipeline isn't in the pre-generated dict, create one dynamically
    if pipeline_name not in PIPELINE_TEST_FUNCTIONS:
        if pipeline_name in SUPPORTED_PIPELINES:
            PIPELINE_TEST_FUNCTIONS[pipeline_name] = _create_pipeline_test_func(
                pipeline_name
            )
        else:
            # Pipeline not in config - create function anyway for flexibility
            logger.debug(f"Creating test function for unlisted pipeline: {pipeline_name}")
            return _create_pipeline_test_func(pipeline_name)

    return PIPELINE_TEST_FUNCTIONS.get(pipeline_name)


# Create module-level aliases for backward compatibility
def _create_module_aliases():
    """Create module-level function aliases for backward compatibility."""
    aliases = {}
    for pipeline_name in SUPPORTED_PIPELINES:
        func_name = f"run_{pipeline_name}_test"
        aliases[func_name] = PIPELINE_TEST_FUNCTIONS.get(pipeline_name)
    return aliases


_aliases = _create_module_aliases()

# Expose common aliases at module level for backward compatibility
if "detection" in PIPELINE_TEST_FUNCTIONS:
    run_detection_test = PIPELINE_TEST_FUNCTIONS["detection"]
if "pose_estimation" in PIPELINE_TEST_FUNCTIONS:
    run_pose_estimation_test = PIPELINE_TEST_FUNCTIONS["pose_estimation"]
if "depth" in PIPELINE_TEST_FUNCTIONS:
    run_depth_test = PIPELINE_TEST_FUNCTIONS["depth"]
if "instance_segmentation" in PIPELINE_TEST_FUNCTIONS:
    run_instance_segmentation_test = PIPELINE_TEST_FUNCTIONS["instance_segmentation"]
if "simple_detection" in PIPELINE_TEST_FUNCTIONS:
    run_simple_detection_test = PIPELINE_TEST_FUNCTIONS["simple_detection"]
if "face_recognition" in PIPELINE_TEST_FUNCTIONS:
    run_face_recognition_test = PIPELINE_TEST_FUNCTIONS["face_recognition"]
if "multisource" in PIPELINE_TEST_FUNCTIONS:
    run_multisource_test = PIPELINE_TEST_FUNCTIONS["multisource"]
if "reid_multisource" in PIPELINE_TEST_FUNCTIONS:
    run_reid_multisource_test = PIPELINE_TEST_FUNCTIONS["reid_multisource"]
if "tiling" in PIPELINE_TEST_FUNCTIONS:
    run_tiling_test = PIPELINE_TEST_FUNCTIONS["tiling"]
if "paddle_ocr" in PIPELINE_TEST_FUNCTIONS:
    run_paddle_ocr_test = PIPELINE_TEST_FUNCTIONS["paddle_ocr"]
if "clip" in PIPELINE_TEST_FUNCTIONS:
    run_clip_test = PIPELINE_TEST_FUNCTIONS["clip"]
