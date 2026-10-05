# region imports
# Standard library imports
import os
import logging
import re

# Third-party imports
import pytest

# Local application-specific imports
from test_utils import (
    run_pipeline_module_with_args,
    run_pipeline_pythonpath_with_args,
    run_pipeline_cli_with_args,
)
from hailo_apps.python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.python.core.common.defines import HAILO8_ARCH, HAILO8L_ARCH, RESOURCES_ROOT_PATH_DEFAULT
# endregion imports

# Configure logging as needed.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_run_everything')
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)


# ============================================================================
# Helper functions (previously in old test_utils module)
# ============================================================================

def get_pipeline_args(suite='default'):
    """Build pipeline args for a given test suite."""
    args = []
    if suite == 'mode-train':
        args.extend(["--mode", "train"])
    elif suite == 'mode-delete':
        args.extend(["--mode", "delete"])
    elif suite == 'usb_camera':
        args.extend(["--input", "usb"])
    # default suite has no extra args
    return args


def check_qos_performance_warning(stdout, stderr):
    """Check for QoS performance warnings in output."""
    combined = ""
    if stdout:
        combined += stdout.decode(errors='replace').lower()
    if stderr:
        combined += stderr.decode(errors='replace').lower()
    qos_matches = re.findall(r'qos', combined)
    count = len(qos_matches)
    return (count >= 100, count)


def check_hailo8l_on_hailo8_warning(stdout, stderr):
    """Check for HailoRT warning when running Hailo8L model on Hailo8."""
    combined = ""
    if stdout:
        combined += stdout.decode(errors='replace')
    if stderr:
        combined += stderr.decode(errors='replace')
    return "warning" in combined.lower() and ("hailo8l" in combined.lower() or "8l" in combined.lower())


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture
def pipeline():
    return {
        'name': 'face_recognition',
        'module': 'hailo_apps.python.pipeline_apps.face_recognition.face_recognition',
        'script': 'hailo_apps/python/pipeline_apps/face_recognition/face_recognition.py',
        'cli': 'hailo-face-recon'
    }

# Map each run method label to its corresponding function.
run_methods = {
    'module': run_pipeline_module_with_args,
    'pythonpath': run_pipeline_pythonpath_with_args,
    'cli': run_pipeline_cli_with_args
}

@pytest.mark.parametrize('run_method_name', list(run_methods.keys()))
def test_train(pipeline, run_method_name):
    test_name = 'test_train'
    args = get_pipeline_args(suite='mode-train')
    log_file_path = os.path.join(log_dir, f"{pipeline['name']}_{test_name}_{run_method_name}.log")

    if run_method_name == 'module':
        stdout, stderr = run_methods[run_method_name](pipeline['module'], args, log_file_path)
    elif run_method_name == 'pythonpath':
        stdout, stderr = run_methods[run_method_name](pipeline['script'], args, log_file_path)
    elif run_method_name == 'cli':
        stdout, stderr = run_methods[run_method_name](pipeline['cli'], args, log_file_path)
    else:
        pytest.fail(f"Unknown run method: {run_method_name}")

    out_str = stdout.decode().lower() if stdout else ""
    err_str = stderr.decode().lower() if stderr else ""
    print(f"Completed: {test_name}, {pipeline['name']}, {run_method_name}: {out_str}")
    assert 'error' not in err_str, f"{pipeline['name']} ({run_method_name}) reported an error in {test_name}: {err_str}"
    assert 'traceback' not in err_str, f"{pipeline['name']} ({run_method_name}) traceback in {test_name} : {err_str}"
    # Check for QoS performance issues
    has_qos_warning, qos_count = check_qos_performance_warning(stdout, stderr)
    if has_qos_warning:
        logger.warning(f"Performance issue detected: QoS messages: {qos_count} total (>=100) for {pipeline['name']} ({run_method_name}) {test_name}")

@pytest.mark.parametrize('run_method_name', list(run_methods.keys()))
def test_default(pipeline, run_method_name):
    test_name = 'test_default'
    args = get_pipeline_args(suite='default')
    log_file_path = os.path.join(log_dir, f"{pipeline['name']}_{test_name}_{run_method_name}.log")

    if run_method_name == 'module':
        stdout, stderr = run_methods[run_method_name](pipeline['module'], args, log_file_path)
    elif run_method_name == 'pythonpath':
        stdout, stderr = run_methods[run_method_name](pipeline['script'], args, log_file_path)
    elif run_method_name == 'cli':
        stdout, stderr = run_methods[run_method_name](pipeline['cli'], args, log_file_path)
    else:
        pytest.fail(f"Unknown run method: {run_method_name}")

    out_str = stdout.decode().lower() if stdout else ""
    err_str = stderr.decode().lower() if stderr else ""
    print(f"Completed: {test_name}, {pipeline['name']}, {run_method_name}: {out_str}")
    assert 'error' not in err_str, f"{pipeline['name']} ({run_method_name}) reported an error in {test_name}: {err_str}"
    assert 'traceback' not in err_str, f"{pipeline['name']} ({run_method_name}) traceback in {test_name} : {err_str}"
    # Check for QoS performance issues
    has_qos_warning, qos_count = check_qos_performance_warning(stdout, stderr)
    if has_qos_warning:
        logger.warning(f"Performance issue detected: QoS messages: {qos_count} total (>=100) for {pipeline['name']} ({run_method_name}) {test_name}")

@pytest.mark.parametrize('run_method_name', list(run_methods.keys()))
def test_cli_usb(pipeline, run_method_name):
    test_name = 'test_cli_usb'
    args = get_pipeline_args(suite='usb_camera')
    log_file_path = os.path.join(log_dir, f"{pipeline['name']}_{test_name}_{run_method_name}.log")

    if run_method_name == 'module':
        stdout, stderr = run_methods[run_method_name](pipeline['module'], args, log_file_path)
    elif run_method_name == 'pythonpath':
        stdout, stderr = run_methods[run_method_name](pipeline['script'], args, log_file_path)
    elif run_method_name == 'cli':
        stdout, stderr = run_methods[run_method_name](pipeline['cli'], args, log_file_path)
    else:
        pytest.fail(f"Unknown run method: {run_method_name}")

    out_str = stdout.decode().lower() if stdout else ""
    err_str = stderr.decode().lower() if stderr else ""
    print(f"Completed: {test_name}, {pipeline['name']}, {run_method_name}: {out_str}")
    assert 'error' not in err_str, f"{pipeline['name']} ({run_method_name}) reported an error in {test_name}: {err_str}"
    assert 'traceback' not in err_str, f"{pipeline['name']} ({run_method_name}) traceback in {test_name} : {err_str}"
    # Check for QoS performance issues
    has_qos_warning, qos_count = check_qos_performance_warning(stdout, stderr)
    if has_qos_warning:
        logger.warning(f"Performance issue detected: QoS messages: {qos_count} total (>=100) for {pipeline['name']} ({run_method_name}) {test_name}")

@pytest.mark.parametrize('run_method_name', list(run_methods.keys()))
def test_delete(pipeline, run_method_name):
    test_name = 'test_delete'
    args = get_pipeline_args(suite='mode-delete')
    log_file_path = os.path.join(log_dir, f"{pipeline['name']}_{test_name}_{run_method_name}.log")

    if run_method_name == 'module':
        stdout, stderr = run_methods[run_method_name](pipeline['module'], args, log_file_path)
    elif run_method_name == 'pythonpath':
        stdout, stderr = '', ''  # can delete only once
    elif run_method_name == 'cli':
        stdout, stderr = '', ''  # can delete only once
    else:
        pytest.fail(f"Unknown run method: {run_method_name}")

    out_str = stdout.decode().lower() if stdout else ""
    err_str = stderr.decode().lower() if stderr else ""
    print(f"Completed: {test_name}, {pipeline['name']}, {run_method_name}: {out_str}")
    assert 'error' not in err_str, f"{pipeline['name']} ({run_method_name}) reported an error in {test_name}: {err_str}"
    assert 'traceback' not in err_str, f"{pipeline['name']} ({run_method_name}) traceback in {test_name} : {err_str}"
    # Check for QoS performance issues
    has_qos_warning, qos_count = check_qos_performance_warning(stdout, stderr)
    if has_qos_warning:
        logger.warning(f"Performance issue detected: QoS messages: {qos_count} total (>=100) for {pipeline['name']} ({run_method_name}) {test_name}")


def run_hailo8l_models_on_hailo8_face_recon(model_names, extra_args=None):
    """Helper to run Hailo8L models on Hailo 8 for face recognition pipeline.

    Face recognition requires multiple models (detector + embedder), so all
    model HEF paths are passed together.

    Args:
        model_names: List of model names to pass together
        extra_args: Additional arguments to pass to the pipeline

    Returns:
        tuple: (stdout, stderr, success)
    """
    hailo_arch = detect_hailo_arch()
    if hailo_arch != HAILO8_ARCH:
        logger.warning(f"Not running on Hailo 8 architecture (current: {hailo_arch})")
        return b"", b"", False

    # Create logs directory
    h8l_log_dir = "logs/h8l_on_h8_face_recon_tests"
    os.makedirs(h8l_log_dir, exist_ok=True)

    # Build HEF paths for all models and pass them together
    args = []
    for model_name in model_names:
        hef_full_path = os.path.join(RESOURCES_ROOT_PATH_DEFAULT, "models", HAILO8L_ARCH, f"{model_name}.hef")
        args.extend(["--hef-path", hef_full_path])
    if extra_args:
        args.extend(extra_args)

    model_label = "_".join(model_names)
    log_file_path = os.path.join(h8l_log_dir, f"face_recon_{model_label}.log")

    try:
        logger.info(f"Testing face recognition with Hailo8L models: {model_names} on Hailo 8")
        stdout, stderr = run_pipeline_cli_with_args("hailo-face-recon", args, log_file_path)

        # Check for errors
        err_str = stderr.decode().lower() if stderr else ""
        success = "error" not in err_str and "traceback" not in err_str

        # Check for HailoRT warning (expected for Hailo8L on Hailo8)
        has_warning = check_hailo8l_on_hailo8_warning(stdout, stderr)
        if not has_warning:
            logger.warning(f"Expected HailoRT warning not found for {model_names} on Hailo 8")

        # Check for QoS performance issues
        has_qos_warning, qos_count = check_qos_performance_warning(stdout, stderr)
        if has_qos_warning:
            logger.warning(f"Performance issue detected: QoS messages: {qos_count} total (>=100) for {model_names}")

        return stdout, stderr, success

    except Exception as e:
        logger.error(f"Exception while testing {model_names} on Hailo 8: {e}")
        return b"", str(e).encode(), False


def test_hailo8l_models_on_hailo8_face_recon():
    """Test Hailo8L models on Hailo 8 for face recognition pipeline."""
    hailo_arch = detect_hailo_arch()
    if hailo_arch != HAILO8_ARCH:
        pytest.skip(f"Skipping Hailo-8L model test on {hailo_arch}")

    # Face recognition requires both detector and embedder models together
    h8l_models = ["scrfd_2.5g", "arcface_mobilefacenet"]

    logger.info("Running Hailo8L model test on Hailo 8 for face recognition pipeline")

    stdout, stderr, success = run_hailo8l_models_on_hailo8_face_recon(h8l_models)

    # Check for QoS performance issues
    has_qos_warning, qos_count = check_qos_performance_warning(stdout, stderr)
    if has_qos_warning:
        logger.warning(f"Performance issue detected: QoS messages: {qos_count} total (>=100)")

    assert success, (
        f"Failed Hailo8L models for face recognition: {h8l_models}\n"
        f"Error: {stderr.decode() if stderr else ''}"
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
