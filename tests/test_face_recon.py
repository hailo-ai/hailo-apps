# region imports
# Standard library imports
import os
import logging

# Third-party imports
import pytest

# Local application-specific imports
from hailo_apps.hailo_app_python.core.common.test_utils import (
    run_pipeline_module_with_args, 
    run_pipeline_pythonpath_with_args, 
    run_pipeline_cli_with_args, 
    get_pipeline_args,
    check_hailo8l_on_hailo8_warning,
    check_qos_performance_warning,
)
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.common.defines import HAILO8_ARCH, HAILO8L_ARCH, RESOURCES_ROOT_PATH_DEFAULT
# endregion imports

# Configure logging as needed.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_run_everything')
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Define pipeline configurations.
@pytest.fixture
def pipeline():
    return {
        'name': 'face_recognition',
        'module': 'hailo_apps.hailo_app_python.apps.face_recognition.face_recognition',
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


def run_hailo8l_model_on_hailo8_face_recon(model_name, extra_args=None):
    """Helper function to run a Hailo8L model on Hailo 8 architecture for face recognition pipeline.
    
    Args:
        model_name: Name of the Hailo8L model to run
        extra_args: Additional arguments to pass to the pipeline
    
    Returns:
        tuple: (stdout, stderr, success)
    """
    hailo_arch = detect_hailo_arch()
    if hailo_arch != HAILO8_ARCH:
        logger.warning(f"Not running on Hailo 8 architecture (current: {hailo_arch})")
        return b"", b"", False

    # Create logs directory
    log_dir = "logs/h8l_on_h8_face_recon_tests"
    os.makedirs(log_dir, exist_ok=True)

    # Build full HEF path for Hailo8L model
    hef_full_path = os.path.join(RESOURCES_ROOT_PATH_DEFAULT, "models", HAILO8L_ARCH, f"{model_name}.hef")
    
    # Prepare CLI arguments
    args = ["--hef-path", hef_full_path]
    if extra_args:
        args.extend(extra_args)

    # Create log file path
    log_file_path = os.path.join(log_dir, f"face_recon_{model_name}.log")

    try:
        logger.info(f"Testing face recognition with Hailo8L model: {model_name} on Hailo 8")
        stdout, stderr = run_pipeline_cli_with_args("hailo-face-recon", args, log_file_path)

        # Check for errors
        err_str = stderr.decode().lower() if stderr else ""
        success = "error" not in err_str and "traceback" not in err_str
        
        # Check for HailoRT warning (expected for Hailo8L on Hailo8)
        has_warning = check_hailo8l_on_hailo8_warning(stdout, stderr)
        if not has_warning:
            logger.warning(f"Expected HailoRT warning not found for {model_name} on Hailo 8")
        
        # Check for QoS performance issues
        has_qos_warning, qos_count = check_qos_performance_warning(stdout, stderr)
        if has_qos_warning:
            logger.warning(f"Performance issue detected: QoS messages: {qos_count} total (>=100) for {model_name}")
        
        return stdout, stderr, success

    except Exception as e:
        logger.error(f"Exception while testing {model_name} on Hailo 8: {e}")
        return b"", str(e).encode(), False


def test_hailo8l_models_on_hailo8_face_recon():
    """Test Hailo8L models on Hailo 8 for face recognition pipeline."""
    hailo_arch = detect_hailo_arch()
    if hailo_arch != HAILO8_ARCH:
        pytest.skip(f"Skipping Hailo-8L model test on {hailo_arch}")

    # Define Hailo8L models that can be used with face recognition
    h8l_models = ["scrfd_2.5g", "arcface_mobilefacenet_h8l"]
    
    logger.info(f"Running Hailo8L model test on Hailo 8 for face recognition pipeline")
    
    failed_models = []
    
    for model in h8l_models:
        stdout, stderr, success = run_hailo8l_model_on_hailo8_face_recon(model)
        
        # Check for QoS performance issues
        has_qos_warning, qos_count = check_qos_performance_warning(stdout, stderr)
        if has_qos_warning:
            logger.warning(f"Performance issue detected: QoS messages: {qos_count} total (>=100) for {model}")
        
        if not success:
            failed_models.append({
                "model": model,
                "stderr": stderr.decode() if stderr else "",
                "stdout": stdout.decode() if stdout else "",
            })
            logger.error(f"Failed to run {model} with face recognition")
        else:
            logger.info(f"Successfully ran {model} with face recognition")

    # Assert that all models passed
    if failed_models:
        failure_details = "\n".join(
            [f"Model: {fail['model']}\nError: {fail['stderr']}\n" for fail in failed_models]
        )
        pytest.fail(f"Failed Hailo8L models for face recognition:\n{failure_details}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])