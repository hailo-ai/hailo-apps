"""Pipeline test utilities."""

import os
import signal
import subprocess
import time

import pytest
from pathlib import Path

from .defines import TERM_TIMEOUT, TEST_RUN_TIME, RESOURCES_ROOT_PATH_DEFAULT, RESOURCES_VIDEOS_DIR_NAME, BASIC_PIPELINES_VIDEO_EXAMPLE_NAME


def get_pipeline_args(
    suite="default",
    hef_path=None,
    override_usb_camera=None,
    override_video_input=None,
    override_labels_json=None,
):
    """Returns a list of additional arguments based on the specified test suite.

    Supported suites (comma separated):
      - "usb_camera": Set the '--input' argument to the USB camera device
                     determined by get_usb_video_devices().
      - "rpi_camera": Set the '--input' argument to "rpi".
      - "hef_path":   Set the '--hef-path' argument to the user-specified HEF path
                     using the USER_HEF environment variable (or a fallback value).
      - "video_file": Set the '--input' argument to a video file ("resources/example.mp4").
      - "disable_sync": Append the flag "--disable-sync".
      - "disable_callback": Append the flag "--disable-callback".
      - "show_fps": Append the flag "--show-fps".
      - "dump_dot": Append the flag "--dump-dot".
      - "labels": Append the flag "--labels-json" followed by "resources/labels.json".
      - "mode-train": Set the '--mode' argument to train.
      - "mode-delete": Set the '--mode' argument to delete.
      - "mode-run": Set the '--mode' argument to run.

    If suite is "default", returns an empty list (i.e. no extra test arguments).
    """
    # Start with no extra arguments.
    args = []
    if suite == "default":
        return args

    suite_names = [s.strip() for s in suite.split(",")]
    for s in suite_names:
        if s == "usb_camera":
            # If override_usb_camera is provided, use it; otherwise, get the USB camera device.
            if override_usb_camera:
                device = override_usb_camera
            else:
                device = "usb"
            # Append or override --input (here we simply add the argument)
            args += ["--input", device]
        elif s == "rpi_camera":
            args += ["--input", "rpi"]
        elif s == "hef_path":
            hef = hef_path
            args += ["--hef-path", hef]
        elif s == "video_file":
            # If override_video_input is provided, use it; otherwise, use the default video file.
            if override_video_input:
                video_file = override_video_input
            else:
                video_file = "resources/example.mp4"
            # Append or override --input (here we simply add the argument)
            args += ["--input", video_file]
        elif s == "disable_sync":
            args.append("--disable-sync")
        elif s == "disable_callback":
            args.append("--disable-callback")
        elif s == "show_fps":
            args.append("--show-fps")
        elif s == "dump_dot":
            args.append("--dump-dot")
        elif s == "labels":
            # If override_labels_json is provided, use it; otherwise, use the default json file.
            if override_labels_json:
                json_file = override_labels_json
            else:
                json_file = "resources/labels.json"
            # Append or override --input (here we simply add the argument)
            args += ["--labels-json", json_file]
        elif s == "mode-train":
            args += ["--mode", "train"]
        elif s == "mode-delete":
            args += ["--mode", "delete"]
        elif s == "mode-run":
            args += ["--mode", "run"]
        elif s == "single_scaling":  # for tiling pipeline
            args += ["--single_scaling"]
        elif s == "sources":  # for multisource pipeline
            args += ["--sources", f"/dev/video0,{str(Path(RESOURCES_ROOT_PATH_DEFAULT) / RESOURCES_VIDEOS_DIR_NAME / BASIC_PIPELINES_VIDEO_EXAMPLE_NAME)}"]
    return args


def run_pipeline_generic(
    cmd: list[str], log_file: str, run_time: int = TEST_RUN_TIME, term_timeout: int = TERM_TIMEOUT
):
    """Run a command, terminate after run_time, capture logs."""
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(run_time)
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=term_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail(f"Command didn't terminate: {' '.join(cmd)}")
        out, err = proc.communicate()
        f.write("stdout:\n" + out.decode() + "\n")
        f.write("stderr:\n" + err.decode() + "\n")
        return out, err


def run_pipeline_module_with_args(module: str, args: list[str], log_file: str, **kwargs):
    return run_pipeline_generic(["python", "-u", "-m", module, *args], log_file, **kwargs)


def run_pipeline_pythonpath_with_args(script: str, args: list[str], log_file: str, **kwargs):
    env = os.environ.copy()
    env["PYTHONPATH"] = "./hailo-apps"
    return run_pipeline_generic(["python", "-u", script, *args], log_file, **kwargs)


def run_pipeline_cli_with_args(cli: str, args: list[str], log_file: str, **kwargs):
    return run_pipeline_generic([cli, *args], log_file, **kwargs)


def safe_decode(data: bytes, errors: str = 'replace') -> str:
    """Safely decode bytes to string, handling encoding errors gracefully.
    
    Args:
        data: Bytes to decode
        errors: Error handling strategy ('replace', 'ignore', or 'strict')
    
    Returns:
        Decoded string, or empty string if decoding fails
    """
    if not data:
        return ""
    try:
        return data.decode(errors=errors)
    except Exception:
        # Fallback to ignore if replace fails
        try:
            return data.decode(errors='ignore')
        except Exception:
            return ""


def check_hailo8l_on_hailo8_warning(stdout: bytes, stderr: bytes) -> bool:
    """Check if the HailoRT warning about Hailo8L HEF on Hailo8 device is present.
    
    Args:
        stdout: Standard output from the pipeline
        stderr: Standard error from the pipeline
    
    Returns:
        bool: True if the warning is found, False otherwise
    """
    warning_pattern = "HEF was compiled for Hailo8L device, while the device itself is Hailo8"
    try:
        output = (stdout.decode(errors='replace') if stdout else "") + (stderr.decode(errors='replace') if stderr else "")
        return warning_pattern in output
    except Exception:
        # If decoding fails, try with ignore errors
        try:
            output = (stdout.decode(errors='ignore') if stdout else "") + (stderr.decode(errors='ignore') if stderr else "")
            return warning_pattern in output
        except Exception:
            return False


def check_qos_performance_warning(stdout: bytes, stderr: bytes) -> tuple[bool, int]:
    """Check for QoS messages indicating performance issues.
    
    Args:
        stdout: Standard output from the pipeline
        stderr: Standard error from the pipeline
    
    Returns:
        tuple: (has_warning, qos_count) where has_warning is True if QoS >= 100, 
               and qos_count is the number of QoS messages found
    """
    import re
    try:
        output = (stdout.decode(errors='replace') if stdout else "") + (stderr.decode(errors='replace') if stderr else "")
    except Exception:
        # If decoding fails, try with ignore errors
        try:
            output = (stdout.decode(errors='ignore') if stdout else "") + (stderr.decode(errors='ignore') if stderr else "")
        except Exception:
            return False, 0
    
    # Look for "QoS messages: X total" pattern
    pattern = r"QoS messages:\s*(\d+)\s+total"
    matches = re.findall(pattern, output)
    
    if matches:
        # Get the highest count found (in case there are multiple)
        qos_count = max(int(match) for match in matches)
        has_warning = qos_count >= 100
        return has_warning, qos_count
    
    return False, 0
