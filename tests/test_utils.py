"""
Test Utilities for Pipeline Execution

This module provides utilities for creating and running pipeline tests
based on configuration.
"""

import logging
import os
import signal
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

import pytest

from hailo_apps.python.core.common.defines import (
    HAILO_ARCH_KEY,
    RESOURCES_ROOT_PATH_DEFAULT,
    TERM_TIMEOUT,
    TEST_RUN_TIME,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Runner Functions
# =============================================================================

def run_pipeline_generic(
    cmd: list[str], log_file: str, run_time: int = TEST_RUN_TIME, term_timeout: int = TERM_TIMEOUT
):
    """Run a command, terminate after run_time, capture logs.

    Features early failure detection - if process exits before run_time,
    we detect it immediately rather than waiting the full duration.

    Reads stdout/stderr in background threads to prevent pipe buffer overflow
    that would cause the process to block on write operations.
    """
    # Thread-safe buffers for collecting output
    stdout_buffer = []
    stderr_buffer = []
    stdout_lock = threading.Lock()
    stderr_lock = threading.Lock()

    def read_stdout(pipe):
        """Read stdout in background thread."""
        try:
            for line in iter(pipe.readline, b''):
                if line:
                    with stdout_lock:
                        stdout_buffer.append(line)
        except Exception as e:
            logger.debug(f"Error reading stdout: {e}")
        finally:
            pipe.close()

    def read_stderr(pipe):
        """Read stderr in background thread."""
        try:
            for line in iter(pipe.readline, b''):
                if line:
                    with stderr_lock:
                        stderr_buffer.append(line)
        except Exception as e:
            logger.debug(f"Error reading stderr: {e}")
        finally:
            pipe.close()

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,  # Line buffered
        )

        # Start background threads to read stdout/stderr
        stdout_thread = threading.Thread(target=read_stdout, args=(proc.stdout,), daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, args=(proc.stderr,), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        # Poll for early exit while waiting for run_time
        # Check every 0.5 seconds if process is still running
        poll_interval = 0.5
        elapsed = 0.0
        early_exit = False

        while elapsed < run_time:
            time.sleep(poll_interval)
            elapsed += poll_interval

            # Check if process exited early (indicates crash/error)
            if proc.poll() is not None:
                early_exit = True
                logger.warning(f"Process exited early after {elapsed:.1f}s (return code: {proc.returncode})")
                break

        # If process still running after run_time, terminate gracefully
        if not early_exit:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=term_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                pytest.fail(f"Command didn't terminate: {' '.join(cmd)}")

        # Wait for reader threads to finish (with timeout)
        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)

        # Collect all output
        with stdout_lock:
            out = b''.join(stdout_buffer)
        with stderr_lock:
            err = b''.join(stderr_buffer)

        # Write to log file
        f.write("stdout:\n" + out.decode(errors='replace') + "\n")
        f.write("stderr:\n" + err.decode(errors='replace') + "\n")

        # Log early exit information
        if early_exit:
            f.write(f"\n[EARLY EXIT] Process exited after {elapsed:.1f}s with code {proc.returncode}\n")

        return out, err


def run_pipeline_module_with_args(module: str, args: list[str], log_file: str, **kwargs):
    """Run a pipeline as a Python module."""
    return run_pipeline_generic(["python", "-u", "-m", module, *args], log_file, **kwargs)


def run_pipeline_pythonpath_with_args(script: str, args: list[str], log_file: str, **kwargs):
    """Run a pipeline script using the current environment (setup_env sets PYTHONPATH)."""
    return run_pipeline_generic(["python3", "-u", script, *args], log_file, **kwargs)


def run_pipeline_cli_with_args(cli: str, args: list[str], log_file: str, **kwargs):
    """Run a pipeline via CLI entry point."""
    return run_pipeline_generic([cli, *args], log_file, **kwargs)


# Map run method names to functions
RUN_METHOD_FUNCTIONS = {
    "module": run_pipeline_module_with_args,
    "pythonpath": run_pipeline_pythonpath_with_args,
    "cli": run_pipeline_cli_with_args,
}


def build_hef_path(model: str, architecture: str, resources_root: Optional[str] = None) -> str:
    """Build full path to HEF file.

    Args:
        model: Model name (without .hef extension)
        architecture: Architecture (hailo8, hailo8l, hailo10h)
        resources_root: Resources root path (defaults to RESOURCES_ROOT_PATH_DEFAULT)

    Returns:
        Full path to HEF file
    """
    if resources_root is None:
        resources_root = RESOURCES_ROOT_PATH_DEFAULT

    hef_file = f"{model}.hef"
    return os.path.join(resources_root, "models", architecture, hef_file)


def build_test_args(
    config: Dict,
    pipeline_config: Dict,
    model: str,
    architecture: str,
    test_suite: str = "default",
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Build command-line arguments for a test.

    Args:
        config: Full test configuration
        pipeline_config: Pipeline-specific configuration
        model: Model name
        architecture: Architecture name
        test_suite: Test suite name (default: "default")
        extra_args: Additional arguments to append

    Returns:
        List of command-line arguments
    """
    args = []

    # Add HEF path
    hef_path = build_hef_path(model, architecture)
    args.extend(["--hef-path", hef_path])

    # Add test suite arguments
    test_suites = config.get("test_suites", {})
    suite_config = test_suites.get(test_suite, {})
    suite_args = suite_config.get("args", [])

    # Replace placeholders in suite args
    resources_root = config.get("resources", {}).get("root_path", RESOURCES_ROOT_PATH_DEFAULT)
    suite_args = [
        arg.replace("${HEF_PATH}", hef_path)
        .replace("${RESOURCES_ROOT}", resources_root)
        for arg in suite_args
    ]

    args.extend(suite_args)

    # Add extra arguments if provided
    if extra_args:
        args.extend(extra_args)

    return args


def run_pipeline_test(
    pipeline_config: Dict,
    model: str,
    architecture: str,
    run_method: str,
    args: List[str],
    log_file: str,
    run_time: Optional[int] = None,
    term_timeout: Optional[int] = None,
) -> Tuple[bytes, bytes, bool]:
    """Run a pipeline test and return results.

    Args:
        pipeline_config: Pipeline configuration
        model: Model name
        architecture: Architecture name
        run_method: Run method ("module", "pythonpath", or "cli")
        args: Command-line arguments
        log_file: Path to log file
        run_time: Optional run time override
        term_timeout: Optional termination timeout override

    Returns:
        Tuple of (stdout, stderr, success)
    """
    run_func = RUN_METHOD_FUNCTIONS.get(run_method)
    if not run_func:
        logger.error(f"Unknown run method: {run_method}")
        return b"", b"Unknown run method".encode(), False

    # Set hailo_arch environment variable for this test
    # This ensures the subprocess uses the correct architecture (important for h8l_on_h8 tests)
    # Note: Must use HAILO_ARCH_KEY ("hailo_arch") to match what apps read via os.getenv(HAILO_ARCH_KEY)
    original_hailo_arch = os.environ.get(HAILO_ARCH_KEY)
    os.environ[HAILO_ARCH_KEY] = architecture
    logger.debug(f"Set {HAILO_ARCH_KEY}={architecture} for test")

    try:
        kwargs = {}
        if run_time is not None:
            kwargs["run_time"] = run_time
        if term_timeout is not None:
            kwargs["term_timeout"] = term_timeout

        if run_method == "module":
            stdout, stderr = run_func(pipeline_config["module"], args, log_file, **kwargs)
        elif run_method == "pythonpath":
            stdout, stderr = run_func(pipeline_config["script"], args, log_file, **kwargs)
        elif run_method == "cli":
            stdout, stderr = run_func(pipeline_config["cli"], args, log_file, **kwargs)
        else:
            return b"", b"Invalid run method".encode(), False

        # Check for errors
        err_str = stderr.decode().lower() if stderr else ""
        out_str = stdout.decode().lower() if stdout else ""
        combined_output = (err_str + " " + out_str).lower()

        success = "error" not in err_str and "traceback" not in err_str

        # Check for FPS output when --show-fps is enabled
        # The actual log format is "FPS measurement: X.XX" (case-sensitive check on lowercase output)
        if "--show-fps" in args or "-f" in args:
            if "fps measurement:" not in combined_output:
                logger.warning(f"FPS flag enabled but FPS output not found in logs: {log_file}")
                # Don't fail the test, just warn - FPS might not appear immediately

        # Check for QOS messages only in stderr (error logs)
        # QOS handling inside the pipeline is normal and expected - we only care about QOS errors
        if "qos" in err_str:
            logger.warning(f"QOS error messages detected in stderr")

        # but the pipeline still runs normally. Both enabled and disabled states result
        # in successful pipeline execution, so we cannot distinguish them from output.
        if "--disable-callback" in args:
            logger.debug("Testing with callback disabled - Python callback will not be invoked")
        else:
            logger.debug("Testing with callback enabled - Python callback will be invoked per frame")

        return stdout, stderr, success

    except Exception as e:
        logger.error(f"Exception running pipeline test: {e}")
        return b"", str(e).encode(), False

    finally:
        # Restore original hailo_arch environment variable
        if original_hailo_arch is not None:
            os.environ[HAILO_ARCH_KEY] = original_hailo_arch
        elif HAILO_ARCH_KEY in os.environ:
            del os.environ[HAILO_ARCH_KEY]
        logger.debug(f"Restored {HAILO_ARCH_KEY} to {original_hailo_arch}")


def get_log_file_path(
    config: Dict,
    test_type: str,
    pipeline_name: str,
    architecture: Optional[str] = None,
    model: Optional[str] = None,
    run_method: Optional[str] = None,
    test_suite: Optional[str] = None,
) -> str:
    """Get log file path for a test.

    Args:
        config: Test configuration
        test_type: Type of test (e.g., "pipeline", "h8l_on_h8", "human_verification")
        pipeline_name: Pipeline name
        architecture: Optional architecture name
        model: Optional model name
        run_method: Optional run method name
        test_suite: Optional test suite name

    Returns:
        Full path to log file
    """
    log_config = config.get("logging", {})
    base_dir = log_config.get("base_dir", "logs")
    subdirs = log_config.get("subdirectories", {})

    # Get appropriate subdirectory
    if test_type in subdirs:
        log_dir = subdirs[test_type]
    else:
        log_dir = base_dir

    # Ensure directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Build filename
    parts = [pipeline_name]
    if architecture:
        parts.append(architecture)
    if model:
        parts.append(model)
    if run_method:
        parts.append(run_method)
    if test_suite and test_suite != "default":
        parts.append(test_suite)

    filename = "_".join(parts) + ".log"
    return os.path.join(log_dir, filename)

