"""Installation-related utilities.

This module provides utilities for detecting system configuration, package versions,
and managing installation-related tasks for Hailo applications.

Function Organization:
    - Internal Helpers: Low-level utilities used only within this module
    - Core Detection: Architecture and package detection (used by multiple modules)
    - Test Utilities: Functions primarily used by test files
    - set_env Utilities: Functions used by set_env.py for environment configuration
    - post_install Utilities: Functions used by post_install.py
"""

import platform
import shlex
import subprocess

from .defines import (
    ARM_NAME_I,
    ARM_POSSIBLE_NAME,
    HAILO8_ARCH,
    HAILO8_ARCH_CAPS,
    HAILO8L_ARCH,
    HAILO8L_ARCH_CAPS,
    HAILO10H_ARCH,
    HAILO10H_ARCH_CAPS,
    HAILO15H_ARCH_CAPS,
    HAILO_FW_CONTROL_CMD,
    HAILO_TAPPAS_CORE,
    HAILO_TAPPAS_CORE_PYTHON_NAMES,
    HAILORT_PACKAGE_NAME,
    HAILORT_PACKAGE_NAME_RPI,
    LINUX_SYSTEM_NAME_I,
    PIP_CMD,
    RPI_NAME_I,
    RPI_POSSIBLE_NAME,
    UNKNOWN_NAME_I,
    X86_NAME_I,
    X86_POSSIBLE_NAME,
)
from .hailo_logger import get_logger

hailo_logger = get_logger(__name__)


# =============================================================================
# Internal Helper Functions
# Used only within this module by other functions
# =============================================================================


def _detect_pkg_config_version(pkg_name: str) -> str:
    """Internal: Get package version from pkg-config."""
    hailo_logger.debug(f"Detecting pkg-config version for: {pkg_name}")
    try:
        version = subprocess.check_output(
            ["pkg-config", "--modversion", pkg_name], stderr=subprocess.DEVNULL, text=True
        )
        version = version.strip()
        hailo_logger.debug(f"Found version {version} for package {pkg_name}")
        return version
    except subprocess.CalledProcessError:
        hailo_logger.warning(f"Package {pkg_name} not found in pkg-config.")
        return ""


def _auto_detect_pkg_config(pkg_name: str) -> bool:
    """Internal: Check if a package exists in pkg-config."""
    hailo_logger.debug(f"Checking if {pkg_name} exists in pkg-config.")
    try:
        subprocess.check_output(
            ["pkg-config", "--exists", pkg_name], stderr=subprocess.DEVNULL, text=True
        )
        hailo_logger.debug(f"Package {pkg_name} exists in pkg-config.")
        return True
    except subprocess.CalledProcessError:
        hailo_logger.debug(f"Package {pkg_name} does not exist in pkg-config.")
        return False


def _detect_pip_package_installed(pkg: str) -> bool:
    """Internal: Check if a pip package is installed."""
    hailo_logger.debug(f"Checking if pip package is installed: {pkg}")
    try:
        result = subprocess.run(
            [PIP_CMD, "show", pkg],
            check=False,
            capture_output=True,
            text=True,
        )
        installed = result.returncode == 0
        hailo_logger.debug(f"Pip package {pkg} installed: {installed}")
        return installed
    except Exception as e:
        hailo_logger.exception(f"Error checking pip package {pkg}: {e}")
        return False


def _run_command_with_output(cmd: list[str]) -> str:
    """Internal: Run a command and return its stdout."""
    hailo_logger.debug(f"Running command with output: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        hailo_logger.error(f"Command failed: {' '.join(cmd)}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout


# =============================================================================
# Core Detection Functions
# Used by: set_env.py, tests, gstreamer_app.py, download_resources.py,
#          reid_multisource_pipeline.py, core.py, conftest.py, test_runner.py
# =============================================================================


def detect_host_arch() -> str:
    """Detect the host system architecture.

    Returns:
        str: One of 'x86', 'rpi', 'arm', or 'unknown'
    """
    hailo_logger.debug("Detecting host architecture.")
    machine_name = platform.machine().lower()
    system_name = platform.system().lower()
    hailo_logger.debug(f"Machine: {machine_name}, System: {system_name}")

    if machine_name in X86_POSSIBLE_NAME:
        hailo_logger.info("Detected host architecture: x86")
        return X86_NAME_I
    if machine_name in ARM_POSSIBLE_NAME:
        if system_name == LINUX_SYSTEM_NAME_I and platform.uname().node in RPI_POSSIBLE_NAME:
            hailo_logger.info("Detected host architecture: Raspberry Pi")
            return RPI_NAME_I
        hailo_logger.info("Detected host architecture: ARM")
        return ARM_NAME_I
    hailo_logger.warning("Unknown host architecture.")
    return UNKNOWN_NAME_I


def detect_hailo_arch() -> str | None:
    """Detect the connected Hailo device architecture.

    Returns:
        str | None: One of 'hailo8', 'hailo8l', 'hailo10h', or None if detection fails
    """
    hailo_logger.debug("Detecting Hailo architecture using hailortcli.")
    try:
        args = shlex.split(HAILO_FW_CONTROL_CMD)
        res = subprocess.run(args, check=False, capture_output=True, text=True)
        if res.returncode != 0:
            hailo_logger.error(f"hailortcli failed with code {res.returncode}")
            return None
        for line in res.stdout.splitlines():
            if HAILO8L_ARCH_CAPS in line:
                hailo_logger.debug("Detected Hailo architecture: HAILO8L")
                return HAILO8L_ARCH
            if HAILO8_ARCH_CAPS in line:
                hailo_logger.debug("Detected Hailo architecture: HAILO8")
                return HAILO8_ARCH
            if HAILO10H_ARCH_CAPS in line or HAILO15H_ARCH_CAPS in line:
                hailo_logger.debug("Detected Hailo architecture: HAILO10H")
                return HAILO10H_ARCH
    except Exception as e:
        hailo_logger.exception(f"Error detecting Hailo architecture: {e}")
        assert False, "Error detecting Hailo architecture. Is Hailo Installed?"
    hailo_logger.warning("Could not determine Hailo architecture.")
    assert False, "Could not determine Hailo architecture. Is Hailo connected?"


def detect_system_pkg_version(pkg_name: str) -> str:
    """Detect the version of a system package (dpkg).

    Args:
        pkg_name: Name of the system package

    Returns:
        str: Version string or empty string if not found
    """
    hailo_logger.debug(f"Detecting system package version for: {pkg_name}")
    try:
        version = subprocess.check_output(
            ["dpkg-query", "-W", "-f=${Version}", pkg_name], stderr=subprocess.DEVNULL, text=True
        )
        version = version.strip()
        hailo_logger.debug(f"Found version {version} for system package {pkg_name}")
        return version
    except subprocess.CalledProcessError:
        hailo_logger.warning(f"System package {pkg_name} is not installed.")
        return ""


def detect_pkg_installed(pkg_name: str) -> bool:
    """Check if a system package is installed.

    Args:
        pkg_name: Name of the system package

    Returns:
        bool: True if installed, False otherwise
    """
    hailo_logger.debug(f"Checking if system package is installed: {pkg_name}")
    try:
        subprocess.check_output(["dpkg", "-s", pkg_name])
        hailo_logger.debug(f"Package {pkg_name} is installed.")
        return True
    except subprocess.CalledProcessError:
        hailo_logger.debug(f"Package {pkg_name} is not installed.")
        return False


def get_hailort_package_name() -> str:
    """Get the appropriate HailoRT package name based on host architecture.

    Returns:
        str: Package name ('hailort' or 'h10-hailort' for RPI)
    """
    host_arch = detect_host_arch()
    if host_arch == RPI_NAME_I:
        hailo_logger.debug(f"Using RPI-specific HailoRT package: {HAILORT_PACKAGE_NAME_RPI}")
        return HAILORT_PACKAGE_NAME_RPI
    hailo_logger.debug(f"Using default HailoRT package: {HAILORT_PACKAGE_NAME}")
    return HAILORT_PACKAGE_NAME


# =============================================================================
# Test Utility Functions
# Used by: test_sanity_check.py
# =============================================================================


def auto_detect_hailort_python_bindings() -> bool:
    """Check if HailoRT Python bindings are installed.

    Returns:
        bool: True if bindings are installed, False otherwise
    """
    hailo_logger.debug("Detecting HailoRT Python bindings.")
    pkg_name = get_hailort_package_name()
    if _detect_pip_package_installed(pkg_name):
        hailo_logger.info("Detected HailoRT Python bindings installed.")
        return True
    hailo_logger.warning("HailoRT Python bindings not found.")
    return False


def auto_detect_hailort_version() -> str:
    """Detect the installed HailoRT version.

    Returns:
        str: Version string or None if not detected
    """
    hailo_logger.debug("Detecting installed HailoRT version.")
    pkg_name = get_hailort_package_name()
    if detect_pkg_installed(pkg_name):
        return detect_system_pkg_version(pkg_name)
    else:
        hailo_logger.warning("Could not detect HailoRT version, please install HailoRT.")
        return None


def auto_detect_tappas_installed() -> bool:
    """Check if hailo-tappas-core is installed.

    Returns:
        bool: True if TAPPAS core is installed, False otherwise
    """
    hailo_logger.debug("Checking if TAPPAS core is installed.")
    if (
        detect_pkg_installed(HAILO_TAPPAS_CORE)
        or _auto_detect_pkg_config(HAILO_TAPPAS_CORE)
        or _auto_detect_pkg_config("hailo-all")
    ):
        hailo_logger.info("Detected TAPPAS core installation")
        return True
    else:
        hailo_logger.warning("TAPPAS core not detected.")
        return False


def auto_detect_installed_tappas_python_bindings() -> bool:
    """Check if TAPPAS core Python bindings are installed.

    Returns:
        bool: True if bindings are installed, False otherwise
    """
    hailo_logger.debug("Detecting installed TAPPAS Python bindings.")
    for pkg in HAILO_TAPPAS_CORE_PYTHON_NAMES:
        if _detect_pip_package_installed(pkg):
            hailo_logger.info(f"Detected {pkg} Python bindings.")
            return True
    hailo_logger.warning("Could not detect TAPPAS Python bindings.")
    return False


# =============================================================================
# set_env.py Utility Functions
# Used by: set_env.py for environment configuration
# =============================================================================


def auto_detect_tappas_version() -> str:
    """Detect TAPPAS core version.

    Returns:
        str: Version string or None if not detected
    """
    hailo_logger.debug("Detecting TAPPAS core version")
    version = _detect_pkg_config_version(HAILO_TAPPAS_CORE)
    if version:
        return version
    hailo_logger.warning("Could not detect TAPPAS version.")
    return None


def auto_detect_tappas_postproc_dir() -> str:
    """Detect TAPPAS post-processing directory.

    Returns:
        str: Path to the postproc directory or empty string if not found
    """
    hailo_logger.debug("Detecting TAPPAS post-processing directory")
    try:
        return _run_command_with_output(
            ["pkg-config", "--variable=tappas_postproc_lib_dir", HAILO_TAPPAS_CORE]
        )
    except Exception as e:
        hailo_logger.error(f"Could not detect TAPPAS postproc directory: {e}")
        return ""


