"""
Sanity Check Tests - Environment & Runtime Validation

This module provides quick sanity checks to validate that the runtime environment
is properly configured before running any actual pipeline tests.

Tests cover:
- Python environment and packages
- Hailo runtime and device detection
- GStreamer installation and plugins
- Environment configuration and variables

Run with: pytest tests/test_sanity_check.py -v
Run only sanity tests: pytest -m sanity -v
"""

import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ============================================================================
# IMPORTS WITH FALLBACKS
# ============================================================================

try:
    from hailo_apps.python.core.common.defines import (
        DEFAULT_DOTENV_PATH,
        RESOURCES_ROOT_PATH_DEFAULT,
        HAILO8_ARCH,
        HAILO8L_ARCH,
        HAILO10H_ARCH,
    )
    from hailo_apps.python.core.common.core import load_environment
    from hailo_apps.python.core.common.installation_utils import (
        detect_hailo_arch,
        detect_host_arch,
        detect_pkg_installed,
        auto_detect_tappas_installed,
        auto_detect_hailort_python_bindings,
        auto_detect_installed_tappas_python_bindings,
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    # Fallback values
    DEFAULT_DOTENV_PATH = "/usr/local/hailo/resources/.env"
    RESOURCES_ROOT_PATH_DEFAULT = "/usr/local/hailo/resources"
    HAILO8_ARCH = "hailo8"
    HAILO8L_ARCH = "hailo8l"
    HAILO10H_ARCH = "hailo10h"


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sanity-tests")


# ============================================================================
# SECTION 1: HAILO APPS PACKAGE TESTS
# ============================================================================

@pytest.mark.sanity
class TestHailoAppsPackage:
    """Tests for hailo_apps package installation and imports."""
    
    def test_hailo_apps_importable(self):
        """Verify hailo_apps package can be imported."""
        if not IMPORTS_AVAILABLE:
            pytest.fail(f"hailo_apps package cannot be imported: {IMPORT_ERROR}")
        
        try:
            import hailo_apps
            logger.info("hailo_apps package is importable")
        except ImportError as e:
            pytest.fail(f"Failed to import hailo_apps: {e}")
    
    def test_hailo_apps_installed_via_pip(self):
        """Verify hailo-apps is installed via pip."""
        result = subprocess.run(
            ["pip", "list"], 
            check=False, 
            capture_output=True, 
            text=True
        )
        
        if "hailo-apps" in result.stdout:
            logger.info("hailo-apps package is installed via pip")
        else:
            logger.warning(
                "hailo-apps package not found in pip list. "
                "Run 'pip install -e .' to install in development mode."
            )


# ============================================================================
# SECTION 2: PYTHON ENVIRONMENT TESTS
# ============================================================================

@pytest.mark.sanity
class TestPythonEnvironment:
    """Tests for Python environment and required packages."""
    
    def test_python_version(self):
        """Verify Python 3.8+ is installed."""
        assert sys.version_info >= (3, 8), (
            f"Python 3.8 or higher is required. "
            f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
        )
        logger.info(f"Python version: {sys.version}")
    
    @pytest.mark.parametrize("package,import_name", [
        ("gi", "gi"),
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("PyYAML", "yaml"),
        ("python-dotenv", "dotenv"),
    ])
    def test_critical_python_package(self, package, import_name):
        """Verify critical Python packages are installed."""
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            logger.info(f"{package} is installed (version: {version})")
        except ImportError:
            pytest.fail(f"Critical package missing: {package} (import: {import_name})")
    
    @pytest.mark.parametrize("package", [
        "setproctitle",
    ])
    def test_optional_python_package(self, package):
        """Check optional Python packages (warning only, no failure)."""
        try:
            importlib.import_module(package)
            logger.info(f"Optional package {package} is installed")
        except ImportError:
            logger.warning(f"Optional package {package} is not installed")
    
    def test_hailort_python_bindings(self):
        """Verify HailoRT Python bindings are installed."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("hailo_apps not importable, skipping HailoRT check")
        
        if auto_detect_hailort_python_bindings():
            logger.info("HailoRT Python bindings are installed")
        else:
            pytest.fail(
                "HailoRT Python bindings not found. "
                "Install hailort or h10-hailort package."
            )
    
    def test_tappas_python_bindings(self):
        """Verify TAPPAS Python bindings are installed."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("hailo_apps not importable, skipping TAPPAS check")
        
        if auto_detect_installed_tappas_python_bindings():
            logger.info("TAPPAS Python bindings are installed")
        else:
            logger.warning(
                "TAPPAS Python bindings not found. "
                "Some functionality may be limited."
            )


# ============================================================================
# SECTION 3: HAILO RUNTIME TESTS
# ============================================================================

@pytest.mark.sanity
@pytest.mark.requires_device
class TestHailoRuntime:
    """Tests for Hailo runtime and device detection."""
    
    def test_hailort_cli_available(self):
        """Verify hailortcli is in PATH and executable."""
        try:
            result = subprocess.run(
                ["hailortcli", "--version"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"hailortcli version: {result.stdout.strip()}")
        except FileNotFoundError:
            pytest.skip("hailortcli not found - skipping on non-Hailo system")
        except subprocess.CalledProcessError as e:
            pytest.fail(f"hailortcli failed: {e}")
    
    def test_hailo_device_detected(self, detected_hailo_arch):
        """Verify a Hailo device is detected."""
        if detected_hailo_arch is None:
            pytest.skip("No Hailo device detected - skipping device-specific tests")
        
        assert detected_hailo_arch in [HAILO8_ARCH, HAILO8L_ARCH, HAILO10H_ARCH], (
            f"Invalid Hailo architecture detected: {detected_hailo_arch}"
        )
        logger.info(f"Detected Hailo architecture: {detected_hailo_arch}")
    
    def test_hailo_architecture_valid(self, detected_hailo_arch):
        """Verify detected architecture is a known valid type."""
        if detected_hailo_arch is None:
            pytest.skip("No Hailo device detected")
        
        valid_archs = {HAILO8_ARCH, HAILO8L_ARCH, HAILO10H_ARCH}
        assert detected_hailo_arch in valid_archs, (
            f"Unknown architecture: {detected_hailo_arch}. "
            f"Expected one of: {valid_archs}"
        )


# ============================================================================
# SECTION 4: GSTREAMER TESTS
# ============================================================================

@pytest.mark.sanity
@pytest.mark.requires_gstreamer
class TestGStreamer:
    """Tests for GStreamer installation and plugins."""
    
    def test_gstreamer_installed(self):
        """Verify GStreamer 1.0+ is installed."""
        try:
            result = subprocess.run(
                ["gst-inspect-1.0", "--version"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"GStreamer version: {result.stdout.strip()}")
        except FileNotFoundError:
            pytest.fail("GStreamer (gst-inspect-1.0) not found in PATH")
        except subprocess.CalledProcessError as e:
            pytest.fail(f"GStreamer check failed: {e}")
    
    @pytest.mark.parametrize("element", [
        "videotestsrc",
        "appsink",
        "videoconvert",
        "autovideosink",
        "filesrc",
        "decodebin",
    ])
    def test_critical_gst_element(self, element):
        """Verify critical GStreamer elements are available."""
        result = subprocess.run(
            ["gst-inspect-1.0", element],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            pytest.fail(f"Critical GStreamer element missing: {element}")
        logger.info(f"GStreamer element available: {element}")
    
    @pytest.mark.requires_device
    @pytest.mark.parametrize("element", [
        "hailonet",
        "hailofilter",
    ])
    def test_hailo_gst_element(self, element, detected_hailo_arch):
        """Verify Hailo GStreamer elements are available."""
        if detected_hailo_arch is None:
            pytest.skip("No Hailo device detected - skipping Hailo GStreamer checks")
        
        result = subprocess.run(
            ["gst-inspect-1.0", element],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            pytest.fail(
                f"Hailo GStreamer element missing: {element}. "
                f"Ensure TAPPAS is properly installed."
            )
        logger.info(f"Hailo GStreamer element available: {element}")


# ============================================================================
# SECTION 5: ENVIRONMENT CONFIGURATION TESTS
# ============================================================================

@pytest.mark.sanity
class TestEnvironmentConfiguration:
    """Tests for environment configuration and variables."""
    
    def test_dotenv_exists(self, dotenv_path, repo_resources_symlink):
        """Verify .env file exists at expected location."""
        # Check default location first
        if dotenv_path.exists():
            logger.info(f".env file found at: {dotenv_path}")
            return
        
        # Check repo resources symlink location
        repo_dotenv = repo_resources_symlink / ".env"
        if repo_dotenv.exists():
            logger.info(f".env file found at repo location: {repo_dotenv}")
            return
        
        logger.warning(
            f".env file not found at {dotenv_path} or {repo_dotenv}. "
            f"Run 'hailo-post-install' to create it."
        )
    
    def test_host_arch_detection(self, detected_host_arch):
        """Verify host architecture can be detected."""
        assert detected_host_arch != "unknown", "Could not detect host architecture"
        assert detected_host_arch in ["x86", "arm", "rpi"], (
            f"Unknown host architecture: {detected_host_arch}"
        )
        logger.info(f"Detected host architecture: {detected_host_arch}")
    
    def test_tappas_installed_detection(self):
        """Verify TAPPAS core installation can be detected."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("hailo_apps not importable")
        
        tappas_installed = auto_detect_tappas_installed()
        if tappas_installed:
            logger.info("Detected TAPPAS core installation")
        else:
            logger.warning(
                "Could not detect TAPPAS installation. "
                "Ensure hailo-tappas-core is installed."
            )
    
    def test_tappas_postproc_env_var(self):
        """Verify TAPPAS_POST_PROC_DIR or TAPPAS_POSTPROC_PATH is set if TAPPAS installed."""
        if not IMPORTS_AVAILABLE:
            pytest.skip("hailo_apps not importable")
        
        # Check both environment variable names
        postproc_path = os.environ.get("TAPPAS_POST_PROC_DIR") or os.environ.get("TAPPAS_POSTPROC_PATH")
        
        if postproc_path:
            if os.path.isdir(postproc_path):
                logger.info(f"TAPPAS postprocess directory: {postproc_path}")
            else:
                logger.warning(f"TAPPAS postprocess path set but directory doesn't exist: {postproc_path}")
        else:
            # Check if TAPPAS is installed
            tappas_installed = auto_detect_tappas_installed()
            if tappas_installed:
                logger.warning(
                    "TAPPAS core is installed but postprocess path not set. "
                    "Run 'hailo-set-env' or 'hailo-post-install' to configure."
                )


# ============================================================================
# SECTION 6: HOST ARCHITECTURE SPECIFIC TESTS
# ============================================================================

@pytest.mark.sanity
class TestHostArchitectureSpecific:
    """Tests specific to host architecture (x86, ARM, RPi)."""
    
    def test_rpi_camera_module(self, detected_host_arch):
        """On Raspberry Pi: verify picamera2 is available."""
        if detected_host_arch != "rpi":
            pytest.skip("Not running on Raspberry Pi")
        
        try:
            import picamera2
            logger.info("picamera2 is available for RPi camera support")
        except ImportError:
            logger.warning(
                "picamera2 not installed. "
                "RPi camera module support will be unavailable. "
                "Install with: pip install picamera2"
            )
    
    def test_libcamera_available(self, detected_host_arch):
        """On Raspberry Pi: check if libcamera is available."""
        if detected_host_arch != "rpi":
            pytest.skip("Not running on Raspberry Pi")
        
        result = subprocess.run(
            ["which", "libcamera-hello"],
            check=False,
            capture_output=True
        )
        if result.returncode == 0:
            logger.info("libcamera tools available")
        else:
            logger.warning("libcamera tools not found in PATH")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    pytest.main(["-v", __file__, "-m", "sanity"])
