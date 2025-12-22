"""
Pytest Configuration and Shared Fixtures for Hailo Apps Tests.

This module provides:
- Shared fixtures for resources, models, videos, etc.
- Test markers for categorization

All configuration loading is handled by the unified config_manager.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ============================================================================
# IMPORTS - Using Unified Config Manager
# ============================================================================

try:
    from hailo_apps.config import config_manager
    from hailo_apps.config.config_manager import ConfigPaths
    from hailo_apps.python.core.common.defines import (
        HAILO8_ARCH,
        HAILO8L_ARCH,
        HAILO10H_ARCH,
        RESOURCES_ROOT_PATH_DEFAULT,
        RESOURCES_MODELS_DIR_NAME,
        RESOURCES_VIDEOS_DIR_NAME,
        RESOURCES_SO_DIR_NAME,
        RESOURCES_JSON_DIR_NAME,
        RESOURCES_PHOTOS_DIR_NAME,
        DEFAULT_DOTENV_PATH,
    )
    from hailo_apps.python.core.common.installation_utils import (
        detect_hailo_arch,
        detect_host_arch,
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    # Fallback defaults if imports fail
    config_manager = None
    ConfigPaths = None
    HAILO8_ARCH = "hailo8"
    HAILO8L_ARCH = "hailo8l"
    HAILO10H_ARCH = "hailo10h"
    RESOURCES_ROOT_PATH_DEFAULT = "/usr/local/hailo/resources"
    RESOURCES_MODELS_DIR_NAME = "models"
    RESOURCES_VIDEOS_DIR_NAME = "videos"
    RESOURCES_SO_DIR_NAME = "so"
    RESOURCES_JSON_DIR_NAME = "json"
    RESOURCES_PHOTOS_DIR_NAME = "photos"
    DEFAULT_DOTENV_PATH = "/usr/local/hailo/resources/.env"
    detect_hailo_arch = None
    detect_host_arch = None
    IMPORTS_AVAILABLE = False


# ============================================================================
# CONFIGURATION PATHS (using unified ConfigPaths when available)
# ============================================================================

CONFIG_DIR = REPO_ROOT / "hailo_apps" / "config"
RESOURCES_CONFIG_PATH = CONFIG_DIR / "resources_config.yaml"
POSTPROCESS_MESON_PATH = REPO_ROOT / "hailo_apps" / "postprocess" / "cpp" / "meson.build"


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "sanity: Quick environment sanity checks")
    config.addinivalue_line("markers", "installation: Installation validation tests")
    config.addinivalue_line("markers", "resources: Resource file validation tests")
    config.addinivalue_line("markers", "requires_device: Tests requiring a Hailo device")
    config.addinivalue_line("markers", "requires_gstreamer: Tests requiring GStreamer")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_meson_shared_libraries() -> List[str]:
    """Parse meson.build to extract shared library names.

    Returns:
        List of shared library filenames (e.g., ['libyolo_hailortpp_postprocess.so', ...])
    """
    if not POSTPROCESS_MESON_PATH.exists():
        return []

    with open(POSTPROCESS_MESON_PATH, "r") as f:
        content = f.read()

    # Match shared_library('name', ...) pattern
    pattern = r"shared_library\s*\(\s*['\"]([^'\"]+)['\"]"
    matches = re.findall(pattern, content)

    # Convert to .so filename format: libNAME.so
    return [f"lib{name}.so" for name in matches]


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def resources_config() -> Dict:
    """Fixture providing parsed resources configuration.
    
    Returns a dictionary with 'videos', 'images', 'apps' structure
    for backward compatibility with existing tests.
    """
    if config_manager is None:
        return {"videos": [], "images": [], "apps": {}}
    
    result = {
        "videos": config_manager.get_videos(),
        "images": config_manager.get_images(),
        "apps": {},
    }
    
    # Build apps structure for backward compatibility
    for app_name in config_manager.get_available_apps():
        app_info = {
            "models": {},
            "json": config_manager.get_json_files(app_name),
        }
        
        # Get models per architecture
        for arch in [HAILO8_ARCH, HAILO8L_ARCH, HAILO10H_ARCH]:
            default_models = config_manager.get_model_names(app_name, arch, tier="default")
            extra_models = config_manager.get_model_names(app_name, arch, tier="extra")
            
            if default_models or extra_models:
                app_info["models"][arch] = {
                    "default": default_models,
                    "extra": extra_models,
                }
        
        result["apps"][app_name] = app_info
    
    return result


@pytest.fixture(scope="session")
def expected_videos() -> List[str]:
    """Fixture providing list of expected video filenames."""
    if config_manager is None:
        return []
    return config_manager.get_videos()


@pytest.fixture(scope="session")
def expected_images() -> List[str]:
    """Fixture providing list of expected image filenames."""
    if config_manager is None:
        return []
    return config_manager.get_images()


@pytest.fixture(scope="session")
def expected_so_files() -> List[str]:
    """Fixture providing list of expected .so filenames from meson.build."""
    return parse_meson_shared_libraries()


@pytest.fixture(scope="session")
def detected_hailo_arch() -> Optional[str]:
    """Fixture providing detected Hailo architecture (or None)."""
    if detect_hailo_arch is None:
        return None
    try:
        return detect_hailo_arch()
    except Exception:
        return None


@pytest.fixture(scope="session")
def detected_host_arch() -> str:
    """Fixture providing detected host architecture."""
    if detect_host_arch is None:
        return "unknown"
    try:
        return detect_host_arch()
    except Exception:
        return "unknown"


@pytest.fixture(scope="session")
def resources_root_path() -> Path:
    """Fixture providing the resources root path."""
    return Path(RESOURCES_ROOT_PATH_DEFAULT)


@pytest.fixture(scope="session")
def repo_resources_symlink() -> Path:
    """Fixture providing the repo resources symlink path."""
    return REPO_ROOT / "resources"


@pytest.fixture(scope="session")
def dotenv_path() -> Path:
    """Fixture providing the .env file path."""
    return Path(DEFAULT_DOTENV_PATH)


@pytest.fixture(scope="session")
def expected_models_for_arch(detected_hailo_arch) -> List[Tuple[str, str]]:
    """Fixture providing expected models for the detected architecture.

    Returns list of (app_name, model_name) tuples.
    """
    if not detected_hailo_arch or config_manager is None:
        return []

    models = []
    for app_name in config_manager.get_available_apps():
        for model_name in config_manager.get_model_names(
            app_name, detected_hailo_arch, tier="default"
        ):
            models.append((app_name, model_name))

    return models


@pytest.fixture(scope="session")
def expected_json_files() -> List[str]:
    """Fixture providing expected JSON files from the shared json section.

    Returns list of JSON filename strings.
    """
    if config_manager is None:
        return []
    return config_manager.get_all_json_files()
