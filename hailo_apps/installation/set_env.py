"""Environment configuration module for Hailo installation.

This module configures environment variables for hailo-apps.
It assumes all required packages (HailoRT, TAPPAS) are already installed.

Environment Variables Set:
    - HOST_ARCH: Host system architecture (x86, rpi, arm)
    - HAILO_ARCH: Hailo device architecture (hailo8, hailo8l, hailo10h)
    - HAILORT_VERSION: Installed HailoRT version
    - TAPPAS_VERSION: Installed TAPPAS version
    - TAPPAS_POSTPROC_PATH: Path to TAPPAS postprocess libraries
    - MODEL_ZOO_VERSION: Model Zoo version based on Hailo architecture
    - RESOURCES_PATH: Path to resources directory (symlink)
    - RESOURCES_ROOT_PATH: Path to resources root (/usr/local/hailo/resources)
    - VIRTUAL_ENV_NAME: Name of the virtual environment
    - HAILO_APPS_PATH: Full path to the hailo-apps repository
    - HAILO_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Try to import logger from hailo_apps, fallback to simple logger
try:
    from hailo_apps.python.core.common.hailo_logger import get_logger
except ImportError:
    import logging

    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

from hailo_apps.config.config_manager import get_main_config

# Try to import defines and installation utils from hailo_apps
try:
    from hailo_apps.python.core.common.defines import (
        DEFAULT_DOTENV_PATH,
        DEFAULT_RESOURCES_SYMLINK_PATH,
        HAILO10H_ARCH,
        HAILO_ARCH_DEFAULT,
        HAILO_ARCH_KEY,
        HAILO_APPS_PATH_KEY,
        HAILO_LOG_LEVEL_KEY,
        HAILORT_VERSION_KEY,
        HOST_ARCH_KEY,
        MODEL_ZOO_VERSION_KEY,
        REPO_ROOT,
        RESOURCES_PATH_KEY,
        RESOURCES_ROOT_PATH_DEFAULT,
        TAPPAS_POSTPROC_PATH_KEY,
        TAPPAS_VERSION_KEY,
        VALID_H8_MODEL_ZOO_VERSION,
        VALID_H10_MODEL_ZOO_VERSION,
        VIRTUAL_ENV_NAME_DEFAULT,
        VIRTUAL_ENV_NAME_KEY,
    )
    from hailo_apps.python.core.common.installation_utils import (
        auto_detect_tappas_postproc_dir,
        auto_detect_tappas_version,
        detect_hailo_arch,
        detect_host_arch,
        detect_system_pkg_version,
        get_hailort_package_name,
    )
except ImportError:
    # Fallback: import from path
    import importlib.util

    current_file = Path(__file__).resolve()
    defines_path = current_file.parent.parent.parent / "python" / "core" / "common" / "defines.py"
    installation_utils_path = (
        current_file.parent.parent.parent / "python" / "core" / "common" / "installation_utils.py"
    )

    if defines_path.exists():
        spec = importlib.util.spec_from_file_location("defines", defines_path)
        defines_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(defines_module)
        DEFAULT_DOTENV_PATH = defines_module.DEFAULT_DOTENV_PATH
        DEFAULT_RESOURCES_SYMLINK_PATH = defines_module.DEFAULT_RESOURCES_SYMLINK_PATH
        HAILO10H_ARCH = defines_module.HAILO10H_ARCH
        HAILO_ARCH_DEFAULT = defines_module.HAILO_ARCH_DEFAULT
        HAILO_ARCH_KEY = defines_module.HAILO_ARCH_KEY
        HAILO_APPS_PATH_KEY = defines_module.HAILO_APPS_PATH_KEY
        HAILO_LOG_LEVEL_KEY = defines_module.HAILO_LOG_LEVEL_KEY
        HAILORT_VERSION_KEY = defines_module.HAILORT_VERSION_KEY
        HOST_ARCH_KEY = defines_module.HOST_ARCH_KEY
        MODEL_ZOO_VERSION_KEY = defines_module.MODEL_ZOO_VERSION_KEY
        REPO_ROOT = defines_module.REPO_ROOT
        RESOURCES_PATH_KEY = defines_module.RESOURCES_PATH_KEY
        RESOURCES_ROOT_PATH_DEFAULT = defines_module.RESOURCES_ROOT_PATH_DEFAULT
        TAPPAS_POSTPROC_PATH_KEY = defines_module.TAPPAS_POSTPROC_PATH_KEY
        TAPPAS_VERSION_KEY = defines_module.TAPPAS_VERSION_KEY
        VALID_H8_MODEL_ZOO_VERSION = defines_module.VALID_H8_MODEL_ZOO_VERSION
        VALID_H10_MODEL_ZOO_VERSION = defines_module.VALID_H10_MODEL_ZOO_VERSION
        VIRTUAL_ENV_NAME_DEFAULT = defines_module.VIRTUAL_ENV_NAME_DEFAULT
        VIRTUAL_ENV_NAME_KEY = defines_module.VIRTUAL_ENV_NAME_KEY
    else:
        raise ImportError(f"Could not find defines.py at {defines_path}")

    if installation_utils_path.exists():
        spec = importlib.util.spec_from_file_location("installation_utils", installation_utils_path)
        installation_utils_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installation_utils_module)
        auto_detect_tappas_postproc_dir = installation_utils_module.auto_detect_tappas_postproc_dir
        auto_detect_tappas_version = installation_utils_module.auto_detect_tappas_version
        detect_hailo_arch = installation_utils_module.detect_hailo_arch
        detect_host_arch = installation_utils_module.detect_host_arch
        detect_system_pkg_version = installation_utils_module.detect_system_pkg_version
        get_hailort_package_name = installation_utils_module.get_hailort_package_name
    else:
        raise ImportError(f"Could not find installation_utils.py at {installation_utils_path}")

hailo_logger = get_logger(__name__)

# Additional environment variable keys
RESOURCES_ROOT_PATH_KEY = "resources_root_path"


def _ensure_env_file(env_path: Path) -> None:
    """Ensure .env file exists and is writable."""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if not env_path.is_file():
        print(f"🔧 Creating .env file at {env_path}")
        env_path.touch()
    os.chmod(env_path, 0o666)


def _write_env_file(env_path: Path, env_vars: Dict[str, Optional[str]]) -> None:
    """Write environment variables to .env file."""
    with open(env_path, "w") as f:
        f.write("# Hailo Apps Infrastructure Environment Configuration\n")
        f.write("# Auto-generated by set_env.py - Do not edit manually\n")
        f.write(f"# Generated at: {__import__('datetime').datetime.now().isoformat()}\n\n")
        
        for key, value in env_vars.items():
            if value is not None:
                f.write(f"{key}={value}\n")
    
    print(f"✅ Persisted environment variables to {env_path}")


def _get_hailort_version() -> str:
    """Get installed HailoRT version."""
    pkg_name = get_hailort_package_name()
    version = detect_system_pkg_version(pkg_name)
    if version:
        return version
    hailo_logger.error("HailoRT version not detected. Is HailoRT installed?")
    sys.exit(1)


def _get_tappas_version() -> str:
    """Get installed TAPPAS version."""
    version = auto_detect_tappas_version()
    if version:
        return version
    hailo_logger.error("TAPPAS version not detected. Is TAPPAS core installed?")
    sys.exit(1)


def _get_model_zoo_version(hailo_arch: str) -> str:
    """Get Model Zoo version based on Hailo architecture."""
    if hailo_arch == HAILO10H_ARCH:
        return VALID_H10_MODEL_ZOO_VERSION[0] if VALID_H10_MODEL_ZOO_VERSION else "v5.1.1"
    return VALID_H8_MODEL_ZOO_VERSION[0] if VALID_H8_MODEL_ZOO_VERSION else "v2.17.0"


def _get_hailo_arch() -> str | None:
    """Get Hailo device architecture.
    
    Returns:
        The detected architecture string, or None if detection fails.
    """
    try:
        arch = detect_hailo_arch()
        if arch:
            return arch
    except Exception as e:
        hailo_logger.warning(f"Could not detect Hailo architecture: {e}")
    return None


def configure_environment(config: Dict, env_path: Path) -> None:
    """Configure environment variables based on config and detected values.
    
    Args:
        config: Configuration dictionary from config file.
        env_path: Path to .env file.
    """
    # Ensure .env file exists
    _ensure_env_file(env_path)
    
    # Get nested config values
    resources_config = config.get('resources', {})
    venv_config = config.get('venv', {})
    
    # Detect values
    host_arch = detect_host_arch()
    hailo_arch = _get_hailo_arch()
    
    if not hailo_arch:
        hailo_logger.error("Could not detect Hailo architecture.")
        print(
            "\n❌ ERROR: Could not detect Hailo device architecture.\n"
            "   Please ensure:\n"
            "   - A Hailo device is connected\n"
            "   - The HailoRT driver is installed and loaded\n"
            "   - You have permissions to access the device\n"
            "\n   Run 'hailortcli fw-control identify' to check device connectivity.\n",
            file=sys.stderr
        )
        sys.exit(1)
    
    hailort_version = _get_hailort_version()
    tappas_version = _get_tappas_version()
    tappas_postproc_dir = auto_detect_tappas_postproc_dir()
    model_zoo_version = _get_model_zoo_version(hailo_arch)
    
    # Get repo root
    try:
        repo_root = str(REPO_ROOT)
    except NameError:
        repo_root = str(Path(__file__).resolve().parent.parent.parent)
    
    # Get log level from config (default: INFO)
    log_level = config.get('log_level', 'INFO').upper()
    
    # Build environment variables dict
    env_vars = {
        HOST_ARCH_KEY: host_arch,
        HAILO_ARCH_KEY: hailo_arch,
        HAILORT_VERSION_KEY: hailort_version,
        TAPPAS_VERSION_KEY: tappas_version,
        TAPPAS_POSTPROC_PATH_KEY: tappas_postproc_dir.strip() if tappas_postproc_dir else "",
        MODEL_ZOO_VERSION_KEY: model_zoo_version,
        RESOURCES_PATH_KEY: resources_config.get('path', DEFAULT_RESOURCES_SYMLINK_PATH),
        RESOURCES_ROOT_PATH_KEY: RESOURCES_ROOT_PATH_DEFAULT,
        VIRTUAL_ENV_NAME_KEY: venv_config.get('name', VIRTUAL_ENV_NAME_DEFAULT),
        HAILO_APPS_PATH_KEY: repo_root,
        HAILO_LOG_LEVEL_KEY: log_level,
    }
    
    # Update os.environ
    os.environ.update({k: v for k, v in env_vars.items() if v})
    
    # Write to .env file
    _write_env_file(env_path, env_vars)
    
    # Print summary
    print("\n📋 Environment Configuration Summary:")
    print("─" * 50)
    for key, value in env_vars.items():
        if value:
            print(f"  {key}: {value}")
    print("─" * 50)


# Legacy functions for backward compatibility
def handle_dot_env(env_path: Optional[Path] = None) -> Path:
    """Create and ensure .env file exists."""
    path = Path(env_path) if env_path else Path(DEFAULT_DOTENV_PATH)
    _ensure_env_file(path)
    return path


def set_environment_vars(config: Dict, env_path: Optional[Path] = None) -> None:
    """Set environment variables from config."""
    path = Path(env_path) if env_path else Path(DEFAULT_DOTENV_PATH)
    configure_environment(config, path)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Set environment variables for Hailo installation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables Set:
  HOST_ARCH             Host system architecture (x86, rpi, arm)
  HAILO_ARCH            Hailo device architecture (hailo8, hailo8l, hailo10h)
  HAILORT_VERSION       Installed HailoRT version
  TAPPAS_VERSION        Installed TAPPAS version
  TAPPAS_POSTPROC_PATH  Path to TAPPAS postprocess libraries
  MODEL_ZOO_VERSION     Model Zoo version (auto-selected based on architecture)
  RESOURCES_PATH        Path to resources symlink
  RESOURCES_ROOT_PATH   Path to resources root (/usr/local/hailo/resources)
  HAILO_APPS_PATH Full path to the repository
  VIRTUAL_ENV_NAME      Name of the virtual environment
  HAILO_LOG_LEVEL       Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
"""
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the config file (optional)"
    )
    parser.add_argument(
        "--env-path", type=str, default=DEFAULT_DOTENV_PATH, help="Path to the .env file"
    )

    args = parser.parse_args()

    # Load config
    config = get_main_config()

    # Configure environment
    configure_environment(config, Path(args.env_path))


if __name__ == "__main__":
    main()
