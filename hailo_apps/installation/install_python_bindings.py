"""Hailo Python bindings installer.

This module provides a Python entry point for installing Hailo Python bindings
(hailort and tappas-core wheels) into the current environment.

Usage:
    hailo-install-python-bindings hailo8     # Install bindings for Hailo-8
    hailo-install-python-bindings hailo10h   # Install bindings for Hailo-10H
    hailo-install-python-bindings hailo8 --download-only  # Download only
"""

import argparse
import subprocess
import sys
from pathlib import Path

from hailo_apps.python.core.common.defines import REPO_ROOT


def get_installer_script_path() -> Path:
    """Get path to the hailo_installer_python.sh script.
    
    Returns:
        Path to the installer script.
    
    Raises:
        FileNotFoundError: If the script is not found.
    """
    # Try repo location first
    repo_script = REPO_ROOT / "scripts" / "hailo_installer_python.sh"
    if repo_script.exists():
        return repo_script
    
    # Try package location (when installed via pip)
    try:
        import hailo_apps
        if hailo_apps.__file__ is not None:
            package_dir = Path(hailo_apps.__file__).parent.parent
            package_script = package_dir / "scripts" / "hailo_installer_python.sh"
            if package_script.exists():
                return package_script
    except (ImportError, AttributeError, TypeError):
        pass
    
    # Try installed data-files location (pip install from git)
    import sys
    data_file_locations = [
        Path(sys.prefix) / "share" / "hailo-apps" / "scripts" / "hailo_installer_python.sh",
        Path("/usr/local/share/hailo-apps/scripts/hailo_installer_python.sh"),
        Path("/usr/share/hailo-apps/scripts/hailo_installer_python.sh"),
    ]
    for script_path in data_file_locations:
        if script_path.exists():
            return script_path
    
    raise FileNotFoundError(
        "Could not find hailo_installer_python.sh script. "
        "Please ensure hailo-apps is installed correctly or run from the repository root."
    )


def main():
    """Main entry point for hailo-install-python-bindings command."""
    parser = argparse.ArgumentParser(
        description="Install Hailo Python bindings (hailort and tappas-core wheels).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    hailo-install-python-bindings hailo8        # Install for Hailo-8
    hailo-install-python-bindings hailo10h      # Install for Hailo-10H
    hailo-install-python-bindings hailo8 --download-only  # Download only
    hailo-install-python-bindings hailo10h --hailort-version 5.2.0
    
Note:
    This requires the hailo_installer_python.sh script which is part of
    the hailo-apps repository. When using pip-installed hailo-apps,
    you may need to clone the repository for full installer functionality.
"""
    )
    parser.add_argument(
        "arch",
        choices=["hailo8", "hailo10h"],
        help="Target Hailo hardware architecture"
    )
    parser.add_argument(
        "--hailort-version", "-r",
        help="Override HailoRT version"
    )
    parser.add_argument(
        "--tappas-core-version", "-t",
        help="Override TAPPAS core version"
    )
    parser.add_argument(
        "--download-only", "-o",
        action="store_true",
        help="Only download wheels, do not install"
    )
    parser.add_argument(
        "--download-dir", "-D",
        help="Directory to download wheels to (default: /usr/local/hailo/resources/packages)"
    )
    parser.add_argument(
        "--no-hailort", "-H",
        action="store_true",
        help="Skip HailoRT download/install"
    )
    parser.add_argument(
        "--no-tappas", "-N",
        action="store_true",
        help="Skip TAPPAS core download/install"
    )
    parser.add_argument(
        "--dry-run", "-y",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    args = parser.parse_args()
    
    try:
        script_path = get_installer_script_path()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nTo install Hailo Python bindings manually, run:", file=sys.stderr)
        print(f"  ./scripts/hailo_installer_python.sh {args.arch}", file=sys.stderr)
        sys.exit(1)
    
    # Build command
    cmd = ["bash", str(script_path), args.arch]
    
    if args.hailort_version:
        cmd.extend(["--hailort-version", args.hailort_version])
    if args.tappas_core_version:
        cmd.extend(["--tappas-core-version", args.tappas_core_version])
    if args.download_only:
        cmd.append("--download-only")
    if args.download_dir:
        cmd.extend(["--download-dir", args.download_dir])
    if args.no_hailort:
        cmd.append("--no-hailort")
    if args.no_tappas:
        cmd.append("--no-tappas")
    if args.dry_run:
        cmd.append("--dry-run")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
