#!/bin/bash
# Cleanup all installation artifacts from hailo-apps.
# Run this BEFORE re-running ./install.sh when upgrading in-place.
# This ensures stale build artifacts (e.g. old TAPPAS .so references in
# the Meson/Ninja build cache) don't break recompilation.
#
# Usage:
#   sudo ./scripts/cleanup_installation.sh
#   sudo ./install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "Cleaning hailo-apps installation artifacts..."

# Python package build artifacts
sudo rm -rf hailo_apps.egg-info/ build/ dist/

# Virtual environment
sudo rm -rf venv_hailo_apps/

# Resources symlink and system resources
sudo rm -rf resources
sudo rm -rf /usr/local/hailo/resources/

# C++ postprocess build artifacts (stale build.ninja causes ninja errors on upgrade)
sudo rm -rf hailo_apps/postprocess/build.release/

# HailoRT logs
sudo rm -f hailort.log hailo_apps/postprocess/hailort.log

# Installation session logs (preserve app/test log subdirs)
sudo rm -f logs/install_*.log

# Python bytecode caches
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Pytest cache
sudo rm -rf .pytest_cache/

echo "Cleanup complete. You can now run: sudo ./install.sh"
