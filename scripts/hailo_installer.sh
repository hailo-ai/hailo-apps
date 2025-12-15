#!/bin/bash
# Hailo Runtime Installer Script
# This script downloads and installs all Hailo runtime requirements
# from the deb server. It performs several checks:
#   - Checks system architecture (x86_64, aarch64, or Raspberry Pi)
#   - For Raspberry Pi: if 'hailo-all' is not installed, points to RPi docs and exits.
#   - Validates Python version (supported: 3.10, 3.11, 3.12, 3.13)
#   - Checks the kernel version (warns if not officially supported)
#   - Downloads and installs the following:
#       * HailoRT driver deb
#       * HailoRT deb
#       * Tapas core deb
#       * HailoRT Python bindings whl
#       * Tapas core Python bindings whl
#
# The deb server is hosted at: http://dev-public.hailo.ai/
# Owner: Sergii Tishchenko
#
#

set -euo pipefail

# --- Configurable Variables ---

# Base URL of the deb server
BASE_URL="http://dev-public.hailo.ai/2025_10"
declare -a DOWNLOADED_URLS=()


# Default version numbers for packages (if using --version, you can adjust these)

HAILORT_VERSION_H8="4.23.0"
TAPPAS_CORE_VERSION_H8="5.1.0"
HAILORT_VERSION_H10="5.1.0"
TAPPAS_CORE_VERSION_H10="5.1.0"

HAILORT_VERSION=""
TAPPAS_CORE_VERSION=""


# Defaults (can be overridden by flags)
HW_ARCHITECTURE=""               # hailo8 | hailo10h
VENV_NAME="venv_hailo_apps"
DOWNLOAD_ONLY="false"
OUTPUT_DIR_BASE="/usr/local/hailo/resources/packages"

PY_TAG_OVERRIDE=""

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --hailort-version=VER           Override HailoRT version
  --tappas-core-version=VER       Override TAPPAS Core version
  --venv-name=NAME                Virtualenv name (install mode only) [default: $VENV_NAME]
  --hw-arch=hailo8|hailo10h       Target hardware (affects version defaults & folder) [default: $HW_ARCHITECTURE]
  --download-only                 Only download packages, do NOT install
  --output-dir=DIR                Base output directory for downloads [default: $OUTPUT_DIR_BASE]
  --py-tag=TAG                    Wheel tag (e.g. cp311-cp311). Useful with --download-only
  -h|--help                       Show this help
EOF
}


# Parse optional command-line flag to override version numbers (e.g., --version=4.20.0)
# For a more complex versioning scheme, you might also separate HailoRT and TAPPAS versions.
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --hailort-version=*)
            HAILORT_VERSION="${1#*=}"
            ;;
        --tappas-core-version=*)
            TAPPAS_CORE_VERSION="${1#*=}"
            ;;
        --venv-name=*)
            VENV_NAME="${1#*=}"
            ;;
        --hw-arch=*)
            HW_ARCHITECTURE="${1#*=}"
            if [[ "$HW_ARCHITECTURE" != "hailo8" && "$HW_ARCHITECTURE" != "hailo10h" ]]; then
                echo "Invalid hardware architecture specified. Use 'hailo8' or 'hailo10h'."
                exit 1
            fi
            if [[ "$HW_ARCHITECTURE" == "hailo8" ]]; then
                HAILORT_VERSION="$HAILORT_VERSION_H8"
                TAPPAS_CORE_VERSION="$TAPPAS_CORE_VERSION_H8"
            elif [[ "$HW_ARCHITECTURE" == "hailo10h" ]]; then
                HAILORT_VERSION="$HAILORT_VERSION_H10"
                TAPPAS_CORE_VERSION="$TAPPAS_CORE_VERSION_H10"
            fi
            ;;
        --download-only)
            DOWNLOAD_ONLY="true"
            ;;
        --output-dir=*)
            OUTPUT_DIR_BASE="${1#*=}"
            ;;
        --py-tag=*)
            PY_TAG_OVERRIDE="${1#*=}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done


TARGET_DIR="${OUTPUT_DIR_BASE}/${HW_ARCHITECTURE}"
mkdir -p "$TARGET_DIR"
echo "Download target directory: $TARGET_DIR"
HW_NAME=""
# Determine hardware name based on architecture
if [[ "$HW_ARCHITECTURE" == "hailo8" ]]; then
  HW_NAME="Hailo8"
else
  HW_NAME="Hailo10H"
fi
BASE_URL="${BASE_URL}/${HW_NAME}"

# --- Functions ---
download_file() {
  local rel="$1"
  local url="${BASE_URL}/${rel}"
  local dst="${TARGET_DIR}/${rel}"

  echo "Downloading ${rel}"
  mkdir -p "$(dirname "$dst")"
  if ! wget -q --show-progress "$url" -O "$dst"; then
    echo "Retrying ${rel}..."
    wget "$url" -O "$dst"
  fi
  DOWNLOADED_URLS+=("$url")
}

install_file() {
  local file="$1"
  local path="${TARGET_DIR}/${file}"

  if [[ "$DOWNLOAD_ONLY" == "true" ]]; then
    echo "[download-only] Skipping install for $file"
    return
  fi

  echo "Installing $file..."
  if [[ "$file" == *.deb ]]; then
    sudo dpkg -i "$path"
  elif [[ "$file" == *.whl ]]; then
    if python3 -c "import site; import os; print(os.access(site.getsitepackages()[0], os.W_OK))" 2>/dev/null | grep -q "True"; then
      echo "Installing system-wide (writable site-packages)"
      python3 -m pip install "$path" --break-system-packages
    else
      echo "Installing with --user flag (user site-packages)"
      python3 -m pip install "$path" --user --break-system-packages
    fi
  else
    echo "Unknown file type: $file"
  fi
}

# -------- System info / tags --------
ARCH="$(uname -m)"
KERNEL="$(uname -r)"

if [[ "$ARCH" == "aarch64" && "$KERNEL" == *"rpi"* ]]; then
    ARCH="rpi"
fi
echo "Detected architecture: $ARCH"

# Ubuntu version detection
UBUNTU_VERSION=""
if [[ -f /etc/os-release ]]; then
  . /etc/os-release
  if [[ "$ID" == "ubuntu" ]]; then
    UBUNTU_VERSION="$VERSION_ID"
    echo "Detected Ubuntu version: $UBUNTU_VERSION"
  fi
fi


# Raspberry Pi detection (same behavior as before, but skip entirely if download-only)
if [[ "$ARCH" == *"arm"* && "$DOWNLOAD_ONLY" != "true" ]]; then
  if [[ -f /proc/device-tree/model ]]; then
    MODEL="$(tr -d '\0' < /proc/device-tree/model || true)"
    if [[ "$MODEL" == *"Raspberry Pi"* ]]; then
      echo "Raspberry Pi detected."
      if ! command -v hailo-all &>/dev/null; then
        echo "hailo-all is not installed. See RPi docs: https://www.raspberrypi.com/documentation/computers/ai.html"
        exit 1
      else
        echo "hailo-all already installed. This installer does not auto-install on RPi."
        exit 0
      fi
    fi
  fi
fi

# Python & kernel checks (skip installs when download-only)
PY_TAG=""
if [[ -n "$PY_TAG_OVERRIDE" ]]; then
  PY_TAG="$PY_TAG_OVERRIDE"
else
  if command -v python3 &>/dev/null; then
    PYTHON_VERSION="$(python3 --version 2>&1 | awk '{print $2}')"
    echo "Detected Python: $PYTHON_VERSION"
    if [[ "$PYTHON_VERSION" =~ ^3\.(10|11|12|13) ]]; then
      PY_VER_MAJOR="$(echo "$PYTHON_VERSION" | cut -d. -f1)"
      PY_VER_MINOR="$(echo "$PYTHON_VERSION" | cut -d. -f2)"
      PY_TAG="cp${PY_VER_MAJOR}${PY_VER_MINOR}-cp${PY_VER_MAJOR}${PY_VER_MINOR}"
    else
      if [[ "$DOWNLOAD_ONLY" == "true" ]]; then
        echo "Unsupported Python version ($PYTHON_VERSION). Falling back to cp310-cp310 for download-only."
        PY_TAG="cp310-cp310"
      else
        echo "Unsupported Python version. Supported: 3.10/3.11/3.12/3.13"
        exit 1
      fi
    fi
  else
    if [[ "$DOWNLOAD_ONLY" == "true" ]]; then
      echo "python3 not found. Falling back to cp310-cp310 for download-only."
      PY_TAG="cp310-cp310"
    else
      echo "python3 is required."
      exit 1
    fi
  fi
fi
echo "Using wheel tag: $PY_TAG"

if [[ "$DOWNLOAD_ONLY" != "true" ]]; then
  KERNEL_VERSION="$(uname -r)"
  echo "Kernel version: $KERNEL_VERSION"
  OFFICIAL_KERNEL_PREFIX="6.5.0"
  if [[ "$KERNEL_VERSION" != "$OFFICIAL_KERNEL_PREFIX"* ]]; then
    echo "Warning: Kernel $KERNEL_VERSION may not be officially supported."
  fi

  echo "Installing build-essential..."
  sudo apt-get update && sudo apt-get install -y build-essential

  echo "Installing deps for hailo-tappas-core..."
  sudo apt-get update && sudo apt-get install -y \
    ffmpeg python3-virtualenv gcc-12 g++-12 python-gi-dev pkg-config libcairo2-dev \
    libgirepository1.0-dev libgstreamer1.0-dev cmake libgstreamer-plugins-base1.0-dev \
    libzmq3-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-bad \
    gstreamer1.0-libav libopencv-dev python3-opencv rapidjson-dev
fi

# -------- Build file lists --------
common_files=(
  "hailort-pcie-driver_${HAILORT_VERSION}_all.deb"
  "hailo_tappas_core_python_binding-${TAPPAS_CORE_VERSION}-py3-none-any.whl"
)

ARCH_FILES=()
case "$ARCH" in
  x86_64|amd64)
    echo "Configuring AMD64 package names..."
    ARCH_FILES+=("hailort_${HAILORT_VERSION}_amd64.deb")
    # Select Ubuntu-specific package if available
    if [[ -n "$UBUNTU_VERSION" ]]; then
      if [[ "$UBUNTU_VERSION" == "22.04" ]]; then
        ARCH_FILES+=("hailo-tappas-core_${TAPPAS_CORE_VERSION}_amd64_ub22.deb")
        echo "Using Ubuntu 22.04 specific package"
      elif [[ "$UBUNTU_VERSION" == "24.04" ]]; then
        ARCH_FILES+=("hailo-tappas-core_${TAPPAS_CORE_VERSION}_amd64_ub24.deb")
        echo "Using Ubuntu 24.04 specific package"
      else
        ARCH_FILES+=("hailo-tappas-core_${TAPPAS_CORE_VERSION}_amd64.deb")
        echo "Using generic AMD64 package for Ubuntu $UBUNTU_VERSION"
      fi
    else
      ARCH_FILES+=("hailo-tappas-core_${TAPPAS_CORE_VERSION}_amd64.deb")
      echo "Using generic AMD64 package (no Ubuntu detection)"
    fi
    ARCH_FILES+=("hailort-${HAILORT_VERSION}-${PY_TAG}-linux_x86_64.whl")
    ;;
  aarch64|arm64)
    echo "Configuring ARM64 package names..."
    ARCH_FILES+=("hailort_${HAILORT_VERSION}_arm64.deb")
    ARCH_FILES+=("hailo-tappas-core_${TAPPAS_CORE_VERSION}_arm64.deb")
    ARCH_FILES+=("hailort-${HAILORT_VERSION}-${PY_TAG}-linux_aarch64.whl")
    ;;
  rpi)
    echo "Configuring rpi  package names..."
    ARCH_FILES+=("hailort_${HAILORT_VERSION}_arm64.deb")
    if [[ "${TAPPAS_CORE_VERSION}" == "5.0.0" ]]; then
      # Special-case naming for 5.0.0
      ARCH_FILES+=("hailo-tappas-core-5.0.0v_5.0.0_arm64.deb")
    else
      ARCH_FILES+=("hailo-tappas-core_${TAPPAS_CORE_VERSION}_arm64.deb")
    fi
    ARCH_FILES+=("hailort-${HAILORT_VERSION}-${PY_TAG}-linux_aarch64.whl")
    ;;
  *)
    echo "Unsupported architecture: $ARCH"
    exit 1
    ;;
esac

# -------- Download --------
echo "Downloading common files..."
for f in "${common_files[@]}"; do
  echo "$f"
  download_file "$f"
done

echo "Downloading arch-specific files..."
for f in "${ARCH_FILES[@]}"; do
  echo "$f"
  download_file "$f"
done

echo "All files downloaded to: ${TARGET_DIR}"

# ---- Print direct links ----
if ((${#DOWNLOADED_URLS[@]})); then
  echo
  echo "=== Direct download links ==="
  for u in "${DOWNLOADED_URLS[@]}"; do
    echo "$u"
  done
  echo "============================="
  echo
fi

# -------- Install (skipped if download-only) --------
if [[ "$DOWNLOAD_ONLY" == "true" ]]; then
  echo "[download-only] Done. Packages saved under ${TARGET_DIR}"
  exit 0
fi

echo "Starting installation..."
install_file "${common_files[0]}"       # PCIe driver
install_file "${ARCH_FILES[0]}"         # HailoRT deb
install_file "${ARCH_FILES[1]}"         # Tappas Core deb
install_file "${common_files[1]}"       # Tappas Core Python bindings (any)
install_file "${ARCH_FILES[2]}"         # HailoRT wheel

echo "Installation complete."
