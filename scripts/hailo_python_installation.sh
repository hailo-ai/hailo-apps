#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Hailo Python Wheels: downloader & installer
# Works with the main installer (calls this with --hailort-version/--tappas-core-version).
# Presets for H8/H10; overrideable via flags.
# ------------------------------------------------------------------------------

# Script root (repo root) – used to locate config.yaml if present
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Base URL of the deb server (will be set based on hardware architecture)
BASE_URL=""

# Default version numbers for packages (if using --version, you can adjust these)

HAILORT_VERSION_H8="4.23.0"
TAPPAS_CORE_VERSION_H8="5.1.0"
HAILORT_VERSION_H10="5.1.1"
TAPPAS_CORE_VERSION_H10="5.1.0"


HAILORT_VERSION=""
TAPPAS_CORE_VERSION=""

# Behavior flags
HW_ARCHITECTURE=""          # hailo8 | hailo10h (affects defaults if versions not passed)
DOWNLOAD_DIR="/usr/local/hailo/resources/packages"
DOWNLOAD_ONLY=false
DRY_RUN=false
QUIET=false

INSTALL_HAILORT=false
INSTALL_TAPPAS=false
NO_TAPPAS=false

# Pip flag presets
PIP_SYS_FLAGS=(--break-system-packages --disable-pip-version-check --no-input --prefer-binary)
PIP_USER_FLAGS=(--user --break-system-packages --disable-pip-version-check --no-input --prefer-binary)
PIP_VENV_FLAGS=(--disable-pip-version-check --no-input --prefer-binary)


usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --arch=(hailo8|hailo10h)       Choose hardware preset for default versions
  --hailort-version=VER          Force a specific HailoRT wheel version (overrides preset)
  --tappas-core-version=VER      Force a specific TAPPAS core wheel version (overrides preset)
  --no-tappas                    Skip TAPPAS core download/install
  --base-url=URL                 Override base URL (default: ${BASE_URL})
  --download-dir=DIR             Where to place wheels (default: ${DOWNLOAD_DIR})
  --download-only                Only download wheels; do not install
  --dry-run                      Show what would be done without actually doing it
  -d, --default                  Install both HailoRT and TAPPAS packages
  -q, --quiet                    Less output
  -h, --help                     Show this help

Notes:
- If you pass neither --hailort-version nor --tappas-core-version, the chosen --arch preset is used.
- If you pass only one of them, only that package is downloaded/installed.
- Dry-run mode shows all actions without downloading or installing anything.
EOF
}

log() { $QUIET || echo -e "$*"; }

# -------------------- Helper functions (defined early for context reporting) --------------------
is_venv() {
  # True for venv/virtualenv/conda (prefix differs) or real_prefix set
  python3 - "$@" <<'PY'
import sys
print("1" if (getattr(sys, "real_prefix", None) or sys.prefix != sys.base_prefix) else "0")
PY
}

site_writable() {
  # True if system site-packages dir is writable
  python3 - "$@" <<'PY'
import os, site, sys
try:
    paths = site.getsitepackages()
except Exception:
    # Fallback (rare, but just in case)
    paths = [sys.prefix + "/lib/python%s.%s/site-packages" % sys.version_info[:2]]
print("1" if (paths and os.access(paths[0], os.W_OK)) else "0")
PY
}

# -------------------- Helper functions (defined early for context reporting) --------------------
log() { $QUIET || echo -e "$*"; }

is_venv() {
  # True for venv/virtualenv/conda (prefix differs) or real_prefix set
  python3 - "$@" <<'PY'
import sys
print("1" if (getattr(sys, "real_prefix", None) or sys.prefix != sys.base_prefix) else "0")
PY
}

site_writable() {
  # True if system site-packages dir is writable
  python3 - "$@" <<'PY'
import os, site, sys
try:
    paths = site.getsitepackages()
except Exception:
    # Fallback (rare, but just in case)
    paths = [sys.prefix + "/lib/python%s.%s/site-packages" % sys.version_info[:2]]
print("1" if (paths and os.access(paths[0], os.W_OK)) else "0")
PY
}

get_installed_version() {
  local pkg="$1"
  python3 - <<PY 2>/dev/null
import subprocess, sys
pkg = sys.argv[1]
try:
    out = subprocess.check_output([sys.executable, "-m", "pip", "show", pkg], text=True)
    for line in out.splitlines():
        if line.startswith("Version:"):
            print(line.split(":",1)[1].strip())
            sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
}

# -------------------- Parse flags --------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch=*)
      HW_ARCHITECTURE="${1#*=}"
      if [[ "$HW_ARCHITECTURE" != "hailo8" && "$HW_ARCHITECTURE" != "hailo10h" ]]; then
          echo "Invalid architecture specified. Use 'hailo8' or 'hailo10h'."
          exit 1
      fi
      shift
      ;;
    --hw-arch=*)
      # Backward compatibility (deprecated)
      HW_ARCHITECTURE="${1#*=}"
      if [[ "$HW_ARCHITECTURE" != "hailo8" && "$HW_ARCHITECTURE" != "hailo10h" ]]; then
          echo "Invalid architecture specified. Use 'hailo8' or 'hailo10h'."
          exit 1
      fi
      shift
      ;;

    --hailort-version=*)
      HAILORT_VERSION="${1#*=}"
      shift
      ;;
    --tappas-core-version=*)
      TAPPAS_CORE_VERSION="${1#*=}"
      shift
      ;;
    --base-url=*)
      BASE_URL="${1#*=}"
      shift
      ;;
    --download-dir=*)
      DOWNLOAD_DIR="${1#*=}"
      shift
      ;;
    --download-only)
      DOWNLOAD_ONLY=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -q|--quiet)
      QUIET=true
      shift
      ;;
    -d | --default)
      INSTALL_HAILORT=true
      INSTALL_TAPPAS=true
      shift
      ;;
    --no-tappas)
      NO_TAPPAS=true
      INSTALL_TAPPAS=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

# Require architecture flag
if [[ -z "$HW_ARCHITECTURE" ]]; then
  echo "Error: --arch is required (hailo8|hailo10h)."
  exit 1
fi

# Set BASE_URL based on hardware architecture (unless overridden by --base-url)
if [[ -z "$BASE_URL" ]]; then
  if [[ "$HW_ARCHITECTURE" == "hailo8" ]]; then
    BASE_URL="http://dev-public.hailo.ai/2025_10"
  elif [[ "$HW_ARCHITECTURE" == "hailo10h" ]]; then
    BASE_URL="http://dev-public.hailo.ai/2025_12"
  fi
fi

# -------------------- Resolve desired versions --------------------
# Prefer: user flag override; otherwise choose by architecture.
if [[ -z ${HAILORT_VERSION+x} || -z "$HAILORT_VERSION" ]]; then
  case "$HW_ARCHITECTURE" in
    hailo8)   HAILORT_VERSION="$HAILORT_VERSION_H8" ;;   # e.g., 4.23.0
    hailo10h) HAILORT_VERSION="5.1.1" ;;                 # latest for H10H
    *)        HAILORT_VERSION="$HAILORT_VERSION_H8" ;;
  esac
fi

if [[ -z ${TAPPAS_CORE_VERSION+x} || -z "$TAPPAS_CORE_VERSION" ]]; then
  case "$HW_ARCHITECTURE" in
    hailo8)   TAPPAS_CORE_VERSION="$TAPPAS_CORE_VERSION_H8" ;;
    hailo10h) TAPPAS_CORE_VERSION="$TAPPAS_CORE_VERSION_H10" ;;
    *)        TAPPAS_CORE_VERSION="$TAPPAS_CORE_VERSION_H8" ;;
  esac
fi

if [[ "$NO_TAPPAS" == true ]]; then
  TAPPAS_CORE_VERSION=""
  INSTALL_TAPPAS=false
fi

# If user specified only one version, we install only that one; otherwise decide based on installed state.
if [[ -n "$HAILORT_VERSION" ]]; then
  if [[ "$INSTALL_HAILORT" == false ]]; then
    installed_hailort="$(get_installed_version hailort || true)"
    if [[ -z "$installed_hailort" || "$installed_hailort" != "$HAILORT_VERSION" ]]; then
      INSTALL_HAILORT=true
    fi
  fi
fi

if [[ -n "$TAPPAS_CORE_VERSION" ]]; then
  if [[ "$INSTALL_TAPPAS" == false ]]; then
    # Try multiple known package names
    installed_tappas=""
    for pkg in hailo-tappas-core-python-binding hailo-tappas-core tappas-core hailo_tappas_core_python_binding; do
      ver="$(get_installed_version "$pkg" || true)"
      if [[ -n "$ver" ]]; then
        installed_tappas="$ver"
        break
      fi
    done
    if [[ -z "$installed_tappas" || "$installed_tappas" != "$TAPPAS_CORE_VERSION" ]]; then
      INSTALL_TAPPAS=true
    fi
  fi
fi

if [[ "$INSTALL_HAILORT" == false && "$INSTALL_TAPPAS" == false ]]; then
  log "Nothing to do (versions already match or not requested)."
  exit 0
fi

HW_FOLDER_NAME=""
# Determine hardware name based on architecture
if [[ "$HW_ARCHITECTURE" == "hailo8" ]]; then
  HW_FOLDER_NAME="Hailo8"
  HW_FOLDER_SECONDARY="Hailo10"
else
  HW_FOLDER_NAME="Hailo10"
  HW_FOLDER_SECONDARY="Hailo8"
fi



# -------------------- Compute tags --------------------
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
PY_TAG="cp${PY_MAJOR}${PY_MINOR}-cp${PY_MAJOR}${PY_MINOR}"

# Map uname -m to wheel platform tag
UNAME_M="$(uname -m)"
case "$UNAME_M" in
  x86_64)  ARCH_TAG="linux_x86_64" ;;
  aarch64) ARCH_TAG="linux_aarch64" ;;
  *)
    echo "Unsupported architecture: $UNAME_M"
    exit 1
    ;;
esac

if [[ "$DRY_RUN" == true ]]; then
  if [[ -e "$DOWNLOAD_DIR" ]]; then
    if [[ ! -d "$DOWNLOAD_DIR" ]]; then
      log "[DRY-RUN] Error: $DOWNLOAD_DIR exists and is not a directory."
      exit 1
    else
      log "[DRY-RUN] Directory $DOWNLOAD_DIR already exists."
    fi
  else
    log "[DRY-RUN] Would create download directory: $DOWNLOAD_DIR"
  fi
else
  if [[ -e "$DOWNLOAD_DIR" ]]; then
    if [[ ! -d "$DOWNLOAD_DIR" ]]; then
      echo "Error: $DOWNLOAD_DIR exists and is not a directory."
      exit 1
    else
      echo "Directory $DOWNLOAD_DIR already exists."
    fi
  else
    echo "Creating download directory: $DOWNLOAD_DIR"
    mkdir -p "$DOWNLOAD_DIR"
  fi
fi

if [[ "$DRY_RUN" == true ]]; then
  log ""
  log "══════════════════════════════════════════════════════════════"
  log "  DRY-RUN MODE - No changes will be made"
  log "══════════════════════════════════════════════════════════════"
  log ""
fi

# -------------------- Permission and Context Info --------------------
log ""
log "────────────────────────────────────────────────────────────────"
log "  Execution Context"
log "────────────────────────────────────────────────────────────────"
log "→ Running as user     = $(whoami) (UID: $(id -u))"
log "→ Effective groups    = $(id -Gn | tr ' ' ', ')"
log "→ Is root?            = $([[ $(id -u) -eq 0 ]] && echo 'yes' || echo 'no')"
log "→ Python executable   = $(which python3)"
log "→ Python version      = $(python3 --version 2>&1)"
log "→ Pip version         = $(python3 -m pip --version 2>&1 | head -1)"
log "→ Virtual env active? = $([[ "$(is_venv)" == "1" ]] && echo 'yes' || echo 'no')"
if [[ "$(is_venv)" == "1" ]]; then
  log "→ Virtual env path    = ${VIRTUAL_ENV:-$(python3 -c 'import sys; print(sys.prefix)')}"
fi
log "→ Site-packages writable? = $([[ "$(site_writable)" == "1" ]] && echo 'yes' || echo 'no')"
log "→ Download dir writable?  = $([[ -w "$(dirname "$DOWNLOAD_DIR")" || -w "$DOWNLOAD_DIR" ]] && echo 'yes' || echo 'no (may need sudo)')"
log "────────────────────────────────────────────────────────────────"
log ""

log "→ BASE_URL            = $BASE_URL"
log "→ ARCH preset         = $HW_ARCHITECTURE"
log "→ Python tag          = $PY_TAG"
log "→ Wheel arch tag      = $ARCH_TAG"
$INSTALL_HAILORT && log "→ HailoRT version     = $HAILORT_VERSION"
$INSTALL_TAPPAS && log "→ TAPPAS core version = $TAPPAS_CORE_VERSION"
log "→ Download dir        = $DOWNLOAD_DIR"
log "→ Download only?      = $DOWNLOAD_ONLY"
log "→ Dry-run mode?       = $DRY_RUN"

# -------------------- Download helper --------------------
fetch() {
  local url="$1"
  local out="$2"

  if [[ "$DRY_RUN" == true ]]; then
    if [[ -f "$out" ]]; then
      log "  [DRY-RUN] Would skip (already exists): $(basename "$out")"
    else
      log "  [DRY-RUN] Would download: $url"
      log "  [DRY-RUN]            to: $out"
    fi
    return 0
  fi

  if [[ -f "$out" ]]; then
    log "  - Exists: $(basename "$out")"
    return 0
  fi
  log "  - GET $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "$out" "$url"
  else
    wget -q --tries=3 --timeout=20 -O "$out" "$url"
  fi
}

install_pip_package() {
  local package_path="$1"

  # Detect venv
  local in_venv
  in_venv="$(python3 - <<'PY'
import sys
print("1" if (getattr(sys, "real_prefix", None) or sys.prefix != sys.base_prefix) else "0")
PY
)"

  if [[ "$DRY_RUN" == true ]]; then
    if [[ "$in_venv" == "1" ]]; then
      log "  [DRY-RUN] Would install into active virtual environment: $(basename "$package_path")"
    elif [[ "$(site_writable)" == "1" ]]; then
      log "  [DRY-RUN] Would install system-wide: $(basename "$package_path")"
    else
      log "  [DRY-RUN] Would install with --user: $(basename "$package_path")"
    fi
    return 0
  fi

  if [[ "$in_venv" == "1" ]]; then
    echo "Installing into active virtual environment"
    python3 -m pip install "${PIP_VENV_FLAGS[@]}" --upgrade -- "$package_path"
  elif [[ "$(site_writable)" == "1" ]]; then
    echo "Installing system-wide"
    python3 -m pip install "${PIP_SYS_FLAGS[@]}" --upgrade -- "$package_path"
  else
    echo "Installing with --user (+ --break-system-packages)"
    python3 -m pip install "${PIP_USER_FLAGS[@]}" --upgrade -- "$package_path"
  fi
}


# -------------------- Download wheels --------------------
HW_FOLDER_NAME=""
# Determine hardware name based on architecture
if [[ "$HW_ARCHITECTURE" == "hailo8" ]]; then
  HW_FOLDER_NAME="Hailo8"
  HW_FOLDER_SECONDARY="Hailo10"
else
  HW_FOLDER_NAME="Hailo10"
  HW_FOLDER_SECONDARY="Hailo8"
fi

if [[ "$INSTALL_TAPPAS" == true ]]; then

  TAPPAS_FILE="hailo_tappas_core_python_binding-${TAPPAS_CORE_VERSION}-py3-none-any.whl"
  TAPPAS_URL="${BASE_URL}/${HW_FOLDER_NAME}/${TAPPAS_FILE}"

  if ! fetch "$TAPPAS_URL" "${DOWNLOAD_DIR}/${TAPPAS_FILE}"; then
    echo "Failed from primary ($HW_FOLDER_NAME). Trying secondary: ${HW_FOLDER_SECONDARY}"
    TAPPAS_URL="${BASE_URL}/${HW_FOLDER_SECONDARY}/${TAPPAS_FILE}"
    if ! fetch "$TAPPAS_URL" "${DOWNLOAD_DIR}/${TAPPAS_FILE}"; then
      echo "Failed from both primary and secondary folders. Check URL(s) and network."
      exit 1
    fi
  fi
fi

if [[ "$INSTALL_HAILORT" == true ]]; then
  HAILORT_FILE="hailort-${HAILORT_VERSION}-${PY_TAG}-${ARCH_TAG}.whl"
  HAILORT_URL="${BASE_URL}/${HW_FOLDER_NAME}/${HAILORT_FILE}"
  fetch "$HAILORT_URL" "${DOWNLOAD_DIR}/${HAILORT_FILE}"
fi

if [[ "$DOWNLOAD_ONLY" == true ]]; then
  if [[ "$DRY_RUN" == true ]]; then
    log "✅ [DRY-RUN] Download(s) would complete (download-only mode)."
  else
    log "✅ Download(s) complete (download-only)."
  fi
  exit 0
fi

# -------------------- Install into current environment --------------------
log "→ Upgrading pip / wheel / setuptools…"

if [[ "$DRY_RUN" == true ]]; then
  if [[ "$(is_venv)" == "1" ]]; then
    log "  [DRY-RUN] Would upgrade pip/setuptools/wheel in virtual environment"
  elif [[ "$(site_writable)" == "1" ]]; then
    log "  [DRY-RUN] Would upgrade pip/setuptools/wheel system-wide"
  else
    log "  [DRY-RUN] Would upgrade pip/setuptools/wheel with --user"
  fi
else
  if [[ "$(is_venv)" == "1" ]]; then
    echo "Upgrading in virtual environment"
    python3 -m pip install "${PIP_VENV_FLAGS[@]}" --upgrade pip setuptools wheel >/dev/null
  elif [[ "$(site_writable)" == "1" ]]; then
    echo "Upgrading system-wide"
    python3 -m pip install "${PIP_SYS_FLAGS[@]}" --upgrade pip setuptools wheel >/dev/null
  else
    echo "Upgrading with --user (+ --break-system-packages)"
    python3 -m pip install "${PIP_USER_FLAGS[@]}" --upgrade pip setuptools wheel >/dev/null
  fi
fi


if [[ "$INSTALL_HAILORT" == true ]]; then
  log "→ Installing HailoRT wheel…"
  install_pip_package "${DOWNLOAD_DIR}/${HAILORT_FILE}"
fi

if [[ "$INSTALL_TAPPAS" == true ]]; then
  log "→ Installing TAPPAS core wheel…"
  install_pip_package "${DOWNLOAD_DIR}/${TAPPAS_FILE}"
fi

# -------------------- Verification --------------------
verify_installation() {
  local package_name="$1"
  local expected_version="$2"
  local display_name="$3"

  # Try to get the installed version
  local installed_version
  installed_version=$(python3 -m pip show "$package_name" 2>/dev/null | grep "^Version:" | awk '{print $2}')

  if [[ -z "$installed_version" ]]; then
    log "  ❌ $display_name: NOT INSTALLED"
    return 1
  elif [[ "$installed_version" == "$expected_version" ]]; then
    log "  ✅ $display_name: $installed_version (matches expected)"
    return 0
  else
    log "  ⚠️  $display_name: $installed_version (expected: $expected_version)"
    return 0  # Still installed, just different version
  fi
}

verify_wheel_file() {
  local wheel_path="$1"
  local display_name="$2"

  if [[ -f "$wheel_path" ]]; then
    local file_size
    file_size=$(stat -c%s "$wheel_path" 2>/dev/null || stat -f%z "$wheel_path" 2>/dev/null || echo "unknown")
    log "  ✅ $display_name: exists (${file_size} bytes)"
    return 0
  else
    log "  ❌ $display_name: NOT FOUND at $wheel_path"
    return 1
  fi
}

if [[ "$DRY_RUN" == true ]]; then
  log ""
  log "══════════════════════════════════════════════════════════════"
  log "  DRY-RUN COMPLETE - No changes were made"
  log "══════════════════════════════════════════════════════════════"
else
  log ""
  log "────────────────────────────────────────────────────────────────"
  log "  Verification"
  log "────────────────────────────────────────────────────────────────"

  VERIFICATION_FAILED=false

  # Verify downloaded wheel files exist
  log "→ Checking downloaded wheel files:"
  if [[ "$INSTALL_HAILORT" == true ]]; then
    if ! verify_wheel_file "${DOWNLOAD_DIR}/${HAILORT_FILE}" "HailoRT wheel"; then
      VERIFICATION_FAILED=true
    fi
  fi
  if [[ "$INSTALL_TAPPAS" == true ]]; then
    if ! verify_wheel_file "${DOWNLOAD_DIR}/${TAPPAS_FILE}" "TAPPAS wheel"; then
      VERIFICATION_FAILED=true
    fi
  fi

  # Verify installed packages (only if not download-only)
  if [[ "$DOWNLOAD_ONLY" != true ]]; then
    log ""
    log "→ Checking installed packages:"
    if [[ "$INSTALL_HAILORT" == true ]]; then
      if ! verify_installation "hailort" "$HAILORT_VERSION" "HailoRT"; then
        VERIFICATION_FAILED=true
      fi
    fi
    if [[ "$INSTALL_TAPPAS" == true ]]; then
      # TAPPAS package name in pip might differ
      if ! verify_installation "hailo-tappas-core-python-binding" "$TAPPAS_CORE_VERSION" "TAPPAS Core"; then
        # Try alternative package name
        verify_installation "hailo_tappas_core_python_binding" "$TAPPAS_CORE_VERSION" "TAPPAS Core" || VERIFICATION_FAILED=true
      fi
    fi

    # Show where packages are installed
    log ""
    log "→ Installation locations:"
    if [[ "$INSTALL_HAILORT" == true ]]; then
      hailort_location=$(python3 -m pip show hailort 2>/dev/null | grep "^Location:" | awk '{print $2}')
      if [[ -n "$hailort_location" ]]; then
        log "  HailoRT: $hailort_location"
      fi
    fi
    if [[ "$INSTALL_TAPPAS" == true ]]; then
      tappas_location=$(python3 -m pip show hailo-tappas-core-python-binding 2>/dev/null | grep "^Location:" | awk '{print $2}')
      if [[ -z "$tappas_location" ]]; then
        tappas_location=$(python3 -m pip show hailo_tappas_core_python_binding 2>/dev/null | grep "^Location:" | awk '{print $2}')
      fi
      if [[ -n "$tappas_location" ]]; then
        log "  TAPPAS Core: $tappas_location"
      fi
    fi
  fi

  log "────────────────────────────────────────────────────────────────"
  log ""

  if [[ "$VERIFICATION_FAILED" == true ]]; then
    log "⚠️  Installation completed with warnings - some verifications failed"
  else
    log "✅ Installation complete and verified successfully!"
  fi
fi
