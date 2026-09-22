#!/usr/bin/env bash
#===============================================================================
# Hailo Apps Infrastructure - Single-File Installation Script
#===============================================================================
# A self-contained installation script with comprehensive error handling,
# debug logging, and config.yaml integration.
#
# Features:
#   - Loads settings from hailo_apps/config/config.yaml
#   - Comprehensive debug logging with timestamps
#   - Dry-run mode to preview actions
#   - Better error messages with troubleshooting hints
#   - Step-by-step progress tracking
#
# Usage:
#   sudo ./install_simple.sh [OPTIONS]
#
#===============================================================================

set -uo pipefail

# Self-elevate to root if not already running as one, preserving the caller's
# environment (PATH, VIRTUAL_ENV, etc.) via `sudo -E`. This must happen before
# any other sudo call, since environment variables lost to a plain `sudo`
# (without -E) cannot be recovered afterwards. This lets an already-active
# virtual environment (e.g. pre-activated inside the Hailo AI Software Suite
# Docker) remain visible to the installer, without users needing to
# remember to pass `-E` themselves.
# Skip elevation for --help/-h, which doesn't need root.
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    _skip_elevation=false
    for _arg in "$@"; do
        [[ "$_arg" == "-h" || "$_arg" == "--help" ]] && _skip_elevation=true && break
    done
    if [[ "${_skip_elevation}" != true ]]; then
        exec sudo -E -- "$0" "$@"
    fi
    unset _skip_elevation _arg
fi

#===============================================================================
# CONSTANTS
#===============================================================================

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly CONFIG_FILE="${SCRIPT_DIR}/hailo_apps/config/config.yaml"

# TAPPAS Core GStreamer resources (auto-downloaded when missing, unless --skip-gstreamer)
readonly TAPPAS_RESOURCES_VERSION="5.4.0"
readonly TAPPAS_DEB_URL_AMD64="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/installation_files/v${TAPPAS_RESOURCES_VERSION}/hailo-tappas-core_${TAPPAS_RESOURCES_VERSION}_amd64.deb"
readonly TAPPAS_DEB_URL_ARM64="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/installation_files/v${TAPPAS_RESOURCES_VERSION}/hailo-tappas-core_${TAPPAS_RESOURCES_VERSION}_arm64.deb"
readonly TAPPAS_WHL_URL="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/installation_files/v${TAPPAS_RESOURCES_VERSION}/hailo_tappas_core_python_binding-${TAPPAS_RESOURCES_VERSION}-py3-none-any.whl"
readonly TAPPAS_DOWNLOAD_DIR="/tmp/hailo_tappas_download"

# Log file path (not readonly - may be updated if log dir not writable)
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/install_${TIMESTAMP}.log"

# Terminal colors (with fallback for non-color terminals)
if [[ -t 1 ]] && command -v tput &>/dev/null && [[ $(tput colors 2>/dev/null || echo 0) -ge 8 ]]; then
    readonly RED=$'\033[0;31m'
    readonly GREEN=$'\033[0;32m'
    readonly YELLOW=$'\033[1;33m'
    readonly BLUE=$'\033[0;34m'
    readonly CYAN=$'\033[0;36m'
    readonly MAGENTA=$'\033[0;35m'
    readonly BOLD=$'\033[1m'
    readonly DIM=$'\033[2m'
    readonly NC=$'\033[0m'
else
    readonly RED='' GREEN='' YELLOW='' BLUE='' CYAN='' MAGENTA='' BOLD='' DIM='' NC=''
fi

#===============================================================================
# COMMAND LINE OPTIONS
#===============================================================================

DRY_RUN=false
FORCE_CLEANUP=false
NO_INSTALL=false
NO_SYSTEM_PYTHON=false
SKIP_GSTREAMER=false
PYHAILORT_PATH=""
PYTAPPAS_PATH=""

# Reuse PyHailoRT from the HailoRT Docker image when available.
# The HailoRT release container installs hailo_platform in this dedicated venv.
CONTAINER_PYHAILORT_VENV="${CONTAINER_PYHAILORT_VENV:-/local/workspace/hailo_platform_venv}"
CONTAINER_PYHAILORT_SITE_PACKAGES=""
CONTAINER_PYHAILORT_VERSION=""

# Configuration variables (populated from config.yaml)
VENV_NAME=""
DOWNLOAD_GROUP=""
RESOURCES_ROOT=""
RESOURCES_DIRS=""
USE_SYSTEM_SITE_PACKAGES=true
SYSTEM_PACKAGES=()
ENV_FILE=""

# Detected values
ORIGINAL_USER=""
ORIGINAL_GROUP=""
HAILORT_VERSION=""
HAILO_ARCH=""
MODEL_ZOO_VER=""

# Step tracking
CURRENT_STEP=0
TOTAL_STEPS=7
declare -a STEP_RESULTS=()

#===============================================================================
# LOGGING FUNCTIONS
#===============================================================================

# Check if we can write to log directory
LOG_ENABLED=false

# Initialize logging directory
init_logging() {
    # Try to create log directory
    if mkdir -p "${LOG_DIR}" 2>/dev/null; then
        if touch "${LOG_FILE}" 2>/dev/null; then
            LOG_ENABLED=true
            # Create log file header
            {
                echo "========================================"
                echo "Hailo Apps Infrastructure Installation"
                echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
                echo "Script: ${SCRIPT_NAME}"
                echo "User: ${SUDO_USER:-$(whoami)}"
                echo "========================================"
                echo ""
            } >> "${LOG_FILE}" 2>/dev/null
        fi
    fi

    if [[ "${LOG_ENABLED}" != true ]]; then
        # Use a temp file for logging if log dir not writable
        LOG_FILE="/tmp/hailo_install_${TIMESTAMP}.log"
        if touch "${LOG_FILE}" 2>/dev/null; then
            LOG_ENABLED=true
        fi
    fi
}

# Log to file with timestamp
log_to_file() {
    [[ "${LOG_ENABLED}" != true ]] && return 0
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${timestamp}] [${level}] ${message}" >> "${LOG_FILE}" 2>/dev/null || true
}

# Console + file logging functions
log_info() {
    echo -e "${GREEN}ℹ️  $*${NC}"
    log_to_file "INFO" "$*"
}

log_success() {
    echo -e "${GREEN}✅ $*${NC}"
    log_to_file "SUCCESS" "$*"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $*${NC}"
    log_to_file "WARNING" "$*"
}

log_error() {
    echo -e "${RED}❌ $*${NC}" >&2
    log_to_file "ERROR" "$*"
}

log_debug() {
    log_to_file "DEBUG" "$*"
}

log_dry_run() {
    echo -e "${MAGENTA}[DRY-RUN]${NC} Would execute: $*"
    log_to_file "DRY-RUN" "$*"
}

# Step header logging
log_step() {
    local step_num="$1"
    local step_name="$2"
    CURRENT_STEP=$step_num
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}  Step ${step_num}/${TOTAL_STEPS}: ${step_name}${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    log_to_file "STEP" "Step ${step_num}/${TOTAL_STEPS}: ${step_name}"
}

# Record step result
record_step_result() {
    local status="$1"
    local message="${2:-}"
    STEP_RESULTS+=("${status}:${message}")
}

#===============================================================================
# ERROR HANDLING
#===============================================================================

# Trap for unexpected errors
trap_error() {
    local exit_code=$?
    local line_no=${1:-unknown}
    local command="${BASH_COMMAND:-unknown}"

    log_error "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_error "UNEXPECTED ERROR"
    log_error "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_error "Exit code: ${exit_code}"
    log_error "Line: ${line_no}"
    log_error "Command: ${command}"
    log_error ""
    log_error "Check the log file for details: ${LOG_FILE}"
    log_error "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Cleanup if needed
    if [[ -n "${ORIGINAL_USER:-}" ]]; then
        log_debug "Attempting ownership cleanup..."
        fix_ownership "${SCRIPT_DIR}" 2>/dev/null || true
    fi

    exit "${exit_code}"
}

# Enable error trap (after logging is initialized)
enable_error_trap() {
    trap 'trap_error ${LINENO}' ERR
}

# Disable error trap (for sections where we handle errors manually)
disable_error_trap() {
    trap - ERR
}

#===============================================================================
# UTILITY FUNCTIONS
#===============================================================================

# Execute command as the original user (not root)
# Preserves PATH and VIRTUAL_ENV so an already-active virtual environment
# (e.g. pre-activated inside the Hailo AI Software Suite Docker) is still
# visible to the original user's shell instead of being wiped by sudo's
# default env_reset behavior. Run the installer with 'sudo -E ./install.sh'
# to make sure these variables are available to preserve in the first place.
as_original_user() {
    if [[ ${EUID:-$(id -u)} -eq 0 && -n "${SUDO_USER:-}" ]]; then
        log_debug "Running as user ${SUDO_USER}: $*"
        sudo -n -u "$SUDO_USER" -H --preserve-env=PATH,VIRTUAL_ENV -- "$@"
    else
        "$@"
    fi
}

# Extract the version number (e.g. 4.24.0) from a wheel filename such as
# hailort-4.24.0-cp312-cp312-linux_x86_64.whl
extract_wheel_version() {
    local whl="$1"
    local name
    name="$(basename "$whl")"
    if [[ "$name" =~ -([0-9]+\.[0-9]+\.[0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

# Fix ownership of files/directories to original user and group
fix_ownership() {
    local target="$1"
    if [[ -e "$target" && -n "${ORIGINAL_USER:-}" && -n "${ORIGINAL_GROUP:-}" ]]; then
        log_debug "Fixing ownership of ${target} to ${ORIGINAL_USER}:${ORIGINAL_GROUP}"
        chown -R "${ORIGINAL_USER}:${ORIGINAL_GROUP}" "$target" 2>/dev/null || true
    fi
}

# Execute or show what would be executed
run_cmd() {
    if [[ "${DRY_RUN}" == true ]]; then
        log_dry_run "$*"
        return 0
    fi
    log_debug "Executing: $*"
    "$@"
}

# Execute as user or show what would be executed
run_as_user() {
    if [[ "${DRY_RUN}" == true ]]; then
        log_dry_run "(as ${ORIGINAL_USER}) $*"
        return 0
    fi
    log_debug "Executing as ${ORIGINAL_USER}: $*"
    as_original_user "$@"
}

# Check if a command exists
command_exists() {
    command -v "$1" &>/dev/null
}

# Detect PyHailoRT preinstalled in a HailoRT/Suite Docker image.
# The release image may keep hailo_platform in a dedicated venv (commonly at
# /local/workspace/hailo_platform_venv, but not guaranteed), or it may already
# be importable from the currently active environment (e.g. the Suite Docker's
# pre-activated venv, or the container's system python3). Try each in turn.
detect_container_pyhailort() {
    CONTAINER_PYHAILORT_SITE_PACKAGES=""
    CONTAINER_PYHAILORT_VERSION=""

    local candidate_python=""

    # 1) Known/expected dedicated venv locations for the PyHailoRT binding.
    local venv_candidates=(
        "${CONTAINER_PYHAILORT_VENV}"
        "/local/workspace/hailo_platform_venv"
        "/opt/hailo_platform_venv"
        "/root/hailo_platform_venv"
    )
    local venv_dir
    for venv_dir in "${venv_candidates[@]}"; do
        [[ -n "$venv_dir" && -x "${venv_dir}/bin/python3" ]] || continue
        if "${venv_dir}/bin/python3" -c 'import hailo_platform' >/dev/null 2>&1; then
            candidate_python="${venv_dir}/bin/python3"
            break
        fi
    done

    # 2) A dedicated venv wasn't found at a known path — search a shallow
    # depth under common container roots for one named like it.
    if [[ -z "$candidate_python" ]]; then
        local found_dir
        found_dir=$(find /local/workspace /opt /root -maxdepth 3 -type d -iname "hailo_platform_venv" 2>/dev/null | head -1) || true
        if [[ -n "$found_dir" && -x "${found_dir}/bin/python3" ]] \
           && "${found_dir}/bin/python3" -c 'import hailo_platform' >/dev/null 2>&1; then
            candidate_python="${found_dir}/bin/python3"
        fi
    fi

    # 3) Fall back to whatever python3 is already active for the original
    # user (covers an already-active venv, e.g. the Suite Docker, or the
    # container's system python3 having the binding installed directly).
    if [[ -z "$candidate_python" ]] && as_original_user python3 -c 'import hailo_platform' >/dev/null 2>&1; then
        candidate_python="python3"
    fi

    if [[ -z "$candidate_python" ]]; then
        return 1
    fi

    CONTAINER_PYHAILORT_SITE_PACKAGES=$(
        as_original_user "${candidate_python}" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null
    ) || true

    if [[ -z "${CONTAINER_PYHAILORT_SITE_PACKAGES}" \
       || ! -d "${CONTAINER_PYHAILORT_SITE_PACKAGES}" ]]; then
        CONTAINER_PYHAILORT_SITE_PACKAGES=""
        return 1
    fi

    # The PyHailoRT wheel/distribution is named "hailort". If distribution
    # metadata is unavailable, the caller can fall back to the HailoRT .deb
    # version that was already detected.
    CONTAINER_PYHAILORT_VERSION=$(
        as_original_user "${candidate_python}" -c \
            'import importlib.metadata as m; print(m.version("hailort"))' \
            2>/dev/null
    ) || true

    return 0
}

# Validate detected versions against config
validate_versions() {
    local hailort_ver="$1"
    local tappas_ver="$2"
    local hailo_arch_val="${3:-}"
    local valid=true

    # Validate HailoRT version
    if [[ -n "$hailort_ver" && "$hailort_ver" != "-1" && -n "${VALID_HAILORT_VERSIONS:-}" ]]; then
        local found=false
        for valid_ver in $VALID_HAILORT_VERSIONS; do
            if [[ "$hailort_ver" == "$valid_ver"* ]]; then
                found=true
                break
            fi
        done
        if [[ "$found" != true ]]; then
            log_warning "HailoRT version $hailort_ver is not in valid versions: $VALID_HAILORT_VERSIONS"
            valid=false
        fi
    fi

    # Validate TAPPAS version
    if [[ -n "$tappas_ver" && "$tappas_ver" != "-1" && -n "${VALID_TAPPAS_VERSIONS:-}" ]]; then
        local found=false
        for valid_ver in $VALID_TAPPAS_VERSIONS; do
            if [[ "$tappas_ver" == "$valid_ver"* ]]; then
                found=true
                break
            fi
        done
        if [[ "$found" != true ]]; then
            log_warning "TAPPAS version $tappas_ver is not in valid versions: $VALID_TAPPAS_VERSIONS"
            valid=false
        fi
    fi

    # Validate Hailo architecture
    if [[ -n "$hailo_arch_val" && "$hailo_arch_val" != "unknown" && -n "${VALID_HAILO_ARCH:-}" ]]; then
        local found=false
        for valid_arch in $VALID_HAILO_ARCH; do
            if [[ "$hailo_arch_val" == "$valid_arch" ]]; then
                found=true
                break
            fi
        done
        if [[ "$found" != true ]]; then
            log_warning "Hailo architecture $hailo_arch_val is not in valid architectures: $VALID_HAILO_ARCH"
            valid=false
        fi
    fi

    # Validate HailoRT + TAPPAS combination for architecture
    if [[ -n "$hailo_arch_val" && "$hailo_arch_val" != "unknown" \
       && -n "$hailort_ver" && "$hailort_ver" != "-1" \
       && -n "$tappas_ver" && "$tappas_ver" != "-1" \
       && -n "${VALID_COMBINATIONS:-}" ]]; then
        local combo="${hailort_ver}:${tappas_ver}"
        local combo_found=false
        for valid_combo in $VALID_COMBINATIONS; do
            if [[ "$combo" == "$valid_combo" ]]; then
                combo_found=true
                break
            fi
        done
        if [[ "$combo_found" != true ]]; then
            log_warning "HailoRT $hailort_ver + TAPPAS $tappas_ver is not a valid combination for $hailo_arch_val"
            log_warning "Valid combinations for $hailo_arch_val: $VALID_COMBINATIONS"
            valid=false
        else
            log_debug "Version combination $combo is valid for $hailo_arch_val"
        fi
    fi

    if [[ "$valid" != true ]]; then
        log_error "Version validation failed - installation cannot continue"
        log_error "Please ensure all components match the supported versions listed in config.yaml"
        return 1
    fi
    
    return 0
}

# Get Model Zoo version for a given Hailo architecture
# For H10: Derives from HailoRT version (5.1.1 -> v5.1.0, 5.2.0 -> v5.2.0)
# For H8/H8L: Uses static mapping v2.17.0
get_model_zoo_version() {
    local arch="$1"
    local hailort_ver="${HAILORT_VERSION:-}"
    local mz_version=""

    case "$arch" in
        hailo8|hailo8l)
            # H8/H8L: Derive from HailoRT version
            # HailoRT 4.24.x -> Model Zoo v2.19.0
            # HailoRT 4.23.x (default) -> Model Zoo v2.18.0
            if [[ "$hailort_ver" == 4.24.* ]]; then
                mz_version="v2.19.0"
            else
                mz_version="v2.18.0"
            fi
            ;;
        hailo10h)
            # H10: Derive from HailoRT version
            # HailoRT 5.4.x -> Model Zoo v5.4.0
            # HailoRT 5.3.x -> Model Zoo v5.3.0
            # HailoRT 5.2.x -> Model Zoo v5.2.0
            # HailoRT 5.1.x (default) -> Model Zoo v5.1.0
            if [[ "$hailort_ver" == 5.4.* ]]; then
                mz_version="v5.4.0"
            elif [[ "$hailort_ver" == 5.3.* ]]; then
                mz_version="v5.3.0"
            elif [[ "$hailort_ver" == 5.2.* ]]; then
                mz_version="v5.2.0"
            else
                mz_version="v5.1.0"
            fi
            ;;
        *)
            mz_version=""
            ;;
    esac

    echo "$mz_version"
}

# Validate Model Zoo version for architecture
validate_model_zoo_version() {
    local arch="$1"
    local mz_version="$2"
    local valid=true

    if [[ -z "$mz_version" || -z "$arch" ]]; then
        return 0
    fi

    # Determine which valid versions to check based on architecture
    local valid_versions=""
    case "$arch" in
        hailo8|hailo8l)
            valid_versions="${VALID_MZ_H8_VERSIONS:-v2.17.0}"
            ;;
        hailo10h)
            valid_versions="${VALID_MZ_H10_VERSIONS:-v5.1.0}"
            ;;
    esac

    if [[ -n "$valid_versions" ]]; then
        local found=false
        for valid_ver in $valid_versions; do
            if [[ "$mz_version" == "$valid_ver" ]]; then
                found=true
                break
            fi
        done
        if [[ "$found" != true ]]; then
            log_warning "Model Zoo version $mz_version is not valid for $arch (valid: $valid_versions)"
            valid=false
        fi
    fi

    if [[ "$valid" != true ]]; then
        return 1
    fi
    return 0
}

# Safe file removal with ownership fix fallback
safe_remove() {
    local target="$1"
    if [[ ! -e "$target" ]]; then
        return 0
    fi

    log_debug "Removing: ${target}"

    if run_as_user rm -rf "${target}" 2>/dev/null; then
        return 0
    fi

    log_debug "Regular removal failed, fixing ownership first..."
    fix_ownership "${target}"
    run_as_user rm -rf "${target}"
}

#===============================================================================
# CONFIGURATION LOADING (Pure Bash YAML Parser)
#===============================================================================

# Get a simple YAML value: yaml_get "key" or yaml_get "section.key"
# Handles: key: value, key: "value", key: 'value'
yaml_get() {
    local key="$1"
    local file="$2"
    local value=""

    if [[ "$key" == *.* ]]; then
        # Nested key like "venv.name" - find section then key
        local section="${key%%.*}"
        local subkey="${key#*.}"

        # Find section and get indented value using sed
        value=$(sed -n "/^${section}:/,/^[a-zA-Z_]/{ /^  ${subkey}:/{ s/^  ${subkey}:[ \t]*//; s/^[\"\x27]//; s/[\"\x27]$//; s/[ \t]*#.*$//; p; q; }}" "$file")
    else
        # Simple key at root level
        value=$(sed -n "/^${key}:/{ s/^${key}:[ \t]*//; s/^[\"\x27]//; s/[\"\x27]$//; s/[ \t]*#.*$//; p; q; }" "$file")
    fi

    echo "$value"
}

# Get a YAML list as space-separated values: yaml_get_list "section.key"
yaml_get_list() {
    local key="$1"
    local file="$2"
    local result=""

    if [[ "$key" == *.* ]]; then
        local section="${key%%.*}"
        local subkey="${key#*.}"

        # Extract list items from nested section
        result=$(sed -n "/^${section}:/,/^[a-zA-Z_]/{
            /^  ${subkey}:/,/^  [a-zA-Z_]/{
                /^    - /{
                    s/^    - //
                    s/^[\"\x27]//
                    s/[\"\x27]$//
                    s/[ \t]*#.*$//
                    p
                }
            }
        }" "$file" | tr '\n' ' ')
    else
        # List at root level
        result=$(sed -n "/^${key}:/,/^[a-zA-Z_]/{
            /^  - /{
                s/^  - //
                s/^[\"\x27]//
                s/[\"\x27]$//
                s/[ \t]*#.*$//
                p
            }
        }" "$file" | tr '\n' ' ')
    fi

    # Trim trailing space
    echo "${result% }"
}

# Get a YAML mapping section as "key=value" pairs
yaml_get_mapping() {
    local section="$1"
    local file="$2"

    sed -n "/^${section}:/,/^[a-zA-Z_]/{
        /^  [a-zA-Z0-9_]*:/{
            s/^  \([a-zA-Z0-9_]*\):[ \t]*/\1=/
            s/[\"\x27]//g
            s/[ \t]*#.*$//
            p
        }
    }" "$file" | tr '\n' ' ' | sed 's/ $//'
}

# Load configuration from config.yaml (pure bash - no Python required)
load_config() {
    log_debug "Loading configuration from: ${CONFIG_FILE}"

    if [[ ! -f "${CONFIG_FILE}" ]]; then
        log_error "Config file not found: ${CONFIG_FILE}"
        log_error "The config.yaml file is required for installation."
        return 1
    fi

    # Parse YAML config using bash
    log_debug "Parsing config.yaml..."

    # Extract venv settings (CLI args take precedence over config values)
    VENV_NAME="${VENV_NAME:-$(yaml_get "venv.name" "${CONFIG_FILE}")}"
    local cfg_use_system_site_packages
    cfg_use_system_site_packages=$(yaml_get "venv.use_system_site_packages" "${CONFIG_FILE}")

    # Handle boolean for use_system_site_packages (only if not already set by CLI)
    if [[ -z "${USE_SYSTEM_SITE_PACKAGES}" ]]; then
        case "${cfg_use_system_site_packages,,}" in
            true|yes|1) USE_SYSTEM_SITE_PACKAGES=true ;;
            false|no|0) USE_SYSTEM_SITE_PACKAGES=false ;;
            *) USE_SYSTEM_SITE_PACKAGES=true ;;
        esac
    fi

    # Extract resources settings (CLI args take precedence over config values)
    RESOURCES_ROOT=$(yaml_get "resources.root" "${CONFIG_FILE}")
    RESOURCES_SYMLINK_NAME=$(yaml_get "resources.path" "${CONFIG_FILE}")
    DOWNLOAD_GROUP="${DOWNLOAD_GROUP:-$(yaml_get "resources.download_group" "${CONFIG_FILE}")}"
    ENV_FILE=$(yaml_get "resources.env_file" "${CONFIG_FILE}")

    # Extract system packages (list)
    local cfg_system_packages
    cfg_system_packages=$(yaml_get_list "system_packages" "${CONFIG_FILE}")
    IFS=' ' read -r -a SYSTEM_PACKAGES <<< "$cfg_system_packages"

    # Extract resource directories to create (list)
    RESOURCES_DIRS=$(yaml_get_list "resources.dirs" "${CONFIG_FILE}")

    # Extract valid versions
    VALID_HAILORT_VERSIONS=$(yaml_get_list "valid_versions.hailort" "${CONFIG_FILE}")
    VALID_TAPPAS_VERSIONS=$(yaml_get_list "valid_versions.tappas" "${CONFIG_FILE}")
    VALID_HAILO_ARCH=$(yaml_get_list "valid_versions.hailo_arch" "${CONFIG_FILE}")
    VALID_HOST_ARCH=$(yaml_get_list "valid_versions.host_arch" "${CONFIG_FILE}")

    # Extract model zoo mapping
    MODEL_ZOO_MAPPING=$(yaml_get_mapping "model_zoo_mapping" "${CONFIG_FILE}")

    # Extract valid model zoo versions
    VALID_MZ_H8_VERSIONS=$(yaml_get_list "valid_model_zoo_versions.h8" "${CONFIG_FILE}")
    VALID_MZ_H10_VERSIONS=$(yaml_get_list "valid_model_zoo_versions.h10" "${CONFIG_FILE}")

    # Extract valid version combinations per architecture (compact "hailort:tappas" format)
    # For the detected arch, load the combo string (e.g. "5.2.0:5.2.0 5.3.0:5.3.0")
    # This is loaded later in check_prerequisites once arch is known
    VALID_COMBINATIONS_HAILO8=$(yaml_get "valid_combinations.hailo8" "${CONFIG_FILE}")
    VALID_COMBINATIONS_HAILO8L=$(yaml_get "valid_combinations.hailo8l" "${CONFIG_FILE}")
    VALID_COMBINATIONS_HAILO10H=$(yaml_get "valid_combinations.hailo10h" "${CONFIG_FILE}")

    log_success "Configuration loaded from config.yaml"
    log_debug "  VENV_NAME=${VENV_NAME}"
    log_debug "  USE_SYSTEM_SITE_PACKAGES=${USE_SYSTEM_SITE_PACKAGES}"
    log_debug "  RESOURCES_ROOT=${RESOURCES_ROOT}"
    log_debug "  RESOURCES_SYMLINK_NAME=${RESOURCES_SYMLINK_NAME}"
    log_debug "  DOWNLOAD_GROUP=${DOWNLOAD_GROUP}"
    log_debug "  ENV_FILE=${ENV_FILE}"
    log_debug "  SYSTEM_PACKAGES=${SYSTEM_PACKAGES[*]}"
    log_debug "  VALID_HAILORT_VERSIONS=${VALID_HAILORT_VERSIONS}"
    log_debug "  VALID_TAPPAS_VERSIONS=${VALID_TAPPAS_VERSIONS}"
    log_debug "  VALID_HAILO_ARCH=${VALID_HAILO_ARCH}"
    log_debug "  MODEL_ZOO_MAPPING=${MODEL_ZOO_MAPPING}"

    return 0
}

#===============================================================================
# SHOW HELP
#===============================================================================

show_help() {
    cat << EOF
${BOLD}Hailo Apps Infrastructure - Single-File Installer${NC}

${BOLD}USAGE:${NC}
    ./$SCRIPT_NAME [OPTIONS]

    The script self-elevates with 'sudo -E' if not already run as root,
    preserving your environment (e.g. an active virtual environment).
    Running it directly with 'sudo $SCRIPT_NAME' also works.

${BOLD}OPTIONS:${NC}
    -n, --venv-name NAME        Virtual environment name (default: from config or venv_hailo_apps)
    -ph, --pyhailort PATH       Path to custom PyHailoRT wheel file
    -pt, --pytappas PATH        Path to custom PyTappas wheel file
    --all                       Download all available models/resources
    -x, --no-install            Skip Python package installation
    --no-system-python          Don't use system site-packages in venv
    --skip-gstreamer             Skip TAPPAS checks, Python TAPPAS install, compile, and post_install
                                (downloads resources directly, no C++ compilation)
    --dry-run                   Show what would be done without executing
    --force-cleanup             Run cleanup script before installation (removes venv,
                                build caches, and resources — useful for upgrades)
    -h, --help                  Show this help message

${BOLD}CONFIGURATION:${NC}
    Settings are loaded from: ${CONFIG_FILE}
    CLI arguments override config file values.

${BOLD}EXAMPLES:${NC}
    ./$SCRIPT_NAME                              # Standard installation
    ./$SCRIPT_NAME --dry-run                    # Preview what would be done
    ./$SCRIPT_NAME --all                        # Install with all models
    ./$SCRIPT_NAME -x                           # Skip Python package installation
    ./$SCRIPT_NAME -n my_venv --all             # Custom venv name + all models
    ./$SCRIPT_NAME --force-cleanup              # Clean stale artifacts then install

${BOLD}LOG FILES:${NC}
    Installation logs: ${LOG_DIR}/
    Current session:   ${LOG_FILE}

${BOLD}REQUIREMENTS:${NC}
    - Must be run with sudo (not as root directly)
    - Hailo PCI driver must be installed (.deb)
    - HailoRT must be installed (.deb)
    - HailoRT Python binding must be available either from the HailoRT container
      environment (${CONTAINER_PYHAILORT_VENV}) or from a supplied .whl

    Unless --skip-gstreamer is passed, TAPPAS Core (.deb) and its Python
    binding (.whl) are downloaded and installed automatically (v${TAPPAS_RESOURCES_VERSION})
    if not already present. Use --pytappas to supply a custom wheel instead.

    Download the Hailo driver and HailoRT packages from the Hailo Developer Zone:
    https://hailo.ai/developer-zone/

EOF
}

#===============================================================================
# PARSE ARGUMENTS
#===============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -n|--venv-name)
                VENV_NAME="$2"
                shift 2
                ;;
            -ph|--pyhailort)
                PYHAILORT_PATH="$2"
                shift 2
                ;;
            -pt|--pytappas)
                PYTAPPAS_PATH="$2"
                shift 2
                ;;
            --all)
                DOWNLOAD_GROUP="all"
                shift
                ;;
            -x|--no-install)
                NO_INSTALL=true
                shift
                ;;
            --no-system-python)
                NO_SYSTEM_PYTHON=true
                USE_SYSTEM_SITE_PACKAGES=false
                shift
                ;;
            --skip-gstreamer)
                SKIP_GSTREAMER=true
                shift
                ;;
            --force-cleanup)
                FORCE_CLEANUP=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information."
                exit 1
                ;;
        esac
    done
}

#===============================================================================
# STEP 1: USER DETECTION
#===============================================================================

detect_user_and_group() {
    log_step 1 "User Detection"

    # Check if running with sudo
    if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
        log_error "This script requires sudo privileges"
        echo ""
        echo "Please run with: sudo $SCRIPT_NAME $*"
        record_step_result "FAILED" "Not running as root"
        return 1
    fi

    # Check if running as root directly (not via sudo)
    if [[ -z "${SUDO_USER:-}" ]]; then
        log_error "This script must be run as a regular user (it self-elevates via sudo), not as root directly"
        echo ""
        echo "Please run with: ./$SCRIPT_NAME"
        echo "Do not use: su -c or login as root"
        record_step_result "FAILED" "Running as root directly"
        return 1
    fi

    ORIGINAL_USER="${SUDO_USER}"
    ORIGINAL_GROUP="$(id -gn "${SUDO_USER}")"

    log_success "Detected user: ${ORIGINAL_USER}"
    log_success "Detected primary group: ${ORIGINAL_GROUP}"

    if [[ "${ORIGINAL_USER}" == "${ORIGINAL_GROUP}" ]]; then
        log_debug "User's primary group matches username"
    else
        log_info "User's primary group differs from username: ${ORIGINAL_GROUP}"
    fi

    export ORIGINAL_USER ORIGINAL_GROUP
    record_step_result "SUCCESS" "User: ${ORIGINAL_USER}, Group: ${ORIGINAL_GROUP}"
    return 0
}

#===============================================================================
# TAPPAS GSTREAMER RESOURCES (auto-download)
#===============================================================================

# Download a file from a URL to a destination path using curl or wget
download_file() {
    local url="$1"
    local dest="$2"

    if command_exists curl; then
        curl -fsSL --retry 3 --connect-timeout 15 -o "$dest" "$url"
    elif command_exists wget; then
        wget -q --tries=3 --timeout=15 -O "$dest" "$url"
    else
        log_error "Neither curl nor wget is available to download files"
        return 1
    fi
}

# Download and install the TAPPAS Core .deb (matching host architecture) and
# the TAPPAS Core Python binding .whl, unless already installed/provided.
ensure_gstreamer_resources() {
    if [[ "${SKIP_GSTREAMER}" == true ]]; then
        log_debug "Skipping TAPPAS GStreamer resources (--skip-gstreamer)"
        return 0
    fi

    log_info "Checking TAPPAS GStreamer resources (v${TAPPAS_RESOURCES_VERSION})..."

    # Detect architecture (amd64/arm64) for the correct .deb
    local arch
    arch="$(dpkg --print-architecture 2>/dev/null || true)"
    if [[ -z "$arch" ]]; then
        case "$(uname -m)" in
            x86_64) arch="amd64" ;;
            aarch64|arm64) arch="arm64" ;;
            *) arch="$(uname -m)" ;;
        esac
    fi

    # --- TAPPAS Core .deb ---
    # Check dpkg first (native .deb install), then fall back to pkg-config
    # (e.g. the Hailo AI Software Suite Docker builds/registers TAPPAS Core
    # without a dpkg entry).
    local tappas_found=false
    if dpkg -l 2>/dev/null | grep -qE "^ii\s+(hailo-apps-core|hailo-tappas-core|hailo-tappas|tappas-core|tappas)\b"; then
        tappas_found=true
    elif command_exists pkg-config; then
        for pc in hailo-apps-core hailo-tappas-core hailo_tappas tappas-core tappas; do
            if pkg-config --exists "$pc" 2>/dev/null; then
                tappas_found=true
                break
            fi
        done
    fi

    if [[ "${tappas_found}" == true ]]; then
        log_success "TAPPAS Core already installed, skipping download"
    else
        local deb_url=""
        case "$arch" in
            amd64) deb_url="${TAPPAS_DEB_URL_AMD64}" ;;
            arm64) deb_url="${TAPPAS_DEB_URL_ARM64}" ;;
            *)
                log_error "No TAPPAS Core package available for architecture: ${arch}"
                log_error "Install hailo-tappas-core manually or use --skip-gstreamer"
                return 1
                ;;
        esac

        local deb_path="${TAPPAS_DOWNLOAD_DIR}/$(basename "$deb_url")"
        mkdir -p "${TAPPAS_DOWNLOAD_DIR}"

        log_info "Downloading TAPPAS Core (.deb, ${arch})..."
        if ! download_file "$deb_url" "$deb_path"; then
            log_error "Failed to download TAPPAS Core package: ${deb_url}"
            return 1
        fi

        log_info "Installing TAPPAS Core package"
        apt-get update -qq 2>/dev/null || log_warning "apt-get update had warnings (continuing anyway)"
        if ! apt-get install -y "${deb_path}"; then
            log_error "Failed to install TAPPAS Core package: ${deb_path}"
            return 1
        fi
        log_success "TAPPAS Core installed"
    fi

    # --- TAPPAS Core Python binding .whl ---
    if [[ -n "${PYTAPPAS_PATH}" ]]; then
        log_debug "Custom PyTappas wheel provided (--pytappas ${PYTAPPAS_PATH}), skipping auto-download"
        return 0
    fi

    # Skip the download if the binding is already importable in the original
    # user's Python environment (e.g. pre-installed in the Suite Docker's
    # active venv). Downloading a newer wheel here would otherwise create a
    # version mismatch against the already-installed TAPPAS Core package.
    if as_original_user python3 -c 'import hailo_platform' >/dev/null 2>&1; then
        log_success "TAPPAS Core Python binding already importable (hailo_platform), skipping download"
        return 0
    fi

    local whl_path="${TAPPAS_DOWNLOAD_DIR}/$(basename "${TAPPAS_WHL_URL}")"
    mkdir -p "${TAPPAS_DOWNLOAD_DIR}"

    log_info "Downloading TAPPAS Core Python binding (.whl)..."
    if ! download_file "${TAPPAS_WHL_URL}" "$whl_path"; then
        log_error "Failed to download TAPPAS Core Python binding: ${TAPPAS_WHL_URL}"
        return 1
    fi

    # Make readable by the original (non-root) user for the later pip install
    chown "${ORIGINAL_USER}:${ORIGINAL_GROUP}" "$whl_path" 2>/dev/null || true
    PYTAPPAS_PATH="$whl_path"
    log_success "TAPPAS Core Python binding downloaded to ${whl_path}"

    return 0
}

#===============================================================================
# STEP 2: PREREQUISITES CHECK
#===============================================================================

check_prerequisites() {
    log_step 2 "Prerequisites Check"

    local check_script="${SCRIPT_DIR}/scripts/check_installed_packages.sh"

    if [[ ! -f "$check_script" ]]; then
        log_error "Prerequisites check script not found: $check_script"
        record_step_result "FAILED" "Check script missing"
        return 1
    fi

    if [[ "${DRY_RUN}" == true ]]; then
        log_dry_run "Running: ${check_script}"
        log_info "Would check: Hailo driver, HailoRT, TAPPAS, Python bindings"
        log_dry_run "Would download/install TAPPAS Core .deb and Python binding .whl if missing (v${TAPPAS_RESOURCES_VERSION})"
        record_step_result "SKIPPED" "Dry-run mode"
        return 0
    fi

    # Auto-download and install TAPPAS Core (.deb) and Python binding (.whl)
    # if not already present, before checking installed versions below.
    if ! ensure_gstreamer_resources; then
        record_step_result "FAILED" "TAPPAS GStreamer resources setup failed"
        return 1
    fi

    # --- Get installed driver versions from dpkg (always available) ---
    local pcie_driver_version="-1"
    local usb_driver_version="-1"
    local hailort_version="-1"
    local pyhailort_version="-1"
    local tappas_version="-1"
    local tappas_python_version="-1"

    local _v
    _v=$(dpkg-query -W -f='${Status} ${Version}' hailort-pcie-driver 2>/dev/null) || true
    [[ "$_v" == install\ ok\ installed\ * ]] && pcie_driver_version="${_v##* }"

    _v=$(dpkg-query -W -f='${Status} ${Version}' hailort-usb-driver 2>/dev/null) || true
    [[ "$_v" == install\ ok\ installed\ * ]] && usb_driver_version="${_v##* }"

    _v=$(dpkg-query -W -f='${Status} ${Version}' hailort 2>/dev/null) || true
    if [[ "$_v" == install\ ok\ installed\ * ]]; then
        hailort_version="${_v##* }"
        HAILORT_VERSION="$hailort_version"
    fi

    # --- Check 1: hailortcli scan — is any Hailo device physically present? ---
    log_info "Checking for connected Hailo device..."
    local scan_pci_device=""
    local scan_usb_device=""
    disable_error_trap
    if command -v hailortcli >/dev/null 2>&1; then
        local scan_output=""
        scan_output=$(hailortcli scan 2>/dev/null) || true
        # HailoRT 5.x: "pci/0000:04:00.0" or "usb/002:016"
        # HailoRT 4.x: "0000:04:00.0" (bare PCI BDF, no prefix)
        scan_pci_device=$(echo "$scan_output" | grep -oE 'pci/[^ ]+' | head -1) || true
        scan_usb_device=$(echo "$scan_output" | grep -oE 'usb/[^ ]+' | head -1) || true
        if [[ -z "$scan_pci_device" ]]; then
            scan_pci_device=$(echo "$scan_output" | grep -oE '[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]' | head -1) || true
        fi
    fi
    local device_on_bus=false
    [[ -n "$scan_pci_device" || -n "$scan_usb_device" ]] && device_on_bus=true

    # --- Check 2: hailortcli fw-control identify — can HailoRT communicate? ---
    local identify_output=""
    if [[ "$device_on_bus" == true ]] && command -v hailortcli >/dev/null 2>&1; then
        identify_output=$(hailortcli fw-control identify 2>/dev/null | tr -d '\000') || true
    fi

    enable_error_trap

    # =========================================================================
    # CASE C: Device found by scan AND identify succeeded
    # =========================================================================
    if [[ -n "$identify_output" ]]; then
        # Parse arch from identify output
        if echo "$identify_output" | grep -qi "HAILO8L"; then
            HAILO_ARCH="hailo8l"
        elif echo "$identify_output" | grep -qi "HAILO8"; then
            HAILO_ARCH="hailo8"
        elif echo "$identify_output" | grep -qiE "HAILO10H|HAILO15H"; then
            HAILO_ARCH="hailo10h"
        fi

        # Select arch-appropriate driver type and version
        local driver_type="" driver_version="-1"
        if [[ "$HAILO_ARCH" == "hailo10h" ]]; then
            if [[ "$usb_driver_version" != "-1" ]]; then
                driver_type="USB"; driver_version="$usb_driver_version"
            elif [[ "$pcie_driver_version" != "-1" ]]; then
                driver_type="PCIe"; driver_version="$pcie_driver_version"
            fi
        else
            if [[ "$pcie_driver_version" != "-1" ]]; then
                driver_type="PCIe"; driver_version="$pcie_driver_version"
            elif [[ "$usb_driver_version" != "-1" ]]; then
                driver_type="USB"; driver_version="$usb_driver_version"
            fi
        fi

        log_success "Hailo device detected (scan: ${scan_usb_device:-$scan_pci_device})"
        log_success "HailoRT ${hailort_version} identified device successfully"

        # Get TAPPAS and Python binding versions from check script
        local summary_line
        disable_error_trap
        summary_line=$(as_original_user "$check_script" 2>&1 | sed -n 's/^SUMMARY: //p')
        enable_error_trap
        for pair in $summary_line; do
            local key="${pair%%=*}" value="${pair#*=}"
            case "$key" in
                tappas-core)   tappas_version="$value" ;;
                pyhailort)     pyhailort_version="$value" ;;
                tappas-python) tappas_python_version="$value" ;;
            esac
        done

        # If the user supplied an explicit wheel via --pyhailort/--pytappas, that
        # wheel is what will actually be installed later (Step 6), so its version
        # takes precedence over whatever pyhailort/tappas-python happens to be
        # detected right now (which may be stale or absent since the venv/wheel
        # install hasn't happened yet at this point in the flow).
        if [[ -n "${PYHAILORT_PATH}" ]]; then
            local whl_ver
            whl_ver="$(extract_wheel_version "${PYHAILORT_PATH}")"
            if [[ -n "$whl_ver" ]]; then
                pyhailort_version="$whl_ver"
            fi
        fi
        if [[ -n "${PYTAPPAS_PATH}" ]]; then
            local whl_ver
            whl_ver="$(extract_wheel_version "${PYTAPPAS_PATH}")"
            if [[ -n "$whl_ver" ]]; then
                tappas_python_version="$whl_ver"
            fi
        fi

        # Load valid combinations for detected arch
        MODEL_ZOO_VER=$(get_model_zoo_version "${HAILO_ARCH}")
        case "${HAILO_ARCH}" in
            hailo8)   VALID_COMBINATIONS="${VALID_COMBINATIONS_HAILO8:-}" ;;
            hailo8l)  VALID_COMBINATIONS="${VALID_COMBINATIONS_HAILO8L:-}" ;;
            hailo10h) VALID_COMBINATIONS="${VALID_COMBINATIONS_HAILO10H:-}" ;;
            *) VALID_COMBINATIONS="" ;;
        esac

        log_info "Detected versions:"
        log_info "  Driver: ${driver_type} ${driver_version}"
        log_info "  HailoRT: ${hailort_version}"
        if [[ "${SKIP_GSTREAMER}" == true ]]; then
            log_info "  TAPPAS: skipped (--skip-gstreamer)"
        else
            log_info "  TAPPAS: ${tappas_version}"
        fi
        [[ -n "${MODEL_ZOO_VER}" ]] && log_info "  Model Zoo Version: ${MODEL_ZOO_VER} (for ${HAILO_ARCH})"

        # Validate TAPPAS combo
        local failed=false
        if [[ "${SKIP_GSTREAMER}" == true ]]; then
            validate_versions "$hailort_version" "-1" "${HAILO_ARCH}" || failed=true
        else
            validate_versions "$hailort_version" "$tappas_version" "${HAILO_ARCH}" || failed=true
        fi
        [[ -n "${MODEL_ZOO_VER}" ]] && validate_model_zoo_version "${HAILO_ARCH}" "${MODEL_ZOO_VER}" || true

        # Reuse PyHailoRT from the HailoRT Docker image when no explicit wheel
        # was supplied. The release image keeps hailo_platform in a dedicated
        # venv, so --system-site-packages alone cannot expose it to our new venv.
        if [[ -z "${PYHAILORT_PATH}" ]] && detect_container_pyhailort; then
            log_success "Found existing PyHailoRT in ${CONTAINER_PYHAILORT_VENV}"
            log_debug "PyHailoRT site-packages: ${CONTAINER_PYHAILORT_SITE_PACKAGES}"

            if [[ "$pyhailort_version" == "-1" ]]; then
                if [[ -n "${CONTAINER_PYHAILORT_VERSION}" ]]; then
                    pyhailort_version="${CONTAINER_PYHAILORT_VERSION}"
                else
                    # The container is a versioned HailoRT release image, so if
                    # the Python distribution metadata is unavailable, use the
                    # detected HailoRT package version for compatibility checks.
                    pyhailort_version="${hailort_version}"
                fi
            fi
        fi

        # Check required components
        local missing_components=()
        [[ "$pyhailort_version" == "-1" && -z "${PYHAILORT_PATH}" ]] && missing_components+=("HailoRT Python binding")
        if [[ "${SKIP_GSTREAMER}" != true ]]; then
            [[ "$tappas_version" == "-1" ]] && missing_components+=("TAPPAS Core (.deb)")
            [[ "$tappas_python_version" == "-1" && -z "${PYTAPPAS_PATH}" ]] && missing_components+=("TAPPAS Core Python binding (.whl)")
        fi
        if [[ ${#missing_components[@]} -gt 0 ]]; then
            log_error "Missing required components:"
            for c in "${missing_components[@]}"; do log_error "  • ${c}"; done
            failed=true
        fi

        # Cross-check: deb and Python wheel versions should match
        if [[ "$hailort_version" != "-1" && "$pyhailort_version" != "-1" \
           && "$hailort_version" != "$pyhailort_version" ]]; then
            log_warning "HailoRT version mismatch: system package=$hailort_version, Python wheel=$pyhailort_version"
            log_warning "The HailoRT deb and Python binding should have matching versions."
            failed=true
        fi
        if [[ "${SKIP_GSTREAMER}" != true \
           && "$tappas_version" != "-1" && "$tappas_python_version" != "-1" \
           && "$tappas_version" != "$tappas_python_version" ]]; then
            log_warning "TAPPAS version mismatch: system package=$tappas_version, Python wheel=$tappas_python_version"
            log_warning "The tappas-core deb and Python binding should have matching versions."
            failed=true
        fi

        if [[ "$failed" == true ]]; then
            record_step_result "FAILED" "Version validation failed"
            return 1
        fi

        log_success "Prerequisites check passed"
        record_step_result "SUCCESS" "All required components found"
        return 0
    fi

    # =========================================================================
    # CASE N8: Device found by scan but identify failed
    # =========================================================================
    if [[ "$device_on_bus" == true ]]; then
        local found_device found_type found_driver_ver
        if [[ -n "$scan_usb_device" ]]; then
            found_device="$scan_usb_device"; found_type="USB"; found_driver_ver="$usb_driver_version"
        else
            found_device="$scan_pci_device"; found_type="PCIe"; found_driver_ver="$pcie_driver_version"
        fi

        log_warning "Hailo ${found_type} device found (${found_device}) but not recognized by HailoRT"

        if [[ "$found_driver_ver" != "-1" && "$hailort_version" != "-1" && \
              "$found_driver_ver" != "$hailort_version" ]]; then
            log_error "HailoRT ${hailort_version} does not match ${found_type} driver ${found_driver_ver}"
            log_error "Reinstall matching packages"
        else
            log_error "Driver version matches but device not responding"
            log_error "Reconnect device or reboot and retry"
        fi

        record_step_result "FAILED" "Device not recognized by HailoRT"
        return 1
    fi

    # =========================================================================
    # CASE N: Scan found nothing — software check only
    # =========================================================================
    log_error "No Hailo device detected (scan found no devices)"
    log_warning "Software check only:"

    local any_driver_installed=false
    local inferred_arch="" inferred_device_name=""

    # Show driver status and infer arch
    if [[ "$pcie_driver_version" != "-1" && "$usb_driver_version" != "-1" ]]; then
        log_info "Driver: PCIe ${pcie_driver_version}, USB ${usb_driver_version}"
        any_driver_installed=true
        if [[ "$usb_driver_version" == "$hailort_version" ]]; then
            inferred_arch=$(infer_arch_from_version "$usb_driver_version")
        else
            inferred_arch=$(infer_arch_from_version "$pcie_driver_version")
        fi
    elif [[ "$usb_driver_version" != "-1" ]]; then
        log_success "Driver: USB ${usb_driver_version}"
        any_driver_installed=true
        inferred_arch=$(infer_arch_from_version "$usb_driver_version")
    elif [[ "$pcie_driver_version" != "-1" ]]; then
        log_success "Driver: PCIe ${pcie_driver_version}"
        any_driver_installed=true
        inferred_arch=$(infer_arch_from_version "$pcie_driver_version")
    else
        log_error "No Hailo driver installed"
    fi

    # Map inferred arch to friendly device name
    case "$inferred_arch" in
        hailo10h) inferred_device_name="Hailo-10H" ;;
        hailo8l)  inferred_device_name="Hailo-8L" ;;
        hailo8)   inferred_device_name="Hailo-8" ;;
    esac

    # Show HailoRT status and check version match against the matching driver
    local matched_driver_type="" matched_driver_ver="-1"
    if [[ "$usb_driver_version" != "-1" && "$hailort_version" == "$usb_driver_version" ]]; then
        matched_driver_type="USB"; matched_driver_ver="$usb_driver_version"
    elif [[ "$pcie_driver_version" != "-1" && "$hailort_version" == "$pcie_driver_version" ]]; then
        matched_driver_type="PCIe"; matched_driver_ver="$pcie_driver_version"
    fi

    if [[ "$hailort_version" != "-1" ]]; then
        if [[ "$any_driver_installed" == true && -z "$matched_driver_type" ]]; then
            local mismatch_ver="${usb_driver_version}"
            [[ "$mismatch_ver" == "-1" ]] && mismatch_ver="$pcie_driver_version"
            log_error "HailoRT: ${hailort_version} (does not match driver ${mismatch_ver})"
        else
            log_success "HailoRT: ${hailort_version} (matches ${matched_driver_type} driver)"
        fi
    else
        log_error "HailoRT not installed"
    fi

    # Final error message — include connection type if known
    local connection_type=""
    [[ "$matched_driver_type" == "USB" ]] && connection_type=" USB"
    [[ "$matched_driver_type" == "PCIe" ]] && connection_type=" PCIe"

    if [[ -n "$inferred_device_name" ]]; then
        log_error "No ${inferred_device_name}${connection_type} device detected — reconnect device or reboot and retry"
    else
        log_error "Prerequisites check failed — missing required components"
    fi

    record_step_result "FAILED" "No Hailo device detected"
    return 1
}

# Helper: infer Hailo arch from driver/hailort version string
infer_arch_from_version() {
    local ver="$1"
    if [[ "$ver" == 4.22* || "$ver" == 4.23* || "$ver" == 4.24* ]]; then
        echo "hailo8"
    elif [[ "$ver" == 5.* ]]; then
        echo "hailo10h"
    else
        echo ""
    fi
}

#===============================================================================
# STEP 3: SYSTEM PACKAGES
#===============================================================================

install_system_packages() {
    log_step 3 "System Package Installation"

    if [[ "${DRY_RUN}" == true ]]; then
        log_dry_run "apt-get install -y ${SYSTEM_PACKAGES[*]}"
        record_step_result "SKIPPED" "Dry-run mode"
        return 0
    fi

    log_info "Installing system packages: ${SYSTEM_PACKAGES[*]}"

    # Update apt cache
    log_debug "Updating apt cache..."
    if ! apt-get update -qq 2>/dev/null; then
        log_warning "apt-get update had warnings (continuing anyway)"
    fi

    # Install packages
    local failed_packages=()
    for pkg in "${SYSTEM_PACKAGES[@]}"; do
        log_debug "Installing: ${pkg}"
        if ! apt-get install -y -qq "$pkg" 2>/dev/null; then
            log_warning "Failed to install: ${pkg}"
            failed_packages+=("$pkg")
        fi
    done

    if [[ ${#failed_packages[@]} -gt 0 ]]; then
        log_warning "Some packages failed to install: ${failed_packages[*]}"
        log_info "Installation will continue - these may not be required"
    else
        log_success "System packages installed"
    fi

    record_step_result "SUCCESS" "Packages installed"
    return 0
}

#===============================================================================
# STEP 4: RESOURCES SETUP (before venv so packages dir is available)
#===============================================================================

setup_resources() {
    log_step 4 "Resources Directory Setup"

    if [[ "${DRY_RUN}" == true ]]; then
        log_dry_run "mkdir -p ${RESOURCES_ROOT}/{${RESOURCES_DIRS// /,}}"
        log_dry_run "chown -R ${ORIGINAL_USER}:${ORIGINAL_GROUP} ${RESOURCES_ROOT}"
        log_dry_run "touch ${ENV_FILE}"
        record_step_result "SKIPPED" "Dry-run mode"
        return 0
    fi

    log_info "Creating resources directories at ${RESOURCES_ROOT}..."

    # Create directory structure from config.yaml
    for dir in ${RESOURCES_DIRS}; do
        mkdir -p "${RESOURCES_ROOT}/${dir}"
        log_debug "  Created: ${RESOURCES_ROOT}/${dir}"
    done

    # Set ownership
    chown -R "${ORIGINAL_USER}:${ORIGINAL_GROUP}" "${RESOURCES_ROOT}"

    # Set permissions (775 for group access)
    chmod -R 775 "${RESOURCES_ROOT}"

    # Remove existing .env file
    if [[ -f "${ENV_FILE}" ]]; then
        log_debug "Removing existing .env file"
        safe_remove "${ENV_FILE}"
    fi

    # Create new .env file at resources root
    run_as_user touch "${ENV_FILE}"
    run_as_user chmod 644 "${ENV_FILE}"
    log_debug "Created .env file at ${ENV_FILE}"

    log_success "Resources directories created"
    log_info "  Owner: ${ORIGINAL_USER}:${ORIGINAL_GROUP}"
    log_info "  Location: ${RESOURCES_ROOT}"
    log_info "  Environment file: ${ENV_FILE}"

    record_step_result "SUCCESS" "Resources at ${RESOURCES_ROOT}"
    return 0
}

#===============================================================================
# STEP 5: VIRTUAL ENVIRONMENT SETUP
#===============================================================================

setup_virtual_environment() {
    log_step 5 "Virtual Environment Setup"

    local venv_path="${SCRIPT_DIR}/${VENV_NAME}"

    # Remove existing virtualenv
    if [[ -d "${venv_path}" ]]; then
        log_info "Removing existing virtualenv at ${venv_path}"
        safe_remove "${venv_path}"
    fi

    # Clean up build artifacts
    log_info "Cleaning up build artifacts..."
    disable_error_trap
    run_as_user find "${SCRIPT_DIR}" -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
    run_as_user rm -rf "${SCRIPT_DIR}/build/" "${SCRIPT_DIR}/dist/" 2>/dev/null || true
    enable_error_trap
    log_debug "Build artifacts cleaned"

    # Resolve which python3 interpreter to use for creating the new venv.
    # If a virtualenv is already active (e.g. the Hailo AI Software Suite
    # Docker pre-activates its own venv), building a venv from that
    # interpreter nests it inside the active one, which can produce a
    # broken or incomplete environment. Bypass it and use the underlying
    # base/system interpreter instead.
    local python_bin="python3"
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        log_warning "An active virtual environment was detected: ${VIRTUAL_ENV}"
        local base_prefix
        base_prefix=$(python3 -c 'import sys; print(getattr(sys, "base_prefix", sys.prefix))' 2>/dev/null) || base_prefix=""
        if [[ -n "$base_prefix" && -x "${base_prefix}/bin/python3" ]]; then
            python_bin="${base_prefix}/bin/python3"
            log_info "Using the base system interpreter instead: ${python_bin}"
        else
            log_warning "Could not resolve a base system interpreter; proceeding with 'python3' (may create a nested venv)"
        fi
    fi

    # Create virtual environment
    local venv_args=""
    if [[ "${USE_SYSTEM_SITE_PACKAGES}" == true && "${NO_SYSTEM_PYTHON}" != true ]]; then
        log_info "Creating virtualenv '${VENV_NAME}' (with system site-packages)..."
        venv_args="--system-site-packages"
    else
        log_info "Creating virtualenv '${VENV_NAME}' (without system site-packages)..."
    fi

    if [[ "${DRY_RUN}" == true ]]; then
        log_dry_run "${python_bin} -m venv ${venv_args} '${venv_path}'"
        record_step_result "SKIPPED" "Dry-run mode"
        return 0
    fi

    if ! run_as_user "${python_bin}" -m venv ${venv_args} "${venv_path}"; then
        log_error "Failed to create virtual environment"
        log_info "Troubleshooting:"
        log_info "  • Ensure python3-venv is installed: sudo apt install python3-venv"
        log_info "  • Check disk space: df -h ${SCRIPT_DIR}"
        log_info "  • Check permissions: ls -la ${SCRIPT_DIR}"
        record_step_result "FAILED" "venv creation failed"
        return 1
    fi

    # Verify venv creation
    if [[ ! -f "${venv_path}/bin/activate" ]]; then
        log_error "Virtual environment created but activate script not found"
        log_error "Expected: ${venv_path}/bin/activate"
        record_step_result "FAILED" "activate script missing"
        return 1
    fi

    # If PyHailoRT is provided by the HailoRT release container's dedicated
    # venv, expose that site-packages directory to venv_hailo_apps using a .pth
    # file. This reuses the existing hailo_platform installation without
    # copying or reinstalling the wheel.
    if [[ -n "${CONTAINER_PYHAILORT_SITE_PACKAGES}" && -z "${PYHAILORT_PATH}" ]]; then
        local venv_python="${venv_path}/bin/python3"
        local venv_site_packages=""
        local pyhailort_pth=""

        venv_site_packages=$(
            run_as_user "${venv_python}" -c 'import site; print(site.getsitepackages()[0])'
        ) || true

        if [[ -z "${venv_site_packages}" || ! -d "${venv_site_packages}" ]]; then
            log_error "Could not determine site-packages for ${VENV_NAME}"
            record_step_result "FAILED" "venv site-packages detection failed"
            return 1
        fi

        pyhailort_pth="${venv_site_packages}/hailo_platform_container.pth"
        log_info "Reusing container PyHailoRT from ${CONTAINER_PYHAILORT_SITE_PACKAGES}"

        if ! run_as_user bash -c \
            "printf '%s\n' '${CONTAINER_PYHAILORT_SITE_PACKAGES}' > '${pyhailort_pth}'"; then
            log_error "Failed to expose container PyHailoRT to ${VENV_NAME}"
            record_step_result "FAILED" "PyHailoRT path setup failed"
            return 1
        fi

        if ! run_as_user "${venv_python}" -c 'import hailo_platform' >/dev/null 2>&1; then
            log_error "Container PyHailoRT is not importable from ${VENV_NAME}"
            log_error "Source site-packages: ${CONTAINER_PYHAILORT_SITE_PACKAGES}"
            record_step_result "FAILED" "PyHailoRT import failed in venv"
            return 1
        fi

        log_success "Container PyHailoRT available in ${VENV_NAME}"
    fi

    log_success "Virtual environment created at ${venv_path}"
    record_step_result "SUCCESS" "venv: ${venv_path}"
    return 0
}

#===============================================================================
# STEP 6: PYTHON PACKAGE INSTALLATION
#===============================================================================

install_python_packages() {
    log_step 6 "Python Package Installation"

    local venv_path="${SCRIPT_DIR}/${VENV_NAME}"
    local venv_activate="${venv_path}/bin/activate"

    if [[ "${DRY_RUN}" == true ]]; then
        log_dry_run "source ${venv_activate}"
        log_dry_run "pip install --upgrade pip setuptools wheel"
        [[ -n "$PYHAILORT_PATH" ]] && log_dry_run "pip install '${PYHAILORT_PATH}'"
        if [[ -n "$PYTAPPAS_PATH" && "${SKIP_GSTREAMER}" != true ]]; then
            log_dry_run "pip install '${PYTAPPAS_PATH}'"
        fi
        log_dry_run "pip install -e ."
        record_step_result "SKIPPED" "Dry-run mode"
        return 0
    fi

    # Install custom wheel files if provided
    if [[ -n "$PYHAILORT_PATH" ]]; then
        log_info "Installing custom HailoRT binding: ${PYHAILORT_PATH}"
        if [[ ! -f "$PYHAILORT_PATH" ]]; then
            log_error "HailoRT wheel file not found: ${PYHAILORT_PATH}"
            record_step_result "FAILED" "PyHailoRT wheel not found"
            return 1
        fi
        if ! run_as_user bash -c "source '${venv_activate}' && pip install '${PYHAILORT_PATH}'"; then
            log_error "Failed to install PyHailoRT wheel"
            record_step_result "FAILED" "PyHailoRT install failed"
            return 1
        fi
    fi

    if [[ -n "$PYTAPPAS_PATH" ]]; then
        if [[ "${SKIP_GSTREAMER}" == true ]]; then
            log_warning "Ignoring PyTappas wheel (--skip-gstreamer): ${PYTAPPAS_PATH}"
            PYTAPPAS_PATH=""
        fi
    fi

    if [[ -n "$PYTAPPAS_PATH" ]]; then
        log_info "Installing custom TAPPAS binding: ${PYTAPPAS_PATH}"
        if [[ ! -f "$PYTAPPAS_PATH" ]]; then
            log_error "TAPPAS wheel file not found: ${PYTAPPAS_PATH}"
            record_step_result "FAILED" "PyTappas wheel not found"
            return 1
        fi
        if ! run_as_user bash -c "source '${venv_activate}' && pip install '${PYTAPPAS_PATH}'"; then
            log_error "Failed to install PyTappas wheel"
            record_step_result "FAILED" "PyTappas install failed"
            return 1
        fi
    fi

    # Install Hailo Python packages into venv (only from user-provided wheels).
    # Without --pyhailort, PyHailoRT may come from system site-packages or from
    # the HailoRT release container venv exposed in Step 5 via a .pth file.

    # Upgrade pip/setuptools/wheel
    log_info "Upgrading pip, setuptools, and wheel..."
    if ! run_as_user bash -c "source '${venv_activate}' && python3 -m pip install --upgrade pip setuptools wheel"; then
        log_warning "pip upgrade had issues (continuing anyway)"
    fi

    # Install the hailo_apps package in editable mode
    log_info "Installing hailo_apps package (editable mode)..."
    if ! run_as_user bash -c "source '${venv_activate}' && pip install -e '${SCRIPT_DIR}'"; then
        log_error "Failed to install hailo_apps package"
        log_info "Troubleshooting:"
        log_info "  • Check setup.py or pyproject.toml exists"
        log_info "  • Check for syntax errors in package code"
        log_info "  • Review pip output above for specific errors"
        record_step_result "FAILED" "hailo_apps install failed"
        return 1
    fi

    log_success "Python packages installed"
    record_step_result "SUCCESS" "Packages installed"
    return 0
}

#===============================================================================
# STEP 7: POST-INSTALLATION
#===============================================================================

# Setup resources symlink (from config parameters)
# This creates: ${SCRIPT_DIR}/${RESOURCES_SYMLINK_NAME} -> ${RESOURCES_ROOT}
setup_resources_symlink() {
    local symlink_path="${SCRIPT_DIR}/${RESOURCES_SYMLINK_NAME}"
    local target_path="${RESOURCES_ROOT}"

    log_info "Setting up resources symlink..."
    log_debug "  Symlink: ${symlink_path}"
    log_debug "  Target:  ${target_path}"

    if [[ "${DRY_RUN}" == true ]]; then
        log_dry_run "ln -sf ${target_path} ${symlink_path}"
        return 0
    fi

    # Verify target exists
    if [[ ! -d "${target_path}" ]]; then
        log_error "Resources root does not exist: ${target_path}"
        log_error "Run setup_resources step first or check config.yaml resources.root"
        return 1
    fi

    # Remove existing path if present (symlink, file, or directory)
    if [[ -e "${symlink_path}" || -L "${symlink_path}" ]]; then
        log_debug "Removing existing: ${symlink_path}"
        rm -rf "${symlink_path}"
    fi

    # Create symlink
    ln -s "${target_path}" "${symlink_path}"

    # Set ownership to user (not root)
    chown -h "${ORIGINAL_USER}:${ORIGINAL_GROUP}" "${symlink_path}"

    log_success "Created symlink: ${symlink_path} -> ${target_path}"
    return 0
}

run_post_install() {
    log_step 7 "Post-Installation"

    local venv_path="${SCRIPT_DIR}/${VENV_NAME}"
    local venv_activate="${venv_path}/bin/activate"

    # Fix permissions before running as user
    log_debug "Fixing ownership of project directory..."
    fix_ownership "${SCRIPT_DIR}"
    fix_ownership "${RESOURCES_ROOT}"

    if [[ "${SKIP_GSTREAMER}" == true ]]; then
        log_info "Running minimal post-installation (--skip-gstreamer)"
        
        # Create resources symlink
        if ! setup_resources_symlink; then
            log_error "Failed to create resources symlink"
            record_step_result "FAILED" "Symlink creation failed"
            return 1
        fi
        
        # Build download args
        local download_args=""
        if [[ "$DOWNLOAD_GROUP" == "all" ]]; then
            download_args="--all"
        elif [[ -n "$DOWNLOAD_GROUP" ]]; then
            download_args="--group '${DOWNLOAD_GROUP}'"
        fi
        
        if [[ "${DRY_RUN}" == true ]]; then
            log_dry_run "source ${venv_activate} && hailo-download-resources ${download_args}"
            record_step_result "SKIPPED" "Dry-run mode"
            return 0
        fi
        
        # Run download_resources directly (skip compile and full post_install)
        log_info "Downloading resources (skipping compile and post_install)..."
        log_info "Download group: ${DOWNLOAD_GROUP}"
        
        disable_error_trap
        local download_exit=0
        
        run_as_user bash -c "
            export PYTHONUNBUFFERED=1 && \
            source '${venv_activate}' && \
            cd '${SCRIPT_DIR}' && \
            stdbuf -oL -eL hailo-download-resources ${download_args} 2>&1
        " | while IFS= read -r line; do
            log_debug "  $line"
            echo "$line"
        done
        
        download_exit=${PIPESTATUS[0]}
        enable_error_trap
        
        if [[ $download_exit -ne 0 ]]; then
            log_error "Resource download failed (exit code: ${download_exit})"
            log_info "You can retry manually:"
            log_info "  source ${SCRIPT_DIR}/setup_env.sh"
            log_info "  hailo-download-resources ${download_args}"
            record_step_result "FAILED" "Download failed: ${download_exit}"
            return 1
        fi
        
        log_success "Resources downloaded (no-tappas mode)"
        record_step_result "SUCCESS" "Resources downloaded (no-tappas)"
        return 0
    fi

    # Build post-install command
    local post_install_args="--group '${DOWNLOAD_GROUP}'"
    if [[ "$DOWNLOAD_GROUP" == "all" ]]; then
        post_install_args="--all"
    fi

    if [[ "${DRY_RUN}" == true ]]; then
        log_dry_run "source ${venv_activate} && hailo-post-install ${post_install_args}"
        log_info "Would run as user ${ORIGINAL_USER}:"
        log_info "  • Configure environment variables (.env)"
        log_info "  • Create symlink: ${RESOURCES_SYMLINK_NAME} -> ${RESOURCES_ROOT}"
        log_info "  • Download resources (group: ${DOWNLOAD_GROUP})"
        log_info "  • Compile C++ postprocess modules"
        record_step_result "SKIPPED" "Dry-run mode"
        return 0
    fi

    log_info "Running post-installation as user: ${ORIGINAL_USER}"
    log_info "Download group: ${DOWNLOAD_GROUP}"

    # Setup resources symlink (from config parameters)
    if ! setup_resources_symlink; then
        log_error "Failed to create resources symlink"
        record_step_result "FAILED" "Symlink creation failed"
        return 1
    fi

    log_info "Continue running Post-Installation steps..."

    # Run post-install with real-time output
    disable_error_trap
    local post_install_exit=0

    # Use unbuffered output to ensure real-time progress display
    # PYTHONUNBUFFERED=1 ensures Python outputs are unbuffered (most reliable for Python)
    # stdbuf -oL -eL ensures line-buffered output as fallback (flush on each line)
    run_as_user bash -c "
        export PYTHONUNBUFFERED=1 && \
        source '${venv_activate}' && \
        cd '${SCRIPT_DIR}' && \
        stdbuf -oL -eL hailo-post-install ${post_install_args} 2>&1
    " | while IFS= read -r line; do
        # Log to file for debugging
        log_debug "  $line"
        # Print to console immediately (unbuffered)
        echo "$line"
    done

    # Capture the exit code from the pipe
    post_install_exit=${PIPESTATUS[0]}

    enable_error_trap

    if [[ $post_install_exit -ne 0 ]]; then
        log_error "Post-installation failed (exit code: ${post_install_exit})"
        echo ""
        log_info "Common causes and solutions:"
        log_info "  • Stale build cache (upgrade/downgrade): Re-run with: sudo $SCRIPT_NAME --force-cleanup"
        log_info "  • Network issues: Check internet connection for resource downloads"
        log_info "  • Permission issues: Try running: sudo chown -R ${ORIGINAL_USER}:${ORIGINAL_GROUP} ${SCRIPT_DIR}"
        log_info "  • C++ compilation: Ensure meson and ninja-build are installed"
        log_info "  • Missing dependencies: Check the error messages above"
        echo ""
        log_info "You can retry post-installation manually:"
        log_info "  source ${SCRIPT_DIR}/setup_env.sh"
        log_info "  hailo-post-install --group '${DOWNLOAD_GROUP}'"

        record_step_result "FAILED" "Exit code: ${post_install_exit}"
        return 1
    fi

    log_success "Post-installation completed"
    record_step_result "SUCCESS" "Post-install done"
    return 0
}

#===============================================================================
# VERIFICATION
#===============================================================================

verify_installation() {
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}  Installation Verification${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    local venv_path="${SCRIPT_DIR}/${VENV_NAME}"
    local venv_activate="${venv_path}/bin/activate"
    local all_ok=true

    if [[ "${DRY_RUN}" == true ]]; then
        log_info "Skipping verification in dry-run mode"
        return 0
    fi

    # Check venv
    echo -n "  📁 Virtual environment: "
    if [[ -f "${venv_activate}" ]]; then
        echo -e "${GREEN}✅ OK${NC}"
    else
        echo -e "${RED}❌ Missing${NC}"
        all_ok=false
    fi

    # Check hailo_apps import
    echo -n "  🐍 hailo_apps package: "
    if run_as_user bash -c "source '${venv_activate}' && python3 -c 'import hailo_apps'" 2>/dev/null; then
        echo -e "${GREEN}✅ OK${NC}"
    else
        echo -e "${RED}❌ Import failed${NC}"
        all_ok=false
    fi

    # Check HailoRT binding
    echo -n "  📦 HailoRT Python binding: "
    if run_as_user bash -c "source '${venv_activate}' && python3 -c 'import hailo_platform'" 2>/dev/null; then
        echo -e "${GREEN}✅ OK${NC}"
        all_ok=false
    fi

    # Check TAPPAS binding
    echo -n "  📦 TAPPAS Core Python binding: "
    if [[ "${SKIP_GSTREAMER}" == true ]]; then
        echo -e "${YELLOW}⚠️  Skipped (--skip-gstreamer)${NC}"
    else
        if run_as_user bash -c "source '${venv_activate}' && python3 -c 'import hailo'" 2>/dev/null; then
            echo -e "${GREEN}✅ OK${NC}"
            all_ok=false
        fi
    fi


    # Show detected architecture and Model Zoo version
    if [[ -n "${HAILO_ARCH:-}" && "${HAILO_ARCH}" != "unknown" ]]; then
        echo "  🏗️  Hailo Architecture: ${HAILO_ARCH}"
        if [[ -n "${MODEL_ZOO_VER:-}" ]]; then
            echo "  📦 Model Zoo Version: ${MODEL_ZOO_VER}"
        fi
    fi

    # Check resources symlink
    echo -n "  📁 Resources symlink: "
    if [[ -L "${SCRIPT_DIR}/resources" && -d "${SCRIPT_DIR}/resources" ]]; then
        echo -e "${GREEN}✅ OK${NC}"

        # Count models
        local model_count=0
        model_count=$(find "${SCRIPT_DIR}/resources/models" -name "*.hef" 2>/dev/null | wc -l)
        echo "      Found ${model_count} model files (.hef)"
    else
        echo -e "${YELLOW}⚠️  Not created${NC}"
    fi

    # Check .env file (at resources root)
    echo -n "  📄 Environment file: "
    if [[ -f "${ENV_FILE}" ]]; then
        echo -e "${GREEN}✅ OK (${ENV_FILE})${NC}"
    else
        echo -e "${YELLOW}⚠️  Not created${NC}"
    fi

    # Check compiled libraries
    echo -n "  🔨 C++ postprocess libs: "
    if [[ -d "${RESOURCES_ROOT}/so" ]]; then
        local so_count=0
        so_count=$(find "${RESOURCES_ROOT}/so" -name "*.so" 2>/dev/null | wc -l)
        if [[ $so_count -gt 0 ]]; then
            echo -e "${GREEN}✅ ${so_count} libraries${NC}"
        else
            echo -e "${YELLOW}⚠️  No .so files found${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Directory not found${NC}"
    fi

    echo ""

    if [[ "$all_ok" == true ]]; then
        return 0
    else
        return 1
    fi
}

#===============================================================================
# PRINT SUMMARY
#===============================================================================

print_summary() {
    echo ""
    echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  Installation Summary${NC}"
    echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
    echo ""

    local step_names=(
        "User Detection"
        "Prerequisites Check"
        "System Packages"
        "Resources Setup"
        "Virtual Environment"
        "Python Packages"
        "Post-Installation"
    )

    local all_success=true
    local i=0

    for result in "${STEP_RESULTS[@]}"; do
        local status="${result%%:*}"
        local message="${result#*:}"
        local icon

        case "$status" in
            SUCCESS) icon="${GREEN}✅${NC}" ;;
            FAILED) icon="${RED}❌${NC}"; all_success=false ;;
            SKIPPED) icon="${YELLOW}⏭️ ${NC}" ;;
            *) icon="❓" ;;
        esac

        printf "  %b %-25s %b\n" "$icon" "${step_names[$i]:-Step $((i+1))}" "${DIM}${message}${NC}"
        ((i++))
    done

    echo ""

    if [[ "${DRY_RUN}" == true ]]; then
        echo -e "${YELLOW}${BOLD}This was a DRY RUN - no changes were made${NC}"
        echo ""
    fi

    if [[ "$all_success" == true ]]; then
        echo -e "${GREEN}${BOLD}✅ Installation completed successfully!${NC}"
        echo ""
        echo "Virtual environment: ${SCRIPT_DIR}/${VENV_NAME}"
        echo "To activate:         source ${SCRIPT_DIR}/setup_env.sh"
    else
        echo -e "${RED}${BOLD}❌ Installation completed with errors${NC}"
        echo ""
        echo "Please review the errors above and try again."
    fi

    echo ""
    echo "Log file: ${LOG_FILE}"
    echo ""
}

#===============================================================================
# MAIN
#===============================================================================

main() {
    # Parse arguments first (before any output)
    parse_arguments "$@"

    # Initialize logging
    init_logging

    # Show banner
    echo ""
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║      Hailo Apps Infrastructure - Single-File Installer           ║${NC}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    if [[ "${DRY_RUN}" == true ]]; then
        echo -e "${MAGENTA}${BOLD}🔍 DRY-RUN MODE - No changes will be made${NC}"
        echo ""
    fi

    # Enable error trap
    enable_error_trap

    # Load configuration from config.yaml (required)
    if ! load_config; then
        log_error "Failed to load configuration. Cannot continue."
        exit 1
    fi

    # Show configuration summary
    log_info "Configuration:"
    log_info "  Virtual Environment: ${VENV_NAME}"
    log_info "  Download Group: ${DOWNLOAD_GROUP}"
    log_info "  Resources Root: ${RESOURCES_ROOT}"
    log_info "  System Site-Packages: ${USE_SYSTEM_SITE_PACKAGES}"
    log_info "  Log File: ${LOG_FILE}"

    # Show valid versions from config (if loaded)
    if [[ -n "${VALID_HAILORT_VERSIONS:-}" ]]; then
        log_debug "Valid HailoRT versions: ${VALID_HAILORT_VERSIONS}"
    fi
    if [[ -n "${VALID_TAPPAS_VERSIONS:-}" ]]; then
        log_debug "Valid TAPPAS versions: ${VALID_TAPPAS_VERSIONS}"
    fi
    if [[ -n "${VALID_HAILO_ARCH:-}" ]]; then
        log_debug "Valid Hailo architectures: ${VALID_HAILO_ARCH}"
    fi
    if [[ -n "${MODEL_ZOO_MAPPING:-}" ]]; then
        log_debug "Model Zoo mapping: ${MODEL_ZOO_MAPPING}"
    fi

    # Force cleanup before installation if requested
    if [[ "${FORCE_CLEANUP}" == true ]]; then
        local cleanup_script="${SCRIPT_DIR}/scripts/cleanup_installation.sh"
        log_info "Running cleanup before installation (--force-cleanup)..."
        if [[ ! -x "${cleanup_script}" ]]; then
            log_error "Cleanup script not found or not executable: ${cleanup_script}"
            exit 1
        fi
        if [[ "${DRY_RUN}" == true ]]; then
            log_dry_run "sudo ${cleanup_script}"
        else
            if ! bash "${cleanup_script}" > /dev/null; then
                log_error "Cleanup script failed. Aborting installation."
                exit 1
            fi
            log_success "Cleanup completed"
        fi
        echo ""
    fi

    # Run installation steps
    local failed=false

    # Step 1: User detection
    if ! detect_user_and_group; then
        failed=true
    fi

    # Step 2: Prerequisites check
    if [[ "$failed" != true ]]; then
        if ! check_prerequisites; then
            failed=true
        fi
    fi

    # Step 3: System packages
    if [[ "$failed" != true ]]; then
        if ! install_system_packages; then
            failed=true
        fi
    fi

    # Step 4: Resources setup (before venv so packages dir is available)
    if [[ "$failed" != true ]]; then
        if ! setup_resources; then
            failed=true
        fi
    fi

    # Step 5: Virtual environment
    if [[ "$failed" != true ]]; then
        if ! setup_virtual_environment; then
            failed=true
        fi
    fi

    # Step 6: Python packages
    if [[ "$failed" != true ]]; then
        if ! install_python_packages; then
            failed=true
        fi
    fi

    # Step 7: Post-installation
    if [[ "$failed" != true ]]; then
        if ! run_post_install; then
            failed=true
        fi
    fi

    # Final ownership fix
    if [[ "${DRY_RUN}" != true && "$failed" != true ]]; then
        log_debug "Fixing final ownership..."
        fix_ownership "${SCRIPT_DIR}"
    fi

    # Verification
    if [[ "$failed" != true ]]; then
        verify_installation || true
    fi

    # Print summary
    print_summary

    if [[ "$failed" == true ]]; then
        exit 1
    fi

    exit 0
}

# Run main
main "$@"