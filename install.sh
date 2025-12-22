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

#===============================================================================
# CONSTANTS
#===============================================================================

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly CONFIG_FILE="${SCRIPT_DIR}/hailo_apps/config/config.yaml"

# Log file path (not readonly - may be updated if log dir not writable)
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/install_${TIMESTAMP}.log"

# Terminal colors (with fallback for non-color terminals)
if [[ -t 1 ]] && command -v tput &>/dev/null && [[ $(tput colors 2>/dev/null || echo 0) -ge 8 ]]; then
    readonly RED='\033[0;31m'
    readonly GREEN='\033[0;32m'
    readonly YELLOW='\033[1;33m'
    readonly BLUE='\033[0;34m'
    readonly CYAN='\033[0;36m'
    readonly MAGENTA='\033[0;35m'
    readonly BOLD='\033[1m'
    readonly DIM='\033[2m'
    readonly NC='\033[0m'
else
    readonly RED='' GREEN='' YELLOW='' BLUE='' CYAN='' MAGENTA='' BOLD='' DIM='' NC=''
fi

#===============================================================================
# COMMAND LINE OPTIONS
#===============================================================================

DRY_RUN=false
NO_INSTALL=false
NO_SYSTEM_PYTHON=false
NO_TAPPAS_REQUIRED=false
PYHAILORT_PATH=""
PYTAPPAS_PATH=""

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
INSTALL_HAILORT=false
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
as_original_user() {
    if [[ ${EUID:-$(id -u)} -eq 0 && -n "${SUDO_USER:-}" ]]; then
        log_debug "Running as user ${SUDO_USER}: $*"
        sudo -n -u "$SUDO_USER" -H -- "$@"
    else
        "$@"
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

    if [[ "$valid" != true ]]; then
        log_warning "Version validation warnings above - installation will continue"
    fi
}

# Get Model Zoo version for a given Hailo architecture
get_model_zoo_version() {
    local arch="$1"
    local mz_version=""

    if [[ -z "${MODEL_ZOO_MAPPING:-}" ]]; then
        # Default mapping if not loaded from config
        case "$arch" in
            hailo8|hailo8l) mz_version="v2.17.0" ;;
            hailo10h) mz_version="v5.1.0" ;;
            *) mz_version="" ;;
        esac
    else
        # Parse MODEL_ZOO_MAPPING (format: "hailo8=v2.17.0 hailo8l=v2.17.0 hailo10h=v5.1.1")
        for mapping in $MODEL_ZOO_MAPPING; do
            local key="${mapping%%=*}"
            local value="${mapping#*=}"
            if [[ "$key" == "$arch" ]]; then
                mz_version="$value"
                break
            fi
        done
    fi

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
            valid_versions="${VALID_MZ_H10_VERSIONS:-v5.1.1}"
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

    # Extract venv settings
    VENV_NAME=$(yaml_get "venv.name" "${CONFIG_FILE}")
    local cfg_use_system_site_packages
    cfg_use_system_site_packages=$(yaml_get "venv.use_system_site_packages" "${CONFIG_FILE}")

    # Handle boolean for use_system_site_packages
    case "${cfg_use_system_site_packages,,}" in
        true|yes|1) USE_SYSTEM_SITE_PACKAGES=true ;;
        false|no|0) USE_SYSTEM_SITE_PACKAGES=false ;;
        *) USE_SYSTEM_SITE_PACKAGES=true ;;
    esac

    # Extract resources settings
    RESOURCES_ROOT=$(yaml_get "resources.root" "${CONFIG_FILE}")
    RESOURCES_SYMLINK_NAME=$(yaml_get "resources.path" "${CONFIG_FILE}")
    DOWNLOAD_GROUP=$(yaml_get "resources.download_group" "${CONFIG_FILE}")
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
    sudo $SCRIPT_NAME [OPTIONS]

${BOLD}OPTIONS:${NC}
    -n, --venv-name NAME        Virtual environment name (default: from config or venv_hailo_apps)
    -ph, --pyhailort PATH       Path to custom PyHailoRT wheel file
    -pt, --pytappas PATH        Path to custom PyTappas wheel file
    --all                       Download all available models/resources
    -x, --no-install            Skip Python package installation
    --no-system-python          Don't use system site-packages in venv
    --no-tappas-required        Skip TAPPAS checks, Python TAPPAS install, compile, and post_install
                                (downloads resources directly, no C++ compilation)
    --dry-run                   Show what would be done without executing
    -h, --help                  Show this help message

${BOLD}CONFIGURATION:${NC}
    Settings are loaded from: ${CONFIG_FILE}
    CLI arguments override config file values.

${BOLD}EXAMPLES:${NC}
    sudo $SCRIPT_NAME                     # Standard installation
    sudo $SCRIPT_NAME --dry-run           # Preview what would be done
    sudo $SCRIPT_NAME --all               # Install with all models
    sudo $SCRIPT_NAME -x                  # Skip Python package installation
    sudo $SCRIPT_NAME -n my_venv --all    # Custom venv name + all models

${BOLD}LOG FILES:${NC}
    Installation logs: ${LOG_DIR}/
    Current session:   ${LOG_FILE}

${BOLD}REQUIREMENTS:${NC}
    - Must be run with sudo (not as root directly)
    - Hailo PCI driver must be installed
    - HailoRT must be installed

    Use 'sudo ./scripts/hailo_installer.sh' to install missing components.

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
            --no-tappas-required)
                NO_TAPPAS_REQUIRED=true
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
        log_error "This script must be run with sudo, not as root directly"
        echo ""
        echo "Please run with: sudo $SCRIPT_NAME"
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
        log_info "Would check: Hailo driver, HailoRT installations"
        record_step_result "SKIPPED" "Dry-run mode"
        return 0
    fi

    log_info "Checking installed Hailo components..."

    local summary_line
    disable_error_trap
    summary_line=$(as_original_user "$check_script" 2>&1 | sed -n 's/^SUMMARY: //p')
    enable_error_trap

    if [[ -z "$summary_line" ]]; then
        log_error "Could not get package summary from check script"
        log_debug "This usually means the check script failed or returned unexpected output"
        record_step_result "FAILED" "No SUMMARY output"
        return 1
    fi

    log_debug "SUMMARY line: $summary_line"

    # Parse the summary line
    local driver_version="-1"
    local hailort_version="-1"
    local pyhailort_version="-1"
    local tappas_version="-1"

    # Parse key=value pairs
    for pair in $summary_line; do
        local key="${pair%%=*}"
        local value="${pair#*=}"
        case "$key" in
            hailo_arch) HAILO_ARCH="$value" ;;
            hailo_pci|hailo1x_pci) driver_version="$value" ;;
            hailort) hailort_version="$value"; HAILORT_VERSION="$value" ;;
            pyhailort) pyhailort_version="$value" ;;
            tappas-core) tappas_version="$value" ;;
        esac
    done

    # Determine Model Zoo version based on architecture
    if [[ -n "${HAILO_ARCH:-}" && "${HAILO_ARCH}" != "unknown" ]]; then
        MODEL_ZOO_VER=$(get_model_zoo_version "${HAILO_ARCH}")
    fi
    local model_zoo_version="${MODEL_ZOO_VER}"

    log_info "Detected versions:"
    log_info "  Hailo Architecture: ${HAILO_ARCH:-unknown}"
    log_info "  Driver: ${driver_version}"
    log_info "  HailoRT: ${hailort_version}"
    if [[ "${NO_TAPPAS_REQUIRED}" == true ]]; then
        log_info "  TAPPAS: skipped (--no-tappas-required)"
    else
        log_info "  TAPPAS: ${tappas_version}"
    fi
    if [[ -n "$model_zoo_version" ]]; then
        log_info "  Model Zoo Version: ${model_zoo_version} (for ${HAILO_ARCH})"
    fi

    # Validate versions against config (including architecture)
    if [[ "${NO_TAPPAS_REQUIRED}" == true ]]; then
        validate_versions "$hailort_version" "-1" "${HAILO_ARCH:-}"
    else
        validate_versions "$hailort_version" "$tappas_version" "${HAILO_ARCH:-}"
    fi

    # Validate Model Zoo version if we have both arch and MZ version
    if [[ -n "$model_zoo_version" && -n "${HAILO_ARCH:-}" ]]; then
        validate_model_zoo_version "${HAILO_ARCH}" "$model_zoo_version" || true
    fi

    # Check required components
    local missing_components=()

    if [[ "$driver_version" == "-1" ]]; then
        missing_components+=("Hailo PCI driver")
    fi
    if [[ "$hailort_version" == "-1" ]]; then
        missing_components+=("HailoRT")
    fi

    if [[ ${#missing_components[@]} -gt 0 ]]; then
        log_error "Missing required components:"
        for component in "${missing_components[@]}"; do
            log_error "  • ${component}"
        done
        echo ""
        log_info "To install missing components, run:"
        log_info "    sudo ./scripts/hailo_installer.sh"
        record_step_result "FAILED" "Missing: ${missing_components[*]}"
        return 1
    fi

    # Check Python bindings
    if [[ "$pyhailort_version" == "-1" ]]; then
        log_warning "Python HailoRT binding not installed - will be installed in virtualenv"
        INSTALL_HAILORT=true
    fi

    if [[ "${NO_INSTALL}" == true ]]; then
        log_info "Skipping Python package installation (--no-install flag)"
        INSTALL_HAILORT=false
    fi

    log_success "Prerequisites check passed"
    record_step_result "SUCCESS" "All required components found"
    return 0
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

    # Create virtual environment
    local venv_args=""
    if [[ "${USE_SYSTEM_SITE_PACKAGES}" == true && "${NO_SYSTEM_PYTHON}" != true ]]; then
        log_info "Creating virtualenv '${VENV_NAME}' (with system site-packages)..."
        venv_args="--system-site-packages"
    else
        log_info "Creating virtualenv '${VENV_NAME}' (without system site-packages)..."
    fi

    if [[ "${DRY_RUN}" == true ]]; then
        log_dry_run "python3 -m venv ${venv_args} '${venv_path}'"
        record_step_result "SKIPPED" "Dry-run mode"
        return 0
    fi

    if ! run_as_user python3 -m venv ${venv_args} "${venv_path}"; then
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
        if [[ -n "$PYTAPPAS_PATH" && "${NO_TAPPAS_REQUIRED}" != true ]]; then
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
        INSTALL_HAILORT=false
    fi

    if [[ -n "$PYTAPPAS_PATH" ]]; then
        if [[ "${NO_TAPPAS_REQUIRED}" == true ]]; then
            log_warning "Ignoring PyTappas wheel (--no-tappas-required): ${PYTAPPAS_PATH}"
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

    # Install Hailo Python packages if needed
    if [[ "${INSTALL_HAILORT}" == true ]]; then
        local install_script="${SCRIPT_DIR}/scripts/hailo_python_installation.sh"

        if [[ -f "$install_script" ]]; then
            log_info "Installing Hailo Python packages..."
            local flags=""

            if [[ -z "${HAILO_ARCH:-}" || "${HAILO_ARCH}" == "unknown" ]]; then
                log_error "HAILO_ARCH is required for Python package installation (hailo8 or hailo10h)."
                record_step_result "FAILED" "Missing HAILO_ARCH for Python install"
                return 1
            fi

            case "${HAILO_ARCH}" in
                hailo8|hailo8l) flags="${flags} --arch=hailo8" ;;
                hailo10h) flags="${flags} --arch=hailo10h" ;;
                *)
                    log_error "Unsupported HAILO_ARCH value: ${HAILO_ARCH}. Expected hailo8/hailo8l/hailo10h."
                    record_step_result "FAILED" "Unsupported HAILO_ARCH"
                    return 1
                    ;;
            esac

            if [[ "${INSTALL_HAILORT}" == true && -n "${HAILORT_VERSION}" && "${HAILORT_VERSION}" != "-1" ]]; then
                flags="${flags} --hailort-version=${HAILORT_VERSION}"
                log_debug "Installing HailoRT version: ${HAILORT_VERSION}"
            fi
            if [[ "${NO_TAPPAS_REQUIRED}" == true ]]; then
                flags="${flags} --no-tappas"
            fi

            log_debug "Running: ${install_script} ${flags}"
            if ! run_as_user bash -c "source '${venv_activate}' && '${install_script}' ${flags}"; then
                log_warning "Hailo Python package installation had issues"
                log_info "Continuing with installation - packages may be available from system"
            fi
        else
            log_warning "Hailo Python installation script not found: ${install_script}"
            log_info "Skipping Hailo Python package installation"
        fi
    fi

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

    if [[ "${NO_TAPPAS_REQUIRED}" == true ]]; then
        log_info "Running minimal post-installation (--no-tappas-required)"
        
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
    if [[ "${NO_TAPPAS_REQUIRED}" == true ]]; then
        echo -e "${YELLOW}⚠️  Skipped (--no-tappas-required)${NC}"
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
