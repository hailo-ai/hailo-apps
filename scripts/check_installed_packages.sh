#!/bin/bash
# Check installed Hailo packages and their versions

set -euo pipefail

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print error message and exit
error_exit() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
    exit 1
}

# Detect if a Python package is installed via pip and return its version
detect_pip_pkg_version() {
    local pkg="$1"
    # Try various methods to get the package version
    pip3 list 2>/dev/null | grep -i "^$pkg " | awk '{print $2}' || \
    python3 -m pip list 2>/dev/null | grep -i "^$pkg " | awk '{print $2}' || \
    python3 -c "import pkg_resources; print(pkg_resources.get_distribution('$pkg').version)" 2>/dev/null || \
    echo ""
}

# Check if hailo-all pip package is installed
is_hailo_all_installed() {
    detect_pip_pkg_version "hailo-all"
}

# Compare two version strings
# Returns 0 if version1 >= version2, 1 otherwise
# Usage: compare_versions "4.23.0" "4.22.0"
compare_versions() {
    local version1="$1"
    local version2="$2"
    
    # If either version is invalid, return failure
    if [[ "$version1" == "-1" || "$version1" == "unknown" || -z "$version1" ]]; then
        return 1
    fi
    if [[ "$version2" == "-1" || "$version2" == "unknown" || -z "$version2" ]]; then
        return 1
    fi
    
    # Use sort -V for version comparison
    local result=$(printf '%s\n%s\n' "$version1" "$version2" | sort -V | head -n1)
    [[ "$result" == "$version2" ]] && return 0 || return 1
}

# Detect Hailo architecture using hailortcli, hardware detection, or version inference
# Arguments: driver_version, hailort_version (optional, for fallback inference)
# Returns: "hailo8", "hailo10h", "hailo8l", or "unknown"
detect_hailo_arch() {
    local arch="unknown"
    local detection_method=""
    local driver_version="${1:-}"
    local hailort_version="${2:-}"
    
    # Method 1: Try hardware detection via PCI devices (works even without packages)
    if command -v lspci >/dev/null 2>&1; then
        local pci_output
        # Try basic lspci first
        pci_output=$(lspci 2>/dev/null | grep -i "hailo" || true)
        
        if [[ -n "$pci_output" ]]; then
            detection_method="pci"
            # Check PCI output for architecture hints
            if echo "$pci_output" | grep -qiE "hailo-8|hailo8"; then
                # Try to distinguish between Hailo8L and Hailo8
                if echo "$pci_output" | grep -qiE "hailo-8l|hailo8l"; then
                    arch="hailo8l"
                    echo "[OK]   Detected Hailo architecture: HAILO8L (via PCI device)"
                else
                    arch="hailo8"
                    echo "[OK]   Detected Hailo architecture: HAILO8 (via PCI device)"
                fi
            elif echo "$pci_output" | grep -qiE "hailo-10|hailo10|hailo-15|hailo15"; then
                arch="hailo10h"
                echo "[OK]   Detected Hailo architecture: HAILO10H (via PCI device)"
            else
                # Try more detailed PCI info (-v flag) to get subsystem/model info
                local pci_detailed
                pci_detailed=$(lspci -v 2>/dev/null | grep -iA 5 "hailo" || true)
                
                if echo "$pci_detailed" | grep -qiE "hailo-8|hailo8"; then
                    if echo "$pci_detailed" | grep -qiE "hailo-8l|hailo8l"; then
                        arch="hailo8l"
                        echo "[OK]   Detected Hailo architecture: HAILO8L (via PCI device detailed info)"
                    else
                        arch="hailo8"
                        echo "[OK]   Detected Hailo architecture: HAILO8 (via PCI device detailed info)"
                    fi
                elif echo "$pci_detailed" | grep -qiE "hailo-10|hailo10|hailo-15|hailo15"; then
                    arch="hailo10h"
                    echo "[OK]   Detected Hailo architecture: HAILO10H (via PCI device detailed info)"
                else
                    # PCI device found but can't determine architecture from name
                    echo "[INFO] Hailo PCI device detected: $pci_output"
                fi
            fi
        fi
    fi
    
    # Method 2: Try hailortcli if available (most reliable when device is connected)
    if [[ "$arch" == "unknown" ]] && command -v hailortcli >/dev/null 2>&1; then
        detection_method="hailortcli"
        local fw_output
        local fw_exit_code=0
        fw_output=$(hailortcli fw-control identify 2>&1) || fw_exit_code=$?
        
        if [[ $fw_exit_code -eq 0 && -n "$fw_output" ]]; then
            # Check for Hailo8L first (more specific)
            if echo "$fw_output" | grep -qi "HAILO8L"; then
                arch="hailo8l"
                echo "[OK]   Detected Hailo architecture: HAILO8L (via hailortcli)"
            # Check for Hailo8
            elif echo "$fw_output" | grep -qi "HAILO8"; then
                arch="hailo8"
                echo "[OK]   Detected Hailo architecture: HAILO8 (via hailortcli)"
            # Check for Hailo10H or Hailo15H
            elif echo "$fw_output" | grep -qiE "HAILO10H|HAILO15H"; then
                arch="hailo10h"
                echo "[OK]   Detected Hailo architecture: HAILO10H (via hailortcli)"
            else
                echo "[WARN] Could not determine Hailo architecture from hailortcli output"
                # If hailortcli ran but returned no device info, it might mean no device is connected
                if echo "$fw_output" | grep -qiE "no.*device|not.*found|not.*connected"; then
                    echo "[INFO] No Hailo device detected via hailortcli (device may not be connected)"
                fi
            fi
        else
            echo "[WARN] Failed to run hailortcli fw-control identify (exit code: $fw_exit_code)"
            # If command failed, check if it's because no device is connected
            if echo "$fw_output" | grep -qiE "no.*device|not.*found|not.*connected"; then
                echo "[INFO] No Hailo device connected"
            fi
        fi
    elif [[ "$arch" == "unknown" ]]; then
        echo "[INFO] hailortcli not found (Hailo packages may not be installed)"
    fi
    
    # Method 3: Check for device files
    if [[ "$arch" == "unknown" ]] && [[ -d /dev ]] && ls /dev/hailo* &>/dev/null; then
        detection_method="device_files"
        echo "[INFO] Hailo device files found in /dev/, but cannot determine architecture without PCI info or hailortcli"
    fi
    
    # Method 4: Fallback - infer architecture from installed package versions
    # This is useful when packages are installed but no device is connected
    if [[ "$arch" == "unknown" && -n "$driver_version" && -n "$hailort_version" ]]; then
        if [[ "$driver_version" != "-1" && "$driver_version" != "unknown" && \
              "$hailort_version" != "-1" && "$hailort_version" != "unknown" ]]; then
            detection_method="version_inference"
            
            # Check if versions indicate Hailo8 (4.22.x or 4.23.x)
            if [[ ("$driver_version" == 4.22* || "$driver_version" == 4.23*) && \
                  ("$hailort_version" == 4.22* || "$hailort_version" == 4.23*) ]]; then
                arch="hailo8"
                echo "[INFO] Inferred Hailo architecture: HAILO8 (from driver/hailort version 4.22.x/4.23.x)"
                echo "[INFO] Note: This is inferred from package versions. Connect device to confirm via hailortcli."
            # Check if versions indicate Hailo10H (>= 5.0.0)
            elif compare_versions "$driver_version" "5.0.0" && compare_versions "$hailort_version" "5.0.0"; then
                arch="hailo10h"
                echo "[INFO] Inferred Hailo architecture: HAILO10H (from driver/hailort version >= 5.0.0)"
                echo "[INFO] Note: This is inferred from package versions. Connect device to confirm via hailortcli."
            fi
        fi
    fi
    
    # If we couldn't detect architecture, provide helpful message
    if [[ "$arch" == "unknown" ]]; then
        if [[ -z "$detection_method" ]]; then
            echo "[INFO] No Hailo packages or hardware detected"
            echo "[INFO] Architecture detection requires either:"
            echo "[INFO]   1. Hailo packages installed (hailortcli)"
            echo "[INFO]   2. Hailo hardware connected and packages installed"
        elif [[ "$detection_method" == "hailortcli" ]]; then
            echo "[INFO] Architecture could not be determined - device may not be connected"
            echo "[INFO] Connect a Hailo device and run 'hailortcli fw-control identify' to detect architecture"
        fi
    fi
    
    echo "hailo_arch=$arch"
}

# Validate driver and hailort versions based on Hailo architecture
# Arguments: architecture, driver_version, hailort_version
validate_versions_for_arch() {
    local arch="$1"
    local driver_version="$2"
    local hailort_version="$3"
    local errors=0
    local packages_installed=false
    
    if [[ "$arch" == "unknown" ]]; then
        echo "[INFO] Unknown Hailo architecture, skipping version validation"
        return 0
    fi
    
    # Check if any packages are installed
    if [[ "$driver_version" != "-1" && "$driver_version" != "unknown" && -n "$driver_version" ]]; then
        packages_installed=true
    fi
    if [[ "$hailort_version" != "-1" && "$hailort_version" != "unknown" && -n "$hailort_version" ]]; then
        packages_installed=true
    fi
    
    # If no packages are installed, skip validation gracefully
    if [[ "$packages_installed" == "false" ]]; then
        echo "[INFO] Hailo packages not installed, skipping version validation for $arch"
        echo "[INFO] To validate versions, please install:"
        echo "[INFO]   - For Hailo8/Hailo8L: driver and hailort version 4.22.x or 4.23.x"
        echo "[INFO]   - For Hailo10H: driver and hailort version >= 5.0.0"
        return 0
    fi
    
    if [[ "$arch" == "hailo8" || "$arch" == "hailo8l" ]]; then
        # For Hailo8: driver and hailort should be 4.22 or 4.23
        local validations_done=0
        
        # Check driver version
        if [[ "$driver_version" != "-1" && "$driver_version" != "unknown" && -n "$driver_version" ]]; then
            validations_done=$((validations_done + 1))
            # Check if version starts with 4.22 or 4.23
            if [[ "$driver_version" == 4.22* || "$driver_version" == 4.23* ]]; then
                echo "[OK]   Driver version $driver_version is valid for $arch (4.22.x or 4.23.x)"
            else
                echo "[ERROR] Driver version $driver_version is invalid for $arch. Expected 4.22.x or 4.23.x"
                errors=$((errors + 1))
            fi
        fi
        
        # Check hailort version
        if [[ "$hailort_version" != "-1" && "$hailort_version" != "unknown" && -n "$hailort_version" ]]; then
            validations_done=$((validations_done + 1))
            # Check if version starts with 4.22 or 4.23
            if [[ "$hailort_version" == 4.22* || "$hailort_version" == 4.23* ]]; then
                echo "[OK]   HailoRT version $hailort_version is valid for $arch (4.22.x or 4.23.x)"
            else
                echo "[ERROR] HailoRT version $hailort_version is invalid for $arch. Expected 4.22.x or 4.23.x"
                errors=$((errors + 1))
            fi
        fi
        
        # If partial installations, provide guidance
        if [[ $validations_done -eq 0 ]]; then
            echo "[INFO] Could not validate versions - driver or hailort not detected"
        elif [[ $validations_done -eq 1 ]]; then
            if [[ "$driver_version" == "-1" || "$driver_version" == "unknown" || -z "$driver_version" ]]; then
                echo "[INFO] Driver version not detected, only HailoRT found"
            fi
            if [[ "$hailort_version" == "-1" || "$hailort_version" == "unknown" || -z "$hailort_version" ]]; then
                echo "[INFO] HailoRT version not detected, only driver found"
            fi
        fi
        
    elif [[ "$arch" == "hailo10h" ]]; then
        # For Hailo10H: driver and hailort should be >= 5.0.0
        local min_version="5.0.0"
        local validations_done=0
        
        # Check driver version
        if [[ "$driver_version" != "-1" && "$driver_version" != "unknown" && -n "$driver_version" ]]; then
            validations_done=$((validations_done + 1))
            if compare_versions "$driver_version" "$min_version"; then
                echo "[OK]   Driver version $driver_version is valid for $arch (>= $min_version)"
            else
                echo "[ERROR] Driver version $driver_version is invalid for $arch. Expected >= $min_version"
                errors=$((errors + 1))
            fi
        fi
        
        # Check hailort version
        if [[ "$hailort_version" != "-1" && "$hailort_version" != "unknown" && -n "$hailort_version" ]]; then
            validations_done=$((validations_done + 1))
            if compare_versions "$hailort_version" "$min_version"; then
                echo "[OK]   HailoRT version $hailort_version is valid for $arch (>= $min_version)"
            else
                echo "[ERROR] HailoRT version $hailort_version is invalid for $arch. Expected >= $min_version"
                errors=$((errors + 1))
            fi
        fi
        
        # If partial installations, provide guidance
        if [[ $validations_done -eq 0 ]]; then
            echo "[INFO] Could not validate versions - driver or hailort not detected"
        elif [[ $validations_done -eq 1 ]]; then
            if [[ "$driver_version" == "-1" || "$driver_version" == "unknown" || -z "$driver_version" ]]; then
                echo "[INFO] Driver version not detected, only HailoRT found"
            fi
            if [[ "$hailort_version" == "-1" || "$hailort_version" == "unknown" || -z "$hailort_version" ]]; then
                echo "[INFO] HailoRT version not detected, only driver found"
            fi
        fi
    fi
    
    return $errors
}

check_kernel_module() {
    local version="-1"
    local module=""
    local module_found="false"

    # Try to find hailo_pci module first
    if lsmod | grep -q "^hailo_pci "; then
        module="hailo_pci"
        module_found="true"
    elif modinfo hailo_pci &>/dev/null; then
        module="hailo_pci"
        module_found="true"
    fi

    # If hailo_pci was not found, check for hailo1x_pci
    if [[ "$module_found" == "false" ]]; then
        if lsmod | grep -q "^hailo1x_pci "; then
            module="hailo1x_pci"
            module_found="true"
        elif modinfo hailo1x_pci &>/dev/null; then
            module="hailo1x_pci"
            module_found="true"
        fi
    fi

    # If a module was found, get its version
    if [[ "$module_found" == "true" ]]; then
        if lsmod | grep -q "^$module "; then
            if modinfo "$module" &>/dev/null; then
                version=$(modinfo "$module" | awk -F ': +' '/^version:/{print $2}')
                echo "[OK]   $module module loaded and installed, version: $version"
            else
                echo "[OK]   $module module loaded (version unknown)"
                version="unknown"
            fi
        else
            version=$(modinfo "$module" | awk -F ': +' '/^version:/{print $2}')
            echo "[OK]   $module module installed (not loaded), version: $version"
        fi
    else
        # Fallback check for the package if neither module is found
        if dpkg -l 2>/dev/null | grep "hailort-pcie-driver" | grep -q "^ii"; then
            version=$(dpkg -l 2>/dev/null | grep "hailort-pcie-driver" | grep "^ii" | awk '{print $3}')
            echo "[OK]   hailort-pcie-driver package installed, version: $version"
        else
            echo "[WARN] hailo_pci/hailo1x_pci module not found, version: -1"
        fi
    fi

    # Always echo the version key=value for downstream parsing
    if [[ -z "$module" ]]; then
        echo "hailo_pci_unified=$version"
    else
        echo "$module=$version"
    fi
}

# Check for hailort installation
check_hailort() {
    local hailort_version="-1"
    
    # Check system installation via apt - handle both regular and versioned packages
    if dpkg -l 2>/dev/null | grep -E "^ii.*hailort(/| )" | head -1 | grep -q .; then
        hailort_version=$(dpkg -l | grep -E "^ii.*hailort(/| )" | head -1 | awk '{print $3}')
        echo "[OK]   hailort (system) version: $hailort_version"
    # Check with hailortcli if available
    elif command -v hailortcli >/dev/null 2>&1; then
        hailort_version=$(hailortcli --version 2>/dev/null | grep -oP 'version \K[0-9\.]+' || echo "-1")
        if [[ "$hailort_version" != "-1" ]]; then
            echo "[OK]   hailort (via hailortcli) version: $hailort_version"
        else
            echo "[WARNING] hailort not installed, version: -1"
        fi
    else
        echo "[WARNING] hailort not installed, version: -1"
    fi
    
    # Return the version
    echo "hailort=$hailort_version"
}

# Check for TAPPAS packages
check_tappas_packages() {
    local version="-1"
    local found=false

    # 1) Check for known Debian packages - handle versioned packages
    if dpkg -l 2>/dev/null | grep -E "^ii.*(hailo-tappas-core|hailo-tappas|tappas-core|tappas)" | head -1 | grep -q .; then
        pkg_line=$(dpkg -l 2>/dev/null | grep -E "^ii.*(hailo-tappas-core|hailo-tappas|tappas-core|tappas)" | head -1)
        pkg_name=$(echo "$pkg_line" | awk '{print $2}')
        version=$(echo "$pkg_line" | awk '{print $3}')
        echo "[OK]   $pkg_name (system) version: $version"
        found=true
    fi

    # 2) Fallback to pkg-config if no dpkg package found
    if ! $found; then
        for pc in hailo-tappas-core hailo_tappas tappas-core tappas; do
            if pkg-config --exists "$pc" 2>/dev/null; then
                if pkg-config --modversion "$pc" &>/dev/null; then
                    version=$(pkg-config --modversion "$pc")
                    echo "[OK]   pkg-config $pc version: $version"
                else
                    echo "[OK]   pkg-config $pc present, version: unknown"
                    version="unknown"
                fi
                found=true
                break
            fi
        done
    fi

    # 3) If still not found
    if ! $found; then
        echo "[MISSING] any of hailo-tappas-core / hailo-tappas / tappas-core (system), version: -1"
    fi

    # 4) Always return a key=value
    echo "tappas-core=$version"
}

# Check for Python HailoRT binding
check_hailort_py() {
    local pyhailort_version="-1"
    
    # First check if hailo-all is installed
    hailo_all_ver=$(is_hailo_all_installed)
    
    # Check pip-distribution
    if ver=$(detect_pip_pkg_version "hailort") && [[ -n "$ver" ]]; then
        pyhailort_version="$ver"
        echo "[OK]   pip 'hailort' version: $pyhailort_version"
        
        # Additional test - try to import in the current environment
        if python3 -c 'import hailo' >/dev/null 2>&1; then
            echo "[OK]   Python import 'hailo' succeeded"
        elif python3 -c 'import hailort' >/dev/null 2>&1; then
            # Try to get the version from the module itself
            module_ver=$(python3 -c 'import hailort; print(getattr(hailort, "__version__", "unknown"))' 2>/dev/null)
            if [[ "$module_ver" != "unknown" && -n "$module_ver" ]]; then
                pyhailort_version="$module_ver"
            fi
            echo "[OK]   Python import 'hailort' succeeded, version: $pyhailort_version"
        else
            echo "[WARNING] pip 'hailort' is installed but cannot be imported in current environment"
        fi
    elif [[ -n "$hailo_all_ver" ]]; then
        pyhailort_version="$hailo_all_ver"
        echo "[OK]   pip 'hailort' is part of hailo-all package: $pyhailort_version"
        
        # Check if it can be imported
        if python3 -c 'import hailo' >/dev/null 2>&1; then
            echo "[OK]   Python import 'hailo' succeeded"
        elif python3 -c 'import hailort' >/dev/null 2>&1; then
            echo "[OK]   Python import 'hailort' succeeded, version: $pyhailort_version"
        else
            echo "[WARNING] hailo-all is installed but 'hailort' module cannot be imported"
        fi
    else
        echo "[MISSING] pip 'hailort', version: -1"
        
        # One last try - maybe it's importable but not visible to pip
        if python3 -c 'import hailo' >/dev/null 2>&1; then
            echo "[OK]   Python import 'hailo' succeeded (not from pip)"
            pyhailort_version="unknown"
        elif python3 -c 'import hailort' >/dev/null 2>&1; then
            module_ver=$(python3 -c 'import hailort; print(getattr(hailort, "__version__", "unknown"))' 2>/dev/null)
            if [[ "$module_ver" != "unknown" && -n "$module_ver" ]]; then
                pyhailort_version="$module_ver"
                echo "[OK]   Python import 'hailort' succeeded (not from pip), version: $pyhailort_version"
            else
                echo "[OK]   Python import 'hailort' succeeded but version unknown"
                pyhailort_version="unknown"
            fi
        else
            echo "[MISSING] Python import 'hailort', version: -1"
        fi
    fi
    
    # Return the version
    echo "pyhailort=$pyhailort_version"
}

# Check for TAPPAS Python binding
check_tappas_core_py() {
    local tappas_python_version="-1"
    
    # First check if hailo-all is installed
    hailo_all_ver=$(is_hailo_all_installed)
    
    # Check pip-distribution with multiple possible package names
    found_version=""
    found_pkg=""
    for pkg in "hailo-tappas-core-python-binding" "tappas-core-python-binding" "hailo-tappas-python-binding" "tappas"; do
        if ver=$(detect_pip_pkg_version "$pkg") && [[ -n "$ver" ]]; then
            tappas_python_version="$ver"
            found_version="$ver"
            found_pkg="$pkg"
            echo "[OK]   pip '$pkg' version: $tappas_python_version"
            break
        fi
    done
    
    if [[ -z "$found_version" ]]; then
        if [[ -n "$hailo_all_ver" ]]; then
            tappas_python_version="$hailo_all_ver"
            echo "[OK]   TAPPAS Python binding is part of hailo-all package: $tappas_python_version"
        else
            echo "[MISSING] TAPPAS Python binding pip package, version: -1"
        fi
    fi
    
    # Check if the module can be imported
    if python3 -c 'import hailo_platform' >/dev/null 2>&1; then
        # Try to get version from the module (but don't overwrite pip version)
        module_ver=$(python3 -c 'import hailo_platform; print(getattr(hailo_platform, "__version__", "unknown"))' 2>/dev/null)
        if [[ "$module_ver" != "unknown" && -n "$module_ver" ]]; then
            # Only use module version if we don't have a pip version
            if [[ -z "$found_version" || "$tappas_python_version" == "-1" ]]; then
                tappas_python_version="$module_ver"
                echo "[OK]   Python import 'hailo_platform' succeeded, version: $tappas_python_version (from module)"
            else
                # Pip version takes precedence, but show module version for reference
                echo "[OK]   Python import 'hailo_platform' succeeded"
                echo "[INFO] Module reports version: $module_ver (pip package version: $tappas_python_version)"
            fi
        else
            echo "[OK]   Python import 'hailo_platform' succeeded, version: $tappas_python_version"
        fi
    else
        if [[ -n "$hailo_all_ver" || -n "$found_version" ]]; then
            echo "[WARNING] TAPPAS Python package is installed but 'hailo_platform' module cannot be imported"
            # Don't reset version to -1 if package is installed
        else
            echo "[MISSING] Python import 'hailo_platform', version: -1"
            tappas_python_version="-1"
        fi
    fi
    
    # Return the version
    echo "tappas-python=$tappas_python_version"
}

# Main function to perform all checks
to_check() {
    echo "=== Hailo Package Detection ==="
    echo ""
    
    # First, silently check for driver and hailort versions to enable version-based inference
    local kernel_output_silent=$(check_kernel_module 2>/dev/null)
    local hailort_output_silent=$(check_hailort 2>/dev/null)
    local kernel_version_silent=$(echo "$kernel_output_silent" | grep -E "^(hailo_pci|hailo_pci_unified|hailo1x_pci)=" | cut -d'=' -f2)
    local hailort_version_silent=$(echo "$hailort_output_silent" | grep "^hailort=" | cut -d'=' -f2)
    
    # Detect Hailo architecture (with version-based fallback if hailortcli fails)
    echo "Hailo Architecture Detection:"
    arch_output=$(detect_hailo_arch "$kernel_version_silent" "$hailort_version_silent")
    local hailo_arch=$(echo "$arch_output" | grep "^hailo_arch=" | cut -d'=' -f2)
    echo "$arch_output" | grep -v "^hailo_arch="
    echo ""
    
    # Display all check results for verbose output
    kernel_output=$(check_kernel_module)
    hailort_output=$(check_hailort)
    tappas_output=$(check_tappas_packages) 
    pyhailort_output=$(check_hailort_py)
    tappas_py_output=$(check_tappas_core_py)
    
    # Display all outputs (filtering out the key=value lines)
    echo "Kernel Module Check:"
    echo "$kernel_output" | grep -vE "^(hailo_pci|hailo_pci_unified|hailo1x_pci)="
    echo ""
    
    echo "HailoRT Check:"
    echo "$hailort_output" | grep -v "^hailort="
    echo ""
    
    echo "TAPPAS Core Check:"
    echo "$tappas_output" | grep -v "^tappas-core="
    echo ""
    
    echo "Python HailoRT Check:"
    echo "$pyhailort_output" | grep -v "^pyhailort="
    echo ""
    
    echo "Python TAPPAS Check:"
    echo "$tappas_py_output" | grep -v "^tappas-python="
    echo ""
    
    # Extract versions from the last line of each output
    local kernel_version=$(echo "$kernel_output" | grep -E "^(hailo_pci|hailo_pci_unified|hailo1x_pci)=" | cut -d'=' -f2)
    local hailort_version=$(echo "$hailort_output" | grep "^hailort=" | cut -d'=' -f2)
    local tappas_version=$(echo "$tappas_output" | grep "^tappas-core=" | cut -d'=' -f2)
    local pyhailort_version=$(echo "$pyhailort_output" | grep "^pyhailort=" | cut -d'=' -f2)
    local tappas_py_version=$(echo "$tappas_py_output" | grep "^tappas-python=" | cut -d'=' -f2)
    
    # Validate versions based on architecture
    if [[ "$hailo_arch" != "unknown" ]]; then
        echo "Version Validation for $hailo_arch:"
        validate_versions_for_arch "$hailo_arch" "$kernel_version" "$hailort_version"
        echo ""
    fi
    
    # Print summary
    echo "================================"
    echo "SUMMARY: hailo_arch=$hailo_arch hailo_pci=$kernel_version hailort=$hailort_version pyhailort=$pyhailort_version tappas-core=$tappas_version tappas-python=$tappas_py_version"
}

# Execute the main function
to_check