#!/usr/bin/env bash
set -e

# ============================================================================
# Hailo Apps Infrastructure Test Runner
# ============================================================================
# This script runs the test suite for hailo-apps.
#
# Test Execution Order:
#   1. Sanity checks (environment validation)
#   2. Installation tests (resource validation)
#   3. Pipeline tests (functional tests)
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh --sanity     # Run only sanity checks
#   ./run_tests.sh --install    # Run only installation tests
#   ./run_tests.sh --pipelines  # Run only pipeline tests
#   ./run_tests.sh --no-download # Skip resource download
#   ./run_tests.sh --help       # Show help
# ============================================================================

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="${SCRIPT_DIR}/tests"
LOG_DIR="${TESTS_DIR}/tests_logs"

# Default options
RUN_SANITY=true
RUN_INSTALL=true
RUN_PIPELINES=true
DOWNLOAD_RESOURCES=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sanity)
            RUN_SANITY=true
            RUN_INSTALL=false
            RUN_PIPELINES=false
            shift
            ;;
        --install)
            RUN_SANITY=false
            RUN_INSTALL=true
            RUN_PIPELINES=false
            shift
            ;;
        --pipelines)
            RUN_SANITY=false
            RUN_INSTALL=false
            RUN_PIPELINES=true
            shift
            ;;
        --no-download)
            DOWNLOAD_RESOURCES=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --sanity       Run only sanity checks (environment validation)"
            echo "  --install      Run only installation tests (resource validation)"
            echo "  --pipelines    Run only pipeline tests (functional tests)"
            echo "  --no-download  Skip resource download step"
            echo "  --help, -h     Show this help message"
            echo ""
            echo "Without options, runs all tests in order: sanity -> install -> pipelines"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "${LOG_DIR}"

echo "============================================================================"
echo "Hailo Apps Infrastructure Test Runner"
echo "============================================================================"
echo ""

# Activate the virtual environment
echo "Activating virtual environment..."
source "${SCRIPT_DIR}/setup_env.sh"

# Install pytest and test dependencies
echo "Installing test requirements..."
python -m pip install --upgrade pip --quiet
python -m pip install -r "${TESTS_DIR}/test_resources/requirements.txt" --quiet

# Download resources for detected architecture only
if [ "$DOWNLOAD_RESOURCES" = true ]; then
    echo ""
    echo "============================================================================"
    echo "Downloading resources for detected architecture..."
    echo "============================================================================"
    # Download default models for all apps (for detected architecture only)
    # Note: This does NOT download hailo8l resources automatically.
    # If you need hailo8l models for h8l_on_h8 tests, run manually:
    #   python -m hailo_apps.installation.download_resources --arch hailo8l
    python -m hailo_apps.installation.download_resources
fi

# Run tests
echo ""
echo "============================================================================"
echo "Running Tests"
echo "============================================================================"

FAILED_TESTS=0

# 1. Sanity Checks - Environment validation
if [ "$RUN_SANITY" = true ]; then
    echo ""
    echo "--- Running Sanity Checks (Environment Validation) ---"
    if python -m pytest "${TESTS_DIR}/test_sanity_check.py" -v --log-cli-level=INFO; then
        echo "✓ Sanity checks passed"
    else
        echo "✗ Sanity checks failed (continuing with remaining tests)"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
fi

# 2. Installation Tests - Resource validation
if [ "$RUN_INSTALL" = true ]; then
    echo ""
    echo "--- Running Installation Tests (Resource Validation) ---"
    if python -m pytest "${TESTS_DIR}/test_installation.py" -v --log-cli-level=INFO; then
        echo "✓ Installation tests passed"
    else
        echo "✗ Installation tests failed (continuing with remaining tests)"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
fi

# 3. Pipeline Tests - Functional tests
if [ "$RUN_PIPELINES" = true ]; then
    echo ""
    echo "--- Running Pipeline Tests (Functional Tests) ---"
    if python -m pytest "${TESTS_DIR}/test_runner.py" -v --log-cli-level=INFO; then
        echo "✓ Pipeline tests passed"
    else
        echo "✗ Pipeline tests failed"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
fi

# Summary
echo ""
echo "============================================================================"
echo "Test Summary"
echo "============================================================================"

if [ $FAILED_TESTS -eq 0 ]; then
    echo "✓ All tests completed successfully!"
    exit 0
else
    echo "✗ ${FAILED_TESTS} test suite(s) failed"
    echo ""
    echo "Check the logs in ${LOG_DIR} for details."
    exit 1
fi
