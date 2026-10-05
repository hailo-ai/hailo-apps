#!/usr/bin/env bash
set -e

# ============================================================================
# Hailo Apps Infrastructure Test Runner
# ============================================================================
# This script runs the test suite for hailo-apps.
#
# Test Execution Order:
#   1. Sanity checks (environment validation + config integrity)
#   2. Installation tests (resource validation)
#   3. Pipeline tests (functional tests)
#   4. Standalone app tests (smoke tests)
#   5. GenAI tests
#
# Usage:
#   ./run_tests.sh              # Run all tests (sanity + install + pipelines)
#   ./run_tests.sh --sanity     # Run only sanity checks
#   ./run_tests.sh --install    # Run only installation tests
#   ./run_tests.sh --pipelines  # Run only pipeline tests
#   ./run_tests.sh --standalone # Run only standalone app tests
#   ./run_tests.sh --genai      # Run only GenAI tests
#   ./run_tests.sh --no-download # Skip resource download
#   ./run_tests.sh --apps detection,pose_estimation # Run only selected apps
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
RUN_STANDALONE=true
RUN_GENAI=false
RUN_CPP=false
DOWNLOAD_RESOURCES=true
APPS_FILTER=""
PYTEST_K_EXPR=""
EXPLICIT_SUITE_SELECTION=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sanity)
            if [ "$EXPLICIT_SUITE_SELECTION" = false ]; then
                RUN_SANITY=false; RUN_INSTALL=false; RUN_PIPELINES=false
                RUN_STANDALONE=false; RUN_GENAI=false
                EXPLICIT_SUITE_SELECTION=true
            fi
            RUN_SANITY=true
            shift
            ;;
        --install)
            if [ "$EXPLICIT_SUITE_SELECTION" = false ]; then
                RUN_SANITY=false; RUN_INSTALL=false; RUN_PIPELINES=false
                RUN_STANDALONE=false; RUN_GENAI=false
                EXPLICIT_SUITE_SELECTION=true
            fi
            RUN_INSTALL=true
            shift
            ;;
        --pipelines)
            if [ "$EXPLICIT_SUITE_SELECTION" = false ]; then
                RUN_SANITY=false; RUN_INSTALL=false; RUN_PIPELINES=false
                RUN_STANDALONE=false; RUN_GENAI=false
                EXPLICIT_SUITE_SELECTION=true
            fi
            RUN_PIPELINES=true
            shift
            ;;
        --standalone)
            if [ "$EXPLICIT_SUITE_SELECTION" = false ]; then
                RUN_SANITY=false; RUN_INSTALL=false; RUN_PIPELINES=false
                RUN_STANDALONE=false; RUN_GENAI=false
                EXPLICIT_SUITE_SELECTION=true
            fi
            RUN_STANDALONE=true
            shift
            ;;
        --genai)
            if [ "$EXPLICIT_SUITE_SELECTION" = false ]; then
                RUN_SANITY=false; RUN_INSTALL=false; RUN_PIPELINES=false
                RUN_STANDALONE=false; RUN_GENAI=false
                EXPLICIT_SUITE_SELECTION=true
            fi
            RUN_GENAI=true
            shift
            ;;
        --cpp)
            if [ "$EXPLICIT_SUITE_SELECTION" = false ]; then
                RUN_SANITY=false; RUN_INSTALL=false; RUN_PIPELINES=false
                RUN_STANDALONE=false; RUN_GENAI=false; RUN_CPP=false
                EXPLICIT_SUITE_SELECTION=true
            fi
            RUN_CPP=true
            shift
            ;;
        --no-download)
            DOWNLOAD_RESOURCES=false
            shift
            ;;
        --apps)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --apps requires a comma-separated list (e.g. --apps detection,pose_estimation)"
                exit 1
            fi
            APPS_FILTER="$2"
            if [ "$EXPLICIT_SUITE_SELECTION" = false ]; then
                RUN_SANITY=false; RUN_INSTALL=false; RUN_PIPELINES=false
                RUN_STANDALONE=false; RUN_GENAI=false; RUN_CPP=false
                EXPLICIT_SUITE_SELECTION=true
            fi
            RUN_PIPELINES=true
            RUN_STANDALONE=true
            # RUN_CPP is NOT auto-enabled by --apps; combine with --cpp to include C++ tests
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --sanity       Run only sanity checks (environment validation)"
            echo "  --install      Run only installation tests (resource validation)"
            echo "  --pipelines    Run only pipeline tests (functional tests)"
            echo "  --standalone   Run standalone app smoke tests"
            echo "  --genai        Run GenAI app tests"
            echo "  --apps LIST    Run only selected pipeline + standalone apps (comma-separated)"
            echo "  --no-download  Skip resource download step"
            echo "  --help, -h     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --apps detection"
            echo "  $0 --apps detection,pose_estimation"
            echo "  $0 --standalone"
            echo "  $0 --genai"
            echo "  $0 --cpp"
            echo "  $0 --cpp --apps object_detection"
            echo "  $0 --cpp --pipelines --standalone"
            echo "  $0 --pipelines --standalone"
            echo ""
            echo "Without options, runs: sanity -> install -> pipelines -> standalone"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build optional pytest app filter expression for pipeline/standalone tests
if [[ -n "$APPS_FILTER" ]]; then
    IFS=',' read -r -a APPS_ARRAY <<< "$APPS_FILTER"
    for raw_app in "${APPS_ARRAY[@]}"; do
        app="$(echo "$raw_app" | xargs)"
        if [[ -n "$app" ]]; then
            if [[ -z "$PYTEST_K_EXPR" ]]; then
                PYTEST_K_EXPR="$app"
            else
                PYTEST_K_EXPR+=" or $app"
            fi
        fi
    done

    if [[ -z "$PYTEST_K_EXPR" ]]; then
        echo "Error: --apps provided but no valid app names were parsed"
        exit 1
    fi
fi

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

# 1. Sanity Checks - Environment validation + config integrity
if [ "$RUN_SANITY" = true ]; then
    echo ""
    echo "--- Running Sanity Checks (Environment Validation) ---"
    if python -m pytest "${TESTS_DIR}/test_sanity_check.py" "${TESTS_DIR}/test_config_integrity.py" -v --log-cli-level=INFO; then
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
    PIPELINE_PYTEST_ARGS=("${TESTS_DIR}/test_runner.py" -v --log-cli-level=INFO)
    if [[ -n "$PYTEST_K_EXPR" ]]; then
        echo "Filtering pipeline tests to apps: ${APPS_FILTER}"
        PIPELINE_PYTEST_ARGS+=( -k "$PYTEST_K_EXPR" )
    fi

    if python -m pytest "${PIPELINE_PYTEST_ARGS[@]}"; then
        echo "✓ Pipeline tests passed"
    else
        echo "✗ Pipeline tests failed"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
fi

# 4. Standalone App Tests - Functional smoke tests
if [ "$RUN_STANDALONE" = true ]; then
    echo ""
    echo "--- Running Standalone App Tests ---"
    STANDALONE_PYTEST_ARGS=("${TESTS_DIR}/test_standalone_runner.py" -v --log-cli-level=INFO)
    if [[ -n "$PYTEST_K_EXPR" ]]; then
        echo "Filtering standalone tests to apps: ${APPS_FILTER}"
        STANDALONE_PYTEST_ARGS+=( -k "$PYTEST_K_EXPR" )
    fi

    if python -m pytest "${STANDALONE_PYTEST_ARGS[@]}"; then
        echo "✓ Standalone tests passed"
    else
        echo "✗ Standalone tests failed"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
fi

# 5. GenAI Tests
if [ "$RUN_GENAI" = true ]; then
    echo ""
    echo "--- Running GenAI Tests ---"
    if python -m pytest "${TESTS_DIR}/test_gen_ai.py" "${TESTS_DIR}/voice_assistant_unit_tests.py" -v --log-cli-level=INFO; then
        echo "✓ GenAI tests passed"
    else
        echo "✗ GenAI tests failed"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
fi

# 6. C++ App Tests — build + image + video + camera
if [ "$RUN_CPP" = true ]; then
    echo ""
    echo "--- Running C++ App Tests (build + image + video) ---"
    CPP_PYTEST_ARGS=("${TESTS_DIR}/test_cpp_runner.py" -v --log-cli-level=INFO
                     -m "cpp")
    if [[ -n "$PYTEST_K_EXPR" ]]; then
        echo "Filtering C++ tests to apps: ${APPS_FILTER}"
        CPP_PYTEST_ARGS+=( -k "$PYTEST_K_EXPR" )
    fi
    if python -m pytest "${CPP_PYTEST_ARGS[@]}"; then
        echo "✓ C++ tests passed"
    else
        echo "✗ C++ tests failed"
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
