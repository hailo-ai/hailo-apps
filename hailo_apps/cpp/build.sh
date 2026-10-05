#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CLEAN=false
ARGS=()
for arg in "$@"; do
    [[ "$arg" == "--rebuild" ]] && CLEAN=true || ARGS+=("$arg")
done
set -- "${ARGS[@]}"

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: ./build.sh [--rebuild] [app1 app2 ...]"
    echo ""
    echo "Options:"
    echo "  --rebuild   Remove build directories before building"
    echo ""
    echo "Apps:"
    echo "  classification, depth_estimation_mono, depth_estimation_stereo,"
    echo "  instance_segmentation, object_detection, onnxrt_hailo_pipeline,"
    echo "  oriented_object_detection, pose_estimation, semantic_segmentation,"
    echo "  zero_shot_classification"
    echo ""
    echo "Examples:"
    echo "  ./build.sh                                                    # build all"
    echo "  ./build.sh object_detection                                   # build one"
    echo "  ./build.sh object_detection instance_segmentation             # build multiple"
    echo "  ./build.sh --rebuild object_detection                         # clean + build one"
    echo "  ./build.sh --rebuild object_detection instance_segmentation   # clean + build multiple"
    echo "  ./build.sh --rebuild                                          # clean + build all"
    exit 0
fi

DEPS_INSTALL_DIR="$SCRIPT_DIR/deps"
YAML_CPP_SUBMODULE="$SCRIPT_DIR/external/yaml-cpp"
if [ ! -f "$YAML_CPP_SUBMODULE/CMakeLists.txt" ]; then
    echo ""
    echo "ERROR: yaml-cpp submodule is not initialized."
    echo "  Run: git submodule update --init --recursive"
    echo ""
    exit 1
fi

# Build yaml-cpp once into hailo_apps/cpp/deps/ (skipped if system or already built)
if pkg-config --exists yaml-cpp 2>/dev/null; then
    echo "-I- yaml-cpp: system ($(pkg-config --modversion yaml-cpp)), skipping"
elif [ -f "$DEPS_INSTALL_DIR/lib/libyaml-cpp.a" ] || [ -f "$DEPS_INSTALL_DIR/lib/libyaml-cpp.so" ]; then
    echo "-I- yaml-cpp: already built, skipping"
else
    echo "=========================================="
    echo " Building yaml-cpp (one-time)"
    echo "=========================================="
    cmake -S "$YAML_CPP_SUBMODULE" \
          -B "$SCRIPT_DIR/external/yaml-cpp/build" \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL_DIR" \
          -DYAML_CPP_BUILD_TESTS=OFF \
          -DYAML_CPP_BUILD_TOOLS=OFF \
          -DYAML_CPP_BUILD_CONTRIB=OFF
    cmake --build "$SCRIPT_DIR/external/yaml-cpp/build" -- -j"$(nproc)"
    cmake --install "$SCRIPT_DIR/external/yaml-cpp/build"
    echo "-I- yaml-cpp: installed to $DEPS_INSTALL_DIR"
fi

CURL_SUBMODULE="$SCRIPT_DIR/external/curl"
if [ ! -f "$CURL_SUBMODULE/CMakeLists.txt" ]; then
    echo ""
    echo "ERROR: curl submodule is not initialized."
    echo "  Run: git submodule update --init --recursive"
    echo ""
    exit 1
fi
if pkg-config --exists libcurl 2>/dev/null; then
    echo "-I- curl: system ($(pkg-config --modversion libcurl)), skipping"
elif [ -f "$DEPS_INSTALL_DIR/lib/libcurl.a" ] || [ -f "$DEPS_INSTALL_DIR/lib/libcurl.so" ]; then
    echo "-I- curl: already built, skipping"
else
    if ! pkg-config --exists openssl 2>/dev/null && [ ! -f /usr/include/openssl/ssl.h ]; then
        echo ""
        echo "ERROR: OpenSSL development headers not found (required to build curl)."
        echo "  Run: sudo apt-get install -y libssl-dev"
        echo ""
        exit 1
    fi
    echo "=========================================="
    echo " Building curl (one-time)"
    echo "=========================================="
    cmake -S "$CURL_SUBMODULE" \
          -B "$SCRIPT_DIR/external/curl/build" \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL_DIR" \
          -DBUILD_CURL_EXE=OFF \
          -DBUILD_SHARED_LIBS=OFF \
          -DCURL_DISABLE_TESTS=ON \
          -DCURL_USE_LIBPSL=OFF
    cmake --build "$SCRIPT_DIR/external/curl/build" -- -j"$(nproc)"
    cmake --install "$SCRIPT_DIR/external/curl/build"
    echo "-I- curl: installed to $DEPS_INSTALL_DIR"
fi

export CMAKE_PREFIX_PATH="$DEPS_INSTALL_DIR:${CMAKE_PREFIX_PATH:-}"
export CXXFLAGS="-w"

ALL_APPS=(
    classification
    depth_estimation_mono
    depth_estimation_stereo
    instance_segmentation
    object_detection
    onnxrt_hailo_pipeline
    oriented_object_detection
    pose_estimation
    semantic_segmentation
    zero_shot_classification
)

# Use args if provided, otherwise build all
if [ $# -gt 0 ]; then
    APPS=()
    for arg in "$@"; do
        APPS+=("${arg%/}")  # strip trailing slash
    done
else
    APPS=("${ALL_APPS[@]}")
fi

FAILED=()

for APP in "${APPS[@]}"; do
    echo ""
    echo "=========================================="
    echo " Building: $APP"
    echo "=========================================="
    if $CLEAN; then
        rm -rf "$SCRIPT_DIR/$APP/build"
        echo "-I- $APP: build dir cleaned"
    fi
    if (cd "$SCRIPT_DIR/$APP" && bash build.sh); then
        echo "-I- $APP: OK"
    else
        echo "-E- $APP: FAILED"
        FAILED+=("$APP")
    fi
done

PASSED=()
for APP in "${APPS[@]}"; do
    [[ ! " ${FAILED[*]} " =~ " ${APP} " ]] && PASSED+=("$APP")
done

echo ""
echo "=========================================="
echo " Build Summary"
echo "=========================================="
if [[ ${#FAILED[@]} -eq 0 ]]; then
    for APP in "${APPS[@]}"; do echo "  ✓  $APP"; done
else
    for APP in "${PASSED[@]}"; do echo "  ✓  $APP"; done
    for APP in "${FAILED[@]}"; do echo "  ✗  $APP"; done
    exit 1
fi
echo "=========================================="
