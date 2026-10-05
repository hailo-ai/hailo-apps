#!/bin/bash
set -e

ONNXRUNTIME_VERSION="1.18.0"

# Auto-download ONNX Runtime if not already set
if [ -z "$ONNXRUNTIME_DIR" ]; then
    OS=$(uname -s)
    ARCH=$(uname -m)

    case "$ARCH" in
        aarch64|arm64) ONNX_ARCH="aarch64" ;;
        x86_64)        ONNX_ARCH="x64" ;;
        *)             echo "-E- Unsupported architecture: $ARCH"; exit 1 ;;
    esac

    case "$OS" in
        Linux)   ONNX_OS="linux";   INSTALL_DIR="/opt/onnxruntime-${ONNXRUNTIME_VERSION}" ;;
        MINGW*|MSYS*|CYGWIN*) ONNX_OS="win"; INSTALL_DIR="C:/opt/onnxruntime-${ONNXRUNTIME_VERSION}" ;;
        *)       echo "-E- Unsupported OS: $OS"; exit 1 ;;
    esac

    if [ -d "$INSTALL_DIR" ]; then
        echo "-I- ONNX Runtime found at $INSTALL_DIR"
    else
        echo "-I- Downloading ONNX Runtime ${ONNXRUNTIME_VERSION} for ${ONNX_OS}-${ONNX_ARCH}..."
        if [ "$ONNX_OS" = "win" ]; then
            ARCHIVE="onnxruntime-win-${ONNX_ARCH}-${ONNXRUNTIME_VERSION}.zip"
            curl -L "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ARCHIVE}" \
                 -o "/tmp/${ARCHIVE}"
            mkdir -p "$INSTALL_DIR"
            unzip -q "/tmp/${ARCHIVE}" -d "$(dirname "$INSTALL_DIR")"
            mv "$(dirname "$INSTALL_DIR")/onnxruntime-win-${ONNX_ARCH}-${ONNXRUNTIME_VERSION}" "$INSTALL_DIR"
            rm -f "/tmp/${ARCHIVE}"
        else
            TARBALL="onnxruntime-linux-${ONNX_ARCH}-${ONNXRUNTIME_VERSION}.tgz"
            curl -L "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${TARBALL}" \
                 -o "/tmp/${TARBALL}"
            sudo tar -xzf "/tmp/${TARBALL}" -C /opt/
            sudo mv "/opt/onnxruntime-linux-${ONNX_ARCH}-${ONNXRUNTIME_VERSION}" "$INSTALL_DIR"
            rm -f "/tmp/${TARBALL}"
        fi
        echo "-I- ONNX Runtime installed to $INSTALL_DIR"
    fi

    export ONNXRUNTIME_DIR="$INSTALL_DIR"
fi

mkdir -p build
cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release \
    ${ONNXRUNTIME_DIR:+-DONNXRUNTIME_DIR="$ONNXRUNTIME_DIR"}
cmake --build build -- -j"$(nproc)"

rm -f hailort.log
