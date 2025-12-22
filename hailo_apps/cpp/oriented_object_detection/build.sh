#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

function init_variables() {
    readonly BUILD_DIR
}

function build() {
    mkdir -p "$BUILD_DIR"

    pushd "$BUILD_DIR"

    cmake ..
    cmake --build . --config release

    popd
}

init_variables
build
