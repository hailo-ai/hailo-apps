#!/usr/bin/env bash
set -euo pipefail

# ONNX Runtime setup is handled by build.sh.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
wget -O yolov8m-seg_post.onnx \
    https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/onnxs/yolov8m-seg_post.onnx
