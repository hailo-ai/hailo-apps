#!/bin/bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# The repo root makes hailo_apps importable in a standalone (not pip-installed) setup.
PYTHONPATH="$(cd ../../../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}" \
    python -m hailo_apps.installation.download_resources --group v2a_demo --arch hailo10h

mkdir -p resources
cd resources

BASE_URL="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/v2a_demo"

FILES=(
    "en_US-joe-medium.onnx"
    "en_US-joe-medium.onnx.json"
    "go_hailo.onnx"
    "hey_hailo.onnx"
    "hey_hailo_v3.onnx"
    "word_embeddings_weight.npy"
)

for file in "${FILES[@]}"; do
    wget "${BASE_URL}/${file}"
done

cd ..
