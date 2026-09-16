#!/bin/bash
# entrypoint.sh — RunPod container startup script
# Downloads the Mistral model if it doesn't exist, then starts the server.
# This solves the chicken-and-egg problem on RunPod where we can't SSH in
# to download models if the container keeps crashing.

set -e

MODEL_DIR="/workspace/models"
MODEL_FILE="mistral-7b-instruct-v0.2.Q4_K_M.secondstate.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MODEL_URL="https://huggingface.co/second-state/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf"

# llama.cpp's loader (threaded/random-access I/O) crashes silently and
# deterministically when reading the model directly off RunPod's
# persistent volume mount - reproduced across two different GGUF sources,
# CPU-only and GPU-offloaded loading, and two GPU types, even though a
# plain sequential `dd` read of the same file succeeds instantly. Keep the
# download cached on the persistent volume, but load from a local copy on
# the container's own ephemeral disk to work around the volume's I/O
# incompatibility.
LOCAL_MODEL_PATH="/app/models/${MODEL_FILE}"

# Create model directories
mkdir -p "${MODEL_DIR}"
mkdir -p "$(dirname "${LOCAL_MODEL_PATH}")"

# Download the model if it doesn't already exist
if [ ! -f "${MODEL_PATH}" ]; then
    echo "=== Model not found at ${MODEL_PATH} ==="
    echo "=== Downloading Mistral 7B (~4.5 GB)... ==="
    curl -L --progress-bar -o "${MODEL_PATH}" "${MODEL_URL}"
    echo "=== Download complete ==="
else
    echo "=== Model already exists at ${MODEL_PATH}, skipping download ==="
fi

echo "=== Copying model to local container disk ==="
cp "${MODEL_PATH}" "${LOCAL_MODEL_PATH}"
echo "=== Local copy ready at ${LOCAL_MODEL_PATH} ==="

# Export the model path so the server can find it
export LLM_MODEL_PATH="${LOCAL_MODEL_PATH}"

# Start the FastAPI server
if ! command -v python3.11 >/dev/null 2>&1; then
    echo "ERROR: python3.11 not found in container."
    exit 1
fi

echo "=== Starting server ==="
cd /app/server
exec python3.11 main.py
