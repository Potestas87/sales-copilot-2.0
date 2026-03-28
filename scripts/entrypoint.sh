#!/bin/bash
# entrypoint.sh — RunPod container startup script
# Downloads the Mistral model if it doesn't exist, then starts the server.
# This solves the chicken-and-egg problem on RunPod where we can't SSH in
# to download models if the container keeps crashing.

set -e

MODEL_DIR="/workspace/models"
MODEL_FILE="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/${MODEL_FILE}"

# Create model directory on the persistent volume
mkdir -p "${MODEL_DIR}"

# Download the model if it doesn't already exist
if [ ! -f "${MODEL_PATH}" ]; then
    echo "=== Model not found at ${MODEL_PATH} ==="
    echo "=== Downloading Mistral 7B (~4.5 GB)... ==="
    curl -L --progress-bar -o "${MODEL_PATH}" "${MODEL_URL}"
    echo "=== Download complete ==="
else
    echo "=== Model already exists at ${MODEL_PATH}, skipping download ==="
fi

# Export the model path so the server can find it
export LLM_MODEL_PATH="${MODEL_PATH}"

# Start the FastAPI server
if ! command -v python3.11 >/dev/null 2>&1; then
    echo "ERROR: python3.11 not found in container."
    exit 1
fi

echo "=== Starting server ==="
cd /app/server
exec python3.11 main.py
