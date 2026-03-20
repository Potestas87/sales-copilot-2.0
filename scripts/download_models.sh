#!/bin/bash
# download_models.sh
# ------------------
# Downloads the required model files into the models/ directory.
# Run this once on any machine (local or RunPod) before starting the server.
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh
#
# Models downloaded:
#   - Mistral 7B Instruct v0.2 (Q4_K_M, 4-bit quantised) ~4.5GB
#   - Whisper large-v3 is downloaded automatically by faster-whisper on first run

set -e  # Exit immediately if any command fails

MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR"

echo "============================================"
echo "  Sales Copilot - Model Downloader"
echo "============================================"
echo "Models will be saved to: $MODELS_DIR"
echo ""

# Mistral 7B Instruct v0.2 (Q4_K_M)
# Source: TheBloke/Mistral-7B-Instruct-v0.2-GGUF on Hugging Face
# Q4_K_M = 4-bit quantisation, good balance of quality vs size and speed
MISTRAL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MISTRAL_FILE="$MODELS_DIR/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

if [ -f "$MISTRAL_FILE" ]; then
    echo "[Mistral 7B] Already downloaded - skipping."
else
    echo "[Mistral 7B] Downloading (~4.5GB) - this will take a few minutes..."
    curl -L --progress-bar -o "$MISTRAL_FILE" "$MISTRAL_URL"
    echo "[Mistral 7B] Done."
fi

echo ""
echo "[Whisper] faster-whisper downloads Whisper automatically on first run."
echo "          No manual download needed."
echo ""
echo "============================================"
echo "  All models ready. You can now start the server."
echo "============================================"