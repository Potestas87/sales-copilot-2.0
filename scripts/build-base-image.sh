#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# build-base-image.sh — Build the Sales Copilot base Docker image on a GPU pod
# ══════════════════════════════════════════════════════════════════════════════
#
# WHY THIS EXISTS
# ───────────────
# The base image contains llama-cpp-python compiled from source with CUDA GPU
# support. Compiling it requires a working NVIDIA driver (libcuda.so.1) at link
# time, which doesn't exist in driverless CI environments like GitHub Actions.
#
# This script is designed to run on a RunPod GPU pod (or any machine with an
# NVIDIA GPU and CUDA toolkit). It builds a pre-compiled wheel, installs all
# dependencies into a clean base image, and pushes it to Docker Hub.
#
# WHEN TO RUN
# ───────────
# Only when you change heavy dependencies:
#   - Upgrade llama-cpp-python or faster-whisper versions
#   - Change CUDA version
#   - Add/remove packages in requirements-server.txt
#
# Day-to-day code changes (server/, config/, scripts/) are handled by the
# GitHub Actions app image workflow — no need to rebuild the base.
#
# HOW TO RUN
# ──────────
# 1. Spin up a RunPod GPU pod (any NVIDIA GPU, e.g. RTX 4090/5090)
#    Template: runpod/pytorch (any version with CUDA devel)
#    Container disk: >= 50 GB
#
# 2. Open the web terminal or SSH in
#
# 3. Clone the repo and run this script:
#      cd /tmp
#      git clone https://github.com/Potestas87/sales-copilot-2.0.git
#      cd sales-copilot-2.0
#      bash scripts/build-base-image.sh
#
# 4. When prompted, enter your Docker Hub access token
#
# 5. Stop and terminate the pod when done
#
# ARCHITECTURE NOTES
# ──────────────────
# - Compiles CUDA kernels for sm_75, sm_80, sm_86, sm_89 (Turing → Ada)
# - Includes PTX for sm_89, which newer GPUs (Blackwell/sm_120a) JIT-compile
#   at first launch via NVIDIA's forward-compatibility mechanism
# - The pod's CUDA toolkit version doesn't need to match the base image —
#   we compile natively on whatever the pod provides
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
DOCKER_USER="${DOCKER_USER:-potestas87}"
IMAGE_NAME="${DOCKER_USER}/sales-copilot-base"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-75;80;86;89}"

echo "═══════════════════════════════════════════════════════════"
echo "  Sales Copilot — Base Image Builder"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Image:  ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  CUDA architectures: ${CUDA_ARCHITECTURES}"
echo ""

# ── Step 1: Verify GPU and CUDA ──────────────────────────────────────────────
echo "▶ Step 1/6: Verifying GPU and CUDA toolkit..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. This script must run on a machine with CUDA toolkit installed."
    exit 1
fi
echo "  nvcc version: $(nvcc --version | grep release | awk '{print $6}')"
echo "  GPU detected:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available, continuing anyway)"
echo ""

# ── Step 2: Build llama-cpp-python wheel with CUDA ───────────────────────────
echo "▶ Step 2/6: Building llama-cpp-python wheel with CUDA support..."
echo "  This is the slow step (~5-15 minutes). Compiling for: ${CUDA_ARCHITECTURES}"
echo ""

WHEEL_DIR="/tmp/llama-wheels"
mkdir -p "${WHEEL_DIR}"

CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    pip wheel --no-binary=llama-cpp-python llama-cpp-python -w "${WHEEL_DIR}/"

WHEEL_FILE=$(ls "${WHEEL_DIR}"/llama_cpp_python-*.whl 2>/dev/null | head -1)
if [ -z "${WHEEL_FILE}" ]; then
    echo "ERROR: Wheel build failed — no .whl file found in ${WHEEL_DIR}"
    exit 1
fi
echo ""
echo "  ✓ Wheel built: ${WHEEL_FILE}"
echo ""

# ── Step 3: Install buildah ──────────────────────────────────────────────────
echo "▶ Step 3/6: Installing buildah..."
if ! command -v buildah &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq buildah > /dev/null 2>&1
fi
echo "  ✓ buildah ready"
echo ""

# ── Step 4: Build the base image ─────────────────────────────────────────────
echo "▶ Step 4/6: Building base image..."

# Create a temporary Dockerfile that installs the pre-compiled wheel
TMPDIR=$(mktemp -d)
cp "${WHEEL_FILE}" "${TMPDIR}/"
cp requirements-server.txt "${TMPDIR}/"
WHEEL_BASENAME=$(basename "${WHEEL_FILE}")

cat > "${TMPDIR}/Dockerfile" << DOCKERFILE
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# System dependencies
RUN apt-get update && apt-get install -y \\
    python3 python3-pip curl \\
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Install pre-compiled llama-cpp-python (CUDA baked in)
COPY ${WHEEL_BASENAME} /tmp/${WHEEL_BASENAME}
RUN python3 -m pip install /tmp/${WHEEL_BASENAME} && rm /tmp/${WHEEL_BASENAME}

# Install remaining Python dependencies
COPY requirements-server.txt /tmp/requirements-server.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-server.txt

LABEL maintainer="${DOCKER_USER}"
LABEL description="Sales Copilot base image — CUDA + llama-cpp-python (GPU) + faster-whisper"
LABEL build.cuda_architectures="${CUDA_ARCHITECTURES}"
LABEL build.date="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
DOCKERFILE

# Try buildah first, fall back to docker
if buildah build -t "${IMAGE_NAME}:${IMAGE_TAG}" "${TMPDIR}" 2>/dev/null; then
    BUILD_TOOL="buildah"
elif docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" "${TMPDIR}" 2>/dev/null; then
    BUILD_TOOL="docker"
else
    echo "ERROR: Neither buildah nor docker could build the image."
    echo "  The pre-compiled wheel is saved at: ${WHEEL_FILE}"
    echo "  You can manually build with it on a machine that has Docker."
    exit 1
fi
echo "  ✓ Image built with ${BUILD_TOOL}"
echo ""

# ── Step 5: Push to Docker Hub ───────────────────────────────────────────────
echo "▶ Step 5/6: Pushing to Docker Hub..."
echo "  You may be prompted for your Docker Hub credentials."
echo ""

if [ "${BUILD_TOOL}" = "buildah" ]; then
    buildah login -u "${DOCKER_USER}" docker.io
    buildah push "${IMAGE_NAME}:${IMAGE_TAG}" "docker://${IMAGE_NAME}:${IMAGE_TAG}"
else
    docker login -u "${DOCKER_USER}"
    docker push "${IMAGE_NAME}:${IMAGE_TAG}"
fi
echo ""
echo "  ✓ Pushed ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# ── Step 6: Cleanup ──────────────────────────────────────────────────────────
echo "▶ Step 6/6: Cleaning up..."
rm -rf "${TMPDIR}" "${WHEEL_DIR}"
echo "  ✓ Done"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "  Base image pushed successfully!"
echo ""
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "  Next steps:"
echo "  1. Push code changes → GitHub Actions builds the app image"
echo "  2. The app image inherits FROM this base image"
echo "  3. Deploy to RunPod"
echo ""
echo "  You can now stop and terminate this build pod."
echo "═══════════════════════════════════════════════════════════"
