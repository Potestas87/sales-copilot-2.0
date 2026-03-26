# ── CUDA 12.8 devel image: compile llama-cpp-python with full GPU support ─────
# RTX 5090 (Blackwell) requires CUDA 12.8+. We use the *devel* base image
# (not runtime) because it ships with nvcc, CUDA headers, and cuBLAS — all
# needed to compile llama-cpp-python from source with GGML_CUDA=ON.
#
# The devel image is larger (~3 GB vs ~1 GB for runtime), but it's the only
# reliable way to get GPU-accelerated llama-cpp-python. Pre-built cu124 wheels
# silently fall back to CPU-only inference on Blackwell GPUs.
#
# DevOps note: In production you'd use a multi-stage build to compile in a
# devel stage and copy artifacts into a slim runtime stage. We keep it single-
# stage here for debuggability — you can SSH in and recompile if needed.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Tell the NVIDIA Container Runtime to expose GPUs and CUDA libraries.
# Without these, the container can't find libcuda.so.1 on cloud GPU providers.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    cmake \
    && rm -rf /var/lib/apt-lists/*

# Upgrade pip + install build toolchain so llama-cpp-python compiles cleanly.
# Without these, pip tries to build scikit-build-core/hatchling from source
# and fails on the dependency chain (pluggy → setuptools-scm → etc).
RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install scikit-build-core hatchling pluggy setuptools-scm ninja

# ── Compile llama-cpp-python from source with CUDA ───────────────────────────
# --no-binary=llama-cpp-python  → forces source compilation for this package
#                                  while still using pre-built wheels for deps
# CMAKE_ARGS="-DGGML_CUDA=ON"  → enables CUDA kernels in the llama.cpp backend
# CUDACXX=/usr/local/cuda/bin/nvcc → tells cmake where to find the CUDA compiler
#
# This takes 5-10 minutes to build. The result is a llama-cpp-python that
# offloads all model layers to GPU (n_gpu_layers=-1).
ENV CMAKE_ARGS="-DGGML_CUDA=ON"
ENV CUDACXX=/usr/local/cuda/bin/nvcc
RUN pip3 install --no-cache-dir --no-binary=llama-cpp-python llama-cpp-python

# ── Install remaining server dependencies ────────────────────────────────────
COPY requirements-server.txt /tmp/requirements-server.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements-server.txt

# ── App setup ────────────────────────────────────────────────────────────────
WORKDIR /app

# Copy server code, config (including sales playbook), and entrypoint
COPY server/   ./server/
COPY config/   ./config/
COPY scripts/entrypoint.sh ./entrypoint.sh

# Models are NOT baked into the image — they're downloaded at first startup
# to the persistent volume (/workspace on RunPod). This keeps the image lean.
RUN mkdir -p /app/models && chmod +x /app/entrypoint.sh

# Expose the FastAPI port
EXPOSE 8000

# Health check — RunPod and any orchestrator can use this to verify readiness.
# start-period=300s gives time to download the model (~4.5GB) on first boot.
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=300s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# The entrypoint script downloads the model if missing, then starts the server.
CMD ["/app/entrypoint.sh"]
