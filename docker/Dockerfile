# ── Single stage: CUDA 12.8 runtime + pre-built cu124 wheel ──────────────────
# RTX 5090 (Blackwell) requires CUDA 12.8+ runtime. We pair the 12.8 runtime
# base image with the cu124 pre-built llama-cpp-python wheel (the newest
# official wheel available). CUDA 12.8 is backward-compatible with code
# compiled against 12.4, and the cu124 wheel includes PTX that the driver
# can JIT-compile for Blackwell's compute capability.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Tell the NVIDIA Container Runtime to pass through GPU drivers and CUDA libraries.
# Without these, the container can't find libcuda.so.1 on cloud GPU providers like RunPod.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install llama-cpp-python from the pre-built CUDA 12.4 wheel index.
# This is the newest official pre-built wheel. Combined with the CUDA 12.8
# runtime base image, this gives us GPU acceleration on RTX 5090.
#
# --only-binary=:all: prevents pip from falling back to source compilation,
# which can OOM on lower-memory build machines. If a wheel isn't available,
# the build fails fast with a clear error instead of compiling.
RUN pip3 install --no-cache-dir --only-binary=:all: \
    llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Install the remaining server dependencies
COPY requirements-server.txt /tmp/requirements-server.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements-server.txt

# ── App setup ──────────────────────────────────────────────────────────────────
WORKDIR /app

# Copy server code and config
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
