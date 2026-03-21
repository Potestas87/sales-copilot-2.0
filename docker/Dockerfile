# ── Single stage: Runtime + pre-built llama-cpp-python wheel ──────────────────
# We use pre-built CUDA wheels for llama-cpp-python instead of compiling from
# source. This is faster (no 10-min compile), more reliable, and keeps the
# Dockerfile simple. The wheel is compiled by the llama-cpp-python project
# against CUDA 12.1 and published at their GitHub releases index.
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install llama-cpp-python from the pre-built CUDA 12.1 wheel index.
# This avoids compiling from source entirely — the wheel already contains
# the GPU kernels compiled against CUDA 12.1.
RUN pip3 install --no-cache-dir \
    llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Install the remaining server dependencies
COPY requirements-server.txt /tmp/requirements-server.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements-server.txt

# ── App setup ──────────────────────────────────────────────────────────────────
WORKDIR /app

# Copy server code and config
# We copy these separately so that changing code doesn't invalidate
# the (expensive) dependency layers above.
COPY server/   ./server/
COPY config/   ./config/

# Models are NOT baked into the image — they're mounted as a volume at runtime.
# Baking them in would add ~8GB to the image and make it impractical to push/pull.
# See docker-compose.yml for the volume mount configuration.
RUN mkdir -p /app/models

# Expose the FastAPI port
EXPOSE 8000

# Health check — RunPod and any orchestrator can use this to verify readiness.
# start-period=120s gives the server time to load models before health checks begin.
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=120s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set working directory to server so relative imports work
WORKDIR /app/server

# Start the server
CMD ["python3", "main.py"]