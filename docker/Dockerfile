# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# We use the CUDA devel image in the build stage because llama-cpp-python
# needs to compile C++ code against the CUDA toolkit (nvcc compiler, headers).
# The devel image is ~7GB but we won't ship it — only the compiled output moves
# to the runtime stage.
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install llama-cpp-python with CUDA support.
# CMAKE_ARGS="-DLLAMA_CUDA=on" tells the build system to compile GPU kernels.
# This step takes 5-10 minutes on first build but is cached by Docker layer
# caching — subsequent builds skip it unless requirements change.
RUN CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 \
    pip3 install --no-cache-dir llama-cpp-python>=0.2.0

# Install the remaining server dependencies
COPY requirements-server.txt /tmp/requirements-server.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements-server.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
# The runtime image is much smaller — it has CUDA libraries but not the
# full compiler toolchain. We copy the compiled Python packages from the
# builder stage, keeping the final image lean.
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime system dependencies only (no compilers)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

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