# ══════════════════════════════════════════════════════════════════════════════
# Dockerfile — Lightweight app image built on top of the base image
# ══════════════════════════════════════════════════════════════════════════════
#
# This image only copies application code and config on top of the pre-built
# base image (which already has CUDA, llama-cpp-python, faster-whisper, etc).
#
# Because there's no compilation, this builds in < 60 seconds on any CI
# runner — including GitHub Actions' free-tier 2-core machines.
#
# Rebuild the BASE image (docker/Dockerfile.base) only when dependencies change.
# Rebuild THIS image on every code push — it's fast and cheap.
#
# Build:
#   docker build -t potestas87/sales-copilot:latest -f docker/Dockerfile .
#
# ══════════════════════════════════════════════════════════════════════════════

FROM potestas87/sales-copilot-base:latest

# ── App setup ────────────────────────────────────────────────────────────────
WORKDIR /app

# Copy server code, config (including sales playbook), and entrypoint
COPY server/   ./server/
COPY config/   ./config/
COPY scripts/entrypoint.sh ./entrypoint.sh

# Models are NOT baked into the image — they're downloaded at first startup
# to the persistent volume (/workspace on RunPod). This keeps the image lean.
RUN mkdir -p /app/models && chmod +x /app/entrypoint.sh

# Fail fast at build-time if critical runtime deps are missing from the base image.
RUN python3.11 --version && python3.11 -c "import fastapi, uvicorn; print('runtime deps ok')"

# Expose the FastAPI port
EXPOSE 8000

# Health check — RunPod and any orchestrator can use this to verify readiness.
# start-period=300s gives time to download the model (~4.5 GB) on first boot.
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=300s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# The entrypoint script downloads the model if missing, then starts the server.
CMD ["/app/entrypoint.sh"]
