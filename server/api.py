"""
api.py
------
FastAPI server — the entry point for all incoming connections from the Mac client.

Responsibilities:
  1. Accept WebSocket connections from the Mac client
  2. Receive raw audio bytes streamed from the VAD filter
  3. Pass audio to the transcription module (faster-whisper)
  4. Pass transcript to the inference module (Mistral 7B)
  5. Send suggestions back to the client over the same WebSocket

Why FastAPI + WebSockets?
  FastAPI is async-native, which means it can handle multiple concurrent
  connections without blocking. WebSockets give us a persistent bidirectional
  channel — the client streams audio in, suggestions stream back out, all on
  one open connection with minimal overhead.

Why a single WebSocket per session?
  Opening a new HTTP connection per audio chunk would add ~100ms of handshake
  overhead each time. One persistent WebSocket connection eliminates that.
"""

import json
import logging
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from transcription import Transcriber
from inference import SuggestionEngine

# ── Logging ────────────────────────────────────────────────────────────────────
# Structured logging is important for a server — it lets you see exactly what's
# happening when you're connected via RunPod's terminal or reading remote logs.
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("api")

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Sales Copilot Server",
    description = "Real-time sales call transcription and suggestion engine",
    version     = "0.1.0",
)

# CORS middleware — allows the Mac client to connect even if origins differ.
# In production you'd lock this down to your specific client IP.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Model loading ──────────────────────────────────────────────────────────────
# Models are loaded once at startup, not per-request.
# Loading a model takes several seconds — doing it per-request would be unusable.
# Both objects are module-level so all WebSocket handlers share them.
log.info("Loading transcription model...")
transcriber = Transcriber()
log.info("Transcription model ready.")

log.info("Loading suggestion engine...")
suggestion_engine = SuggestionEngine()
log.info("Suggestion engine ready.")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """
    Simple endpoint to verify the server is running and models are loaded.
    Useful for RunPod readiness checks and debugging connectivity.

    Call from your Mac with:
        curl http://<your-runpod-host>:8000/health
    """
    return {
        "status":     "ok",
        "models":     {
            "transcriber":       transcriber.model_size,
            "suggestion_engine": suggestion_engine.model_name,
        }
    }


# ── WebSocket handler ──────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint — one connection per active sales call.

    Protocol:
      Client → Server:  Raw float32 audio bytes (numpy array, 16kHz mono)
      Server → Client:  JSON string with transcription and/or suggestion

    Message format sent back to client:
      {
        "transcript": "I'm not sure I need this right now",
        "type":       "objection",            # objection | question | buying_signal | none
        "suggestion": "That's completely understandable..."
      }
    """
    await websocket.accept()
    client_host = websocket.client.host
    log.info(f"Client connected: {client_host}")

    try:
        while True:
            # ── Receive audio ─────────────────────────────────────────────────
            # The client sends raw float32 bytes — we reconstruct the numpy array.
            # frombuffer() does zero-copy conversion — no unnecessary data copying.
            raw_bytes  = await websocket.receive_bytes()
            audio      = np.frombuffer(raw_bytes, dtype=np.float32)

            log.info(f"Received audio: {len(audio)} samples ({len(audio)/16000:.2f}s)")

            # ── Transcribe ────────────────────────────────────────────────────
            transcript = transcriber.transcribe(audio)

            if not transcript.strip():
                log.info("Empty transcription — skipping inference")
                continue

            log.info(f"Transcript: '{transcript}'")

            # ── Generate suggestion ───────────────────────────────────────────
            result = suggestion_engine.analyse(transcript)

            # ── Send response ─────────────────────────────────────────────────
            # Always include the transcript so the client can display a live feed.
            # Only include a suggestion if the engine detected something actionable.
            response = {
                "transcript": transcript,
                "type":       result["type"],
                "suggestion": result.get("suggestion", ""),
            }

            await websocket.send_text(json.dumps(response))
            log.info(f"Sent response: type={result['type']}")

    except WebSocketDisconnect:
        log.info(f"Client disconnected: {client_host}")
    except Exception as e:
        log.error(f"Error handling client {client_host}: {e}", exc_info=True)
        await websocket.close(code=1011)  # 1011 = Internal Error