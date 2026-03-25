"""
api.py
------
FastAPI server — the entry point for all incoming connections from the Mac client.

Responsibilities:
  1. Accept WebSocket connections from the Mac client
  2. Receive speaker-labeled utterance payloads from the VAD filters
  3. Pass audio to the transcription module (faster-whisper)
  4. Maintain per-session context and pass it to the inference module (Mistral 7B)
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

import logging
import base64
import binascii
import os
import time
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from transcription import Transcriber
from inference import SuggestionEngine
from protocol import InferenceMessage, UtteranceMessage

MAX_CONVERSATION_TURNS = int(os.getenv("MAX_CONVERSATION_TURNS", 20))
LATENCY_STATS_WINDOW = int(os.getenv("LATENCY_STATS_WINDOW", 200))
LATENCY_LOG_EVERY = int(os.getenv("LATENCY_LOG_EVERY", 20))

# ── Logging ────────────────────────────────────────────────────────────────────
# Structured logging is important for a server — it lets you see exactly what's
# happening when you're connected via RunPod's terminal or reading remote logs.
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("api")


class LatencyTracker:
    """Rolling latency tracker for p50/p95 visibility."""

    def __init__(self, window_size: int):
        self.window_size = max(1, window_size)
        self._samples: list[float] = []
        self._count = 0

    def add(self, latency_ms: float) -> None:
        self._count += 1
        self._samples.append(max(0.0, latency_ms))
        if len(self._samples) > self.window_size:
            self._samples = self._samples[-self.window_size:]

    def should_log(self) -> bool:
        return self._count % max(1, LATENCY_LOG_EVERY) == 0

    def summary(self) -> dict:
        if not self._samples:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0}
        sorted_samples = sorted(self._samples)
        p50_idx = int(round((len(sorted_samples) - 1) * 0.50))
        p95_idx = int(round((len(sorted_samples) - 1) * 0.95))
        return {
            "count": len(sorted_samples),
            "p50_ms": sorted_samples[p50_idx],
            "p95_ms": sorted_samples[p95_idx],
        }

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
      Client → Server:  JSON envelope containing speaker + sample rate + ts + base64 audio
      Server → Client:  JSON inference message with speaker + transcript + intent + suggestion

    Message format sent back to client:
      {
        "type":       "inference",
        "speaker":    "customer",             # customer | salesperson
        "transcript": "I'm not sure I need this right now",
        "intent":     "objection",            # objection | question | buying_signal | none
        "suggestion": "That's completely understandable...",
        "reasoning_short": "Customer raised pricing concern; reinforce ROI briefly.",
        "confidence": 0.82,
        "latency_ms": 1375.4
      }
    """
    await websocket.accept()
    client_host = websocket.client.host
    log.info(f"Client connected: {client_host}")
    conversation_turns: list[dict] = []
    latency_tracker = LatencyTracker(window_size=LATENCY_STATS_WINDOW)

    try:
        while True:
            # ── Receive utterance payload ─────────────────────────────────────
            raw_message = await websocket.receive_text()
            try:
                payload = UtteranceMessage.model_validate_json(raw_message)
            except ValidationError as e:
                log.warning("Rejecting invalid websocket payload from %s: %s", client_host, e.errors())
                continue

            try:
                raw_bytes = base64.b64decode(payload.audio_b64, validate=True)
                audio = np.frombuffer(raw_bytes, dtype=np.float32)
            except (binascii.Error, ValueError) as e:
                log.warning("Rejecting payload with invalid audio bytes from %s: %s", client_host, e)
                continue
            request_start = time.monotonic()
            end_to_end_latency_ms = max(0.0, (time.time() * 1000.0) - float(payload.ts_ms))

            log.info(
                "Received %s audio: %d samples (%.2fs)",
                payload.speaker,
                len(audio),
                len(audio) / payload.sample_rate,
            )

            # ── Transcribe ────────────────────────────────────────────────────
            transcript = transcriber.transcribe(audio)

            if not transcript.strip():
                log.info("Empty transcription — returning no-op inference")
                response = InferenceMessage(
                    speaker=payload.speaker,
                    transcript="",
                    intent="none",
                    suggestion="",
                    reasoning_short="",
                    confidence=0.0,
                    latency_ms=end_to_end_latency_ms,
                )
                await websocket.send_text(response.model_dump_json())
                continue

            log.info(f"Transcript: '{transcript}'")

            # ── Update rolling per-session memory ─────────────────────────────
            conversation_turns.append(
                {
                    "speaker": payload.speaker,
                    "transcript": transcript,
                    "ts_ms": payload.ts_ms,
                }
            )
            if len(conversation_turns) > MAX_CONVERSATION_TURNS:
                conversation_turns = conversation_turns[-MAX_CONVERSATION_TURNS:]

            # ── Generate suggestion (customer turns only) ─────────────────────
            if payload.speaker == "customer":
                prior_turns = conversation_turns[:-1]
                result = suggestion_engine.analyse(transcript, prior_turns)
            else:
                log.info("Skipping rebuttal inference for salesperson turn.")
                result = {"type": "none", "suggestion": "", "reasoning_short": "", "confidence": 0.0}

            # ── Send response ─────────────────────────────────────────────────
            # Always include the transcript so the client can display a live feed.
            # Suggestions are only generated on customer turns.
            intent = result.get("type", "none")
            if intent not in {"objection", "question", "buying_signal", "none"}:
                intent = "none"
            confidence = float(result.get("confidence", 0.0) or 0.0)
            confidence = max(0.0, min(1.0, confidence))

            response = InferenceMessage(
                speaker=payload.speaker,
                transcript=transcript,
                intent=intent,
                suggestion=result.get("suggestion", ""),
                reasoning_short=result.get("reasoning_short", ""),
                confidence=confidence,
                latency_ms=end_to_end_latency_ms,
            )

            await websocket.send_text(response.model_dump_json())
            processing_ms = (time.monotonic() - request_start) * 1000.0
            latency_tracker.add(end_to_end_latency_ms)
            log.info(
                "Sent response: intent=%s e2e_latency_ms=%.1f processing_ms=%.1f",
                intent,
                end_to_end_latency_ms,
                processing_ms,
            )
            if latency_tracker.should_log():
                stats = latency_tracker.summary()
                log.info(
                    "Latency snapshot (window=%d): p50=%.1fms p95=%.1fms",
                    stats["count"],
                    stats["p50_ms"],
                    stats["p95_ms"],
                )

    except WebSocketDisconnect:
        log.info(f"Client disconnected: {client_host}")
    except Exception as e:
        log.error(f"Error handling client {client_host}: {e}", exc_info=True)
        await websocket.close(code=1011)  # 1011 = Internal Error
