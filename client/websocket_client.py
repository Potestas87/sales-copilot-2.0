"""
websocket_client.py
-------------------
Manages the WebSocket connection from the Mac client to the GPU server.

This module sits between the VAD filter and the display UI:
  VAD filter  →  WebSocketClient.send(audio)  →  GPU server
  GPU server  →  on_response callback          →  display.py

Responsibilities:
  - Open and maintain a persistent WebSocket connection to the server
  - Send audio utterances as speaker-labeled JSON envelopes when VAD emits them
  - Receive JSON responses (transcript + suggestion) from the server
  - Fire a callback so the display layer can show suggestions immediately
  - Reconnect automatically if the connection drops mid-call

Why async?
  WebSocket I/O is inherently asynchronous — we can't block waiting for
  a response while also being ready to send the next utterance. asyncio
  lets us do both concurrently on a single thread without complexity.
"""

import asyncio
import logging
import os
import threading
import time
from typing import Callable, Optional

import numpy as np
import websockets
from dotenv import load_dotenv

from protocol import InferenceMessage, UtteranceMessage

load_dotenv()
log = logging.getLogger("websocket_client")

RECONNECT_DELAY = 3    # Seconds to wait before attempting reconnection
MAX_RETRIES     = 10   # Maximum reconnection attempts before giving up
SEND_QUEUE_MAXSIZE = int(os.getenv("WS_SEND_QUEUE_MAXSIZE", 64))


class WebSocketClient:
    """
    Sends audio utterances to the GPU server and receives suggestion responses.

    Runs an asyncio event loop in a dedicated background thread so the rest
    of the client (audio capture, VAD, UI) can stay synchronous and simple.

    Usage:
        def on_response(data: dict):
            # Called when server sends back a transcript + suggestion
            display.show(data["suggestion"], data["type"])

        client = WebSocketClient(on_response=on_response)
        client.start()

        # From the VAD callback:
        client.send(audio_array)   # thread-safe, non-blocking

        client.stop()
    """

    def __init__(
        self,
        on_response: Callable[[dict], None],
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """
        Args:
            on_response:  Callback fired with each server response dict.
                          Called from the asyncio thread — keep it fast.
                          Dict keys: "transcript", "type", "suggestion"
            host:         Server hostname (defaults to SERVER_HOST in .env)
            port:         Server port    (defaults to SERVER_PORT in .env)
        """
        self.host        = host or os.getenv("SERVER_HOST", "localhost")
        self.port        = int(port or os.getenv("SERVER_PORT", 8000))
        self.on_response = on_response
        # SERVER_URL overrides the constructed URI (needed for RunPod proxy / WSS)
        self._uri        = os.getenv("SERVER_URL") or f"ws://{self.host}:{self.port}/ws"

        # asyncio infrastructure
        self._loop:   Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread]          = None
        self._ws                                          = None
        self._running = False

        # Queue bridges the sync world (VAD callback) and async world (websocket send)
        # The VAD runs in a sync thread; we can't call async functions from there directly.
        # Instead, VAD puts audio into this queue and the async loop drains it.
        self._send_queue: asyncio.Queue = None
        self._send_queue_maxsize = SEND_QUEUE_MAXSIZE

        # Transport metrics (useful for tuning queue size and reconnect behavior)
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "enqueued": 0,
            "sent": 0,
            "received": 0,
            "dropped_not_running": 0,
            "dropped_loop_unavailable": 0,
            "dropped_queue_full": 0,
            "send_failures": 0,
        }

    # ── Public API (called from sync threads) ─────────────────────────────────
    def send(self, audio: np.ndarray, speaker: str = "customer", sample_rate: int = 16000) -> None:
        """
        Queue an audio utterance for sending to the server.
        Thread-safe — can be called from the VAD thread or anywhere else.

        Audio is serialized into a JSON payload with metadata:
          type, speaker, sample_rate, timestamp, and base64-encoded float32 audio.
        """
        if not self._running or self._loop is None:
            self._inc_metric("dropped_not_running")
            log.warning("WebSocketClient not running — dropping audio chunk")
            return

        if self._loop.is_closed():
            self._inc_metric("dropped_loop_unavailable")
            log.warning("WebSocket event loop is closed — dropping audio chunk")
            return

        payload = UtteranceMessage.from_audio(
            audio=audio,
            speaker=speaker,
            sample_rate=sample_rate,
            ts_ms=int(time.time() * 1000),
        ).to_json()

        try:
            self._loop.call_soon_threadsafe(self._enqueue_payload, payload)
        except RuntimeError:
            # Race: loop can close between the is_closed() check and call_soon_threadsafe().
            self._inc_metric("dropped_loop_unavailable")
            log.warning("WebSocket event loop unavailable during enqueue — dropping audio chunk")

    def start(self) -> None:
        """Start the background asyncio thread and open the WebSocket connection."""
        if self._running:
            log.warning("WebSocketClient already running.")
            return

        self._running = True
        self._thread  = threading.Thread(
            target  = self._run_event_loop,
            daemon  = True,
            name    = "websocket-thread",
        )
        self._thread.start()
        log.info(f"WebSocketClient started — connecting to {self._uri}")

    def stop(self) -> None:
        """Signal the background thread to shut down and wait for it."""
        self._running = False
        if self._loop:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass
        if self._thread:
            self._thread.join(timeout=5.0)
        log.info("WebSocketClient stopped. Metrics=%s", self._snapshot_metrics())

    # ── Async internals ────────────────────────────────────────────────────────
    def _run_event_loop(self) -> None:
        """
        Entry point for the background thread.
        Creates a new asyncio event loop (each thread needs its own)
        and runs the main connection coroutine inside it.
        """
        self._loop       = asyncio.new_event_loop()
        self._send_queue = asyncio.Queue(maxsize=self._send_queue_maxsize)
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._connect_with_retry())
        except Exception as e:
            log.error(f"WebSocket event loop crashed: {e}", exc_info=True)
        finally:
            self._loop.close()
            self._loop = None
            self._ws = None
            self._send_queue = None

    async def _connect_with_retry(self) -> None:
        """
        Connection loop — attempts to connect and reconnects on failure.

        Why retry logic?
          In real use, RunPod pods occasionally restart or the network hiccups.
          Without retry logic, one dropped connection ends the session.
          With it, the client silently reconnects within a few seconds.
        """
        retries = 0

        while self._running and retries < MAX_RETRIES:
            try:
                log.info(f"Connecting to {self._uri}...")
                async with websockets.connect(self._uri) as ws:
                    self._ws = ws
                    retries  = 0    # Reset retry count on successful connection
                    log.info("Connected to server.")
                    await self._session(ws)

            except (websockets.exceptions.ConnectionClosed,
                    websockets.exceptions.WebSocketException,
                    OSError) as e:
                retries += 1
                log.warning(f"Connection lost: {e}. Retry {retries}/{MAX_RETRIES} "
                            f"in {RECONNECT_DELAY}s...")
                await asyncio.sleep(RECONNECT_DELAY)

        if retries >= MAX_RETRIES:
            log.error(f"Giving up after {MAX_RETRIES} failed connection attempts.")
            self._running = False

    async def _session(self, ws) -> None:
        """
        Active session — runs two concurrent tasks:
          _sender:   drains the send queue and sends audio to the server
          _receiver: listens for responses and fires the on_response callback

        asyncio.gather() runs them concurrently. If either task raises an
        exception (e.g. connection closed), gather() cancels the other.
        """
        sender_task   = asyncio.create_task(self._sender(ws))
        receiver_task = asyncio.create_task(self._receiver(ws))

        try:
            await asyncio.gather(sender_task, receiver_task)
        except Exception:
            sender_task.cancel()
            receiver_task.cancel()
            raise

    async def _sender(self, ws) -> None:
        """
        Drains the send queue and sends each audio chunk to the server.
        Waits efficiently using queue.get() — zero CPU usage while idle.
        """
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._send_queue.get(),
                    timeout=1.0,
                )
                await ws.send(message)
                self._inc_metric("sent")
            except asyncio.TimeoutError:
                continue    # No audio in queue — loop back and wait
            except Exception:
                self._inc_metric("send_failures")
                raise

    async def _receiver(self, ws) -> None:
        """
        Listens for JSON messages from the server and fires on_response.
        Runs continuously until the connection closes.
        """
        async for message in ws:
            try:
                event = InferenceMessage.from_json(message)
                data = {
                    "transcript": event.transcript,
                    "type": event.intent,
                    "intent": event.intent,
                    "speaker": event.speaker,
                    "suggestion": event.suggestion,
                    "reasoning_short": event.reasoning_short,
                    "confidence": event.confidence,
                    "latency_ms": event.latency_ms,
                }
                log.info(
                    "Received inference: speaker=%s intent=%s latency_ms=%.1f transcript='%s...'",
                    event.speaker,
                    event.intent,
                    event.latency_ms,
                    event.transcript[:50],
                )
                self._inc_metric("received")
                self.on_response(data)
            except Exception as e:
                log.warning(f"Could not parse server message: {e}")

    # ── Queue internals and metrics ────────────────────────────────────────────
    def _enqueue_payload(self, payload: str) -> None:
        """
        Enqueue payload from the event loop thread.

        Drop policy when full:
          - Drop the oldest queued item and enqueue the newest item.
        This favors fresh realtime context over stale utterances.
        """
        if self._send_queue is None:
            self._inc_metric("dropped_loop_unavailable")
            return

        try:
            self._send_queue.put_nowait(payload)
            self._inc_metric("enqueued")
            return
        except asyncio.QueueFull:
            self._inc_metric("dropped_queue_full")

        try:
            self._send_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        try:
            self._send_queue.put_nowait(payload)
            self._inc_metric("enqueued")
        except asyncio.QueueFull:
            self._inc_metric("dropped_queue_full")

    def _inc_metric(self, key: str, value: int = 1) -> None:
        with self._metrics_lock:
            self._metrics[key] = self._metrics.get(key, 0) + value

    def _snapshot_metrics(self) -> dict:
        with self._metrics_lock:
            return dict(self._metrics)
