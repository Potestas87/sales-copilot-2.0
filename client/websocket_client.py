"""
websocket_client.py
-------------------
Manages the WebSocket connection from the Mac client to the GPU server.

This module sits between the VAD filter and the display UI:
  VAD filter  →  WebSocketClient.send(audio)  →  GPU server
  GPU server  →  on_response callback          →  display.py

Responsibilities:
  - Open and maintain a persistent WebSocket connection to the server
  - Send audio utterances as raw bytes when the VAD emits them
  - Receive JSON responses (transcript + suggestion) from the server
  - Fire a callback so the display layer can show suggestions immediately
  - Reconnect automatically if the connection drops mid-call

Why async?
  WebSocket I/O is inherently asynchronous — we can't block waiting for
  a response while also being ready to send the next utterance. asyncio
  lets us do both concurrently on a single thread without complexity.
"""

import asyncio
import json
import logging
import os
import threading
from typing import Callable, Optional

import numpy as np
import websockets
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("websocket_client")

RECONNECT_DELAY = 3    # Seconds to wait before attempting reconnection
MAX_RETRIES     = 10   # Maximum reconnection attempts before giving up


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
        self._uri        = f"ws://{self.host}:{self.port}/ws"

        # asyncio infrastructure
        self._loop:   Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread]          = None
        self._ws                                          = None
        self._running = False

        # Queue bridges the sync world (VAD callback) and async world (websocket send)
        # The VAD runs in a sync thread; we can't call async functions from there directly.
        # Instead, VAD puts audio into this queue and the async loop drains it.
        self._send_queue: asyncio.Queue = None

    # ── Public API (called from sync threads) ─────────────────────────────────
    def send(self, audio: np.ndarray) -> None:
        """
        Queue an audio utterance for sending to the server.
        Thread-safe — can be called from the VAD thread or anywhere else.

        The audio is serialised to bytes here so the conversion happens
        on the calling thread, not the asyncio event loop.
        """
        if not self._running or self._loop is None:
            log.warning("WebSocketClient not running — dropping audio chunk")
            return

        audio_bytes = audio.astype(np.float32).tobytes()

        # asyncio.run_coroutine_threadsafe bridges sync → async safely
        asyncio.run_coroutine_threadsafe(
            self._send_queue.put(audio_bytes),
            self._loop,
        )

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
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
        log.info("WebSocketClient stopped.")

    # ── Async internals ────────────────────────────────────────────────────────
    def _run_event_loop(self) -> None:
        """
        Entry point for the background thread.
        Creates a new asyncio event loop (each thread needs its own)
        and runs the main connection coroutine inside it.
        """
        self._loop       = asyncio.new_event_loop()
        self._send_queue = asyncio.Queue()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._connect_with_retry())
        except Exception as e:
            log.error(f"WebSocket event loop crashed: {e}", exc_info=True)
        finally:
            self._loop.close()

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
                audio_bytes = await asyncio.wait_for(
                    self._send_queue.get(),
                    timeout=1.0,
                )
                await ws.send(audio_bytes)
                log.info(f"Sent {len(audio_bytes)} bytes to server")
            except asyncio.TimeoutError:
                continue    # No audio in queue — loop back and wait

    async def _receiver(self, ws) -> None:
        """
        Listens for JSON messages from the server and fires on_response.
        Runs continuously until the connection closes.
        """
        async for message in ws:
            try:
                data = json.loads(message)
                log.info(f"Received: type={data.get('type')} | "
                         f"transcript='{data.get('transcript', '')[:50]}...'")
                self.on_response(data)
            except json.JSONDecodeError as e:
                log.warning(f"Could not parse server message: {e}")