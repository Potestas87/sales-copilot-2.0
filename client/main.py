"""
main.py
-------
Entry point for the Sales Copilot Mac client.

Wires together all four client modules:
  AudioCapture  →  VADFilter  →  WebSocketClient  →  SuggestionDisplay

Each module runs on its own thread. main() starts them all up,
then hands control to the display's tkinter mainloop (which must
run on the main thread on macOS). Ctrl+C or closing the window
triggers a clean shutdown of all background threads.

Run with:
    python3 client/main.py
"""

import signal
import sys
import threading

from audio_capture   import AudioCapture
from vad             import VADFilter
from websocket_client import WebSocketClient
from display         import SuggestionDisplay


def main():
    print("=" * 50)
    print("  Sales Copilot — starting up")
    print("=" * 50)

    display = SuggestionDisplay()

    # ── Step 1: WebSocket client ───────────────────────────────────────────────
    # Started first so it's ready to receive audio as soon as VAD emits utterances.
    # on_response fires on the websocket thread — display.show() is thread-safe.
    def on_server_response(data: dict):
        display.show(
            suggestion      = data.get("suggestion", ""),
            suggestion_type = data.get("type", "none"),
            transcript      = data.get("transcript", ""),
        )

    ws_client = WebSocketClient(on_response=on_server_response)
    ws_client.start()

    # ── Step 2: VAD filter ─────────────────────────────────────────────────────
    # on_utterance fires on the VAD thread — ws_client.send() is thread-safe.
    def on_utterance(audio):
        ws_client.send(audio)

    capture = AudioCapture()
    vad     = VADFilter(
        input_queue  = capture.audio_queue,
        on_utterance = on_utterance,
    )

    # ── Step 3: Start audio capture and VAD ───────────────────────────────────
    capture.start()
    vad.start()

    print("\nSales Copilot is running.")
    print("Speak to your customer — suggestions will appear in the window.")
    print("Close the window or press Ctrl+C to stop.\n")

    # ── Step 4: Clean shutdown handler ────────────────────────────────────────
    # Ctrl+C in the terminal triggers this. Also fires when the window is closed.
    def shutdown(sig=None, frame=None):
        print("\nShutting down...")
        vad.stop()
        capture.stop()
        ws_client.stop()
        display.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    # ── Step 5: Start display (blocks — must be on main thread) ───────────────
    # This call blocks until the window is closed. All other modules are
    # already running on background threads at this point.
    try:
        display.start()
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()