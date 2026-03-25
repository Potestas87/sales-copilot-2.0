"""
main.py
-------
Entry point for the Sales Copilot Mac client.

Wires together all client modules:
  AudioCapture  →  VADFilter(customer + salesperson)  →  WebSocketClient  →  SuggestionDisplay

Each module runs on its own thread. main() starts them all up,
then hands control to the display's tkinter mainloop (which must
run on the main thread on macOS). Ctrl+C or closing the window
triggers a clean shutdown of all background threads.

Run with:
    python3 client/main.py
"""

import signal
import sys
import os

from audio_capture   import AudioCapture
from vad             import VADFilter
from websocket_client import WebSocketClient
from display         import SuggestionDisplay


def main():
    print("=" * 50)
    print("  Sales Copilot — starting up")
    print("=" * 50)

    # ── Latency tuning knobs (env-configurable) ───────────────────────────────
    customer_speech_threshold = float(os.getenv("CUSTOMER_VAD_SPEECH_THRESHOLD", "0.5"))
    customer_min_speech_chunks = int(os.getenv("CUSTOMER_VAD_MIN_SPEECH_CHUNKS", "3"))
    customer_silence_chunks = int(os.getenv("CUSTOMER_VAD_SILENCE_CHUNKS", "20"))

    sales_speech_threshold = float(os.getenv("SALES_VAD_SPEECH_THRESHOLD", "0.5"))
    sales_min_speech_chunks = int(os.getenv("SALES_VAD_MIN_SPEECH_CHUNKS", "3"))
    sales_silence_chunks = int(os.getenv("SALES_VAD_SILENCE_CHUNKS", "24"))

    print(
        "[Config] Customer VAD -> threshold=%.2f min_speech=%d silence_chunks=%d"
        % (customer_speech_threshold, customer_min_speech_chunks, customer_silence_chunks)
    )
    print(
        "[Config] Sales VAD    -> threshold=%.2f min_speech=%d silence_chunks=%d"
        % (sales_speech_threshold, sales_min_speech_chunks, sales_silence_chunks)
    )

    display = SuggestionDisplay()

    # ── Step 1: WebSocket client ───────────────────────────────────────────────
    # Started first so it's ready to receive audio as soon as VAD emits utterances.
    # on_response fires on the websocket thread — display.show() is thread-safe.
    def on_server_response(data: dict):
        display.show(
            suggestion      = data.get("suggestion", ""),
            suggestion_type = data.get("type", "none"),
            transcript      = data.get("transcript", ""),
            speaker         = data.get("speaker", "customer"),
            confidence      = data.get("confidence", 0.0),
            latency_ms      = data.get("latency_ms", 0.0),
            reasoning_short = data.get("reasoning_short", ""),
        )

    ws_client = WebSocketClient(on_response=on_server_response)
    ws_client.start()

    # ── Step 2: Dual VAD filters (customer + salesperson) ─────────────────────
    # on_utterance callbacks fire on VAD threads — ws_client.send() is thread-safe.
    def on_customer_utterance(audio):
        ws_client.send(audio, speaker="customer")

    def on_sales_utterance(audio):
        ws_client.send(audio, speaker="salesperson")

    capture = AudioCapture()
    customer_vad = VADFilter(
        input_queue  = capture.customer_audio_queue,
        on_utterance = on_customer_utterance,
        speech_threshold = customer_speech_threshold,
        min_speech_chunks = customer_min_speech_chunks,
        silence_chunks = customer_silence_chunks,
    )
    sales_vad = VADFilter(
        input_queue  = capture.sales_audio_queue,
        on_utterance = on_sales_utterance,
        speech_threshold = sales_speech_threshold,
        min_speech_chunks = sales_min_speech_chunks,
        silence_chunks = sales_silence_chunks,
    )

    # ── Step 3: Start audio capture and both VAD pipelines ────────────────────
    capture.start()
    customer_vad.start()
    sales_vad.start()

    print("\nSales Copilot is running.")
    print("Speak to your customer — suggestions will appear in the window.")
    print("Close the window or press Ctrl+C to stop.\n")

    # ── Step 4: Clean shutdown handler ────────────────────────────────────────
    # Ctrl+C in the terminal triggers this. Also fires when the window is closed.
    def shutdown(sig=None, frame=None):
        print("\nShutting down...")
        customer_vad.stop()
        sales_vad.stop()
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
