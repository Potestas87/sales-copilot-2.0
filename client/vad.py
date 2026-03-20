"""
vad.py
------
Voice Activity Detection filter — sits between audio_capture and websocket_client.

Reads raw audio chunks from AudioCapture's queue, detects when the customer
is actually speaking, buffers those chunks, and emits complete utterances
once a pause is detected.

Why this matters:
  - Without VAD, we'd stream silence and background noise to the GPU server,
    wasting bandwidth and GPU time on meaningless transcription.
  - Without utterance buffering, we'd send half-sentences and get garbled output.
  - With VAD, the server only receives clean, complete phrases.

Model: silero-vad (1MB neural net, runs locally on Mac — zero network latency)
"""

import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import torch

# ── VAD constants ──────────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000   # Must match audio_capture.py
VAD_CHUNK_SIZE    = 512     # Silero-vad's required input size at 16kHz (32ms per chunk)

SPEECH_THRESHOLD  = 0.5     # Probability above which a chunk is considered speech
                             # Lower = more sensitive (catches quiet speech, more false positives)
                             # Higher = more strict (misses soft speech, fewer false positives)

MIN_SPEECH_CHUNKS = 3       # Minimum consecutive speech chunks before we start buffering
                             # = 3 × 32ms = ~96ms — filters out clicks and brief noise

SILENCE_CHUNKS    = 24      # Consecutive silence chunks before we emit the utterance
                             # = 24 × 32ms = ~768ms — natural pause between sentences


# ── VADFilter class ────────────────────────────────────────────────────────────
class VADFilter:
    """
    Consumes raw audio chunks from a queue, runs silero-vad on each one,
    and emits complete utterances via a callback when speech ends.

    State machine:
        SILENCE  →  (speech detected for MIN_SPEECH_CHUNKS)  →  SPEAKING
        SPEAKING →  (silence detected for SILENCE_CHUNKS)    →  SILENCE + emit utterance

    Usage:
        def on_utterance(audio: np.ndarray):
            # audio is a complete utterance ready for transcription
            websocket_client.send(audio)

        vad = VADFilter(
            input_queue=capture.audio_queue,
            on_utterance=on_utterance,
        )
        vad.start()
        # runs in background thread
        vad.stop()
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        on_utterance: Callable[[np.ndarray], None],
        speech_threshold: float = SPEECH_THRESHOLD,
        min_speech_chunks: int = MIN_SPEECH_CHUNKS,
        silence_chunks: int = SILENCE_CHUNKS,
    ):
        """
        Args:
            input_queue:       Queue of float32 numpy arrays from AudioCapture.
            on_utterance:      Callback fired with a complete utterance (numpy array).
                               This is what the WebSocket client will hook into.
            speech_threshold:  Silero-vad probability cutoff (0.0–1.0).
            min_speech_chunks: Minimum chunks above threshold to begin buffering.
            silence_chunks:    Chunks below threshold before emitting the utterance.
        """
        self.input_queue       = input_queue
        self.on_utterance      = on_utterance
        self.speech_threshold  = speech_threshold
        self.min_speech_chunks = min_speech_chunks
        self.silence_chunks    = silence_chunks

        self._running          = False
        self._thread: Optional[threading.Thread] = None

        # Internal state
        self._speech_buffer: list[np.ndarray] = []   # chunks accumulated during speech
        self._speech_count   = 0    # consecutive speech chunks seen
        self._silence_count  = 0    # consecutive silence chunks since speech ended
        self._in_speech      = False

        # Load silero-vad model
        print("[VAD] Loading silero-vad model...")
        self._model, _ = torch.hub.load(
            repo_or_dir = "snakers4/silero-vad",
            model       = "silero_vad",
            force_reload = False,
            verbose     = False,
        )
        self._model.eval()
        print("[VAD] Model loaded.")

    # ── Core processing ────────────────────────────────────────────────────────
    def _get_speech_probability(self, chunk: np.ndarray) -> float:
        """
        Run silero-vad on a 512-sample chunk and return speech probability.

        Silero-vad expects a 1D float32 tensor of exactly VAD_CHUNK_SIZE samples.
        We use torch.no_grad() because we're doing inference only — no backprop needed.
        """
        tensor = torch.from_numpy(chunk).unsqueeze(0)   # shape: (1, 512)
        with torch.no_grad():
            prob = self._model(tensor, SAMPLE_RATE).item()
        return prob

    def _process_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        Process one 8000-sample chunk from the audio queue.

        Splits it into 512-sample sub-chunks (silero-vad's required size),
        classifies each one as speech or silence, and manages the state machine.
        """
        for i in range(0, len(audio_chunk) - VAD_CHUNK_SIZE + 1, VAD_CHUNK_SIZE):
            sub_chunk = audio_chunk[i : i + VAD_CHUNK_SIZE]
            prob      = self._get_speech_probability(sub_chunk)
            is_speech = prob >= self.speech_threshold

            if is_speech:
                self._silence_count = 0
                self._speech_count += 1

                # Start buffering once we've seen enough consecutive speech
                if self._speech_count >= self.min_speech_chunks:
                    self._in_speech = True

                if self._in_speech:
                    self._speech_buffer.append(sub_chunk)

            else:
                self._speech_count = 0

                if self._in_speech:
                    # Still buffering — add this silence chunk too so we don't
                    # clip the end of words
                    self._speech_buffer.append(sub_chunk)
                    self._silence_count += 1

                    # Enough silence — the customer has finished their sentence
                    if self._silence_count >= self.silence_chunks:
                        self._emit_utterance()

    def _emit_utterance(self) -> None:
        """
        Concatenate buffered chunks into a single utterance array and fire the callback.
        Resets all state for the next utterance.
        """
        if not self._speech_buffer:
            return

        utterance = np.concatenate(self._speech_buffer)
        duration  = len(utterance) / SAMPLE_RATE

        print(f"[VAD] Utterance detected ({duration:.2f}s) — sending to transcription")

        # Fire the callback (WebSocket client will pick this up)
        self.on_utterance(utterance)

        # Reset state
        self._speech_buffer = []
        self._speech_count  = 0
        self._silence_count = 0
        self._in_speech     = False

    # ── Background thread ──────────────────────────────────────────────────────
    def _run(self) -> None:
        """
        Main loop — runs in a background thread.
        Continuously pulls chunks from the input queue and processes them.
        Blocks on queue.get() so it uses no CPU while waiting for audio.
        """
        print("[VAD] Processing thread started.")
        while self._running:
            try:
                # Block for up to 0.5s waiting for a chunk
                # The timeout lets us check self._running periodically
                audio_chunk = self.input_queue.get(timeout=0.5)
                self._process_chunk(audio_chunk)
            except queue.Empty:
                continue

        # Emit any remaining buffered speech when stopping
        if self._speech_buffer:
            print("[VAD] Flushing remaining speech buffer on shutdown...")
            self._emit_utterance()

        print("[VAD] Processing thread stopped.")

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Start the VAD processing thread."""
        if self._running:
            print("[VAD] Already running.")
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True, name="vad-thread")
        self._thread.start()
        print("[VAD] Started.")

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish cleanly."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[VAD] Stopped.")


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Test the VAD filter standalone — captures from your mic and prints
    whenever a complete utterance is detected.

    Usage:
        python3 client/vad.py
    """
    from audio_capture import AudioCapture

    utterance_count = 0

    def on_utterance(audio: np.ndarray):
        global utterance_count
        utterance_count += 1
        duration = len(audio) / SAMPLE_RATE
        print(f"  -> Utterance #{utterance_count}: {duration:.2f}s, "
              f"{len(audio)} samples — would send to GPU server now")

    print("Starting audio capture + VAD test. Speak into your mic.")
    print("You should see 'Utterance detected' each time you finish a sentence.\n")

    capture = AudioCapture()
    vad     = VADFilter(
        input_queue  = capture.audio_queue,
        on_utterance = on_utterance,
    )

    try:
        capture.start()
        vad.start()
        print("Listening for 30 seconds... (Ctrl+C to stop early)\n")
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        vad.stop()
        capture.stop()
        print(f"\nTest complete. Detected {utterance_count} utterances.")