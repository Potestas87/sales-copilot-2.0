"""
audio_capture.py
----------------
Captures audio from two sources simultaneously:
  1. Your microphone (your own voice)
  2. BlackHole virtual device (customer's voice from the call app)

Only the customer's audio is sent downstream for transcription and
objection detection — we don't need to analyse what the salesperson says.

Audio chunks are placed into a Queue that the VAD module reads from.
This keeps the capture layer decoupled from everything else.

Run this file directly to list your audio devices and find the right indexes:
    python client/audio_capture.py
"""

import queue
import os
import time

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from typing import Optional, Callable

load_dotenv()

# ── Audio constants ────────────────────────────────────────────────────────────
# 16kHz mono is what Whisper expects.
SAMPLE_RATE      = 16000   # Hz  — Whisper's native sample rate
CHUNK_SECONDS    = 0.5     # How many seconds of audio per chunk sent to the queue
CHUNK_SIZE       = int(SAMPLE_RATE * CHUNK_SECONDS)  # = 8000 samples per chunk


def get_device_channels(device_index: int) -> int:
    """
    Ask sounddevice how many input channels a device actually supports.
    This avoids hardcoding channel counts — different mics and virtual
    devices report different values (1 for mono, 2 for stereo, etc.).
    """
    device_info = sd.query_devices(device_index)
    return int(device_info["max_input_channels"])


def find_device_by_name(name: str, kind: str = "input") -> Optional[int]:
    """
    Look up an audio device by name instead of index number.

    Why: macOS reassigns device indexes every time you connect/disconnect
    a device or restart. A name-based lookup is stable and portable.

    Args:
        name:  Substring to match (case-insensitive), e.g. "BlackHole" or "SoloCast"
        kind:  "input" or "output" — filters to the correct direction

    Returns:
        The device index, or None if no match is found.
    """
    devices = sd.query_devices()
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, dev in enumerate(devices):
        if name.lower() in dev["name"].lower() and dev[channel_key] > 0:
            return i
    return None


def resolve_device(env_name: str, env_index: str, kind: str = "input") -> int:
    """
    Resolve an audio device using two strategies:
      1. If a DEVICE_NAME env var is set, find the device by name (stable)
      2. Fall back to a DEVICE_INDEX env var (fragile but simple)

    This gives the best of both worlds — name-based is preferred, but
    index-based still works for quick testing.

    Args:
        env_name:   Env var for device name,  e.g. "MIC_DEVICE_NAME"
        env_index:  Env var for device index,  e.g. "MIC_DEVICE_INDEX"
        kind:       "input" or "output"

    Returns:
        The resolved device index.

    Raises:
        RuntimeError if the device can't be found by either method.
    """
    # Strategy 1: look up by name (preferred)
    device_name = os.getenv(env_name)
    if device_name:
        index = find_device_by_name(device_name, kind)
        if index is not None:
            print(f"[AudioCapture] Found '{device_name}' at index {index}")
            return index
        raise RuntimeError(
            f"Could not find {kind} device matching '{device_name}'. "
            f"Run 'python client/audio_capture.py' to see available devices."
        )

    # Strategy 2: fall back to index
    device_index = os.getenv(env_index)
    if device_index is not None:
        return int(device_index)

    raise RuntimeError(
        f"No device configured. Set {env_name} (recommended) or {env_index} in your .env file."
    )


# ── Device listing helper ──────────────────────────────────────────────────────
def list_audio_devices() -> None:
    """
    Print every audio device sounddevice can see, separated into inputs/outputs.

    Run this first to find the index numbers for your microphone and BlackHole.
    You'll put those numbers in your .env file as:
        MIC_DEVICE_INDEX=<number>
        CALL_AUDIO_DEVICE_INDEX=<number>
    """
    print("\n=== Available Audio Devices ===\n")
    devices = sd.query_devices()
    print("  INPUT DEVICES (can be read from):")
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"    [{i:2}]  {dev['name']}  (channels: {dev['max_input_channels']})")
    print("\n  OUTPUT DEVICES (speakers/virtual):")
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            print(f"    [{i:2}]  {dev['name']}  (channels: {dev['max_output_channels']})")
    print()
    print("-> Set MIC_DEVICE_INDEX and CALL_AUDIO_DEVICE_INDEX in your .env file")
    print("-> BlackHole 2ch should appear as an input device after installation\n")


# ── AudioCapture class ─────────────────────────────────────────────────────────
class AudioCapture:
    """
    Opens two simultaneous audio input streams:

      mic_stream  — your microphone
                    Captured but not forwarded downstream (we only want to
                    analyse the customer, not the salesperson).

      call_stream — BlackHole 2ch virtual device
                    The customer's voice coming through your call app.
                    Each chunk is placed into self.audio_queue for the
                    VAD module to consume.

    Usage:
        capture = AudioCapture()
        capture.start()
        # audio_queue now receives 0.5-second float32 numpy chunks
        chunk = capture.audio_queue.get()
        capture.stop()
    """

    def __init__(
        self,
        mic_device_index: Optional[int] = None,
        call_device_index: Optional[int] = None,
        on_customer_audio: Optional[Callable[[np.ndarray], None]] = None,
    ):
        """
        Args:
            mic_device_index:    Sounddevice index for your microphone.
                                 Falls back to MIC_DEVICE_INDEX in .env.
            call_device_index:   Sounddevice index for BlackHole 2ch.
                                 Falls back to CALL_AUDIO_DEVICE_INDEX in .env.
            on_customer_audio:   Optional callback called with each audio
                                 chunk in addition to the queue. Useful for
                                 debugging or visualising the waveform.
        """
        if mic_device_index is not None:
            self.mic_device_index = mic_device_index
        else:
            self.mic_device_index = resolve_device("MIC_DEVICE_NAME", "MIC_DEVICE_INDEX", "input")

        if call_device_index is not None:
            self.call_device_index = call_device_index
        else:
            self.call_device_index = resolve_device("CALL_DEVICE_NAME", "CALL_AUDIO_DEVICE_INDEX", "input")
        self.on_customer_audio = on_customer_audio

        # Thread-safe queue consumed by the VAD module
        self.audio_queue: queue.Queue = queue.Queue()

        self._running = False
        self._streams = []

    # ── Stream callbacks ───────────────────────────────────────────────────────
    # sounddevice calls these functions from a background audio thread every
    # time a new chunk of audio is ready. They must return quickly — no heavy
    # processing here, just capture and hand off.

    def _mic_callback(self, indata, frames, time_info, status) -> None:
        """Mic audio arrives here. We log errors but don't forward the data."""
        if status:
            print(f"[AudioCapture][MIC] Warning: {status}")
        # Mic audio is not forwarded — we only analyse customer speech

    def _to_mono(self, indata: np.ndarray) -> np.ndarray:
        """
        Convert an audio array to mono float32.
        - If the device is mono (1 channel): just flatten the array
        - If the device is stereo (2 channels): average both channels
        This handles any device regardless of how many channels it reports.
        """
        if indata.shape[1] == 1:
            return indata.copy().flatten().astype(np.float32)
        return indata.copy().mean(axis=1).astype(np.float32)

    def _call_callback(self, indata, frames, time_info, status) -> None:
        """
        Customer audio arrives here every CHUNK_SECONDS.
        We convert whatever channel format the device uses down to mono,
        then put the chunk in the queue for the VAD module.
        """
        if status:
            print(f"[AudioCapture][CALL] Warning: {status}")

        audio_chunk = self._to_mono(indata)
        self.audio_queue.put(audio_chunk)

        if self.on_customer_audio:
            self.on_customer_audio(audio_chunk)

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    def _open_stream(self, device_index: int, callback) -> tuple:
        """
        Try to open an input stream, starting with 1 channel (mono) and
        falling back to the device's reported max channels if mono fails.

        Why: Some devices (like the HyperX SoloCast) report 2 channels to
        the OS but physically only accept 1 channel at certain sample rates.
        Trying mono first is safer — we only need mono for Whisper anyway.

        Returns: (InputStream, channels_used)
        """
        max_channels = get_device_channels(device_index)
        device_name  = sd.query_devices(device_index)["name"]

        for channels in sorted(set([1, max_channels])):  # try 1 first, then max
            try:
                stream = sd.InputStream(
                    device     = device_index,
                    channels   = channels,
                    samplerate = SAMPLE_RATE,
                    blocksize  = CHUNK_SIZE,
                    dtype      = "float32",
                    callback   = callback,
                )
                return stream, channels
            except sd.PortAudioError:
                continue

        raise RuntimeError(
            f"Could not open '{device_name}' (device {device_index}) "
            f"with 1 or {max_channels} channels at {SAMPLE_RATE}Hz. "
            f"Check that the device is connected and not in use by another app."
        )

    def start(self) -> None:
        """
        Open both input streams and begin capturing.
        Channel counts are determined automatically via _open_stream().
        Non-blocking — audio arrives in the background via callbacks.
        """
        if self._running:
            print("[AudioCapture] Already running.")
            return

        self._running = True

        mic_stream,  mic_channels  = self._open_stream(self.mic_device_index,  self._mic_callback)
        call_stream, call_channels = self._open_stream(self.call_device_index, self._call_callback)

        self._streams = [mic_stream, call_stream]
        for stream in self._streams:
            stream.start()

        print(f"[AudioCapture] Mic stream open   -> device {self.mic_device_index} ({mic_channels}ch)")
        print(f"[AudioCapture] Call stream open  -> device {self.call_device_index} ({call_channels}ch)")
        print(f"[AudioCapture] Chunk size: {CHUNK_SIZE} samples ({CHUNK_SECONDS}s @ {SAMPLE_RATE}Hz)")

    def stop(self) -> None:
        """Stop and close both streams cleanly."""
        self._running = False
        for stream in self._streams:
            stream.stop()
            stream.close()
        self._streams = []
        print("[AudioCapture] Stopped.")


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to:
      1. See all available audio devices and their index numbers
      2. Do a 5-second capture test to confirm audio is flowing

    Usage:
        python client/audio_capture.py
    """
    list_audio_devices()

    input("Press Enter to run a 5-second capture test (or Ctrl+C to exit)...\n")

    capture = AudioCapture()

    try:
        capture.start()
        print("Capturing for 5 seconds — speak into your mic and play audio through your call app...\n")
        time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        chunk_count = capture.audio_queue.qsize()
        print(f"\nTest complete. Captured {chunk_count} chunks "
              f"({chunk_count * CHUNK_SECONDS:.1f}s of audio)")
        if chunk_count == 0:
            print("No audio captured — check your CALL_AUDIO_DEVICE_INDEX in .env")