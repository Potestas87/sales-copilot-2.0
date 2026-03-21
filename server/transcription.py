"""
transcription.py
----------------
Wraps faster-whisper to transcribe audio utterances into text.

faster-whisper is a reimplementation of OpenAI's Whisper model using
CTranslate2 — a C++ inference engine optimised for transformer models.
It's 4x faster than the original Whisper and uses significantly less VRAM,
making it the right choice for a real-time use case on a GPU pod.

Model sizes and tradeoffs:
  tiny    — ~1GB VRAM, fastest, least accurate
  base    — ~1GB VRAM, fast, decent accuracy
  small   — ~2GB VRAM, good balance
  medium  — ~5GB VRAM, high accuracy
  large-v3 — ~10GB VRAM, best accuracy, ~2-3s latency on a good GPU

For sales calls, large-v3 is recommended — transcription errors lead to
bad suggestions, and a 2-3s delay is acceptable.
"""

import io
import logging
import os
import numpy as np

from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()
log = logging.getLogger("transcription")


class Transcriber:
    """
    Loads a faster-whisper model and transcribes float32 numpy audio arrays.

    The model is loaded once at instantiation and reused for every transcription
    call — model loading takes ~5-10 seconds, inference takes ~1-3 seconds.

    Usage:
        transcriber = Transcriber()
        text = transcriber.transcribe(audio_array)  # audio_array: float32 np array
        print(text)  # "I'm not sure I really need this right now"
    """

    def __init__(self):
        self.model_size   = os.getenv("WHISPER_MODEL", "large-v3")
        self.device       = "cuda"        # We're on a GPU pod — always use CUDA
        self.compute_type = "float16"     # float16 halves VRAM usage vs float32
                                          # with negligible accuracy loss on modern GPUs

        log.info(f"Loading Whisper model '{self.model_size}' on {self.device} ({self.compute_type})...")

        self._model = WhisperModel(
            self.model_size,
            device       = self.device,
            compute_type = self.compute_type,
        )

        log.info("Whisper model loaded.")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a float32 audio array into text.

        Args:
            audio: 1D float32 numpy array at 16kHz (from the VAD filter)

        Returns:
            Transcribed text as a string, stripped of leading/trailing whitespace.
            Returns empty string if nothing was recognised.

        How it works:
            faster-whisper's transcribe() returns a generator of Segment objects.
            Each Segment has a .text property. We join all segments together —
            for short utterances (5-15s) there's usually just one segment, but
            longer ones may produce several.
        """
        if len(audio) == 0:
            return ""

        # Ensure correct dtype — faster-whisper expects float32
        audio = audio.astype(np.float32)

        # beam_size=5 is the default — balances speed vs accuracy.
        # vad_filter=True uses Whisper's built-in VAD as a second pass to
        # skip any silence that slipped through our client-side VAD filter.
        segments, info = self._model.transcribe(
            audio,
            beam_size  = 5,
            language   = "en",       # Lock to English — faster than auto-detect
            vad_filter = True,       # Second-pass silence filtering on the server
        )

        log.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        # Concatenate all segments into a single string
        transcript = " ".join(segment.text for segment in segments).strip()

        return transcript