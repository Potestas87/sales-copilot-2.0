"""
protocol.py
-----------
Typed websocket message models for client-side transport.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

Speaker = Literal["customer", "salesperson"]
Intent = Literal["objection", "question", "buying_signal", "none"]


@dataclass(frozen=True)
class UtteranceMessage:
    """Client -> Server payload for a single speaker-labeled utterance."""

    type: Literal["utterance"]
    speaker: Speaker
    sample_rate: int
    ts_ms: int
    audio_b64: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_audio(
        cls,
        *,
        audio: np.ndarray,
        speaker: Speaker,
        sample_rate: int,
        ts_ms: int,
    ) -> "UtteranceMessage":
        audio_bytes = audio.astype(np.float32).tobytes()
        return cls(
            type="utterance",
            speaker=speaker,
            sample_rate=sample_rate,
            ts_ms=ts_ms,
            audio_b64=base64.b64encode(audio_bytes).decode("ascii"),
        )


@dataclass(frozen=True)
class InferenceMessage:
    """Server -> Client payload for transcript/inference updates."""

    type: Literal["inference"]
    speaker: Speaker
    transcript: str
    intent: Intent
    suggestion: str
    reasoning_short: str
    confidence: float
    latency_ms: float

    @classmethod
    def from_json(cls, raw_message: str) -> "InferenceMessage":
        parsed = json.loads(raw_message)
        if parsed.get("type") != "inference":
            raise ValueError(f"Unsupported message type: {parsed.get('type')}")
        return cls(
            type="inference",
            speaker=parsed["speaker"],
            transcript=parsed.get("transcript", ""),
            intent=parsed.get("intent", "none"),
            suggestion=parsed.get("suggestion", ""),
            reasoning_short=parsed.get("reasoning_short", ""),
            confidence=float(parsed.get("confidence", 0.0)),
            latency_ms=float(parsed.get("latency_ms", 0.0)),
        )
