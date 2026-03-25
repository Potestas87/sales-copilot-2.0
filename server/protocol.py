"""
protocol.py
-----------
Typed websocket message models for server-side transport validation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Speaker = Literal["customer", "salesperson"]
Intent = Literal["objection", "question", "buying_signal", "none"]


class UtteranceMessage(BaseModel):
    """Client -> Server payload for a single speaker-labeled utterance."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["utterance"]
    speaker: Speaker
    sample_rate: int = Field(gt=0, le=192000)
    ts_ms: int = Field(ge=0)
    audio_b64: str = Field(min_length=1)


class InferenceMessage(BaseModel):
    """Server -> Client payload with transcript and inference result."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["inference"] = "inference"
    speaker: Speaker
    transcript: str = ""
    intent: Intent = "none"
    suggestion: str = ""
    reasoning_short: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    latency_ms: float = Field(ge=0.0, default=0.0)
