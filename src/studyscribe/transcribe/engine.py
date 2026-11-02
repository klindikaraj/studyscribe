"""What a speech recognition engine provides to the transcription pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class TranscriberError(RuntimeError):
    """The engine cannot run; the message says what to do (no traceback needed)."""


@dataclass(frozen=True)
class AsrWord:
    start: float  # seconds, relative to the audio given to the engine
    end: float
    text: str
    probability: float


@dataclass(frozen=True)
class AsrSegment:
    start: float  # seconds, relative to the audio given to the engine
    end: float
    text: str
    words: tuple[AsrWord, ...] = field(default=())
    # faster-whisper computes these per decoding window (up to 30 s), so the segments of one
    # window share them. None when an engine does not provide them.
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


class Transcriber(Protocol):
    engine: str  # e.g. "faster-whisper"
    model: str  # e.g. "dropbox-dash/faster-whisper-large-v3-turbo@0a363e9"
    compute_type: str

    def language_probabilities(self, speech: np.ndarray) -> dict[str, float]:
        """Language probabilities of one chunk of up to 30 s of speech (float32, 16 kHz)."""
        ...

    def transcribe(
        self, audio: np.ndarray, language: str, hints: str | None = None
    ) -> Iterator[AsrSegment]:
        """Segments of `audio` (float32, 16 kHz) with times relative to it. The engine applies
        its own voice activity detection."""
        ...

    def options(self) -> dict:
        """Everything about the engine that shapes its output (stored with the run)."""
        ...
