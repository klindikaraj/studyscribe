"""Voice activity detection: the Silero model and code bundled with faster-whisper (CPU)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# faster-whisper's defaults for vad_filter=True: what decides which audio Whisper hears, and
# what the language detection hears.
SPEECH_VAD = {"threshold": 0.5, "min_silence_duration_ms": 2000, "speech_pad_ms": 400}
# For finding where to cut a window: every pause of 100 ms or more, unpadded.
PAUSE_VAD = {"threshold": 0.5, "min_silence_duration_ms": 100, "speech_pad_ms": 0}

Spans = list[tuple[int, int]]
Vad = Callable[..., Spans]  # (float32 audio, **options) -> speech spans in samples


def speech_spans(audio: np.ndarray, **options) -> Spans:
    """Speech [start, end) sample spans of float32 16 kHz `audio`."""
    if audio.size < 512:  # shorter than one Silero window
        return []
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    chunks = get_speech_timestamps(audio, VadOptions(**options))
    return [(int(c["start"]), int(c["end"])) for c in chunks]
