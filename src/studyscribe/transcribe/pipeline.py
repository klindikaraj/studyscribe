"""Transcription of a session: language detection on speech, then windows cut in pauses.

No database and no CLI here: callers get Window results (and per-segment callbacks) and decide
what to store and show. Step 3 stores them; step 4 can run the same code in a worker process.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from studyscribe.audio.reader import SessionAudio
from studyscribe.config import SAMPLE_RATE
from studyscribe.transcribe import language as lang
from studyscribe.transcribe.engine import Transcriber
from studyscribe.transcribe.filters import apply_filters
from studyscribe.transcribe.levels import normalize, to_float
from studyscribe.transcribe.segments import TranscriptSegment, to_session
from studyscribe.transcribe.vad import PAUSE_VAD, SPEECH_VAD, Vad, speech_spans
from studyscribe.transcribe.windows import MAX_WINDOW_S, SEARCH_S, Cut, choose_cut

DETECTION_BLOCK_S = 300.0  # the language scan reads the session in blocks this long
CONTEXT_SEGMENTS = 5  # segments of the previous windows the filters look back at


@dataclass
class Window:
    start_frame: int
    end_frame: int
    segments: list[TranscriptSegment] = field(default_factory=list)
    cut_in_pause: bool = True  # False: the end had to be put in the quietest stretch

    @property
    def kept(self) -> list[TranscriptSegment]:
        return [s for s in self.segments if s.dropped is None]


def prepare(pcm: np.ndarray, normalize_audio: bool = True) -> np.ndarray:
    """The float copy of `pcm` that the VAD and the engine get."""
    return normalize(pcm) if normalize_audio else to_float(pcm)


def detect_language(
    audio: SessionAudio,
    transcriber: Transcriber,
    start: int = 0,
    end: int | None = None,
    vad: Vad = speech_spans,
    normalize_audio: bool = True,
) -> lang.LanguageGuess | None:
    """Language of the first DETECT_SPEECH_S of speech after `start`; None if there is no
    speech at all (the language is then unknown, never guessed from silence)."""
    end = audio.total_frames if end is None else min(end, audio.total_frames)
    wanted = int(lang.DETECT_SPEECH_S * SAMPLE_RATE)
    block = int(DETECTION_BLOCK_S * SAMPLE_RATE)
    speech: list[np.ndarray] = []
    collected = 0
    position = start
    while position < end and collected < wanted:
        samples = prepare(audio.read(position, min(block, end - position)), normalize_audio)
        for s, e in vad(samples, **SPEECH_VAD):
            piece = samples[s:e][: wanted - collected]
            speech.append(piece)
            collected += piece.size
            if collected >= wanted:
                break
        position += block
    if not collected:
        return None
    sample = np.concatenate(speech)
    probabilities = [transcriber.language_probabilities(c) for c in lang.chunks(sample)]
    return lang.average(probabilities, collected / SAMPLE_RATE)


def transcribe_windows(
    audio: SessionAudio,
    transcriber: Transcriber,
    language: str,
    start: int = 0,
    end: int | None = None,
    hints: str | None = None,
    vad: Vad = speech_spans,
    normalize_audio: bool = True,
    before: list[TranscriptSegment] | None = None,
    on_segment: Callable[[TranscriptSegment], None] | None = None,
    max_window_s: float = MAX_WINDOW_S,
    search_s: float = SEARCH_S,
) -> Iterator[Window]:
    """Transcribe [start, end) of the session window by window.

    `start` is a resume point (the end of the last window done); `before` are the last
    segments before it, for the filters. Each window is yielded once its segments are filtered;
    `on_segment` sees every segment as soon as the engine returns it, before filtering (for
    progress).
    """
    end = audio.total_frames if end is None else min(end, audio.total_frames)
    max_frames = int(max_window_s * SAMPLE_RATE)
    search_frames = int(search_s * SAMPLE_RATE)
    context = list(before or [])[-CONTEXT_SEGMENTS:]
    cursor = start
    while cursor < end:
        stop = min(cursor + max_frames, end)
        samples = prepare(audio.read(cursor, stop - cursor), normalize_audio)
        if stop < end:
            region_start = max(samples.size - search_frames, 0)
            region = samples[region_start:]
            cut = choose_cut(region, cursor + region_start, vad(region, **PAUSE_VAD))
            if cut.frame <= cursor:  # only with a search region as long as the window
                cut = Cut(stop, False)
        else:
            cut = Cut(end, True)
        window = Window(cursor, cut.frame, cut_in_pause=cut.in_pause)
        for asr in transcriber.transcribe(samples[: cut.frame - cursor], language, hints):
            segment = to_session(asr, cursor)
            window.segments.append(segment)
            if on_segment is not None:
                on_segment(segment)
        apply_filters(window.segments, context)
        context = (context + window.segments)[-CONTEXT_SEGMENTS:]
        yield window
        cursor = cut.frame
