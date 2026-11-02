"""Transcript segments on the session timeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from studyscribe.config import SAMPLE_RATE
from studyscribe.transcribe.engine import AsrSegment


@dataclass
class TranscriptSegment:
    start_ms: int  # session timeline: 0 is the first sample of 000001.wav
    end_ms: int
    text: str
    words: list[tuple[int, int, str, float]] = field(default_factory=list)  # ms, ms, text, p
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    dropped: str | None = None  # None: kept; else the filter rule that dropped it

    @property
    def mean_word_probability(self) -> float | None:
        if not self.words:
            return None
        return sum(w[3] for w in self.words) / len(self.words)


def to_session(segment: AsrSegment, window_start_frame: int) -> TranscriptSegment:
    """Map engine times (seconds from the start of the window audio) to session milliseconds."""
    base_ms = window_start_frame * 1000 / SAMPLE_RATE

    def ms(seconds: float) -> int:
        return round(base_ms + seconds * 1000)

    return TranscriptSegment(
        start_ms=ms(segment.start),
        end_ms=ms(segment.end),
        text=segment.text.strip(),
        words=[(ms(w.start), ms(w.end), w.text, w.probability) for w in segment.words],
        avg_logprob=segment.avg_logprob,
        no_speech_prob=segment.no_speech_prob,
        compression_ratio=segment.compression_ratio,
    )
