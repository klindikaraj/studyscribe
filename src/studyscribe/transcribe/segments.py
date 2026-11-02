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


# Longest transcript segment. Each segment is a reference the notes of step 3 link to, so its
# start should be close to what it cites: 12 s is about two spoken sentences (30 words at 150
# words per minute), the upper end of Whisper's usual 3-10 s segments, so ordinary segments stay
# whole and only the long ones (seen: 21 and 30 s after a window start) are split.
MAX_SEGMENT_MS = 12_000
# A piece is not ended at a sentence end before this, so a split does not leave a scrap.
MIN_PIECE_MS = 2_000
SENTENCE_END = (".", "?", "!", "…")
CLAUSE_END = (",", ";", ":")


def split_long(
    segment: TranscriptSegment, max_ms: int = MAX_SEGMENT_MS, min_ms: int = MIN_PIECE_MS
) -> list[TranscriptSegment]:
    """Split a segment longer than `max_ms` at word boundaries, using its word timestamps.

    Each piece ends at the last sentence end (., ?, !) that keeps it within `max_ms`, else at
    the last clause end (, ; :), else at the widest pause between words. The per-window metrics
    are copied to every piece. Segments without words are left whole.
    """
    words = segment.words
    if segment.end_ms - segment.start_ms <= max_ms or len(words) < 2:
        return [segment]
    pieces, first = [], 0
    while first < len(words):
        last = first
        while last + 1 < len(words) and words[last + 1][1] - words[first][0] <= max_ms:
            last += 1
        if last + 1 < len(words):  # the rest does not fit: choose where this piece ends
            last = _best_end(words, first, last, min_ms)
        pieces.append(_piece(segment, words[first : last + 1]))
        first = last + 1
    return pieces


def _best_end(words: list, first: int, last: int, min_ms: int) -> int:
    """Index of the word that ends the piece starting at `first` (at most `last`)."""
    candidates = range(last, first - 1, -1)
    for endings in (SENTENCE_END, CLAUSE_END):
        for i in candidates:
            if words[i][2].rstrip().endswith(endings) and words[i][1] - words[first][0] >= min_ms:
                return i
    # The widest gap between word i and word i + 1.
    return max(range(first, last + 1), key=lambda i: (words[i + 1][0] - words[i][1], i))


def _piece(segment: TranscriptSegment, words: list) -> TranscriptSegment:
    return TranscriptSegment(
        start_ms=words[0][0],
        end_ms=words[-1][1],
        text="".join(w[2] for w in words).strip(),
        words=list(words),
        avg_logprob=segment.avg_logprob,
        no_speech_prob=segment.no_speech_prob,
        compression_ratio=segment.compression_ratio,
    )


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
