"""Anti-hallucination rules (pure). A dropped segment is kept with the rule's name in `dropped`,
so a rule that is too strict shows up in the list of dropped segments.

Thresholds: SILENCE and REPETITION use the values of OpenAI's reference implementation for
long-form audio (faster-whisper's defaults). DUPLICATES, KNOWN_PHRASE and LOW_CONFIDENCE are
heuristics to validate on real recordings.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from studyscribe.transcribe.segments import TranscriptSegment

NO_SPEECH_PROB = 0.6
LOG_PROB = -1.0
COMPRESSION_RATIO = 2.4
DUPLICATE_RUN = 3  # this many consecutive equal texts: all but the first are dropped
ISOLATION_MS = 2000  # no words for this long before and after a segment
LOW_WORD_PROBABILITY = 0.35
SHORT_MS = 1500

SILENCE = "silence"
REPETITION = "repetition"
DUPLICATE = "duplicate"
KNOWN_PHRASE = "known phrase"
LOW_CONFIDENCE = "low confidence"

# Whole-segment hallucinations Whisper learned from subtitle files (normalised text).
KNOWN_PHRASES = frozenset({
    "thank you for watching",
    "thanks for watching",
    "thank you so much for watching",
    "please subscribe",
    "please like and subscribe",
    "subscribe to my channel",
    "subtitles by the amara org community",
    "transcription by castingwords",
    "sottotitoli creati dalla comunita amara org",
    "sottotitoli a cura di qtss",
    "sottotitoli e revisione a cura di qtss",
    "grazie per la visione",
    "grazie a tutti per la visione",
    "iscriviti al canale",
})


def normalized(text: str) -> str:
    """Lower case, no accents, punctuation as spaces, single spaces."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w]+", " ", plain).split())


def apply_filters(
    segments: Sequence[TranscriptSegment], before: Sequence[TranscriptSegment] = ()
) -> None:
    """Set `dropped` on the segments of a window that look hallucinated.

    `before`: the last segments of the previous windows (already decided, not changed), so
    duplicates and isolation are judged across window boundaries. A segment that is last in the
    window has no known successor and counts as isolated after it.
    """
    for segment in segments:
        segment.dropped = None
    sequence = list(before) + list(segments)
    first = len(before)

    texts = [normalized(s.text) for s in sequence]
    run_start = 0
    for i in range(1, len(sequence) + 1):
        if i < len(sequence) and texts[i] == texts[run_start] and texts[i]:
            continue
        if i - run_start >= DUPLICATE_RUN:
            for j in range(max(run_start + 1, first), i):
                sequence[j].dropped = DUPLICATE
        run_start = i

    for i in range(first, len(sequence)):
        segment = sequence[i]
        if segment.dropped:
            continue
        isolated = _isolated(sequence, i)
        if (
            segment.no_speech_prob is not None and segment.avg_logprob is not None
            and segment.no_speech_prob > NO_SPEECH_PROB and segment.avg_logprob < LOG_PROB
        ):
            segment.dropped = SILENCE
        elif segment.compression_ratio is not None and segment.compression_ratio > COMPRESSION_RATIO:
            segment.dropped = REPETITION
        elif texts[i] in KNOWN_PHRASES and isolated:
            segment.dropped = KNOWN_PHRASE
        elif (
            segment.mean_word_probability is not None
            and segment.mean_word_probability < LOW_WORD_PROBABILITY
            and (segment.end_ms - segment.start_ms < SHORT_MS or isolated)
        ):
            segment.dropped = LOW_CONFIDENCE


def _isolated(sequence: Sequence[TranscriptSegment], i: int) -> bool:
    segment = sequence[i]
    if i > 0 and segment.start_ms - sequence[i - 1].end_ms < ISOLATION_MS:
        return False
    if i + 1 < len(sequence) and sequence[i + 1].start_ms - segment.end_ms < ISOLATION_MS:
        return False
    return True
