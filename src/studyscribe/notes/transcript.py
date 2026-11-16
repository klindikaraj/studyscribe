"""A session's transcript as the notes pipeline sees it: numbered lines and the prompt's format.

Lines are the kept segments of the session's transcript, in time order, numbered from 1. The
number is the line's S-number: prompts cite `S<n>` instead of database IDs (cheaper in tokens,
and stable within one transcript), and the code maps it back to the segment and its time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from studyscribe.transcribe.export import clock

MARKER_EVERY_MS = 60_000


@dataclass(frozen=True)
class Word:
    start_ms: int
    end_ms: int
    text: str  # as Whisper wrote it, with its leading space and punctuation
    probability: float


@dataclass(frozen=True)
class Line:
    n: int  # S-number: position among the kept segments, from 1
    segment_id: int
    start_ms: int
    end_ms: int
    text: str
    words: tuple[Word, ...] = ()

    @property
    def label(self) -> str:
        return f"S{self.n}"


def make_lines(rows: Iterable[Mapping[str, Any]]) -> list[Line]:
    """Lines from transcript segment rows (`id`, `start_ms`, `end_ms`, `text`, `words_json`),
    already filtered to the kept segments. Sorted by start time, then id, as the transcript
    exports do."""
    ordered = sorted(rows, key=lambda r: (r["start_ms"], r["id"]))
    return [
        Line(n=n, segment_id=row["id"], start_ms=row["start_ms"], end_ms=row["end_ms"],
             text=row["text"].strip(), words=_words(row["words_json"]))
        for n, row in enumerate(ordered, start=1)
    ]


def format_lines(lines: Sequence[Line]) -> str:
    """The prompt's input: one `S<n> text` per line and an `[H:MM:SS]` marker before the first
    line of every minute, so the model can place topics in time without counting."""
    out: list[str] = []
    next_marker = 0
    for line in lines:
        if line.start_ms >= next_marker:
            out.append(f"[{clock(line.start_ms)}]")
            next_marker = (line.start_ms // MARKER_EVERY_MS + 1) * MARKER_EVERY_MS
        out.append(f"{line.label} {line.text}")
    return "\n".join(out)


def speech_ms(lines: Iterable[Line]) -> int:
    """Total duration of the lines (the speech), used for the per-hour targets."""
    return sum(max(0, line.end_ms - line.start_ms) for line in lines)


def _words(raw: str | None) -> tuple[Word, ...]:
    if not raw:
        return ()
    return tuple(Word(int(w[0]), int(w[1]), str(w[2]), float(w[3])) for w in json.loads(raw))
