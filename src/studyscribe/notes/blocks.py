"""Cuts a transcript into the map's blocks: 10 to 15 minutes each, cut in a pause.

A block is a run of consecutive lines. From the start of a block, the cut goes before the line
that ends the longest gap among the lines starting between 10 and 15 minutes later, so blocks
end where the lecturer pauses rather than mid-sentence. A lecture of at most 15 minutes is one
block, and a last block shorter than 5 minutes joins the previous one (so a block can reach
about 20 minutes). Durations are wall-clock spans on the session timeline, pauses included.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from studyscribe.notes.transcript import Line, speech_ms

MIN_BLOCK_MS = 10 * 60_000
MAX_BLOCK_MS = 15 * 60_000
MIN_LAST_BLOCK_MS = 5 * 60_000


@dataclass(frozen=True)
class Block:
    index: int  # from 1
    lines: tuple[Line, ...]

    @property
    def first_n(self) -> int:
        return self.lines[0].n

    @property
    def last_n(self) -> int:
        return self.lines[-1].n

    @property
    def start_ms(self) -> int:
        return self.lines[0].start_ms

    @property
    def end_ms(self) -> int:
        return self.lines[-1].end_ms

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def speech_ms(self) -> int:
        return speech_ms(self.lines)


def cut_blocks(lines: Sequence[Line], *, min_ms: int = MIN_BLOCK_MS, max_ms: int = MAX_BLOCK_MS,
               min_last_ms: int = MIN_LAST_BLOCK_MS) -> list[Block]:
    """Blocks covering every line once, in order. No lines, no blocks."""
    if not 0 < min_ms <= max_ms:
        raise ValueError("need 0 < min_ms <= max_ms")
    groups: list[list[Line]] = []
    i = 0
    while i < len(lines):
        start = lines[i].start_ms
        if lines[-1].end_ms - start <= max_ms:
            groups.append(list(lines[i:]))
            break
        cut = _cut_index(lines, i, start, min_ms, max_ms)
        groups.append(list(lines[i:cut]))
        i = cut
    if len(groups) >= 2 and groups[-1][-1].end_ms - groups[-1][0].start_ms < min_last_ms:
        groups[-2].extend(groups.pop())
    return [Block(index, tuple(group)) for index, group in enumerate(groups, start=1)]


def _cut_index(lines: Sequence[Line], i: int, start: int, min_ms: int, max_ms: int) -> int:
    """Index of the line the next block starts with."""
    best: int | None = None
    best_gap = -1
    for j in range(i + 1, len(lines)):
        offset = lines[j].start_ms - start
        if offset > max_ms:
            if best is None:
                return j  # nothing starts inside the window: cut at the silence before j
            break
        if offset >= min_ms:
            gap = lines[j].start_ms - lines[j - 1].end_ms
            if gap >= best_gap:  # on a tie the later cut: fewer, longer blocks
                best, best_gap = j, gap
    return best if best is not None else len(lines)
