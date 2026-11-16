import pytest

from studyscribe.notes.blocks import cut_blocks
from studyscribe.notes.transcript import Line

MIN = 60_000


def lines_every(seconds, total_minutes, *, long_gaps=(), start_ms=0):
    """A line every `seconds`, each 1 s shorter than the step; `long_gaps` are times (ms) at
    which the line starts 3 s later than usual (a longer pause before it)."""
    out, t, n = [], start_ms, 1
    step = seconds * 1000
    while t < start_ms + total_minutes * MIN:
        shift = 3000 if any(t == g for g in long_gaps) else 0
        out.append(Line(n, n, t + shift, t + step - 1000, f"line {n}"))
        t += step
        n += 1
    return out


def covered(blocks):
    return [line.n for block in blocks for line in block.lines]


def test_no_lines_no_blocks():
    assert cut_blocks([]) == []


def test_lecture_of_at_most_15_minutes_is_one_block():
    lines = lines_every(10, 15)
    blocks = cut_blocks(lines)
    assert len(blocks) == 1 and blocks[0].first_n == 1 and blocks[0].last_n == len(lines)


def test_cut_at_the_longest_gap_inside_the_window():
    lines = lines_every(10, 40, long_gaps=(12 * MIN, 26 * MIN))
    blocks = cut_blocks(lines)
    assert [b.start_ms for b in blocks] == [0, 12 * MIN + 3000, 26 * MIN + 3000]
    assert covered(blocks) == [line.n for line in lines]
    assert [b.index for b in blocks] == [1, 2, 3]


def test_without_a_longer_gap_the_cut_is_as_late_as_possible():
    blocks = cut_blocks(lines_every(10, 40))
    assert [b.start_ms for b in blocks][:2] == [0, 15 * MIN]


def test_a_short_last_block_joins_the_previous_one():
    lines = lines_every(10, 33, long_gaps=(14 * MIN, 28 * MIN))
    blocks = cut_blocks(lines)
    # 0-14, 14-28 and 28-33: the last is under 5 minutes and joins the second
    assert [b.start_ms for b in blocks] == [0, 14 * MIN + 3000]
    assert blocks[-1].last_n == lines[-1].n


def test_a_long_silence_without_lines_in_the_window_is_the_cut():
    before = lines_every(10, 8)
    after = [Line(len(before) + l.n, len(before) + l.n, l.start_ms, l.end_ms, l.text)
             for l in lines_every(10, 12, start_ms=40 * MIN)]
    blocks = cut_blocks(before + after)
    assert [(b.first_n, b.last_n) for b in blocks] == [(1, len(before)),
                                                       (len(before) + 1, len(before + after))]


def test_block_properties():
    block = cut_blocks(lines_every(10, 2))[0]
    assert block.start_ms == 0 and block.end_ms == block.lines[-1].end_ms
    assert block.duration_ms == block.end_ms
    assert block.speech_ms == 9_000 * len(block.lines)


def test_invalid_window():
    with pytest.raises(ValueError):
        cut_blocks(lines_every(10, 2), min_ms=5, max_ms=4)
