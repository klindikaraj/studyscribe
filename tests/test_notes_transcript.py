import json

from studyscribe.notes.transcript import Line, Word, format_lines, make_lines, speech_ms


def row(id_, start_ms, end_ms, text, words=None):
    return {"id": id_, "start_ms": start_ms, "end_ms": end_ms, "text": text,
            "words_json": json.dumps(words) if words is not None else None}


def test_make_lines_numbers_in_time_order_and_parses_words():
    rows = [
        row(7, 5_000, 9_000, " second line "),
        row(3, 1_000, 4_000, " first line", [[1_000, 1_400, " first", 0.9],
                                              [1_400, 2_000, " line", 0.3]]),
        row(5, 5_000, 6_000, "same start, lower id"),
    ]
    lines = make_lines(rows)
    assert [(l.n, l.segment_id) for l in lines] == [(1, 3), (2, 5), (3, 7)]
    assert lines[0].text == "first line"
    assert lines[0].words == (Word(1_000, 1_400, " first", 0.9), Word(1_400, 2_000, " line", 0.3))
    assert lines[1].words == ()
    assert lines[2].label == "S3"


def test_format_lines_puts_a_marker_before_the_first_line_of_each_minute():
    lines = [Line(1, 1, 0, 5_000, "a"), Line(2, 2, 30_000, 35_000, "b"),
             Line(3, 3, 61_000, 64_000, "c"), Line(4, 4, 185_000, 190_000, "d")]
    assert format_lines(lines).splitlines() == [
        "[0:00:00]", "S1 a", "S2 b", "[0:01:01]", "S3 c", "[0:03:05]", "S4 d"]


def test_format_lines_of_a_block_starts_with_a_marker():
    lines = [Line(41, 9, 3_725_000, 3_730_000, "late")]
    assert format_lines(lines) == "[1:02:05]\nS41 late"


def test_speech_ms_sums_line_durations():
    assert speech_ms([Line(1, 1, 0, 1_500, "a"), Line(2, 2, 10_000, 12_000, "b")]) == 3_500
