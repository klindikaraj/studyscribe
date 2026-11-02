from studyscribe.transcribe.engine import AsrSegment, AsrWord
from studyscribe.transcribe.segments import TranscriptSegment, split_long, to_session


def segment_of(words: list[tuple[float, float, str]]) -> TranscriptSegment:
    ws = [(round(a * 1000), round(b * 1000), text, 0.9) for a, b, text in words]
    return TranscriptSegment(ws[0][0], ws[-1][1], "".join(w[2] for w in ws).strip(), ws,
                             -0.3, 0.05, 1.6)


def evenly(text: str, start: float = 0.0, step: float = 0.5) -> list:
    """One word every `step` seconds, 0.4 s long; words keep their leading space."""
    return [(start + i * step, start + i * step + 0.4, " " + w) for i, w in enumerate(text.split())]


def test_short_segments_are_left_whole() -> None:
    s = segment_of(evenly("This is short."))
    assert split_long(s) == [s]


def test_long_segment_is_split_at_sentence_ends() -> None:
    # 30 s of speech, 60 words, sentences of 10 words (5 s each).
    sentence = "one two three four five six seven eight nine ten."
    s = segment_of(evenly(" ".join([sentence] * 6)))
    pieces = split_long(s)
    assert all(p.end_ms - p.start_ms <= 12_000 for p in pieces)
    assert all(p.text.endswith(".") for p in pieces)
    assert [len(p.words) for p in pieces] == [20, 20, 20]  # two sentences fit in 12 s
    assert " ".join(p.text for p in pieces) == s.text
    assert pieces[0].start_ms == s.start_ms and pieces[-1].end_ms == s.end_ms
    assert all((p.avg_logprob, p.no_speech_prob, p.compression_ratio) == (-0.3, 0.05, 1.6)
               for p in pieces)


def test_without_sentence_ends_clauses_then_pauses() -> None:
    clauses = segment_of(evenly(("a b c d e f g h, " * 4).strip()))  # 16 s, commas every 4 s
    pieces = split_long(clauses)
    assert all(p.text.endswith(",") for p in pieces[:-1])
    assert all(p.end_ms - p.start_ms <= 12_000 for p in pieces)
    # No punctuation at all: the widest pause between words.
    words = evenly("w " * 20) + evenly("x " * 12, start=10.6)  # 0.6 s pause before "x"
    pieces = split_long(segment_of(words))
    assert pieces[0].words[-1][2] == " w" and pieces[1].words[0][2] == " x"


def test_a_sentence_end_too_early_is_not_used() -> None:
    # "Okay." at 0.4 s would leave a scrap: the cut goes to the later clause end.
    words = evenly("Okay. " + "word " * 18 + "here, " + "more " * 10)
    pieces = split_long(segment_of(words))
    assert pieces[0].text != "Okay."
    assert pieces[0].text.endswith("here,")


def test_engine_times_become_session_times() -> None:
    asr = AsrSegment(1.5, 2.25, " Hello there.", (AsrWord(1.5, 1.9, " Hello", 0.9),
                                                  AsrWord(1.9, 2.25, " there.", 0.8)))
    s = to_session(asr, window_start_frame=16_000 * 600)  # window starting at 10:00
    assert (s.start_ms, s.end_ms, s.text) == (601_500, 602_250, "Hello there.")
    assert s.words == [(601_500, 601_900, " Hello", 0.9), (601_900, 602_250, " there.", 0.8)]
