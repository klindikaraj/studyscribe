from studyscribe.transcribe import filters
from studyscribe.transcribe.segments import TranscriptSegment


def seg(start_s: float, end_s: float, text: str = "some words", p: float = 0.9,
        no_speech: float = 0.01, logprob: float = -0.2, ratio: float = 1.3) -> TranscriptSegment:
    start, end = round(start_s * 1000), round(end_s * 1000)
    return TranscriptSegment(start, end, text, [(start, end, text, p)], logprob, no_speech, ratio)


def reasons(segments) -> list:
    return [s.dropped for s in segments]


def test_ordinary_speech_is_kept() -> None:
    segments = [seg(0, 4, "Hello everyone."), seg(4.2, 8, "Today: variables.")]
    filters.apply_filters(segments)
    assert reasons(segments) == [None, None]


def test_silence_rule_needs_both_conditions() -> None:
    segments = [
        seg(0, 3, "a", no_speech=0.8, logprob=-1.2),
        seg(3, 6, "b", no_speech=0.8, logprob=-0.4),  # confident text: kept
        seg(6, 9, "c", no_speech=0.2, logprob=-1.5),
    ]
    filters.apply_filters(segments)
    assert reasons(segments) == [filters.SILENCE, None, None]


def test_repetition_loops_are_dropped() -> None:
    segments = [seg(0, 30, "the the the the the the", ratio=3.1)]
    filters.apply_filters(segments)
    assert reasons(segments) == [filters.REPETITION]


def test_runs_of_three_equal_texts_keep_the_first() -> None:
    segments = [seg(0, 2, "Okay."), seg(2, 4, "okay"), seg(4, 6, "Okay!"), seg(6, 8, "Next.")]
    filters.apply_filters(segments)
    assert reasons(segments) == [None, filters.DUPLICATE, filters.DUPLICATE, None]
    two = [seg(0, 2, "No."), seg(2, 4, "No.")]
    filters.apply_filters(two)
    assert reasons(two) == [None, None]


def test_duplicates_across_windows() -> None:
    before = [seg(0, 2, "Right."), seg(2, 4, "Right.")]
    segments = [seg(4, 6, "Right."), seg(6, 8, "Moving on.")]
    filters.apply_filters(segments, before)
    assert reasons(segments) == [filters.DUPLICATE, None]
    assert reasons(before) == [None, None]  # earlier windows are never changed


def test_known_phrase_only_when_isolated() -> None:
    alone = [seg(0, 10, "Lecture text."), seg(20, 22, "Thanks for watching!"),
             seg(40, 45, "More text.")]
    filters.apply_filters(alone)
    assert reasons(alone) == [None, filters.KNOWN_PHRASE, None]
    spoken = [seg(0, 10, "That's the course."), seg(10.5, 12, "Thanks for watching!")]
    filters.apply_filters(spoken[1:], spoken[:1])
    assert reasons(spoken[1:]) == [None]
    italian = [seg(0, 3, "Sottotitoli creati dalla comunità Amara.org")]
    filters.apply_filters(italian)
    assert reasons(italian) == [filters.KNOWN_PHRASE]


def test_low_confidence_when_short_or_isolated() -> None:
    segments = [
        seg(0, 10, "clear speech"),
        seg(10.2, 11, "mumble", p=0.2),  # short
        seg(11.2, 20, "long and unclear but in the flow", p=0.2),  # long, not isolated
        seg(30, 34, "unclear and alone", p=0.3),  # isolated
    ]
    filters.apply_filters(segments)
    assert reasons(segments) == [None, filters.LOW_CONFIDENCE, None, filters.LOW_CONFIDENCE]


def test_normalized_text() -> None:
    assert filters.normalized("  Sottotitoli creati dalla Comunità Amara.org! ") == (
        "sottotitoli creati dalla comunita amara org"
    )
