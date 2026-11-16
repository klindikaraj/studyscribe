from studyscribe.notes.transcript import Line, Word
from studyscribe.notes.uncertain import UncertainWord, normalize, uncertain_words


def line(n, *words):
    """words: (text, probability); each word lasts 100 ms from n * 10 s."""
    start = n * 10_000
    return Line(n, n, start, start + 100 * len(words), "".join(t for t, _ in words),
                tuple(Word(start + 100 * i, start + 100 * (i + 1), t, p)
                      for i, (t, p) in enumerate(words)))


def test_lists_rare_low_probability_words_with_line_time_and_probability():
    lines = [
        line(1, (" stacchiamoli", 0.95), (" col", 0.9), (" metodo", 0.97), (" di", 0.41),
             (" TEC", 0.35), (" per", 0.31), (" plottare.", 0.9)),
        line(2, (" per", 0.9), (" ogni", 0.9), (" epoca,", 0.9), (" per", 0.95)),
    ]
    # "per" is low too, but said three times; "di" has two letters
    assert uncertain_words(lines) == [UncertainWord("TEC", 1, 10_400, 0.35)]


def test_short_words_and_threshold_are_excluded():
    lines = [line(1, (" at", 0.1), (" the", 0.2), (" var", 0.4), (" keyboard,", 0.38))]
    # "at": 2 letters; "the": 3 letters but listed; "var": probability not below 0.4
    assert [w.word for w in uncertain_words(lines)] == ["the", "keyboard"]


def test_words_said_often_are_taken_as_known():
    lines = [line(1, (" Okay", 0.2)), line(2, (" okay,", 0.9)), line(3, (" OKAY.", 0.9)),
             line(4, (" left.", 0.27)), line(5, (" left", 0.9))]
    # "okay" is said 3 times (case and punctuation ignored); "left" twice: still rare
    assert [(w.word, w.n) for w in uncertain_words(lines)] == [("left", 4)]


def test_digits_do_not_count_as_letters():
    lines = [line(1, (" 23456", 0.1), (" 3D", 0.1), (" mp3s", 0.1))]
    assert [w.word for w in uncertain_words(lines)] == ["mp3s"]


def test_lines_without_words_and_custom_rule():
    lines = [Line(1, 1, 0, 1000, "no words"), line(2, (" maybe", 0.6))]
    assert uncertain_words(lines) == []
    assert [w.word for w in uncertain_words(lines, threshold=0.7)] == ["maybe"]


def test_normalize():
    assert normalize(" Keyboard,") == "keyboard"
    assert normalize("don't") == "dont"
