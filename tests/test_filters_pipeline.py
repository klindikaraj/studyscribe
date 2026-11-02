"""Every filter rule fires through the whole chain: engine -> pipeline (split, filters) ->
database -> transcript-dropped.txt. The real 3 h transcript dropped nothing; this shows that is
because its audio is clean, not because the rules never fire."""

from pathlib import Path

from fakes import FakeTranscriber
from test_transcribe_run import recorded, run, timeline  # noqa: F401 (fixture)

from studyscribe.db import repo
from studyscribe.transcribe import export, filters
from studyscribe.transcribe.engine import AsrSegment, AsrWord

# (text, word probability, no_speech_prob, avg_logprob, compression_ratio) per burst.
SCRIPT = [
    ("Hello everyone, today we talk about arrays.", 0.95, 0.01, -0.2, 1.4),
    ("so so so", 0.6, 0.8, -1.3, 1.1),  # silence rule
    ("the the the the the the the the the the the", 0.9, 0.05, -0.4, 3.2),  # repetition
    ("Okay.", 0.9, 0.01, -0.2, 1.0),
    ("Okay.", 0.9, 0.01, -0.2, 1.0),  # duplicate
    ("Okay.", 0.9, 0.01, -0.2, 1.0),  # duplicate
    ("Thanks for watching!", 0.9, 0.3, -0.5, 1.0),  # known phrase, isolated
    ("mmh hm", 0.2, 0.2, -0.8, 1.0),  # low confidence, short
    ("Let's look at the next example.", 0.95, 0.01, -0.2, 1.3),
]
EXPECTED = [None, filters.SILENCE, filters.REPETITION, None, filters.DUPLICATE,
            filters.DUPLICATE, filters.KNOWN_PHRASE, filters.LOW_CONFIDENCE, None]
# 1 s bursts 0.5 s apart, with 3 s of quiet around the known phrase (isolated).
BURSTS = [(0.5, 1.5), (2.0, 3.0), (3.5, 4.5), (5.0, 6.0), (6.5, 7.5), (8.0, 9.0),
          (12.0, 13.0), (16.0, 17.0), (17.5, 18.5)]


class ScriptedEngine(FakeTranscriber):
    def transcribe(self, audio, language, hints=None):
        for i, seg in enumerate(super().transcribe(audio, language, hints)):
            text, p, no_speech, logprob, ratio = SCRIPT[self._count - 1]
            tokens = text.split()
            step = (seg.end - seg.start) / len(tokens)
            words = tuple(AsrWord(seg.start + k * step, seg.start + (k + 1) * step, " " + t, p)
                          for k, t in enumerate(tokens))
            yield AsrSegment(seg.start, seg.end, text, words, logprob, no_speech, ratio)


def test_each_rule_drops_through_the_whole_chain(db, data_dir: Path, recorded) -> None:  # noqa: F811
    sid = recorded(timeline(BURSTS, seconds=20.0))
    run(db, data_dir, sid, ScriptedEngine(), language="en",
        window_options={"max_window_s": 60.0, "search_s": 5.0})
    rows = repo.list_transcript(db, sid, include_dropped=True)
    assert [r["text"] for r in rows] == [s[0] for s in SCRIPT]
    assert [r["dropped"] for r in rows] == EXPECTED
    kept = repo.list_transcript(db, sid)
    assert [r["text"] for r in kept] == [s[0] for s, e in zip(SCRIPT, EXPECTED) if e is None]

    export.export_text(db, data_dir, sid)
    folder = export.exports_dir(data_dir, sid)
    dropped = (folder / export.DROPPED_FILE).read_text(encoding="utf-8")
    assert dropped.startswith("Segments dropped by the anti-hallucination filters: 6")
    for rule in (filters.SILENCE, filters.REPETITION, filters.DUPLICATE, filters.KNOWN_PHRASE,
                 filters.LOW_CONFIDENCE):
        assert f"] {rule}  (" in dropped
    srt = (folder / export.SRT_FILE).read_text(encoding="utf-8")
    assert "Thanks for watching" not in srt and "the the the" not in srt
    assert srt.count("Okay.") == 1
