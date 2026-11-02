import io
import json
from pathlib import Path

import av
import numpy as np
import pytest
from fakes import FakeTranscriber, loud_spans
from test_transcribe_run import SMALL_WINDOWS, recorded, run  # noqa: F401 (fixture)

from studyscribe import cli
from studyscribe.audio.wav import wav_header
from studyscribe.config import session_dir
from studyscribe.db import repo
from studyscribe.transcribe import export


def seg(start_ms, end_ms, text, words=None, dropped=None, **metrics):
    words = words or [[start_ms, end_ms, " " + text, 0.9]]
    return {"start_ms": start_ms, "end_ms": end_ms, "text": text, "words_json": json.dumps(words),
            "dropped": dropped, "no_speech_prob": metrics.get("no_speech", 0.01),
            "avg_logprob": metrics.get("logprob", -0.2), "compression_ratio": 1.3}


def evenly(text: str, start_ms: int, step_ms: int = 400) -> list:
    return [[start_ms + i * step_ms, start_ms + i * step_ms + 350, " " + w, 0.9]
            for i, w in enumerate(text.split())]


def test_srt_times_and_numbering() -> None:
    text = export.to_srt([seg(1_500, 4_250, "Hello there."), seg(3_725_010, 3_727_000, "Later.")])
    assert text == ("1\n00:00:01,500 --> 00:00:04,250\nHello there.\n\n"
                    "2\n01:02:05,010 --> 01:02:07,000\nLater.\n")


def test_long_segments_become_several_short_cues() -> None:
    sentence = "This segment is far too long to fit on the screen in one subtitle cue"
    words = evenly(sentence + " " + sentence, 10_000)
    cues = export.cues(seg(words[0][0], words[-1][1], sentence + " " + sentence, words))
    assert len(cues) >= 2
    assert all(len(text) <= export.CUE_MAX_CHARS and end - start <= export.CUE_MAX_MS
               for start, end, text in cues)
    assert " ".join(t for _, _, t in cues) == sentence + " " + sentence
    assert export.wrap_cue("a" * 30 + " " + "b" * 30).split("\n") == ["a" * 30, "b" * 30]
    assert export.wrap_cue("short line") == "short line"


def test_markdown_paragraphs_and_header() -> None:
    segments = [seg(0, 4_000, "First."), seg(4_500, 8_000, "Same paragraph."),
                seg(12_000, 15_000, "After a pause."), seg(3_700_000, 3_702_000, "An hour on.")]
    md = export.to_markdown({"title": "Lecture 3", "Course": "Calculus", "Duration": "1:01:42",
                             "Language": "en (detected)"}, segments)
    assert md.startswith("# Lecture 3\n\n- Course: Calculus\n- Duration: 1:01:42\n")
    assert "**[0:00:00]** First. Same paragraph.\n" in md
    assert "**[0:00:12]** After a pause.\n" in md and "**[1:01:40]** An hour on.\n" in md
    assert "audio.flac" in md


def test_long_speech_is_cut_into_paragraphs() -> None:
    segments = [seg(i * 5_000, i * 5_000 + 4_900, f"s{i}.") for i in range(40)]  # 200 s, no pause
    starts = [start for start, _ in export.paragraphs(segments)]
    assert starts == [0, 60_000, 120_000, 180_000]


def test_dropped_list() -> None:
    segments = [seg(0, 2_000, "kept"),
                seg(30_000, 32_000, "Thanks for watching!", dropped="known phrase")]
    text = export.to_dropped(segments)
    assert text.startswith("Segments dropped by the anti-hallucination filters: 1")
    assert "[0:00:30 -> 0:00:32] known phrase  (no_speech 0.01" in text
    assert "mean_word_p 0.90" in text and "Thanks for watching!" in text
    assert export.to_dropped([seg(0, 1, "x")]).startswith("Segments dropped by the "
                                                         "anti-hallucination filters: 0")


def test_files_are_written_from_the_database(db, data_dir, recorded) -> None:  # noqa: F811
    sid = recorded()
    with pytest.raises(export.NoTranscript):
        export.export_text(db, data_dir, sid)
    run(db, data_dir, sid)
    paths = export.export_text(db, data_dir, sid)
    assert [p.name for p in paths] == [export.SRT_FILE, export.MD_FILE, export.DROPPED_FILE]
    srt = paths[0].read_text(encoding="utf-8")
    assert srt.count(" --> ") == 5 and srt.startswith("1\n00:00:00,5")
    md = paths[1].read_text(encoding="utf-8")
    assert md.startswith("# Lecture\n") and "- Language: en (detected) - en 0.90, it 0.10" in md
    assert not list(paths[0].parent.glob("*.tmp"))


def test_audio_flac_has_exactly_the_segment_samples(db, data_dir, start_session) -> None:
    sid = start_session("t")
    rng = np.random.default_rng(3)
    folder = session_dir(data_dir, sid) / "audio"
    parts = []
    for seq, n in ((1, 960_000), (2, 31_337)):  # one full 60 s segment and a short last one
        pcm = rng.integers(-8_000, 8_000, n).astype("<i2")
        (folder / f"{seq:06d}.wav").write_bytes(wav_header(pcm.size * 2) + pcm.tobytes())
        start = (seq - 1) * 60_000
        repo.add_audio_segment(db, sid, seq, f"audio/{seq:06d}.wav", start,
                               start + n * 1000 // 16_000, 0)
        parts.append(pcm)
    path = export.export_audio(db, data_dir, sid)
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        assert (stream.rate, stream.codec_context.name) == (16_000, "flac")
        decoded = np.concatenate([f.to_ndarray().reshape(-1) for f in container.decode(stream)])
    assert np.array_equal(decoded, np.concatenate(parts))


def test_transcribe_writes_the_exports_and_export_rewrites_them(db, data_dir,
                                                                recorded) -> None:  # noqa: F811
    sid = recorded()
    args = cli.build_parser().parse_args(["transcribe", sid])
    out = io.StringIO()
    assert cli.cmd_transcribe(db, data_dir, args, out, load=FakeTranscriber, vad=loud_spans,
                              window_options=SMALL_WINDOWS) == 0
    assert "Exports in" in out.getvalue() and "transcript.md" in out.getvalue()
    md = export.exports_dir(data_dir, sid) / export.MD_FILE
    md.unlink()
    out = io.StringIO()
    assert cli.cmd_export(db, data_dir, cli.build_parser().parse_args(
        ["export", sid, "--audio"]), out) == 0
    assert md.is_file() and (export.exports_dir(data_dir, sid) / export.AUDIO_FILE).is_file()
    out = io.StringIO()
    assert cli.cmd_export(db, data_dir, cli.build_parser().parse_args(["export", "nope"]),
                          out) == 2
