"""scripts/check_notes_export.py: the notes' timestamps against transcript.srt and the audio."""

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from fakes import generate_fake_notes

from studyscribe.notes.export import export_notes
from studyscribe.transcribe import export

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_notes_export.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("check_notes_export", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


@pytest.fixture
def exported(db, data_dir, transcribed):
    sid = transcribed()
    generate_fake_notes(db, data_dir, sid)
    export.export_text(db, data_dir, sid)
    return export_notes(db, data_dir, sid).folder


def fake_flac(path, seconds, rate=16000):
    """A FLAC file that is only a STREAMINFO block saying how long it is."""
    sample_rate = (rate << 4) | 0  # 20 bits of rate, 3 of channels - 1, top bit of bits - 1
    body = (bytes(10) + sample_rate.to_bytes(3, "big")
            + (((15 & 0xF) << 36) | seconds * rate).to_bytes(5, "big") + bytes(16))
    path.write_bytes(b"fLaC" + bytes([0x80]) + len(body).to_bytes(3, "big") + body)


def test_the_notes_of_a_session_match_its_srt(check, exported, capsys):
    assert check.main([str(exported)]) == 0
    out = capsys.readouterr().out
    assert "notes.html:" in out and "OK" in out and "checked against the SRT" in out
    assert "no audio.flac" in out


def test_coverage_and_appendix_links_are_accepted(check, db, data_dir, transcribed, capsys):
    import json

    from studyscribe.db import repo
    from studyscribe.db.connection import transaction

    sid = transcribed()
    notes_id = generate_fake_notes(db, data_dir, sid)
    stats = {"coverage_warnings": [{"block": 2, "last_cited": 100, "uncovered": [101, 144],
                                    "start_ms": 1_010_000, "end_ms": 1_430_000}]}
    with transaction(db):
        db.execute("UPDATE generated_notes SET stats_json = ? WHERE id = ?",
                   (json.dumps(stats), notes_id))
    export.export_text(db, data_dir, sid)
    folder = export_notes(db, data_dir, sid).folder
    page = (folder / "notes.html").read_text(encoding="utf-8")
    assert "potrebbero non coprire" in page and repo.get_notes(db, notes_id) is not None
    assert check.main([str(folder)]) == 0
    assert "OK" in capsys.readouterr().out


def test_a_time_that_is_not_a_cue_start_is_reported(check, exported, capsys):
    page = exported / "notes.html"
    text = page.read_text(encoding="utf-8")
    page.write_text(re.sub(r'data-ms="(\d+)"', lambda m: f'data-ms="{int(m[1]) + 1}"', text,
                           count=1), encoding="utf-8")
    assert check.main([str(exported)]) == 1
    assert "is not the start of any cue" in capsys.readouterr().out


def test_a_link_that_seeks_somewhere_else_is_reported(check, exported, capsys):
    page = exported / "notes.html"
    text = page.read_text(encoding="utf-8")
    page.write_text(text.replace('data-t="', 'data-t="1', 1), encoding="utf-8")
    assert check.main([str(exported)]) == 1
    out = capsys.readouterr().out
    assert "the label does not show" in out and "seeks 1" in out


def test_a_time_past_the_end_of_the_audio_is_reported(check, exported, capsys):
    fake_flac(exported / "audio.flac", seconds=60)
    assert check.main([str(exported)]) == 1
    assert "past the end of the audio (60.0 s)" in capsys.readouterr().out
    fake_flac(exported / "audio.flac", seconds=24 * 60)
    assert check.main([str(exported)]) == 0
    assert "audio.flac: 1440.0 s" in capsys.readouterr().out


def test_the_flac_length_is_read_from_a_real_file(check, tmp_path):
    import av

    path = tmp_path / "a.flac"
    with av.open(str(path), "w", format="flac") as container:
        stream = container.add_stream("flac", rate=16000, layout="mono")
        stream.format = "s16"
        frame = av.AudioFrame.from_ndarray(np.zeros((1, 16000 * 3), dtype=np.int16),
                                           format="s16", layout="mono")
        frame.sample_rate = 16000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    assert check.flac_duration_ms(path) == 3000
    (tmp_path / "b.flac").write_bytes(b"RIFF....")
    assert check.flac_duration_ms(tmp_path / "b.flac") is None


def test_usage_errors(check, tmp_path, capsys):
    assert check.main([]) == 2
    assert check.main([str(tmp_path)]) == 2 and "No transcript.srt" in capsys.readouterr().out
    (tmp_path / "transcript.srt").write_text("", encoding="utf-8")
    assert check.main([str(tmp_path)]) == 2 and "No notes*.html" in capsys.readouterr().out
