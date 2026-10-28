import functools
import re
import sqlite3
import time
import wave
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeSource, ScriptedDefaultOutput, wait_for

from studyscribe.audio.capture import CaptureConfig
from studyscribe.config import db_path, session_dir
from studyscribe.db import repo
from studyscribe.db.connection import open_database
from studyscribe.recording import RecorderHandle
from studyscribe.session import SessionSink, final_status, new_session


class Events:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, kind: str, **fields) -> None:
        self.events.append({"type": kind, **fields})

    def of_type(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["type"] == kind]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def db(data_dir: Path):
    conn = open_database(db_path(data_dir))
    yield conn
    conn.close()


# --- ids and rows ------------------------------------------------------------------------


def test_session_id_is_readable_and_sortable() -> None:
    sid = repo.new_session_id(datetime(2026, 10, 28, 14, 23, 1))
    assert re.fullmatch(r"20261028-142301-[0-9a-f]{6}", sid)
    assert repo.new_session_id(datetime(2026, 10, 28, 14, 23, 1)) != sid


def test_new_session_creates_folder_row_and_course(db, data_dir: Path) -> None:
    sid = new_session(db, data_dir, "Lecture 1", course="Machine Learning")
    assert (session_dir(data_dir, sid) / "audio").is_dir()
    row = repo.get_session(db, sid)
    assert (row["title"], row["status"]) == ("Lecture 1", "recording")
    sid2 = new_session(db, data_dir, "Lecture 2", course="machine learning")
    assert repo.get_session(db, sid2)["course_id"] == row["course_id"]
    assert db.execute("SELECT COUNT(*) FROM courses").fetchone()[0] == 1


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Èlite", "èlite"),  # non-ASCII case: COLLATE NOCASE would treat these as different
        ("Èlite", "Èlite"),  # composed vs combining accent
        ("Straße", "STRASSE"),  # full case folding
        ("  Calculus 1 ", "calculus 1"),
    ],
)
def test_course_key_matches_equivalent_titles(a: str, b: str) -> None:
    assert repo.course_key(a) == repo.course_key(b)


def test_course_key_keeps_different_titles_apart() -> None:
    assert repo.course_key("Èlite") != repo.course_key("Elite")
    assert repo.course_key("Calculus 1") != repo.course_key("Calculus 2")


def test_course_is_reused_whatever_case_and_accent_encoding(db) -> None:
    first = repo.get_or_create_course(db, "Èlite Analysis")
    assert repo.get_or_create_course(db, "èlite analysis") == first
    assert repo.get_or_create_course(db, "ÈLITE ANALYSIS") == first
    row = db.execute("SELECT title, title_key FROM courses").fetchone()
    assert (row["title"], row["title_key"]) == ("Èlite Analysis", "èlite analysis")


def test_new_session_without_course(db, data_dir: Path) -> None:
    sid = new_session(db, data_dir, None)
    assert repo.get_session(db, sid)["course_id"] is None


@pytest.mark.parametrize(
    ("reason", "frames", "status"),
    [
        ("requested", 100, "recorded"),
        ("requested", 0, "recorded"),
        ("parent_exited", 100, "interrupted"),
        ("error", 100, "interrupted"),
        ("error", 0, "failed"),
    ],
)
def test_final_status(reason: str, frames: int, status: str) -> None:
    assert final_status(reason, frames) == status


# --- SessionSink, in-process -------------------------------------------------------------


def test_sink_writes_segments_manifest_and_status(db, data_dir: Path) -> None:
    sid = new_session(db, data_dir, "t")
    provisional = repo.get_session(db, sid)["started_at"]
    time.sleep(0.01)
    events = Events()
    sink = SessionSink(str(data_dir), sid, segment_frames=8_000)  # 0.5 s segments
    sink.start(events)
    for _ in range(25):
        sink.write(np.zeros(1_000, dtype=np.int16))
    sink.write(np.zeros(2_000, dtype=np.int16), gap=True)  # 27000 frames: 3 x 8000 + 3000
    sink.close("requested")

    session = repo.get_session(db, sid)
    assert session["status"] == "recorded"
    assert session["started_at"] > provisional
    assert session["ended_at"] >= session["started_at"]
    rows = repo.list_audio_segments(db, sid)
    assert [(r["seq"], r["path"], r["start_ms"], r["end_ms"], r["gap_ms"]) for r in rows] == [
        (1, "audio/000001.wav", 0, 500, 0),
        (2, "audio/000002.wav", 500, 1000, 0),
        (3, "audio/000003.wav", 1000, 1500, 0),
        (4, "audio/000004.wav", 1500, 1687, 125),
    ]
    for r in rows:
        with wave.open(str(session_dir(data_dir, sid) / r["path"])) as w:
            assert w.getnframes() * 1000 // 16_000 == r["end_ms"] - r["start_ms"]
    assert [e["seq"] for e in events.of_type("segment")] == [1, 2, 3, 4]
    assert events.of_type("warning") == []


def test_database_errors_do_not_stop_the_audio(db, data_dir: Path, monkeypatch) -> None:
    sid = new_session(db, data_dir, "t")

    def broken(*args):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(repo, "add_audio_segment", broken)
    events = Events()
    sink = SessionSink(str(data_dir), sid, segment_frames=8_000)
    sink.start(events)
    sink.write(np.zeros(20_000, dtype=np.int16))
    sink.close("requested")
    assert len(list((session_dir(data_dir, sid) / "audio").iterdir())) == 3
    warnings = [e["message"] for e in events.of_type("warning")]
    assert len(warnings) == 3 and "database is locked" in warnings[0]
    assert repo.get_session(db, sid)["status"] == "recorded"


def test_failure_before_start_marks_the_session_failed(db, data_dir: Path) -> None:
    sid = new_session(db, data_dir, "t")
    SessionSink(str(data_dir), sid).close("error")
    assert repo.get_session(db, sid)["status"] == "failed"


# --- end to end: capture process writing a real session ------------------------------------


def test_capture_process_records_a_session(db, data_dir: Path) -> None:
    sid = new_session(db, data_dir, "end to end")
    config = CaptureConfig(
        sink_factory=functools.partial(SessionSink, str(data_dir), sid, segment_frames=8_000),
        source_factory=functools.partial(FakeSource, frames_per_drain=2_400),
        default_output_factory=ScriptedDefaultOutput,
    )
    handle = RecorderHandle(config)
    handle.start()
    try:
        wait_for(handle, "started")
        time.sleep(1.5)
    finally:
        exitcode = handle.stop()
    assert exitcode == 0
    events = handle.events()
    assert [e for e in events if e["type"] == "warning"] == []

    session = repo.get_session(db, sid)
    assert session["status"] == "recorded"
    rows = repo.list_audio_segments(db, sid)
    assert len(rows) >= 2
    assert rows[0]["start_ms"] == 0
    for prev, cur in zip(rows, rows[1:]):
        assert cur["start_ms"] == prev["end_ms"]
        assert prev["end_ms"] - prev["start_ms"] == 500
    stopped = [e for e in events if e["type"] == "stopped"][0]
    assert rows[-1]["end_ms"] == pytest.approx(stopped["audio_s"] * 1000, abs=1)
