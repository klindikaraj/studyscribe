"""Recovery of sessions left in 'recording' (and of manifest rows lost to database errors).

Segments of FRAMES samples (0.5 s) keep the files small. "Live" sessions rely on locks being
per file handle, which is how Windows behaves.
"""

import functools
import os
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeSource, ScriptedDefaultOutput, wait_for

from studyscribe.audio.capture import CaptureConfig
from studyscribe.audio.wav import wav_header
from studyscribe.config import session_dir
from studyscribe.db import repo
from studyscribe.recording import RecorderHandle
from studyscribe.recovery import recover_all, recover_session
from studyscribe.session import SessionSink, new_session

FRAMES = 8_000
windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="in-process lock conflicts need Windows file locks"
)


def abandoned_session(db, data_dir: Path, title: str = "t") -> str:
    """A session whose recording processes are all gone: row in 'recording', locks free."""
    session_id, owner_lock = new_session(db, data_dir, title)
    owner_lock.release()
    return session_id


def segment(data_dir: Path, session_id: str, seq: int) -> Path:
    return session_dir(data_dir, session_id) / "audio" / f"{seq:06d}.wav"


def write_segment(path: Path, frames: int, header_frames: int | None = None,
                  odd_byte: bool = False) -> Path:
    """A segment on disk; `header_frames` below `frames` is what a kill leaves behind."""
    header_frames = frames if header_frames is None else header_frames
    data = np.ones(frames, dtype="<i2").tobytes() + (b"\x01" if odd_byte else b"")
    path.write_bytes(wav_header(header_frames * 2) + data)
    return path


def manifest(db, session_id: str) -> list[tuple]:
    return [
        (r["seq"], r["path"], r["start_ms"], r["end_ms"], r["gap_ms"])
        for r in repo.list_audio_segments(db, session_id)
    ]


def test_crashed_session_is_repaired_and_marked_interrupted(db, data_dir: Path) -> None:
    sid = abandoned_session(db, data_dir)
    write_segment(segment(data_dir, sid, 1), FRAMES)
    repo.add_audio_segment(db, sid, 1, "audio/000001.wav", 0, 500, 120)
    write_segment(segment(data_dir, sid, 2), FRAMES)  # row lost to a database error
    last = write_segment(segment(data_dir, sid, 3), 3_000, header_frames=0, odd_byte=True)
    killed_at = 1_800_000_000.25
    os.utime(last, (killed_at, killed_at))

    [result] = recover_all(db, data_dir, segment_frames=FRAMES)

    assert (result.session_id, result.status_before, result.status_after) == (
        sid, "recording", "interrupted"
    )
    assert (result.added, result.repaired, result.problems) == ([2, 3], [3], [])
    assert result.duration_ms == 1_187
    assert manifest(db, sid) == [
        (1, "audio/000001.wav", 0, 500, 120),
        (2, "audio/000002.wav", 500, 1_000, 0),
        (3, "audio/000003.wav", 1_000, 1_187, 0),  # gap_ms of a recovered segment is lost
    ]
    with wave.open(str(last)) as w:
        assert w.getnframes() == 3_000
    session = repo.get_session(db, sid)
    # ended_at is the last write before the crash, read before the header repair touched it.
    expected = datetime.fromtimestamp(killed_at, timezone.utc).isoformat(timespec="milliseconds")
    assert session["ended_at"] == expected


def test_recovery_is_idempotent(db, data_dir: Path) -> None:
    sid = abandoned_session(db, data_dir)
    write_segment(segment(data_dir, sid, 1), 2_000, header_frames=0)
    assert len(recover_all(db, data_dir, segment_frames=FRAMES)) == 1
    before = manifest(db, sid)
    assert recover_all(db, data_dir, segment_frames=FRAMES) == []
    assert manifest(db, sid) == before


def test_session_without_audio_is_failed(db, data_dir: Path) -> None:
    sid = abandoned_session(db, data_dir)
    [result] = recover_all(db, data_dir)
    assert result.status_after == "failed"
    session = repo.get_session(db, sid)
    assert session["ended_at"] == session["started_at"]


def test_finished_session_only_gets_missing_rows(db, data_dir: Path) -> None:
    sid = abandoned_session(db, data_dir)
    write_segment(segment(data_dir, sid, 1), FRAMES)
    write_segment(segment(data_dir, sid, 2), 1_000)
    repo.add_audio_segment(db, sid, 1, "audio/000001.wav", 0, 500, 0)
    repo.finish_session(db, sid, "recorded", "2026-10-29T10:00:00.000+00:00")
    [result] = recover_all(db, data_dir, segment_frames=FRAMES)
    assert (result.status_after, result.added, result.repaired) == ("recorded", [2], [])
    assert repo.get_session(db, sid)["ended_at"] == "2026-10-29T10:00:00.000+00:00"


def test_problems_are_reported_and_the_rest_recovered(db, data_dir: Path) -> None:
    sid = abandoned_session(db, data_dir)
    write_segment(segment(data_dir, sid, 1), 5_000)  # short, but not the last one
    segment(data_dir, sid, 2).write_bytes(b"not a segment at all, just some text bytes")
    write_segment(segment(data_dir, sid, 4), FRAMES, header_frames=0)
    (segment(data_dir, sid, 1).parent / "notes.txt").write_text("ignored")
    [result] = recover_all(db, data_dir, segment_frames=FRAMES)
    assert result.added == [1, 4]
    problems = "\n".join(result.problems)
    assert "segment 1 holds 5000 samples" in problems
    assert "segment 2 left out" in problems
    assert "missing from the sequence: [3]" in problems
    assert [row[2:4] for row in manifest(db, sid)] == [(0, 312), (1_500, 2_000)]
    assert result.status_after == "interrupted"


def test_missing_session_folder(db, data_dir: Path) -> None:
    sid = abandoned_session(db, data_dir)
    for path in sorted(session_dir(data_dir, sid).rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    session_dir(data_dir, sid).rmdir()
    [result] = recover_all(db, data_dir)
    assert result.status_after == "failed"
    assert result.problems == ["session folder is missing"]


@windows_only
def test_sessions_still_being_recorded_are_left_alone(db, data_dir: Path) -> None:
    # Owner alive (the CLI that started it).
    sid_owned, owner_lock = new_session(db, data_dir, "owned")
    # Owner gone, capture process alive (it holds the capture lock).
    sid_capturing = abandoned_session(db, data_dir, "capturing")
    sink = SessionSink(str(data_dir), sid_capturing, segment_frames=FRAMES)
    try:
        assert recover_all(db, data_dir) == []
        for sid in (sid_owned, sid_capturing):
            assert repo.get_session(db, sid)["status"] == "recording"
        # The owner itself may recover its session once its capture process is gone.
        result = recover_session(db, data_dir, sid_owned, owner_lock=owner_lock)
        assert result is not None and result.status_after == "failed"
    finally:
        sink.close("requested")
        owner_lock.release()


@windows_only
def test_killed_capture_process_is_recovered_by_its_owner(db, data_dir: Path) -> None:
    sid, owner_lock = new_session(db, data_dir, "killed")
    config = CaptureConfig(
        sink_factory=functools.partial(SessionSink, str(data_dir), sid, segment_frames=FRAMES),
        source_factory=functools.partial(FakeSource, frames_per_drain=2_400),
        default_output_factory=ScriptedDefaultOutput,
    )
    handle = RecorderHandle(config)
    handle.start()
    try:
        wait_for(handle, "started")
        started = time.monotonic()
        time.sleep(1.3)
        # While the capture process lives, nobody recovers the session, not even its owner.
        assert recover_session(db, data_dir, sid, owner_lock=owner_lock) is None
        assert handle.kill() != 0
        elapsed = time.monotonic() - started
        # Other processes still see the owner alive...
        assert recover_all(db, data_dir, segment_frames=FRAMES) == []
        # ...the owner recovers it.
        result = recover_session(
            db, data_dir, sid, owner_lock=owner_lock, segment_frames=FRAMES
        )
    finally:
        if handle.is_alive():  # a failed assertion must not leave pytest waiting for it
            handle.kill()
        owner_lock.release()

    assert result is not None and result.status_after == "interrupted"
    assert result.added and result.problems == []
    rows = repo.list_audio_segments(db, sid)
    assert rows[0]["start_ms"] == 0
    for prev, cur in zip(rows, rows[1:]):
        assert cur["start_ms"] == prev["end_ms"] == prev["start_ms"] + 500
    assert rows[-1]["end_ms"] / 1000 == pytest.approx(elapsed, abs=0.4)
    for row in rows:
        with wave.open(str(session_dir(data_dir, sid) / row["path"])) as w:
            assert w.getnframes() * 1000 // 16_000 == row["end_ms"] - row["start_ms"]
