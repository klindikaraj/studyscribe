"""Repair sessions left behind by a crash, a killed process or a power cut.

A session stays in 'recording' when its capture process died without closing it. For every
such session that nobody is recording any more (both session locks free), recovery:
- reads the modification time of the last segment (the end of the recording);
- repairs the header of each segment missing from the manifest, from the file size: while
  recording the header is patched only every second;
- adds those segments to the manifest: the one open at the crash, plus any whose row could
  not be written because of a database error;
- sets the status to 'interrupted' ('failed' if there is no audio at all) and `ended_at`.
Finished sessions ('recorded', 'interrupted') only get their missing manifest rows.

Known loss: the gap_ms of a segment added here is 0, since it is known only to the process
that wrote the segment.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from studyscribe.audio.wav import ForeignWavError, repair_wav_header
from studyscribe.config import AUDIO_SUBDIR, SEGMENT_FRAMES, session_dir
from studyscribe.db import repo
from studyscribe.db.connection import transaction
from studyscribe.session import CAPTURE_LOCK_BYTE, OWNER_LOCK_BYTE, session_lock
from studyscribe.system.locks import ByteLock

RECOVERABLE = ("recording", "recorded", "interrupted")
_SEGMENT_NAME = re.compile(r"(\d{6})\.wav")


@dataclass
class Recovered:
    session_id: str
    status_before: str
    status_after: str
    added: list[int] = field(default_factory=list)  # segments added to the manifest
    repaired: list[int] = field(default_factory=list)  # segments whose file was fixed
    duration_ms: int = 0  # end of the last segment on the timeline
    problems: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        # Problems alone do not count: they would be reported again on every run.
        return bool(self.added or self.repaired or self.status_after != self.status_before)


def recover_all(
    conn: sqlite3.Connection, data_dir: Path, segment_frames: int = SEGMENT_FRAMES
) -> list[Recovered]:
    """Recover every session that needs it; return what was done (sessions left as they were
    are not listed). Sessions still being recorded are skipped."""
    placeholders = ", ".join("?" * len(RECOVERABLE))
    ids = [
        row["id"]
        for row in conn.execute(
            f"SELECT id FROM sessions WHERE status IN ({placeholders}) ORDER BY id", RECOVERABLE
        )
    ]
    results = []
    for session_id in ids:
        result = recover_session(conn, data_dir, session_id, segment_frames=segment_frames)
        if result is not None and result.changed:
            results.append(result)
    return results


def recover_session(
    conn: sqlite3.Connection,
    data_dir: Path,
    session_id: str,
    owner_lock: ByteLock | None = None,
    segment_frames: int = SEGMENT_FRAMES,
) -> Recovered | None:
    """Recover one session; None if it is still being recorded or not recoverable.

    The process that started the recording passes its `owner_lock` when its capture process
    died: the session is then recovered although that lock is (by itself) still held.
    """
    session = repo.get_session(conn, session_id)
    if session is None or session["status"] not in RECOVERABLE:
        return None
    folder = session_dir(data_dir, session_id)
    if not folder.is_dir():
        return _without_folder(conn, session)
    bytes_to_lock = [CAPTURE_LOCK_BYTE]
    if owner_lock is None or not owner_lock.held:
        bytes_to_lock.insert(0, OWNER_LOCK_BYTE)
    locks: list[ByteLock] = []
    try:
        for byte in bytes_to_lock:
            lock = session_lock(data_dir, session_id, byte)
            if not lock.acquire():
                return None  # still being recorded
            locks.append(lock)
        return _recover(conn, folder, session, segment_frames)
    finally:
        for lock in locks:
            lock.release()


def _recover(
    conn: sqlite3.Connection, folder: Path, session: sqlite3.Row, segment_frames: int
) -> Recovered:
    session_id, status = session["id"], session["status"]
    result = Recovered(session_id, status, status)
    rows = {row["seq"]: row for row in repo.list_audio_segments(conn, session_id)}
    files: dict[int, Path] = {}
    audio_dir = folder / AUDIO_SUBDIR
    if audio_dir.is_dir():
        for path in audio_dir.iterdir():
            match = _SEGMENT_NAME.fullmatch(path.name)
            if match:
                files[int(match.group(1))] = path
    # Before any repair, which would change it.
    last_write = files[max(files)].stat().st_mtime if files else None

    for seq in sorted(rows.keys() - files.keys()):
        result.problems.append(f"segment {seq} is in the manifest but its file is missing")
    numbers = sorted(files.keys() | rows.keys())
    if numbers and numbers != list(range(1, numbers[-1] + 1)):
        missing = sorted(set(range(1, numbers[-1] + 1)) - set(numbers))
        result.problems.append(f"segments missing from the sequence: {missing}")

    new_rows = []
    for seq in sorted(files.keys() - rows.keys()):
        try:
            fix = repair_wav_header(files[seq])
        except (ForeignWavError, OSError) as exc:
            result.problems.append(f"segment {seq} left out: {exc}")
            continue
        if fix.changed:
            result.repaired.append(seq)
        # Segment N starts at (N - 1) full segments by construction.
        start = (seq - 1) * segment_frames
        if fix.frames != segment_frames and seq < max(files):
            result.problems.append(
                f"segment {seq} holds {fix.frames} samples instead of {segment_frames}: "
                "the audio after it is shifted"
            )
        new_rows.append(
            (seq, f"{AUDIO_SUBDIR}/{files[seq].name}", _ms(start, fix.sample_rate),
             _ms(start + fix.frames, fix.sample_rate))
        )

    ends = [row["end_ms"] or 0 for row in rows.values()] + [row[3] for row in new_rows]
    result.duration_ms = max(ends, default=0)
    with transaction(conn):
        for seq, path, start_ms, end_ms in new_rows:
            repo.add_audio_segment(conn, session_id, seq, path, start_ms, end_ms, 0)
            result.added.append(seq)
        if status == "recording":
            result.status_after = "interrupted" if result.duration_ms > 0 else "failed"
            ended_at = (
                datetime.fromtimestamp(last_write, timezone.utc).isoformat(timespec="milliseconds")
                if last_write is not None
                else session["started_at"]
            )
            repo.finish_session(conn, session_id, result.status_after, ended_at)
    return result


def _without_folder(conn: sqlite3.Connection, session: sqlite3.Row) -> Recovered:
    result = Recovered(session["id"], session["status"], session["status"])
    result.problems.append("session folder is missing")
    if session["status"] == "recording":
        result.status_after = "failed"
        with transaction(conn):
            repo.finish_session(conn, session["id"], "failed", session["started_at"])
    return result


def _ms(frames: int, sample_rate: int) -> int:
    return frames * 1000 // sample_rate
