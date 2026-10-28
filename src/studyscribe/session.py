"""Session lifecycle: create the row and folder (parent), record into them (capture process)."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from studyscribe.audio.segments import ClosedSegment, SegmentWriter
from studyscribe.config import AUDIO_SUBDIR, SEGMENT_FRAMES, db_path, session_dir
from studyscribe.db import repo
from studyscribe.db.connection import connect, transaction
from studyscribe.system.locks import ByteLock

Emit = Callable[..., None]

# Two byte locks in <session>/session.lock tell recovery that a session is still being
# recorded: one held by the process that started it (from before its row exists), one by the
# capture process. Recovery touches a session only when it can take both.
LOCK_FILENAME = "session.lock"
OWNER_LOCK_BYTE = 0
CAPTURE_LOCK_BYTE = 1


def session_lock(data_dir: Path, session_id: str, byte: int) -> ByteLock:
    return ByteLock(session_dir(data_dir, session_id) / LOCK_FILENAME, byte)


class NewSession(NamedTuple):
    id: str
    owner_lock: ByteLock  # release it once the capture process has exited


def new_session(
    conn: sqlite3.Connection, data_dir: Path, title: str | None, course: str | None = None
) -> NewSession:
    """Create the session folder, lock it as its owner, then create its row ('recording').

    The owner lock is taken before the row exists, so recovery in another process can never
    mistake a session that is just starting for an abandoned one. `started_at` is provisional:
    the capture process overwrites it when the timeline starts.
    """
    session_id = repo.new_session_id()
    (session_dir(data_dir, session_id) / AUDIO_SUBDIR).mkdir(parents=True)
    lock = session_lock(data_dir, session_id, OWNER_LOCK_BYTE)
    if not lock.acquire():
        raise RuntimeError(f"session {session_id} is locked by another process")
    try:
        with transaction(conn):
            course_id = repo.get_or_create_course(conn, course) if course else None
            repo.create_session(conn, session_id, title, course_id, repo.utc_now_iso())
    except BaseException:
        lock.release()
        raise
    return NewSession(session_id, lock)


def final_status(reason: str, frames: int) -> str:
    if reason in ("requested", "silence_limit"):
        return "recorded"
    if reason == "error" and frames == 0:
        return "failed"
    return "interrupted"


class SessionSink:
    """Capture sink of a real session: 60 s WAV segments plus manifest rows and status in SQLite.

    Built inside the capture process, which holds the session's capture lock until close().
    A database error never stops the recording: it is reported as a warning, the audio is on
    disk anyway and recovery registers it later.
    """

    def __init__(
        self, data_dir: str, session_id: str, segment_frames: int = SEGMENT_FRAMES
    ) -> None:
        self._session_id = session_id
        self._dir = session_dir(Path(data_dir), session_id)
        self._emit: Emit = _ignore
        self._lock = session_lock(Path(data_dir), session_id, CAPTURE_LOCK_BYTE)
        self._lock_error: str | None = None
        try:
            if not self._lock.acquire():
                raise RuntimeError(f"session {session_id} is already being recorded")
        except OSError as exc:  # record anyway: the audio matters more than the lock
            self._lock_error = str(exc)
        self._conn: sqlite3.Connection | None = None
        self._connect_error: str | None = None
        try:
            self._conn = connect(db_path(Path(data_dir)))
        except sqlite3.Error as exc:
            self._connect_error = str(exc)
        self._segments = SegmentWriter(
            self._dir / AUDIO_SUBDIR, on_close=self._register, segment_frames=segment_frames
        )

    def start(self, emit: Emit) -> None:
        self._emit = emit
        if self._lock_error is not None:
            emit(
                "warning",
                message=f"cannot lock the session, recovery may not see it is live: "
                f"{self._lock_error}",
            )
        if self._connect_error is not None:
            emit(
                "warning",
                message=f"database unavailable, recording audio only: {self._connect_error}",
            )
        self._db("record the start time", repo.mark_session_started, repo.utc_now_iso())

    def write(self, pcm: np.ndarray, gap: bool = False) -> None:
        self._segments.write(pcm, gap=gap)

    def close(self, reason: str) -> None:
        try:
            self._segments.close()
        finally:
            status = final_status(reason, self._segments.total_frames)
            self._db("update the session status", repo.finish_session, status, repo.utc_now_iso())
            if self._conn is not None:
                self._conn.close()
            self._lock.release()

    def _register(self, segment: ClosedSegment) -> None:
        path = segment.path.relative_to(self._dir).as_posix()
        self._db(
            f"register segment {segment.seq}",
            repo.add_audio_segment,
            segment.seq,
            path,
            segment.start_ms,
            segment.end_ms,
            segment.gap_ms,
        )
        self._emit(
            "segment",
            seq=segment.seq,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            gap_ms=segment.gap_ms,
        )

    def _db(self, what: str, query: Callable[..., None], *args: Any) -> None:
        if self._conn is None:
            return
        try:
            query(self._conn, self._session_id, *args)
        except sqlite3.Error as exc:
            self._emit("warning", message=f"database: could not {what}: {exc}")


def _ignore(*args: Any, **kwargs: Any) -> None:
    pass
