"""Session lifecycle: create the row and folder (parent), record into them (capture process)."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from studyscribe.audio.segments import ClosedSegment, SegmentWriter
from studyscribe.config import AUDIO_SUBDIR, SEGMENT_FRAMES, db_path, session_dir
from studyscribe.db import repo
from studyscribe.db.connection import connect, transaction

Emit = Callable[..., None]


def new_session(
    conn: sqlite3.Connection, data_dir: Path, title: str | None, course: str | None = None
) -> str:
    """Create the session folder and its row in 'recording' state; return the session ID.

    `started_at` is provisional: the capture process overwrites it when the timeline starts.
    """
    session_id = repo.new_session_id()
    (session_dir(data_dir, session_id) / AUDIO_SUBDIR).mkdir(parents=True)
    with transaction(conn):
        course_id = repo.get_or_create_course(conn, course) if course else None
        repo.create_session(conn, session_id, title, course_id, repo.utc_now_iso())
    return session_id


def final_status(reason: str, frames: int) -> str:
    if reason in ("requested", "silence_limit"):
        return "recorded"
    if reason == "error" and frames == 0:
        return "failed"
    return "interrupted"


class SessionSink:
    """Capture sink of a real session: 60 s WAV segments plus manifest rows and status in SQLite.

    Built inside the capture process. A database error never stops the recording: it is
    reported as a warning, the audio is on disk anyway and recovery registers it later.
    """

    def __init__(
        self, data_dir: str, session_id: str, segment_frames: int = SEGMENT_FRAMES
    ) -> None:
        self._session_id = session_id
        self._dir = session_dir(Path(data_dir), session_id)
        self._emit: Emit = _ignore
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
