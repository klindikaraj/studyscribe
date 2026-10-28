"""Queries on courses, sessions and audio segments."""

from __future__ import annotations

import secrets
import sqlite3
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_session_id(now: datetime | None = None) -> str:
    """Readable, sortable ID in local time, e.g. 20261028-142301-3f9a2c."""
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def get_or_create_course(conn: sqlite3.Connection, title: str) -> int:
    title = title.strip()
    row = conn.execute(
        "SELECT id FROM courses WHERE title = ? COLLATE NOCASE", (title,)
    ).fetchone()
    if row is not None:
        return row["id"]
    cursor = conn.execute(
        "INSERT INTO courses (title, created_at) VALUES (?, ?)", (title, utc_now_iso())
    )
    return cursor.lastrowid


def create_session(
    conn: sqlite3.Connection,
    session_id: str,
    title: str | None,
    course_id: int | None,
    started_at: str,
) -> None:
    conn.execute(
        "INSERT INTO sessions (id, course_id, title, status, started_at)"
        " VALUES (?, ?, ?, 'recording', ?)",
        (session_id, course_id, title, started_at),
    )


def mark_session_started(conn: sqlite3.Connection, session_id: str, started_at: str) -> None:
    conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (started_at, session_id))


def finish_session(
    conn: sqlite3.Connection, session_id: str, status: str, ended_at: str
) -> None:
    conn.execute(
        "UPDATE sessions SET status = ?, ended_at = ? WHERE id = ?",
        (status, ended_at, session_id),
    )


def add_audio_segment(
    conn: sqlite3.Connection,
    session_id: str,
    seq: int,
    path: str,
    start_ms: int,
    end_ms: int,
    gap_ms: int,
) -> None:
    conn.execute(
        "INSERT INTO audio_segments (session_id, seq, path, start_ms, end_ms, gap_ms)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, seq, path, start_ms, end_ms, gap_ms),
    )


def get_session(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()


def list_audio_segments(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM audio_segments WHERE session_id = ? ORDER BY seq", (session_id,)
    ).fetchall()
