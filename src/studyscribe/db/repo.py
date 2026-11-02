"""Queries on courses, sessions and audio segments."""

from __future__ import annotations

import secrets
import sqlite3
import unicodedata
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_session_id(now: datetime | None = None) -> str:
    """Readable, sortable ID in local time, e.g. 20261028-142301-3f9a2c."""
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def course_key(title: str) -> str:
    """Key that makes course titles unique: "Èlite", "èlite" and "Èlite" typed with a combining
    accent are one course.

    NFC first, so composed and decomposed accents match; then case folding (which also maps
    "ß" to "ss"); then NFC again, because case folding can leave decomposed characters.
    """
    folded = unicodedata.normalize("NFC", title.strip()).casefold()
    return unicodedata.normalize("NFC", folded)


def get_or_create_course(conn: sqlite3.Connection, title: str) -> int:
    title = unicodedata.normalize("NFC", title.strip())
    key = course_key(title)
    row = conn.execute("SELECT id FROM courses WHERE title_key = ?", (key,)).fetchone()
    if row is not None:
        return row["id"]
    cursor = conn.execute(
        "INSERT INTO courses (title, title_key, created_at) VALUES (?, ?, ?)",
        (title, key, utc_now_iso()),
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


_SUMMARY = """
    SELECT s.id, s.title, s.status, s.started_at, s.ended_at, c.title AS course,
           COUNT(a.seq) AS segments,
           COALESCE(MAX(a.end_ms), 0) AS duration_ms,
           COALESCE(SUM(a.gap_ms), 0) AS gap_ms
    FROM sessions s
    LEFT JOIN courses c ON c.id = s.course_id
    LEFT JOIN audio_segments a ON a.session_id = s.id
"""


def list_session_summaries(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every session with its course title and, from the manifest, segments/duration/gap."""
    return conn.execute(_SUMMARY + " GROUP BY s.id ORDER BY s.id").fetchall()


def get_session_summary(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    return conn.execute(
        _SUMMARY + " WHERE s.id = ? GROUP BY s.id", (session_id,)
    ).fetchone()


def list_audio_segments(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM audio_segments WHERE session_id = ? ORDER BY seq", (session_id,)
    ).fetchall()


# --- transcription runs and segments -------------------------------------------------------------


def create_run(
    conn: sqlite3.Connection,
    session_id: str,
    engine: str,
    model: str,
    compute_type: str,
    options_json: str,
    audio_ms: int,
    pass_: str = "final",
) -> int:
    cursor = conn.execute(
        "INSERT INTO transcription_runs (session_id, pass, status, engine, model, compute_type,"
        " options_json, audio_ms, started_at) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)",
        (session_id, pass_, engine, model, compute_type, options_json, audio_ms, utc_now_iso()),
    )
    return cursor.lastrowid


def get_run(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM transcription_runs WHERE id = ?", (run_id,)).fetchone()


def get_open_run(
    conn: sqlite3.Connection, session_id: str, pass_: str = "final"
) -> sqlite3.Row | None:
    """The unfinished ('running' or 'interrupted') run of a session, if any."""
    return conn.execute(
        "SELECT * FROM transcription_runs WHERE session_id = ? AND pass = ? AND status <> 'done'",
        (session_id, pass_),
    ).fetchone()


def get_done_run(
    conn: sqlite3.Connection, session_id: str, pass_: str = "final"
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM transcription_runs WHERE session_id = ? AND pass = ? AND status = 'done'",
        (session_id, pass_),
    ).fetchone()


def set_run_status(
    conn: sqlite3.Connection, run_id: int, status: str, error: str | None = None
) -> None:
    conn.execute(
        "UPDATE transcription_runs SET status = ?, error = ? WHERE id = ?", (status, error, run_id)
    )


def set_run_language(
    conn: sqlite3.Connection,
    run_id: int,
    language: str | None,
    source: str | None,
    probs_json: str | None,
) -> None:
    conn.execute(
        "UPDATE transcription_runs SET language = ?, language_source = ?, language_probs = ?"
        " WHERE id = ?",
        (language, source, probs_json, run_id),
    )


def delete_run(conn: sqlite3.Connection, run_id: int) -> None:
    """Delete a run and (cascade) its segments."""
    conn.execute("DELETE FROM transcription_runs WHERE id = ?", (run_id,))


def add_transcript_window(
    conn: sqlite3.Connection,
    run_id: int,
    segments: list[tuple],
    done_ms: int,
    processing_s: float,
) -> None:
    """Insert a window's segments and advance the run. `segments`: (start_ms, end_ms, text,
    words_json, avg_logprob, no_speech_prob, compression_ratio, dropped). Call inside a
    transaction, so the segments and the progress are committed together."""
    conn.executemany(
        "INSERT INTO transcript_segments (run_id, start_ms, end_ms, text, words_json,"
        " avg_logprob, no_speech_prob, compression_ratio, dropped)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(run_id, *segment) for segment in segments],
    )
    conn.execute(
        "UPDATE transcription_runs SET done_ms = ?, processing_s = processing_s + ? WHERE id = ?",
        (done_ms, processing_s, run_id),
    )


def finish_run(conn: sqlite3.Connection, run_id: int) -> None:
    """Make `run_id` the session's transcript, replacing the previous one. Call inside a
    transaction: readers see the old transcript until the commit, then the new one."""
    run = get_run(conn, run_id)
    conn.execute(
        "DELETE FROM transcription_runs WHERE session_id = ? AND pass = ? AND status = 'done'",
        (run["session_id"], run["pass"]),
    )
    conn.execute(
        "UPDATE transcription_runs SET status = 'done', error = NULL, finished_at = ?,"
        " done_ms = audio_ms WHERE id = ?",
        (utc_now_iso(), run_id),
    )
    conn.execute(
        "UPDATE sessions SET asr_engine = ?, asr_model = ?, language = ? WHERE id = ?",
        (run["engine"], run["model"], run["language"], run["session_id"]),
    )


def list_run_segments(
    conn: sqlite3.Connection, run_id: int, include_dropped: bool = True
) -> list[sqlite3.Row]:
    where = "" if include_dropped else " AND dropped IS NULL"
    return conn.execute(
        f"SELECT * FROM transcript_segments WHERE run_id = ?{where} ORDER BY start_ms, id",
        (run_id,),
    ).fetchall()


def list_transcript(
    conn: sqlite3.Connection, session_id: str, include_dropped: bool = False
) -> list[sqlite3.Row]:
    """Segments of the session's current transcript (its done final run); kept ones only
    unless `include_dropped`. Empty if the session has no transcript."""
    run = get_done_run(conn, session_id)
    return [] if run is None else list_run_segments(conn, run["id"], include_dropped)


def list_runs_by_status(conn: sqlite3.Connection, status: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM transcription_runs WHERE status = ? ORDER BY id", (status,)
    ).fetchall()


def transcript_states(conn: sqlite3.Connection) -> dict[str, str]:
    """Session ID -> short transcript state for listings: the language of the done run ('no
    speech' without one), or the open run's status and progress, which take precedence."""
    states = {}
    for row in conn.execute(
        "SELECT session_id, status, language, done_ms, audio_ms FROM transcription_runs"
        " WHERE pass = 'final' ORDER BY status = 'done' DESC, id"
    ):
        if row["status"] == "done":
            states[row["session_id"]] = row["language"] or "no speech"
        else:
            percent = 100 * row["done_ms"] // max(row["audio_ms"], 1)
            states[row["session_id"]] = f"{row['status']} {percent}%"
    return states
