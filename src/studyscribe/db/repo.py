"""Queries on courses, sessions, audio segments, transcripts and notes."""

from __future__ import annotations

import json
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


# --- notes: settings, templates, generations, LLM calls ------------------------------------------


def get_setting(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return default if row is None else row["value"]


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO settings (key, value) VALUES (?, ?)"
                 " ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


def list_settings(conn: sqlite3.Connection) -> dict[str, str]:
    return {row["key"]: row["value"]
            for row in conn.execute("SELECT key, value FROM settings ORDER BY key")}


def upsert_template(conn: sqlite3.Connection, template_id: str, title_json: str,
                    description: str, template_json: str, builtin: bool) -> None:
    conn.execute(
        "INSERT INTO templates (id, title_json, description, template_json, builtin, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET title_json ="
        " excluded.title_json, description = excluded.description, template_json ="
        " excluded.template_json, builtin = excluded.builtin, updated_at = excluded.updated_at",
        (template_id, title_json, description, template_json, int(builtin), utc_now_iso()))


def get_template(conn: sqlite3.Connection, template_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM templates WHERE id = ?", (template_id,)).fetchone()


def list_templates(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM templates ORDER BY builtin DESC, id").fetchall()


def create_notes(conn: sqlite3.Connection, *, session_id: str, transcript_run_id: int,
                 provider: str, model: str, template_id: str, template_json: str,
                 language: str, language_source: str, prompt_version: str, prompt_hash: str,
                 options_json: str, blocks_json: str) -> int:
    cursor = conn.execute(
        "INSERT INTO generated_notes (session_id, transcript_run_id, status, provider, model,"
        " template_id, template_json, language, language_source, prompt_version, prompt_hash,"
        " options_json, blocks_json, started_at)"
        " VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, transcript_run_id, provider, model, template_id, template_json, language,
         language_source, prompt_version, prompt_hash, options_json, blocks_json,
         utc_now_iso()))
    return cursor.lastrowid


def get_notes(conn: sqlite3.Connection, notes_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM generated_notes WHERE id = ?", (notes_id,)).fetchone()


def get_open_notes(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    """The unfinished ('running' or 'interrupted') generation of a session, if any."""
    return conn.execute(
        "SELECT * FROM generated_notes WHERE session_id = ? AND status IN"
        " ('running', 'interrupted')", (session_id,)).fetchone()


def get_latest_done_notes(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM generated_notes WHERE session_id = ? AND status = 'done'"
        " ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()


def list_done_notes(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    """The completed generations of a session, oldest first. A generation's version is its
    position here (from 1): a completed generation never changes, so it never moves."""
    return conn.execute(
        "SELECT * FROM generated_notes WHERE session_id = ? AND status = 'done' ORDER BY id",
        (session_id,)).fetchall()


def list_notes_by_status(conn: sqlite3.Connection, status: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM generated_notes WHERE status = ? ORDER BY id",
                        (status,)).fetchall()


def set_notes_status(conn: sqlite3.Connection, notes_id: int, status: str,
                     error: str | None = None) -> None:
    finished = utc_now_iso() if status in ("done", "failed") else None
    conn.execute("UPDATE generated_notes SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                 (status, error, finished, notes_id))


def set_notes_options(conn: sqlite3.Connection, notes_id: int, options_json: str) -> None:
    conn.execute("UPDATE generated_notes SET options_json = ? WHERE id = ?",
                 (options_json, notes_id))


def finish_notes(conn: sqlite3.Connection, notes_id: int, *, content_json: str,
                 uncertain_json: str, stats_json: str, content_md: str | None = None) -> None:
    conn.execute(
        "UPDATE generated_notes SET status = 'done', error = NULL, finished_at = ?,"
        " content_json = ?, content_md = ?, uncertain_json = ?, stats_json = ? WHERE id = ?",
        (utc_now_iso(), content_json, content_md, uncertain_json, stats_json, notes_id))


def add_llm_call(conn: sqlite3.Connection, notes_id: int, *, stage: str, block: int,
                 attempt: int, status: str, model_version: str | None = None,
                 request_id: str | None = None, input_tokens: int | None = None,
                 output_tokens: int | None = None, thinking_tokens: int | None = None,
                 cached_tokens: int | None = None, cost_usd: float | None = None,
                 list_cost_usd: float | None = None, duration_s: float | None = None,
                 load_s: float | None = None, prefill_s: float | None = None,
                 generate_s: float | None = None, output_json: str | None = None,
                 error: str | None = None) -> int:
    cursor = conn.execute(
        "INSERT INTO llm_calls (notes_id, stage, block, attempt, status, model_version,"
        " request_id, input_tokens, output_tokens, thinking_tokens, cached_tokens, cost_usd,"
        " list_cost_usd, duration_s, load_s, prefill_s, generate_s, output_json, error,"
        " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (notes_id, stage, block, attempt, status, model_version, request_id, input_tokens,
         output_tokens, thinking_tokens, cached_tokens, cost_usd, list_cost_usd, duration_s,
         load_s, prefill_s, generate_s, output_json, error, utc_now_iso()))
    return cursor.lastrowid


def list_llm_calls(conn: sqlite3.Connection, notes_id: int) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM llm_calls WHERE notes_id = ? ORDER BY id",
                        (notes_id,)).fetchall()


def accepted_map_answers(conn: sqlite3.Connection, notes_id: int) -> dict[int, str]:
    """Block index -> the raw answer of its last accepted map call (to resume)."""
    rows = conn.execute(
        "SELECT block, output_json FROM llm_calls WHERE notes_id = ? AND stage = 'map'"
        " AND status = 'ok' AND output_json IS NOT NULL ORDER BY id", (notes_id,))
    return {row["block"]: row["output_json"] for row in rows}


def model_timings(conn: sqlite3.Connection, provider: str, model: str,
                  limit: int = 50) -> list[sqlite3.Row]:
    """Recent successful calls of a model with their timings (speed estimates)."""
    return conn.execute(
        "SELECT c.input_tokens, c.cached_tokens, c.output_tokens, c.prefill_s, c.generate_s"
        " FROM llm_calls c JOIN generated_notes g ON g.id = c.notes_id"
        " WHERE g.provider = ? AND g.model = ? AND c.status = 'ok' AND c.generate_s > 0"
        " ORDER BY c.id DESC LIMIT ?", (provider, model, limit)).fetchall()


def notes_states(conn: sqlite3.Connection) -> dict[str, str]:
    """Session ID -> short notes state for listings: an open generation's status with its
    accepted map blocks ("interrupted 5/15"), else the latest done generation's model, else
    'failed' when every generation failed."""
    open_, done, failed = {}, {}, set()
    for row in conn.execute(
        "SELECT g.session_id, g.status, g.model, g.blocks_json,"
        " (SELECT COUNT(DISTINCT c.block) FROM llm_calls c WHERE c.notes_id = g.id"
        "  AND c.stage = 'map' AND c.status = 'ok') AS maps"
        " FROM generated_notes g ORDER BY g.id"
    ):
        sid = row["session_id"]
        if row["status"] in ("running", "interrupted"):
            blocks = len(json.loads(row["blocks_json"] or "[]"))
            open_[sid] = f"{row['status']} {row['maps']}/{blocks}"
        elif row["status"] == "done":
            done[sid] = f"done ({row['model']})"
        else:
            failed.add(sid)
    return {sid: open_.get(sid) or done.get(sid) or "failed"
            for sid in {*open_, *done, *failed}}
