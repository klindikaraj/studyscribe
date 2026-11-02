"""Schema migrations, tracked with PRAGMA user_version.

MIGRATIONS[i] takes the database from version i to i + 1. Each one runs in its own transaction.
Never edit a migration that has already run somewhere: add a new one.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

V1 = [
    """
    CREATE TABLE courses (
      id          INTEGER PRIMARY KEY,
      title       TEXT NOT NULL,
      title_key   TEXT NOT NULL,
      platform    TEXT,
      url         TEXT,
      created_at  TEXT NOT NULL
    )
    """,
    # title_key is computed in Python (repo.course_key: Unicode NFC + case folding), so
    # `--course` reuses an existing course whatever the case or accents' encoding.
    "CREATE UNIQUE INDEX idx_courses_title_key ON courses(title_key)",
    """
    CREATE TABLE sessions (
      id          TEXT PRIMARY KEY,
      course_id   INTEGER REFERENCES courses(id),
      title       TEXT,
      source_app  TEXT,
      source_url  TEXT,
      status      TEXT NOT NULL CHECK (status IN
                    ('recording', 'recorded', 'interrupted', 'transcribing', 'ready', 'failed')),
      started_at  TEXT NOT NULL,
      ended_at    TEXT,
      sample_rate INTEGER NOT NULL DEFAULT 16000,
      asr_engine  TEXT,
      asr_model   TEXT,
      language    TEXT
    )
    """,
    """
    CREATE TABLE audio_segments (
      session_id  TEXT NOT NULL REFERENCES sessions(id),
      seq         INTEGER NOT NULL,
      path        TEXT NOT NULL,
      start_ms    INTEGER NOT NULL,
      end_ms      INTEGER,
      gap_ms      INTEGER NOT NULL DEFAULT 0,
      transcribed INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY (session_id, seq)
    )
    """,
]

# Transcription (step 2). A run is one transcription of a session; its segments become visible
# to readers only when it is 'done', in the transaction that deletes the previous done run, so a
# re-transcription replaces the old text atomically. sessions.status is not involved.
V2 = [
    """
    CREATE TABLE transcription_runs (
      id              INTEGER PRIMARY KEY,
      session_id      TEXT NOT NULL REFERENCES sessions(id),
      pass            TEXT NOT NULL DEFAULT 'final' CHECK (pass IN ('live', 'final')),
      status          TEXT NOT NULL CHECK (status IN ('running', 'interrupted', 'done')),
      engine          TEXT NOT NULL,
      model           TEXT NOT NULL,
      compute_type    TEXT NOT NULL,
      language        TEXT,
      language_source TEXT CHECK (language_source IN ('forced', 'detected')),
      language_probs  TEXT,
      options_json    TEXT NOT NULL,
      audio_ms        INTEGER NOT NULL,
      done_ms         INTEGER NOT NULL DEFAULT 0,
      started_at      TEXT NOT NULL,
      finished_at     TEXT,
      processing_s    REAL NOT NULL DEFAULT 0,
      error           TEXT
    )
    """,
    "CREATE UNIQUE INDEX idx_runs_done ON transcription_runs(session_id, pass)"
    " WHERE status = 'done'",
    "CREATE UNIQUE INDEX idx_runs_open ON transcription_runs(session_id, pass)"
    " WHERE status <> 'done'",
    """
    CREATE TABLE transcript_segments (
      id                INTEGER PRIMARY KEY,
      run_id            INTEGER NOT NULL REFERENCES transcription_runs(id) ON DELETE CASCADE,
      start_ms          INTEGER NOT NULL,
      end_ms            INTEGER NOT NULL,
      text              TEXT NOT NULL,
      words_json        TEXT,
      avg_logprob       REAL,
      no_speech_prob    REAL,
      compression_ratio REAL,
      dropped           TEXT
    )
    """,
    "CREATE INDEX idx_transcript_run_time ON transcript_segments(run_id, start_ms)",
]

MIGRATIONS: list[list[str]] = [V1, V2]


class SchemaTooNewError(RuntimeError):
    """The database was written by a newer version of StudyScribe."""


def schema_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def migrate(conn: sqlite3.Connection, migrations: Sequence[Sequence[str]] = MIGRATIONS) -> int:
    """Apply pending migrations and return the resulting version.

    `conn` must be in autocommit mode (isolation_level=None). The version is re-read inside each
    write transaction, so two processes migrating at once cannot apply a step twice.
    """
    while True:
        conn.execute("BEGIN IMMEDIATE")
        try:
            version = schema_version(conn)
            if version > len(migrations):
                raise SchemaTooNewError(
                    f"database schema version {version} is newer than this program "
                    f"({len(migrations)})"
                )
            if version == len(migrations):
                conn.execute("COMMIT")
                return version
            for statement in migrations[version]:
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {version + 1}")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
