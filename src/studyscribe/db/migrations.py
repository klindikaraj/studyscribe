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

MIGRATIONS: list[list[str]] = [V1]


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
