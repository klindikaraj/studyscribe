import sqlite3
from pathlib import Path

import pytest

from studyscribe.db.connection import connect, open_database
from studyscribe.db.migrations import MIGRATIONS, SchemaTooNewError, migrate, schema_version


def tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {r["name"] for r in rows}


@pytest.fixture
def db(tmp_path: Path):
    conn = open_database(tmp_path / "app.db")
    yield conn
    conn.close()


def test_fresh_database_reaches_latest_version(db) -> None:
    assert schema_version(db) == len(MIGRATIONS) == 1
    assert {"courses", "sessions", "audio_segments"} <= tables(db)


def test_connection_settings(db) -> None:
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_migrate_is_idempotent(db) -> None:
    assert migrate(db) == 1
    assert migrate(db) == 1


def test_failed_migration_rolls_back_entirely(tmp_path: Path) -> None:
    conn = connect(tmp_path / "app.db")
    broken = [
        MIGRATIONS[0],
        ["CREATE TABLE half_done (id INTEGER)", "THIS IS NOT SQL"],
    ]
    with pytest.raises(sqlite3.OperationalError):
        migrate(conn, broken)
    assert schema_version(conn) == 1  # v1 committed, v2 not
    assert "half_done" not in tables(conn)
    assert not conn.in_transaction
    conn.close()


def test_newer_database_is_refused(tmp_path: Path) -> None:
    conn = connect(tmp_path / "app.db")
    conn.execute("PRAGMA user_version = 99")
    with pytest.raises(SchemaTooNewError):
        migrate(conn)
    conn.close()


def test_session_status_is_checked(db) -> None:
    insert = "INSERT INTO sessions (id, status, started_at) VALUES (?, ?, '2026-10-28')"
    for status in ("recording", "recorded", "interrupted", "transcribing", "ready", "failed"):
        db.execute(insert, (f"id-{status}", status))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(insert, ("id-bad", "paused"))


def test_course_title_key_is_unique_and_required(db) -> None:
    insert = "INSERT INTO courses (title, title_key, created_at) VALUES (?, ?, 'x')"
    db.execute(insert, ("Machine Learning", "machine learning"))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(insert, ("MACHINE LEARNING", "machine learning"))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(insert, ("No key", None))


def test_foreign_keys_are_enforced(db) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO audio_segments (session_id, seq, path, start_ms) VALUES ('nope', 1, 'a', 0)"
        )
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO sessions (id, course_id, status, started_at)"
            " VALUES ('s', 42, 'recording', 'x')"
        )
