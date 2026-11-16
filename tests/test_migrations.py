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
    assert schema_version(db) == len(MIGRATIONS) == 3
    assert {"courses", "sessions", "audio_segments", "transcription_runs",
            "transcript_segments", "settings", "templates", "generated_notes",
            "llm_calls"} <= tables(db)


def test_connection_settings(db) -> None:
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_migrate_is_idempotent(db) -> None:
    assert migrate(db) == 3
    assert migrate(db) == 3


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


def v2_database(path: Path) -> None:
    conn = connect(path)
    migrate(conn, MIGRATIONS[:2])
    conn.execute("INSERT INTO sessions (id, status, started_at) VALUES ('s1', 'recorded', 'x')")
    conn.close()


def test_a_database_with_an_older_schema_is_backed_up_before_migrating(tmp_path: Path) -> None:
    path = tmp_path / "app.db"
    v2_database(path)
    saved = []
    conn = open_database(path, on_backup=saved.append)
    assert schema_version(conn) == 3
    conn.close()
    assert len(saved) == 1 and saved[0].parent == tmp_path / "backups"
    assert saved[0].name.startswith("app-v2-") and saved[0].suffix == ".db"
    backup = connect(saved[0])
    assert schema_version(backup) == 2
    assert backup.execute("SELECT id FROM sessions").fetchone()["id"] == "s1"
    assert "generated_notes" not in tables(backup)
    backup.close()
    # already migrated: no new backup
    open_database(path, on_backup=saved.append).close()
    assert len(saved) == 1


def test_a_new_database_is_not_backed_up(tmp_path: Path) -> None:
    saved = []
    open_database(tmp_path / "app.db", on_backup=saved.append).close()
    assert saved == [] and not (tmp_path / "backups").exists()


def test_notes_tables(db) -> None:
    db.execute("INSERT INTO sessions (id, status, started_at) VALUES ('s1', 'recorded', 'x')")
    row = ("s1", "running", "ollama", "m", "studio", "{}", "it", "transcript", "v5.1", "h", "{}",
           "[]", "now")
    insert = ("INSERT INTO generated_notes (session_id, status, provider, model, template_id,"
              " template_json, language, language_source, prompt_version, prompt_hash,"
              " options_json, blocks_json, started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)")
    notes_id = db.execute(insert, row).lastrowid
    with pytest.raises(sqlite3.IntegrityError):  # one open generation per session
        db.execute(insert, row)
    db.execute("UPDATE generated_notes SET status = 'done' WHERE id = ?", (notes_id,))
    db.execute(insert, row)  # a new one beside the done one
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO llm_calls (notes_id, stage, block, attempt, status, created_at)"
                   " VALUES (?, 'summary', 1, 1, 'ok', 'now')", (notes_id,))
    db.execute("INSERT INTO llm_calls (notes_id, stage, block, attempt, status, created_at)"
               " VALUES (?, 'map', 1, 1, 'ok', 'now')", (notes_id,))
    db.execute("DELETE FROM generated_notes WHERE id = ?", (notes_id,))
    assert db.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 0  # cascade
