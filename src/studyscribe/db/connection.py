"""SQLite connections: WAL, foreign keys, explicit transactions."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from studyscribe.db.migrations import migrate


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the database in autocommit mode; group writes with `transaction()`.

    Does not migrate: the capture process uses this on a database the parent already migrated.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    # Safe with WAL: a crash cannot corrupt the database; a power loss may drop the last commits.
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def open_database(path: Path | str) -> sqlite3.Connection:
    """Connect and apply pending migrations."""
    conn = connect(path)
    try:
        migrate(conn)
    except BaseException:
        conn.close()
        raise
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT, or ROLLBACK if the block raises."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
