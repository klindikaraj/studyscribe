"""SQLite connections: WAL, foreign keys, explicit transactions."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path

from studyscribe.db.migrations import MIGRATIONS, migrate, schema_version

BACKUP_SUBDIR = "backups"


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


def open_database(path: Path | str,
                  on_backup: Callable[[Path], None] | None = None) -> sqlite3.Connection:
    """Connect and apply pending migrations. A database that already has a schema and needs a
    migration is first copied to `backups/` next to it (`on_backup` is told where)."""
    conn = connect(path)
    try:
        saved = backup_before_migration(conn, Path(path))
        if saved is not None and on_backup is not None:
            on_backup(saved)
        migrate(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def backup_before_migration(conn: sqlite3.Connection, path: Path,
                            now: datetime | None = None) -> Path | None:
    """Copy the database to `<folder>/backups/<name>-v<version>-<time>.db` when it has a schema
    older than this program's (a new, empty database is not copied). SQLite's online backup:
    consistent even with the WAL and other connections open."""
    version = schema_version(conn)
    if version == 0 or version >= len(MIGRATIONS):
        return None
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    dest = path.parent / BACKUP_SUBDIR / f"{path.stem}-v{version}-{stamp}.db"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(dest)) as target:
        conn.backup(target)
    return dest


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
