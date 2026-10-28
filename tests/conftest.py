from pathlib import Path

import pytest

from studyscribe.config import db_path
from studyscribe.db.connection import open_database
from studyscribe.session import new_session


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def db(data_dir: Path):
    conn = open_database(db_path(data_dir))
    yield conn
    conn.close()


@pytest.fixture
def start_session(db, data_dir: Path):
    """new_session() that returns the ID and releases the owner lock after the test."""
    locks = []

    def start(title: str | None, course: str | None = None) -> str:
        session_id, lock = new_session(db, data_dir, title, course=course)
        locks.append(lock)
        return session_id

    yield start
    for lock in locks:
        lock.release()
