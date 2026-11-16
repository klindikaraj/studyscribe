import json
from pathlib import Path

import pytest

from studyscribe.config import db_path
from studyscribe.db import repo
from studyscribe.db.connection import open_database, transaction
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


@pytest.fixture
def transcribed(db, start_session):
    """A session with a done transcript: one 9.5 s line every 10 s for `minutes`."""
    def make(minutes=24, language="it", title="Lezione"):
        sid = start_session(title, course="PyTorch")
        with transaction(db):
            run_id = repo.create_run(db, sid, "fake", "fake-model", "none", "{}",
                                     minutes * 60_000)
            repo.set_run_language(db, run_id, language, "detected", None)
            rows = []
            for n in range(minutes * 6):
                start = n * 10_000
                text = "stacchiamoli col metodo di TEC" if n == 4 else f"frase {n}"
                words = [[start, start + 500, f" {w}", 0.35 if w == "TEC" else 0.95]
                         for w in text.split()]
                rows.append((start, start + 9_500, text, json.dumps(words), -0.2, 0.01, 1.2,
                             None))
            repo.add_transcript_window(db, run_id, rows, minutes * 60_000, 1.0)
            repo.finish_run(db, run_id)
        return sid
    return make
