import json

from studyscribe.db import repo
from studyscribe.recovery import recover_notes
from studyscribe.session import NOTES_LOCK_BYTE, session_lock


def generation(db, sid):
    notes_id = repo.create_notes(
        db, session_id=sid, transcript_run_id=None, provider="ollama", model="m",
        template_id="studio", template_json="{}", language="it", language_source="transcript",
        prompt_version="v5.1", prompt_hash="h", options_json="{}",
        blocks_json=json.dumps([[1, 5], [6, 9]]))
    repo.add_llm_call(db, notes_id, stage="map", block=1, attempt=1, status="ok",
                      output_json="{}")
    return notes_id


def test_a_generation_whose_process_died_is_interrupted(db, data_dir, start_session):
    sid = start_session("Lecture")
    notes_id = generation(db, sid)
    results = recover_notes(db, data_dir)
    assert [(r.notes_id, r.maps_done, r.blocks) for r in results] == [(notes_id, 1, 2)]
    row = repo.get_notes(db, notes_id)
    assert row["status"] == "interrupted" and row["error"] == "the notes process stopped"
    assert recover_notes(db, data_dir) == []  # idempotent


def test_a_generation_still_running_is_left_alone(db, data_dir, start_session):
    sid = start_session("Lecture")
    notes_id = generation(db, sid)
    lock = session_lock(data_dir, sid, NOTES_LOCK_BYTE)
    assert lock.acquire()
    try:
        # another handle on the same file, as another process would have
        assert recover_notes(db, data_dir) == []
    finally:
        lock.release()
    assert repo.get_notes(db, notes_id)["status"] == "running"


def test_a_session_without_folder(db, data_dir):
    db.execute("INSERT INTO sessions (id, status, started_at) VALUES ('gone', 'recorded', 'x')")
    notes_id = generation(db, "gone")
    assert [r.notes_id for r in recover_notes(db, data_dir)] == [notes_id]
