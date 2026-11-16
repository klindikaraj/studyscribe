import json

import pytest

from studyscribe.db import repo
from studyscribe.db.connection import transaction


def session(db, sid="s1"):
    db.execute("INSERT INTO sessions (id, status, started_at) VALUES (?, 'recorded', 'x')", (sid,))
    return sid


def notes(db, sid="s1", blocks=3, **changes):
    args = dict(session_id=sid, transcript_run_id=None, provider="ollama", model="qwen3.5:9b",
                template_id="studio", template_json="{}", language="it",
                language_source="transcript", prompt_version="v5.1", prompt_hash="abc",
                options_json="{}", blocks_json=json.dumps([[1, 10]] * blocks))
    args.update(changes)
    return repo.create_notes(db, **args)


def test_settings(db):
    assert repo.get_setting(db, "ollama.num_ctx") is None
    assert repo.get_setting(db, "ollama.num_ctx", "32768") == "32768"
    repo.set_setting(db, "ollama.num_ctx", "16384")
    repo.set_setting(db, "ollama.num_ctx", "49152")
    assert repo.list_settings(db) == {"ollama.num_ctx": "49152"}


def test_templates(db):
    repo.upsert_template(db, "studio", '{"en": "Study"}', "d", "{}", builtin=True)
    repo.upsert_template(db, "mine", '{"en": "Mine"}', "", "{}", builtin=False)
    repo.upsert_template(db, "studio", '{"en": "Study 2"}', "d", "{}", builtin=True)
    assert [t["id"] for t in repo.list_templates(db)] == ["studio", "mine"]
    assert repo.get_template(db, "studio")["title_json"] == '{"en": "Study 2"}'
    assert repo.get_template(db, "nope") is None


def test_generation_lifecycle_and_calls(db):
    session(db)
    notes_id = notes(db)
    assert repo.get_open_notes(db, "s1")["id"] == notes_id
    repo.add_llm_call(db, notes_id, stage="map", block=1, attempt=1, status="invalid",
                      output_json="{bad")
    repo.add_llm_call(db, notes_id, stage="map", block=1, attempt=2, status="ok",
                      output_json='{"a": 1}', input_tokens=10, output_tokens=5, generate_s=0.5)
    repo.add_llm_call(db, notes_id, stage="map", block=2, attempt=1, status="error",
                      error="network")
    assert repo.accepted_map_answers(db, notes_id) == {1: '{"a": 1}'}
    assert len(repo.list_llm_calls(db, notes_id)) == 3
    assert repo.notes_states(db) == {"s1": "running 1/3"}
    repo.set_notes_status(db, notes_id, "interrupted", "killed")
    assert repo.notes_states(db) == {"s1": "interrupted 1/3"}
    assert repo.get_notes(db, notes_id)["finished_at"] is None
    repo.finish_notes(db, notes_id, content_json="{}", uncertain_json="[]", stats_json="{}")
    assert repo.get_open_notes(db, "s1") is None
    assert repo.get_latest_done_notes(db, "s1")["id"] == notes_id
    assert repo.notes_states(db) == {"s1": "done (qwen3.5:9b)"}
    assert [r["output_tokens"] for r in repo.model_timings(db, "ollama", "qwen3.5:9b")] == [5]


def test_states_prefer_open_then_done_then_failed(db):
    for sid in ("a", "b", "c"):
        session(db, sid)
    done = notes(db, "a")
    repo.finish_notes(db, done, content_json="{}", uncertain_json="[]", stats_json="{}")
    notes(db, "a", model="other")  # a newer open one wins
    failed = notes(db, "b")
    repo.set_notes_status(db, failed, "failed", "boom")
    done_c = notes(db, "c")
    repo.finish_notes(db, done_c, content_json="{}", uncertain_json="[]", stats_json="{}")
    repo.set_notes_status(db, notes(db, "c"), "failed", "boom")  # an older done one wins
    assert repo.notes_states(db) == {"a": "running 0/3", "b": "failed",
                                     "c": "done (qwen3.5:9b)"}
    assert [r["id"] for r in repo.list_notes_by_status(db, "failed")] == [failed, failed + 2]


def test_one_open_generation_per_session(db):
    session(db)
    notes(db)
    with pytest.raises(Exception):
        with transaction(db):
            notes(db)
