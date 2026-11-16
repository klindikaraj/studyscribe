"""Generating a session's notes into the database: checks, lifecycle, call records, resume,
restart, lock. Fake LLM: no network, no model."""

import json

import pytest
from fakes import FakeLlm, echo_notes

from studyscribe.db import repo
from studyscribe.notes.providers.base import BadRequest
from studyscribe.notes.runner import (
    CannotGenerate,
    NotesRequest,
    check_provider,
    generate_session_notes,
    import_template,
    load_template,
    prepare,
    sync_templates,
)
from studyscribe.notes.templates import TemplateError, builtin_template, parse_template
from studyscribe.session import NOTES_LOCK_BYTE, session_lock

REQUEST = NotesRequest(provider="ollama", model="fake-1", effort="none", num_ctx=32768)


def generate(db, data_dir, sid, responder=echo_notes, request=REQUEST, **kwargs):
    prepared = prepare(db, sid, request)
    llm = FakeLlm(responder)
    events = []
    result = generate_session_notes(db, data_dir, prepared, llm,
                                    emit=lambda kind, **info: events.append((kind, info)),
                                    **kwargs)
    return result, llm, events


# --- checks ---------------------------------------------------------------------------------------


def test_prepare_checks(db, transcribed, start_session):
    with pytest.raises(CannotGenerate, match="no session nope"):
        prepare(db, "nope", REQUEST)
    bare = start_session("No transcript")
    with pytest.raises(CannotGenerate, match="no transcript yet.*transcribe"):
        prepare(db, bare, REQUEST)
    with pytest.raises(CannotGenerate, match="only 4.8 minutes of speech"):
        prepare(db, transcribed(minutes=5), REQUEST)
    silent = transcribed(language=None)
    with pytest.raises(CannotGenerate, match="--notes-language"):
        prepare(db, silent, REQUEST)
    forced = prepare(db, silent, NotesRequest("ollama", "m", "none", language="en"))
    assert (forced.language, forced.language_source) == ("en", "forced")
    with pytest.raises(CannotGenerate, match="unknown template 'nope'.*studio"):
        prepare(db, silent, NotesRequest("ollama", "m", "none", template_id="nope",
                                         language="it"))


def test_prepare_gives_blocks_lecture_and_estimate(db, transcribed):
    prepared = prepare(db, transcribed(minutes=24), REQUEST)
    assert len(prepared.blocks) == 2 and prepared.language == "it"
    assert prepared.lecture.title == "Lezione" and prepared.lecture.course == "PyTorch"
    assert prepared.estimate.count("map") == 2 and prepared.estimate.count("reduce") == 1
    assert prepared.speech_ms == 24 * 6 * 9_500


def test_check_provider(db):
    llm = FakeLlm(echo_notes)
    assert check_provider(llm, REQUEST) == "fake-1"
    small = NotesRequest("ollama", "m", "none", num_ctx=16384)
    with pytest.raises(CannotGenerate, match="16,384.*does not converge.*config set "
                                             "ollama.num_ctx 32768"):
        check_provider(llm, small)


# --- generation -----------------------------------------------------------------------------------


def test_a_generation_is_stored_with_its_calls(db, data_dir, transcribed):
    sid = transcribed()
    result, llm, events = generate(db, data_dir, sid)
    row = repo.get_notes(db, result.notes_id)
    assert row["status"] == "done" and row["finished_at"]
    assert (row["provider"], row["model"], row["template_id"], row["language"]) == (
        "ollama", "fake-1", "studio", "it")
    calls = repo.list_llm_calls(db, result.notes_id)
    assert [(c["stage"], c["block"], c["status"]) for c in calls] == [
        ("map", 1, "ok"), ("map", 2, "ok"), ("reduce", 1, "ok")]
    assert all(c["input_tokens"] and c["output_tokens"] and c["cost_usd"] == 0 for c in calls)
    content = json.loads(row["content_json"])
    assert content["outline"][0]["ref"]["segment_id"]  # resolved to the segment
    assert json.loads(row["uncertain_json"])[0]["word"] == "TEC"
    stats = json.loads(row["stats_json"])
    assert stats["count_tokens"]["calls"] >= 3 and stats["map_calls"] == 2
    assert stats["output_tokens"] == sum(c["output_tokens"] for c in calls)
    assert [kind for kind, _ in events][0] == "started"
    assert any(kind == "call" for kind, _ in events)
    lock = session_lock(data_dir, sid, NOTES_LOCK_BYTE)  # released
    assert lock.acquire()
    lock.release()


def test_ctrl_c_leaves_it_interrupted_and_the_same_request_resumes(db, data_dir, transcribed):
    sid = transcribed()

    def stop_at_second_map(request, index):
        if index == 1:
            raise KeyboardInterrupt
        return echo_notes(request, index)

    with pytest.raises(KeyboardInterrupt):
        generate(db, data_dir, sid, stop_at_second_map)
    row = repo.get_open_notes(db, sid)
    assert row["status"] == "interrupted" and row["error"] == "interrupted by the user"
    assert repo.notes_states(db)[sid] == "interrupted 1/2"
    result, llm, events = generate(db, data_dir, sid)
    assert result.notes_id == row["id"] and result.resumed_blocks == 1
    assert len(llm.requests) == 2  # map 2 and the reduce
    assert ("resumed", {"notes_id": row["id"], "maps": 1, "blocks": 2}) in events
    assert repo.get_notes(db, row["id"])["status"] == "done"


def test_a_provider_error_leaves_it_interrupted_with_the_message(db, data_dir, transcribed):
    sid = transcribed()

    def broken(request, index):
        raise BadRequest("unknown model", remedy="Pull it.")

    with pytest.raises(BadRequest):
        generate(db, data_dir, sid, broken)
    row = repo.get_open_notes(db, sid)
    assert row["status"] == "interrupted" and row["error"] == "unknown model\nPull it."
    assert [c["status"] for c in repo.list_llm_calls(db, row["id"])] == ["error"]


def test_restart_and_different_options_discard_the_open_generation(db, data_dir, transcribed):
    sid = transcribed()

    def stop(request, index):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        generate(db, data_dir, sid, stop)
    first = repo.get_open_notes(db, sid)["id"]
    with pytest.raises(KeyboardInterrupt):
        generate(db, data_dir, sid, stop, request=NotesRequest(
            "ollama", "fake-1", "high", num_ctx=32768))
    assert repo.get_notes(db, first)["error"] == "discarded: different options"
    second = repo.get_open_notes(db, sid)["id"]
    _, _, events = generate(db, data_dir, sid, request=NotesRequest(
        "ollama", "fake-1", "high", num_ctx=32768, restart=True))
    assert repo.get_notes(db, second)["status"] == "failed"
    assert repo.get_notes(db, second)["error"] == "discarded: --restart"
    assert ("discarded", {"notes_id": second, "reason": "--restart"}) in events
    assert len(repo.list_notes_by_status(db, "failed")) == 2  # every generation is kept


def test_a_session_whose_notes_are_being_generated_is_refused(db, data_dir, transcribed):
    sid = transcribed()
    lock = session_lock(data_dir, sid, NOTES_LOCK_BYTE)
    assert lock.acquire()
    try:
        llm = FakeLlm(echo_notes)
        closed = []
        llm.close = lambda: closed.append(True)
        with pytest.raises(CannotGenerate, match="another process"):
            generate_session_notes(db, data_dir, prepare(db, sid, REQUEST), llm)
        assert closed == [True]
    finally:
        lock.release()


def test_the_provider_is_closed_after_success_and_error(db, data_dir, transcribed):
    sid = transcribed()
    closed = []

    class Closing(FakeLlm):
        def close(self):
            closed.append(True)

    prepared = prepare(db, sid, REQUEST)
    generate_session_notes(db, data_dir, prepared, Closing(echo_notes))

    def broken(request, index):
        raise BadRequest("x")

    with pytest.raises(BadRequest):
        generate_session_notes(db, data_dir, prepare(db, sid, REQUEST), Closing(broken))
    assert closed == [True, True]


# --- templates ------------------------------------------------------------------------------------


def test_templates_sync_and_import(db):
    sync_templates(db)
    assert [t["id"] for t in repo.list_templates(db)] == ["studio"]
    assert load_template(db, "studio") == builtin_template()
    mine = parse_template(json.dumps({
        "id": "mine", "title": {"en": "Mine"},
        "sections": [{"key": "points", "kind": "bullets", "heading": {"en": "Points"},
                      "instructions": "Points.", "per_hour": 10, "min": 1, "max": 5}]}))
    import_template(db, mine)
    assert load_template(db, "mine") == mine
    with pytest.raises(TemplateError, match="built-in"):
        import_template(db, builtin_template())
