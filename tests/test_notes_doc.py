"""A generation of notes read for the exports: versions, the transcript it was written from,
the audio, timestamps. Fake LLM, no network."""

import json
import os

import pytest
from fakes import generate_fake_notes

from studyscribe.db import repo
from studyscribe.db.connection import transaction
from studyscribe.notes.doc import (
    NotesUnavailable,
    iter_refs,
    load_notes,
    seek_seconds,
    timestamps,
)
from studyscribe.transcribe.export import exports_dir


def ref(n, start_ms):
    return {"n": n, "segment_id": n, "start_ms": start_ms}


def test_the_latest_version_is_the_default(db, data_dir, transcribed):
    sid = transcribed()
    first = generate_fake_notes(db, data_dir, sid)
    second = generate_fake_notes(db, data_dir, sid, restart=True)
    doc = load_notes(db, data_dir, sid)
    assert (doc.notes_id, doc.version, doc.total_versions) == (second, 2, 2)
    assert load_notes(db, data_dir, sid, version=1).notes_id == first
    assert doc.model == "fake-1" and doc.effort == "none" and doc.language == "it"
    assert doc.title == "Lezione" and doc.course == "PyTorch"
    assert doc.duration_ms == 0  # no audio was recorded in the fixture


def test_unavailable_notes_say_what_exists(db, data_dir, transcribed):
    sid = transcribed()
    with pytest.raises(NotesUnavailable, match="no session nope"):
        load_notes(db, data_dir, "nope")
    with pytest.raises(NotesUnavailable, match=f"no finished notes.*notes {sid}"):
        load_notes(db, data_dir, sid)
    generate_fake_notes(db, data_dir, sid)
    with pytest.raises(NotesUnavailable) as error:
        load_notes(db, data_dir, sid, version=2)
    assert "no notes version 2" in str(error.value)
    assert "1 (" in str(error.value) and "ollama fake-1, effort none" in str(error.value)
    with pytest.raises(NotesUnavailable, match="no notes version 0"):
        load_notes(db, data_dir, sid, version=0)


def test_unfinished_and_failed_generations_are_not_versions(db, data_dir, transcribed):
    sid = transcribed()
    done = generate_fake_notes(db, data_dir, sid)
    with transaction(db):
        extra = repo.create_notes(
            db, session_id=sid, transcript_run_id=None, provider="ollama", model="m",
            template_id="studio", template_json="{}", language="it",
            language_source="transcript", prompt_version="v", prompt_hash="h", options_json="{}",
            blocks_json="[]")
        repo.set_notes_status(db, extra, "interrupted", "killed")
    doc = load_notes(db, data_dir, sid)
    assert (doc.notes_id, doc.version, doc.total_versions) == (done, 1, 1)


def test_same_transcript_gives_line_texts_and_checks_the_times(db, data_dir, transcribed):
    sid = transcribed()
    generate_fake_notes(db, data_dir, sid)
    doc = load_notes(db, data_dir, sid)
    assert doc.same_transcript and doc.time_mismatches == 0
    assert doc.line_texts[5] == "stacchiamoli col metodo di TEC"
    assert [(w.word, w.n) for w in doc.uncertain] == [("TEC", 5)]


def test_a_time_that_is_not_the_segments_is_counted(db, data_dir, transcribed):
    sid = transcribed()
    generate_fake_notes(db, data_dir, sid)
    row = repo.get_latest_done_notes(db, sid)
    content = json.loads(row["content_json"])
    content["outline"][0]["ref"]["start_ms"] += 5_000
    content["concepts"][0]["refs"][0]["segment_id"] += 1
    with transaction(db):
        db.execute("UPDATE generated_notes SET content_json = ? WHERE id = ?",
                   (json.dumps(content), row["id"]))
    assert load_notes(db, data_dir, sid).time_mismatches == 2


def test_after_a_new_transcription_the_check_is_skipped(db, data_dir, transcribed):
    sid = transcribed()
    generate_fake_notes(db, data_dir, sid)
    with transaction(db):  # what deleting the old run does (ON DELETE SET NULL)
        db.execute("UPDATE generated_notes SET transcript_run_id = NULL")
    doc = load_notes(db, data_dir, sid)
    assert not doc.same_transcript and doc.line_texts is None and doc.time_mismatches == 0


def test_audio_is_detected_and_never_created(db, data_dir, transcribed):
    sid = transcribed()
    generate_fake_notes(db, data_dir, sid)
    assert not load_notes(db, data_dir, sid).audio_present
    folder = exports_dir(data_dir, sid)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "audio.flac").write_bytes(b"fLaC")
    assert load_notes(db, data_dir, sid).audio_present
    os.remove(folder / "audio.flac")
    assert not load_notes(db, data_dir, sid).audio_present
    assert not (folder / "audio.flac").exists()


def test_coverage_and_dropped_come_from_the_stats(db, data_dir, transcribed):
    sid = transcribed()
    notes_id = generate_fake_notes(db, data_dir, sid)
    stats = {"coverage_warnings": [{"block": 2, "last_cited": 225, "uncovered": [226, 241],
                                    "start_ms": 1032722, "end_ms": 1087642}],
             "refs": {"map": {"invalid_refs": 2, "dropped_items": {"concepts": 1}},
                      "reduce": {"invalid_refs": 1, "dropped_items": {}}}}
    with transaction(db):
        db.execute("UPDATE generated_notes SET stats_json = ? WHERE id = ?",
                   (json.dumps(stats), notes_id))
    doc = load_notes(db, data_dir, sid)
    assert doc.coverage[0]["uncovered"] == [226, 241]
    assert doc.dropped == (3, 1)


def test_references_and_timestamps():
    content = {"summary": "text",
               "outline": [{"title": "t", "ref": ref(1, 1_000)}],
               "concepts": [{"name": "n", "explanation": "e",
                             "refs": [ref(9, 66_980), ref(7, 66_100), ref(3, 2_500)]}],
               "open_points": []}
    assert [r["n"] for r in iter_refs(content)] == [1, 9, 7, 3]
    assert seek_seconds(ref(9, 66_980)) == 66  # never after the line starts
    shown = timestamps(content["concepts"][0]["refs"])
    assert [r["n"] for r in shown] == [3, 7]  # time order, one per second
