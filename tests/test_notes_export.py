"""Writing notes.md / notes.html (and the versioned names) to the exports folder."""

import pytest
from fakes import generate_fake_notes

from studyscribe.db import repo
from studyscribe.db.connection import transaction
from studyscribe.notes.export import (
    NotesUnavailable,
    export_notes,
    notes_file_names,
    warnings,
)
from studyscribe.transcribe.export import exports_dir


def test_file_names():
    assert notes_file_names(None) == ("notes.md", "notes.html")
    assert notes_file_names(2) == ("notes-v2.md", "notes-v2.html")


def test_the_latest_notes_are_written_as_notes_md_and_html(db, data_dir, transcribed):
    sid = transcribed()
    generate_fake_notes(db, data_dir, sid)
    result = export_notes(db, data_dir, sid)
    folder = exports_dir(data_dir, sid)
    assert (result.markdown, result.html) == (folder / "notes.md", folder / "notes.html")
    markdown = result.markdown.read_text(encoding="utf-8")
    page = result.html.read_text(encoding="utf-8")
    assert markdown.startswith("# Lezione\n") and "(notes.html#t=" in markdown
    assert page.startswith("<!DOCTYPE html>") and '<audio id="player"' in page
    assert "\r" not in markdown and "\r" not in page
    assert sorted(p.name for p in folder.iterdir()) == ["notes.html", "notes.md"]  # no .tmp


def test_a_version_has_its_own_files_and_links(db, data_dir, transcribed):
    sid = transcribed()
    generate_fake_notes(db, data_dir, sid)
    generate_fake_notes(db, data_dir, sid, restart=True)
    export_notes(db, data_dir, sid)
    result = export_notes(db, data_dir, sid, version=1)
    folder = exports_dir(data_dir, sid)
    assert sorted(p.name for p in folder.iterdir()) == [
        "notes-v1.html", "notes-v1.md", "notes.html", "notes.md"]
    assert "(notes-v1.html#t=" in result.markdown.read_text(encoding="utf-8")
    assert "(notes.html#t=" in (folder / "notes.md").read_text(encoding="utf-8")
    assert result.doc.version == 1 and result.doc.total_versions == 2


def test_writing_again_replaces_the_files(db, data_dir, transcribed):
    sid = transcribed()
    generate_fake_notes(db, data_dir, sid)
    first = export_notes(db, data_dir, sid)
    first.html.write_text("old", encoding="utf-8")
    export_notes(db, data_dir, sid)
    assert first.html.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_nothing_to_export(db, data_dir, transcribed):
    sid = transcribed()
    with pytest.raises(NotesUnavailable, match="no finished notes"):
        export_notes(db, data_dir, sid)
    assert not exports_dir(data_dir, sid).exists()
    generate_fake_notes(db, data_dir, sid)
    with pytest.raises(NotesUnavailable, match="no notes version 3"):
        export_notes(db, data_dir, sid, version=3)
    assert not (exports_dir(data_dir, sid) / "notes-v3.md").exists()


def test_the_audio_file_is_never_created(db, data_dir, transcribed):
    sid = transcribed()
    generate_fake_notes(db, data_dir, sid)
    result = export_notes(db, data_dir, sid)
    assert not (result.folder / "audio.flac").exists()
    assert 'id="no-audio" class="warn">' in result.html.read_text(encoding="utf-8")
    (result.folder / "audio.flac").write_bytes(b"fLaC")
    again = export_notes(db, data_dir, sid)
    assert 'id="no-audio" class="warn" hidden>' in again.html.read_text(encoding="utf-8")


def test_warnings(db, data_dir, transcribed):
    sid = transcribed()
    generate_fake_notes(db, data_dir, sid)
    result = export_notes(db, data_dir, sid)
    [missing] = warnings(result)
    assert f"studyscribe export {sid} --audio" in missing and "audio.flac is not in" in missing
    (result.folder / "audio.flac").write_bytes(b"fLaC")
    assert warnings(export_notes(db, data_dir, sid)) == []
    with transaction(db):
        db.execute("UPDATE generated_notes SET transcript_run_id = NULL")
    [earlier] = warnings(export_notes(db, data_dir, sid))
    assert "earlier transcript" in earlier
    assert repo.get_latest_done_notes(db, sid) is not None


def test_a_failure_to_write_is_an_oserror(db, data_dir, transcribed):
    sid = transcribed()
    generate_fake_notes(db, data_dir, sid)
    folder = exports_dir(data_dir, sid)
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.write_text("a file where the folder should be", encoding="utf-8")
    with pytest.raises(OSError):
        export_notes(db, data_dir, sid)
