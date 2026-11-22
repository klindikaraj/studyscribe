"""The notes commands of the CLI: notes, transcribe --notes, template, config set, the NOTES
column. Fake provider and fake engine: no network, no GPU."""

import io
import json

import pytest
from fakes import FakeLlm, echo_notes

from studyscribe import cli
from studyscribe.db import repo
from studyscribe.notes.keys import FoundKey


def args_for(*argv):
    return cli.build_parser().parse_args(list(argv))


def fake_make(responder=echo_notes, made=None):
    def make(provider, model, key, **kwargs):
        if made is not None:
            made.append((provider, model, key, kwargs))
        return FakeLlm(responder, model=model)
    return make


def notes(db, data_dir, *argv, responder=echo_notes, made=None, **options):
    out = io.StringIO()
    code = cli.cmd_notes(db, data_dir, args_for("notes", *argv), out,
                         make=fake_make(responder, made), **options)
    return code, out.getvalue()


def test_notes_on_a_transcribed_session(db, data_dir, transcribed):
    sid = transcribed()
    made = []
    code, text = notes(db, data_dir, sid, made=made)
    assert code == 0
    assert f'Notes for {sid} "Lezione": language it (transcript), template studio' in text
    assert "2 blocks; requests (estimate): 2 map + 0 partial + 1 final" in text
    assert "(this PC: nothing leaves it), context 32,768 tokens, effort none" in text
    assert "Model check: qwen3.5:9b" in text
    assert "  map 1/2" in text and "  final reduce" in text and "    ok in " in text
    assert "Done (generation #1)" in text and "words to check by listening: 1" in text
    provider, model, key, kwargs = made[0]
    assert (provider, model, key) == ("ollama", "qwen3.5:9b", None)
    assert kwargs["ollama"].num_ctx == 32768 and not kwargs["ollama"].allow_cpu
    listing = io.StringIO()
    cli.cmd_sessions(db, data_dir, listing)
    assert "done (qwen3.5:9b)" in listing.getvalue()


def test_unknown_session_and_missing_transcript(db, data_dir, start_session):
    code, text = notes(db, data_dir, "nope")
    assert code == 2 and "no session nope" in text
    code, text = notes(db, data_dir, start_session("Empty"))
    assert code == 1 and "no transcript yet" in text


def test_a_small_context_is_refused_before_loading(db, data_dir, transcribed):
    sid = transcribed()
    out = io.StringIO()
    assert cli_notes_config(db, "ollama.num_ctx", "16384", out) == 0
    assert "the notes command refuses to run" in out.getvalue()
    code, text = notes(db, data_dir, sid)
    assert code == 1 and "16,384: with less than 32,768" in text
    assert repo.get_open_notes(db, sid) is None  # nothing started


def cli_notes_config(db, name, value, out):
    from studyscribe import cli_notes

    return cli_notes.config_set(db, args_for("config", "set", name, value), out)


def test_config_set_checks_values(db):
    out = io.StringIO()
    assert cli_notes_config(db, "ollama.num_ctx", "big", out) == 2
    assert cli_notes_config(db, "ollama.num_ctx", "1000", out) == 2
    assert cli_notes_config(db, "notes.confirm_usd", "0.2", out) == 0
    assert repo.get_setting(db, "notes.confirm_usd") == "0.2"
    with pytest.raises(SystemExit):
        args_for("config", "set", "unknown.setting", "1")


def test_ctrl_c_says_where_and_the_same_command_resumes(db, data_dir, transcribed):
    sid = transcribed()

    def stop(request, index):
        if index == 1:
            raise KeyboardInterrupt
        return echo_notes(request, index)

    code, text = notes(db, data_dir, sid, responder=stop)
    assert code == 1 and "Interrupted at map 2/2. Run the same command to resume." in text
    listing = io.StringIO()
    cli.cmd_sessions(db, data_dir, listing)
    assert "interrupted 1/2" in listing.getvalue()
    code, text = notes(db, data_dir, sid)
    assert code == 0 and "Resuming generation #1: 1 of 2 map blocks already done." in text


def test_a_cloud_generation_above_the_threshold_asks_first(db, data_dir, transcribed):
    sid = transcribed()
    asked = []
    key = lambda provider: FoundKey("anthropic", "sk-ant-test-key-123456", "test")  # noqa: E731
    with_threshold = cli_notes_config(db, "notes.confirm_usd", "0", io.StringIO())
    assert with_threshold == 0
    code, text = notes(db, data_dir, sid, "--provider", "anthropic", "--model",
                       "claude-sonnet-5-5", confirm=lambda q: asked.append(q) or False,
                       find_key=key)
    assert code == 1 and "Nothing sent." in text and "estimated cost: $" in text
    assert asked and "threshold $0.00" in asked[0]
    code, text = notes(db, data_dir, sid, "--provider", "anthropic", "--model",
                       "claude-sonnet-5-5", "--yes", find_key=key)
    assert code == 0 and "cost $" in text


def test_gemini_free_tier_notice(db, data_dir, transcribed):
    sid = transcribed()
    def key(provider):
        return FoundKey("gemini", "AIzaSyTestKeyNotReal_0123456789abcdefg", "test")

    code, text = notes(db, data_dir, sid, "--provider", "gemini", find_key=key)
    assert code == 0 and "Google may use this transcript" in text
    assert "estimated cost: $0.0000" in text


def test_template_list_and_import(db, tmp_path):
    out = io.StringIO()
    assert cli.cli_notes.cmd_template(db, args_for("template", "list"), out) == 0
    assert "studio (built-in): Study" in out.getvalue()
    assert "  concepts (concepts): 24 per hour, 3-20" in out.getvalue()
    path = tmp_path / "mine.json"
    path.write_text(json.dumps({"id": "mine", "title": {"en": "Mine"}, "sections": [
        {"key": "points", "kind": "bullets", "heading": {"en": "Points"},
         "instructions": "Points.", "per_hour": 10, "min": 1, "max": 5}]}), encoding="utf-8")
    out = io.StringIO()
    assert cli.cli_notes.cmd_template(db, args_for("template", "import", str(path)), out) == 0
    assert "Template 'mine' imported (1 sections)." in out.getvalue()
    path.write_text("{}", encoding="utf-8")
    out = io.StringIO()
    assert cli.cli_notes.cmd_template(db, args_for("template", "import", str(path)), out) == 1
    assert "Not imported:" in out.getvalue()


def test_transcribe_with_notes_runs_the_notes_after_freeing_whisper(db, data_dir,
                                                                     start_session):
    from fakes import FakeTranscriber, loud_spans
    from test_transcribe_run import SMALL_WINDOWS, timeline

    from studyscribe.audio.wav import wav_header
    from studyscribe.config import session_dir
    from studyscribe.db.connection import transaction

    sid = start_session("Short")
    pcm = timeline()
    (session_dir(data_dir, sid) / "audio" / "000001.wav").write_bytes(
        wav_header(pcm.size * 2) + pcm.astype("<i2").tobytes())
    with transaction(db):
        repo.add_audio_segment(db, sid, 1, "audio/000001.wav", 0, pcm.size * 1000 // 16_000, 0)
        repo.finish_session(db, sid, "recorded", repo.utc_now_iso())
    engine = FakeTranscriber()
    out = io.StringIO()
    code = cli.cmd_transcribe(db, data_dir, args_for("transcribe", sid, "--notes"), out,
                              load=lambda: engine, vad=loud_spans,
                              window_options=SMALL_WINDOWS,
                              notes_options={"make": fake_make()})
    text = out.getvalue()
    assert engine.closed
    assert "Whisper's GPU memory is released; now the notes." in text
    # a 10-second recording has too little speech: the notes command says so
    assert code == 1 and "notes need at least 5" in text


# --- exports of the notes -------------------------------------------------------------------------


def export(db, data_dir, *argv):
    out = io.StringIO()
    code = cli.cmd_export(db, data_dir, args_for("export", *argv), out)
    return code, out.getvalue()


def exports_of(data_dir, sid):
    from studyscribe.transcribe.export import exports_dir

    folder = exports_dir(data_dir, sid)
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def test_notes_write_notes_md_and_html_when_they_are_done(db, data_dir, transcribed):
    sid = transcribed()
    code, text = notes(db, data_dir, sid)
    assert code == 0
    assert "Notes version 1 of 1 (qwen3.5:9b" in text and "notes.md, notes.html" in text
    assert "warning: audio.flac is not in" in text and f"studyscribe export {sid} --audio" in text
    assert exports_of(data_dir, sid) == ["notes.html", "notes.md"]  # no audio.flac made up


def test_a_failed_notes_export_does_not_fail_the_generation(db, data_dir, transcribed):
    from studyscribe.transcribe.export import exports_dir

    sid = transcribed()
    folder = exports_dir(data_dir, sid)
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.write_text("a file where the folder should be", encoding="utf-8")
    code, text = notes(db, data_dir, sid)
    assert code == 0 and "Done (generation #1)" in text
    assert "Cannot write the notes exports" in text
    assert repo.get_latest_done_notes(db, sid) is not None


def test_export_has_nothing_to_say_about_notes_that_do_not_exist(db, data_dir, transcribed):
    sid = transcribed()
    code, text = export(db, data_dir, sid)
    assert code == 0 and "Notes version" not in text and "notes" not in text.split("\n", 1)[-1]
    assert exports_of(data_dir, sid) == ["transcript-dropped.txt", "transcript.md",
                                         "transcript.srt"]


def test_export_writes_the_latest_notes(db, data_dir, transcribed):
    sid = transcribed()
    notes(db, data_dir, sid)
    (data_dir / "sessions" / sid / "exports" / "notes.html").unlink()
    code, text = export(db, data_dir, sid)
    assert code == 0 and "Notes version 1 of 1" in text
    assert exports_of(data_dir, sid) == [
        "notes.html", "notes.md", "transcript-dropped.txt", "transcript.md", "transcript.srt"]


def test_export_notes_version_writes_versioned_files(db, data_dir, transcribed):
    sid = transcribed()
    notes(db, data_dir, sid)
    notes(db, data_dir, sid, "--restart")
    code, text = export(db, data_dir, sid, "--notes-version", "1")
    assert code == 0 and "Notes version 1 of 2" in text and "notes-v1.md, notes-v1.html" in text
    assert {"notes-v1.html", "notes-v1.md", "notes.html", "notes.md"} <= set(
        exports_of(data_dir, sid))


def test_export_notes_version_that_does_not_exist_lists_the_versions(db, data_dir, transcribed):
    sid = transcribed()
    notes(db, data_dir, sid)
    code, text = export(db, data_dir, sid, "--notes-version", "5")
    assert code == 1 and "no notes version 5" in text and "Available: 1 (" in text
    assert "notes-v5.md" not in exports_of(data_dir, sid)
    code, text = export(db, data_dir, transcribed(), "--notes-version", "1")
    assert code == 1 and "no finished notes" in text


def test_export_with_audio_in_place_has_no_audio_warning(db, data_dir, transcribed):
    sid = transcribed()
    notes(db, data_dir, sid)
    (data_dir / "sessions" / sid / "exports" / "audio.flac").write_bytes(b"fLaC")
    code, text = export(db, data_dir, sid)
    assert code == 0 and "warning" not in text
