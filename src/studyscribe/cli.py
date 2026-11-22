"""Command-line entry point: record, transcribe, notes, export, sessions, recover, template,
config."""

from __future__ import annotations

import argparse
import functools
import os
import sqlite3
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from studyscribe import __version__, cli_notes
from studyscribe.audio.capture import CaptureConfig
from studyscribe.config import (
    DATA_DIR_ENV,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_MODEL,
    SILENCE_LIMIT_S,
    db_path,
    resolve_data_dir,
    session_dir,
)
from studyscribe.db import repo
from studyscribe.db.connection import open_database
from studyscribe.notes import keys
from studyscribe.notes.providers import DEFAULT_MODELS, DEFAULT_PROVIDER, make_provider
from studyscribe.notes.providers.base import (
    KEY_PROVIDERS,
    InvalidKey,
    LlmError,
    LlmProvider,
    MissingKey,
)
from studyscribe.recording import Event, RecorderHandle
from studyscribe.recovery import (
    Recovered,
    RecoveredRun,
    recover_all,
    recover_notes,
    recover_session,
    recover_transcriptions,
)
from studyscribe.session import SessionSink, new_session
from studyscribe.transcribe import export
from studyscribe.transcribe.engine import Transcriber, TranscriberError
from studyscribe.transcribe.language import describe
from studyscribe.transcribe.runner import CannotTranscribe, transcribe_session

LEVEL_FLOOR_DB = -60.0  # left end of the level bar
LEVEL_BAR_WIDTH = 20
STARTUP_CAP_S = 60.0  # with max_seconds (tests): longest wait for the capture to start


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="studyscribe",
        description="Record online lectures, transcribe them locally and write study notes.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--data-dir",
        metavar="PATH",
        help=f"data directory (default: ${DATA_DIR_ENV}, else %%LOCALAPPDATA%%\\StudyScribe)",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    record = commands.add_parser(
        "record",
        help="record the system audio into a new session (Ctrl+C to stop)",
        description="Record what the PC plays (loopback of the default output) into a new "
        "session. Abandoned sessions are recovered first.",
    )
    record.add_argument("--title", required=True, help="title of the session (the lecture)")
    record.add_argument("--course", help="course the lecture belongs to (created if new)")
    record.add_argument(
        "--silence-limit",
        type=float,
        default=SILENCE_LIMIT_S / 60,
        metavar="MINUTES",
        help=f"stop by itself after this many minutes without sound; 0 = never "
        f"(default {SILENCE_LIMIT_S / 60:g})",
    )
    transcribe = commands.add_parser(
        "transcribe",
        help="transcribe a recorded session locally (GPU)",
        description="Transcribe a recorded session with faster-whisper. An interrupted "
        "transcription is resumed by running the same command again.",
    )
    transcribe.add_argument("session_id", metavar="SESSION_ID")
    transcribe.add_argument(
        "--language", default="auto",
        help="language code, e.g. en or it (default: auto, detected on the first minutes of "
        "speech)",
    )
    transcribe.add_argument("--model", default=DEFAULT_MODEL,
                            help=f"faster-whisper model (default {DEFAULT_MODEL})")
    transcribe.add_argument(
        "--compute-type", default=DEFAULT_COMPUTE_TYPE,
        help=f"default {DEFAULT_COMPUTE_TYPE}; int8_float16 uses less GPU memory",
    )
    transcribe.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    transcribe.add_argument("--restart", action="store_true",
                            help="start from scratch instead of resuming an interrupted run")
    transcribe.add_argument("--hints", metavar="TEXT",
                            help="words to expect (title, technical terms), passed to the model")
    transcribe.add_argument("--notes", action="store_true",
                            help="then write the notes (Whisper's GPU memory is freed first); "
                            "the notes options below apply")
    cli_notes.add_notes_options(transcribe, in_transcribe=True)
    export_parser = commands.add_parser(
        "export",
        help="write the transcript files of a session (exports/ in its folder)",
        description="Write transcript.srt, transcript.md and transcript-dropped.txt from the "
        "session's transcript (transcribe writes them too); --audio also joins the segments "
        "into audio.flac, with the same timeline. When the session has finished notes it also "
        "writes notes.md and notes.html from the latest ones (the page plays audio.flac, so run "
        "--audio once); --notes-version N writes notes-vN.md and notes-vN.html instead.",
    )
    export_parser.add_argument("session_id", metavar="SESSION_ID")
    export_parser.add_argument("--audio", action="store_true",
                               help="also write audio.flac (the whole recording, lossless)")
    export_parser.add_argument("--notes-version", type=int, metavar="N",
                               help="write the notes of version N (the Nth finished generation, "
                               "from 1) as notes-vN.md and notes-vN.html")
    cli_notes.add_commands(commands)
    commands.add_parser(
        "sessions",
        help="list the sessions with status and duration",
        description="List the sessions. Abandoned sessions are recovered first.",
    )
    commands.add_parser(
        "recover",
        help="repair sessions left behind by a crash or a killed process",
        description="Repair sessions left in 'recording' by a process that no longer runs, "
        "and register segments missing from the manifest.",
    )
    config = commands.add_parser(
        "config",
        help="LLM providers for the notes: local Ollama state, cloud API keys",
        description="Show the state of the local provider (Ollama) and manage the API keys of "
        "the cloud providers. Keys are kept in the Windows Credential Manager; environment "
        "variables work as a fallback.",
    )
    config_actions = config.add_subparsers(dest="config_action", metavar="ACTION", required=True)
    set_key = config_actions.add_parser(
        "set-key", help="store a provider's API key (typed or pasted, not shown)",
        description="Read the key without echoing it, check it with a free request (no tokens), "
        "and store it.")
    set_key.add_argument("provider", choices=KEY_PROVIDERS)
    set_key.add_argument("--no-check", action="store_true",
                         help="store the key without checking it online")
    delete_key = config_actions.add_parser("delete-key", help="remove a stored API key")
    delete_key.add_argument("provider", choices=KEY_PROVIDERS)
    config_actions.add_parser(
        "show", help="settings, Ollama's state and where each cloud provider's key comes from "
        "(masked)")
    cli_notes.add_config_set(config_actions)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        # Redirected output uses the ANSI code page: never crash on a title it cannot encode.
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = build_parser()
    args = parser.parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    print(f"Data directory: {data_dir}")
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "config" and args.config_action in ("set-key", "delete-key"):
        return cmd_config(args, sys.stdout)
    with closing(open_database(db_path(data_dir), on_backup=_backup_note)) as conn:
        if args.command == "config":
            if args.config_action == "set":
                return cli_notes.config_set(conn, args, sys.stdout)
            return cmd_config(args, sys.stdout, settings=cli_notes.settings_lines(conn))
        if args.command == "record":
            return cmd_record(conn, data_dir, args, sys.stdout)
        if args.command == "transcribe":
            return cmd_transcribe(conn, data_dir, args, sys.stdout)
        if args.command == "notes":
            return cmd_notes(conn, data_dir, args, sys.stdout)
        if args.command == "template":
            return cli_notes.cmd_template(conn, args, sys.stdout)
        if args.command == "export":
            return cmd_export(conn, data_dir, args, sys.stdout)
        if args.command == "sessions":
            return cmd_sessions(conn, data_dir, sys.stdout)
        return cmd_recover(conn, data_dir, sys.stdout)


def _backup_note(path: Path) -> None:
    print(f"Database backed up before updating its schema: {path}")


# --- commands ----------------------------------------------------------------------------------


def cmd_record(
    conn: sqlite3.Connection,
    data_dir: Path,
    args: argparse.Namespace,
    out: TextIO,
    capture_options: dict[str, Any] | None = None,
    max_seconds: float | None = None,
) -> int:
    """Record until Ctrl+C (or `max_seconds` after the capture started, used by tests); 0 if
    the session is 'recorded'.

    `capture_options` override CaptureConfig fields (tests pass fake sources).
    """
    _recover_everything(conn, data_dir, out)
    session_id, owner_lock = new_session(conn, data_dir, args.title, args.course)
    try:
        print(f"Session {session_id}\nFolder: {session_dir(data_dir, session_id)}", file=out)
        limit_s = args.silence_limit * 60 if args.silence_limit > 0 else None
        if limit_s is not None:
            print(f"Stops by itself after {args.silence_limit:g} min without sound.", file=out)
        config = CaptureConfig(
            sink_factory=functools.partial(SessionSink, str(data_dir), session_id),
            silence_limit_s=limit_s,
            **(capture_options or {}),
        )
        exitcode = _run_capture(config, out, max_seconds)
        if exitcode != 0:
            print("The capture process stopped abnormally: recovering the session.", file=out)
            result = recover_session(conn, data_dir, session_id, owner_lock=owner_lock)
            print_recovery([result] if result else [], out, quiet=True)
    finally:
        owner_lock.release()
    summary = repo.get_session_summary(conn, session_id)
    print(
        f"Session {session_id}: {summary['status']}, {format_duration(summary['duration_ms'])} "
        f"in {summary['segments']} segment(s)" + _gap_note(summary["gap_ms"]),
        file=out,
    )
    return 0 if summary["status"] == "recorded" else 1


def _recover_everything(conn: sqlite3.Connection, data_dir: Path, out: TextIO) -> None:
    print_recovery(recover_all(conn, data_dir), out, quiet=True)
    print_run_recovery(recover_transcriptions(conn, data_dir), out)
    cli_notes.print_notes_recovery(recover_notes(conn, data_dir), out)


def cmd_sessions(conn: sqlite3.Connection, data_dir: Path, out: TextIO) -> int:
    _recover_everything(conn, data_dir, out)
    rows = repo.list_session_summaries(conn)
    if not rows:
        print("No sessions yet.", file=out)
        return 0
    transcripts = repo.transcript_states(conn)
    notes = repo.notes_states(conn)
    table = [("SESSION", "STATUS", "STARTED", "DURATION", "GAP", "TRANSCRIPT", "NOTES", "COURSE",
              "TITLE")]
    for row in rows:
        table.append(
            (
                row["id"],
                row["status"],
                _local_time(row["started_at"]),
                format_duration(row["duration_ms"]),
                f"{row['gap_ms'] / 1000:.1f} s" if row["gap_ms"] else "-",
                transcripts.get(row["id"], "-"),
                notes.get(row["id"], "-"),
                row["course"] or "-",
                row["title"] or "-",
            )
        )
    widths = [max(len(line[i]) for line in table) for i in range(len(table[0]) - 1)]
    for line in table:
        cells = [cell.ljust(width) for cell, width in zip(line, widths)]
        print("  ".join(cells + [line[-1]]), file=out)
    print(f"{len(rows)} session(s)", file=out)
    return 0


def cmd_recover(conn: sqlite3.Connection, data_dir: Path, out: TextIO) -> int:
    results = recover_all(conn, data_dir)
    runs = recover_transcriptions(conn, data_dir)
    notes = recover_notes(conn, data_dir)
    print_recovery(results, out, quiet=bool(runs or notes))
    print_run_recovery(runs, out)
    cli_notes.print_notes_recovery(notes, out)
    return 0


def cmd_transcribe(
    conn: sqlite3.Connection,
    data_dir: Path,
    args: argparse.Namespace,
    out: TextIO,
    load: Callable[[], Transcriber] | None = None,
    notes_options: dict[str, Any] | None = None,
    **options: Any,
) -> int:
    """Transcribe a session; 0 when its transcript is complete (and, with --notes, its notes).

    `load` and `options` (passed to transcribe_session) let tests use a fake engine;
    `notes_options` are passed to the notes command (fake provider).
    """
    language = args.language.lower()
    if language != "auto" and not _known_language(language):
        print(f"Unknown language code {args.language!r} (e.g. en, it, or auto).", file=out)
        return 2
    _recover_everything(conn, data_dir, out)
    if load is None:
        load = functools.partial(_load_whisper, args, out)
    progress = TranscribeProgress(out)
    try:
        result = transcribe_session(
            conn, data_dir, args.session_id, load, language=language, hints=args.hints,
            restart=args.restart, emit=progress.show, **options,
        )
    except CannotTranscribe as exc:
        progress.clear()
        print(f"Cannot transcribe {args.session_id}: {exc}", file=out)
        return 2 if repo.get_session(conn, args.session_id) is None else 1
    except TranscriberError as exc:
        progress.clear()
        print(f"Error: {exc}", file=out)
        return 1
    except KeyboardInterrupt:
        progress.clear()
        run = repo.get_open_run(conn, args.session_id)
        done = "" if run is None else f" at {format_duration(run['done_ms'])}"
        print(f"Interrupted{done}. Run the same command to resume.", file=out)
        return 1
    progress.clear()
    total = format_duration(result.audio_ms)
    if result.language is None:
        print(f"Done: no speech found in {total}.", file=out)
        return 0
    speed = result.audio_ms / 1000 / result.processing_s if result.processing_s else 0
    if result.resumed_from_ms:
        # This attempt did only the rest: say so, and give the speed over all attempts.
        rest = format_duration(result.audio_ms - result.resumed_from_ms)
        timing = (f"{total} ({rest} from {format_duration(result.resumed_from_ms)} in "
                  f"{format_duration(progress.elapsed_ms())} now; "
                  f"{format_duration(result.processing_s * 1000)} of processing in all, "
                  f"{speed:.0f}x real time)")
    else:
        timing = f"{total} in {format_duration(progress.elapsed_ms())} ({speed:.0f}x real time)"
    print(f"Done: {timing}, {result.kept} segments kept, {result.dropped} dropped", file=out)
    code = _write_exports(conn, data_dir, args.session_id, out, audio=False)
    if code != 0 or not getattr(args, "notes", False):
        return code
    print("Whisper's GPU memory is released; now the notes.", file=out)
    return cmd_notes(conn, data_dir, args, out, **(notes_options or {}))


def cmd_notes(conn: sqlite3.Connection, data_dir: Path, args: argparse.Namespace, out: TextIO,
              **options: Any) -> int:
    """`notes`: recovery first, then the generation (see cli_notes). `options` (make,
    confirm, find_key) let tests run without a provider."""
    if args.command == "notes":
        _recover_everything(conn, data_dir, out)
    return cli_notes.cmd_notes(conn, data_dir, args, out, **options)


def cmd_export(conn: sqlite3.Connection, data_dir: Path, args: argparse.Namespace,
               out: TextIO) -> int:
    if repo.get_session(conn, args.session_id) is None:
        print(f"No session {args.session_id} in this data directory.", file=out)
        return 2
    code = _write_exports(conn, data_dir, args.session_id, out, audio=args.audio)
    if code != 0:
        return code
    version = getattr(args, "notes_version", None)
    if version is None and not repo.list_done_notes(conn, args.session_id):
        return 0  # no notes yet: nothing to say about them
    return cli_notes.write_notes_exports(conn, data_dir, args.session_id, out, version=version)


def _write_exports(conn: sqlite3.Connection, data_dir: Path, session_id: str, out: TextIO,
                   audio: bool) -> int:
    try:
        paths = export.export_text(conn, data_dir, session_id)
        if audio:
            print("Joining the segments into audio.flac...", file=out, flush=True)
            paths.append(export.export_audio(conn, data_dir, session_id))
    except export.NoTranscript as exc:
        print(f"Cannot export: {exc}", file=out)
        return 1
    except OSError as exc:
        print(f"Cannot write the exports: {exc}", file=out)
        return 1
    print(f"Exports in {paths[0].parent}: " + ", ".join(p.name for p in paths), file=out)
    return 0


def cmd_config(args: argparse.Namespace, out: TextIO, *, store: keys.KeyStore | None = None,
               environ: Mapping[str, str] | None = None,
               read_secret: Callable[[str], str] | None = None,
               make: Callable[[str, str | None, str], LlmProvider] = make_provider,
               ollama_status: Callable[[], list[str]] | None = None,
               settings: list[str] | None = None) -> int:
    """Providers and API keys. `store`, `environ`, `read_secret`, `make` and `ollama_status` let
    tests run without the real credential store, console, network or Ollama."""
    environ = os.environ if environ is None else environ
    try:
        if args.config_action == "show":
            for line in settings or []:
                print(line, file=out)
            return _config_show(out, store, environ, ollama_status or _ollama_status)
        if args.config_action == "delete-key":
            if keys.delete_key(args.provider, store=store):
                print(f"Key for {args.provider} removed from the {keys.KEYRING_SOURCE}.", file=out)
            else:
                print(f"No stored key for {args.provider}.", file=out)
            return 0
        return _config_set_key(args, out, store, read_secret or _read_secret, make)
    except LlmError as exc:
        print(f"Error: {exc.user_message()}", file=out)
        return 1


def _config_show(out: TextIO, store: keys.KeyStore | None, environ: Mapping[str, str],
                 ollama_status: Callable[[], list[str]]) -> int:
    print(f"Default provider: {DEFAULT_PROVIDER} ({DEFAULT_MODELS[DEFAULT_PROVIDER]})", file=out)
    for line in ollama_status():
        print(line, file=out)
    for provider in KEY_PROVIDERS:
        found = keys.find_key(provider, store=store, environ=environ)
        where = f"{keys.mask(found.key)} from {found.source}" if found else "not set"
        print(f"{provider}: {where}  (default model {DEFAULT_MODELS[provider]})", file=out)
    print("Environment variables, used when no key is stored: "
          + ", ".join(name for names in keys.ENV_VARS.values() for name in names), file=out)
    return 0


def _ollama_status() -> list[str]:
    """Ollama's state for `config show`: reachable, version, models, the default model."""
    from studyscribe.notes.providers.ollama import OllamaOptions, OllamaProvider, is_local

    options = OllamaOptions()
    model = DEFAULT_MODELS["ollama"]
    provider = OllamaProvider(model, options)
    where = options.host + ("" if is_local(options.host) else " (not this PC: text leaves it)")
    try:
        version = provider.server_version()
        models = provider.downloaded_models()
    except LlmError as exc:
        return [f"ollama: {exc.message} {exc.remedy}".rstrip()]
    names = [name for name, _ in models]
    lines = [f"ollama: version {version} at {where}",
             "  models: " + (", ".join(f"{n} ({s / 1e9:.1f} GB)" for n, s in models) or "none")]
    if model in names or f"{model}:latest" in names:
        lines.append(f"  default model {model}: downloaded")
    else:
        lines.append(f"  default model {model}: NOT downloaded (run: ollama pull {model})")
    return lines


def _config_set_key(args: argparse.Namespace, out: TextIO, store: keys.KeyStore | None,
                    read_secret: Callable[[str], str],
                    make: Callable[[str, str | None, str], LlmProvider]) -> int:
    provider = args.provider
    print(f"Get a key at {keys.KEY_PAGES[provider]}", file=out)
    try:
        key = keys.clean_key(read_secret(f"{provider} API key (not shown): "))
    except (EOFError, KeyboardInterrupt):
        print("\nNothing stored.", file=out)
        return 1
    except ValueError as exc:
        print(f"Not stored: {exc}.", file=out)
        return 1
    note = " (not checked)"
    if not args.no_check:
        try:
            name = make(provider, None, key).check()
            note = f" (checked: {name} is available)"
        except (InvalidKey, MissingKey) as exc:
            print(f"Not stored: {exc.message}", file=out)
            return 1
        except LlmError as exc:
            note = f" (could not check it: {exc.message})"
    keys.store_key(provider, key, store=store)
    print(f"Key {keys.mask(key)} stored in the {keys.KEYRING_SOURCE}{note}.", file=out)
    return 0


def _read_secret(prompt: str) -> str:
    if sys.stdin is not None and sys.stdin.isatty():
        import getpass

        return getpass.getpass(prompt)
    return sys.stdin.readline() if sys.stdin is not None else ""


def _known_language(code: str) -> bool:
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    return code in _LANGUAGE_CODES


def _load_whisper(args: argparse.Namespace, out: TextIO) -> Transcriber:
    from studyscribe.transcribe.whisper import FasterWhisperTranscriber

    print(f"Loading {args.model} ({args.compute_type}, {args.device})...", file=out, flush=True)
    started = time.monotonic()
    engine = FasterWhisperTranscriber.load(args.model, args.compute_type, args.device)
    print(f"Model {engine.model} loaded in {time.monotonic() - started:.1f} s", file=out)
    return engine


class TranscribeProgress:
    """Shows transcription events; on a terminal one progress line is redrawn in place."""

    def __init__(
        self, out: TextIO, clock: Callable[[], float] = time.monotonic, live: bool | None = None
    ) -> None:
        self._out = out
        self._clock = clock
        self._live = out.isatty() if live is None else live
        self._started = clock()
        self._audio_ms = 0
        self._from_ms = 0  # where this attempt started (the resume point)
        self._width = 0

    def elapsed_ms(self) -> float:
        return (self._clock() - self._started) * 1000

    def show(self, kind: str, **info: Any) -> None:
        if kind in ("started", "resumed"):
            self._audio_ms = info["audio_ms"]
            self._from_ms = info.get("done_ms", 0)
            self._started = self._clock()
            if kind == "resumed":
                self.message(f"Resuming an interrupted transcription at "
                             f"{format_duration(self._from_ms)}")
        elif kind == "discarded":
            self.message(f"Discarding an interrupted transcription ({info['reason']}), "
                         "starting again")
        elif kind == "language":
            self._language(info)
        elif kind == "segment":
            self._draw(info["end_ms"])
        elif kind == "window" and not self._live:
            self.message(
                f"Transcribed up to {format_duration(info['end_ms'])} of "
                f"{format_duration(self._audio_ms)}: {info['kept']} segments"
                + (f", {info['dropped']} dropped" if info["dropped"] else "")
            )

    def _language(self, info: dict) -> None:
        guess = info["guess"]
        if info["language"] is None:
            self.message("Language: none, no speech found")
        elif info["source"] == "forced":
            self.message(f"Language: {info['language']} (forced)")
        elif guess is None:
            self.message(f"Language: {info['language']} (detected earlier)")
        else:
            self.message(f"Language: {guess.language} (detected on {guess.speech_s:.0f} s of "
                         f"speech: {describe(guess)})")
        for warning in info["warnings"]:
            self.message(f"Warning: {warning}")

    def message(self, text: str) -> None:
        self.clear()
        print(text, file=self._out, flush=True)

    def clear(self) -> None:
        if self._width:
            self._out.write("\r" + " " * self._width + "\r")
            self._out.flush()
            self._width = 0

    def _draw(self, position_ms: int) -> None:
        if not self._live or not self._audio_ms:
            return
        position_ms = min(max(position_ms, self._from_ms), self._audio_ms)
        fraction = position_ms / self._audio_ms
        filled = round(fraction * LEVEL_BAR_WIDTH)
        elapsed = self.elapsed_ms()
        done = position_ms - self._from_ms
        line = (f"Transcribing [{'#' * filled}{'-' * (LEVEL_BAR_WIDTH - filled)}] "
                f"{fraction:4.0%}  {format_duration(position_ms)} / "
                f"{format_duration(self._audio_ms)}  elapsed {format_duration(elapsed)}")
        if done > 0 and elapsed > 0:
            left = elapsed / done * (self._audio_ms - position_ms)
            line += f"  left ~{format_duration(left)}  {done / elapsed:.0f}x"
        self._out.write("\r" + line.ljust(self._width))
        self._out.flush()
        self._width = len(line)


# --- recording loop ----------------------------------------------------------------------------


def _run_capture(config: CaptureConfig, out: TextIO, max_seconds: float | None) -> int | None:
    """Run the capture process, show its events, stop it on Ctrl+C; return its exit code."""
    handle = RecorderHandle(config)
    handle.start()
    print(f"Parent pid {os.getpid()}, capture pid {handle.pid}. Ctrl+C to stop.", file=out)
    status = StatusLine(out)
    # Spawning the process can take seconds on a busy PC: `max_seconds` counts from the start
    # of the capture (the process ends by itself if it cannot start, so the cap is a safety net).
    deadline = None if max_seconds is None else time.monotonic() + STARTUP_CAP_S
    try:
        while handle.is_alive() and (deadline is None or time.monotonic() < deadline):
            for event in handle.events(timeout=0.2):
                status.show(event)
                if event["type"] == "started" and max_seconds is not None:
                    deadline = time.monotonic() + max_seconds
    except KeyboardInterrupt:
        status.message("Stopping...")
    while True:
        try:
            exitcode = handle.stop()
            break
        except KeyboardInterrupt:
            status.message("Still stopping: closing the last segment, please wait.")
    for event in handle.events():
        status.show(event)
    status.clear()
    return exitcode


class StatusLine:
    """Shows capture events: messages on their own lines, plus (on a terminal) one live line
    with elapsed time, level and current segment, redrawn in place."""

    def __init__(
        self,
        out: TextIO,
        clock: Callable[[], float] = time.monotonic,
        live: bool | None = None,
    ) -> None:
        self._out = out
        self._live = out.isatty() if live is None else live
        self._clock = clock
        self._started: float | None = None
        self._device = ""  # device being recorded (or lost last)
        self._level_db = LEVEL_FLOOR_DB
        self._closed_segments = 0
        self._gap_ms = 0
        self._width = 0  # characters of the live line currently on screen

    def show(self, event: Event) -> None:
        kind = event["type"]
        if kind == "started":
            self._started = self._clock()
            self._device = event["device"]
            self.message(
                f"Recording from {event['device']}, {event['rate']} Hz, {event['channels']} ch"
            )
        elif kind == "level":
            self._level_db = event["rms_db"]
            self._draw()
        elif kind == "segment":
            self._closed_segments += 1
            self._gap_ms += event["gap_ms"]
        elif kind == "device_lost":
            self.message(f"Device lost ({event['reason']}): {event['device']}")
        elif kind == "device_opened":
            device = event["device"]
            what = (
                f"Recording from {device} again" if device == self._device
                else f"Switched to {device}"
            )
            self._device = device
            self.message(
                f"{what}, gap {event['gap_ms']} ms"
                + (f" after {event['attempts']} attempts" if event["attempts"] > 1 else "")
            )
        elif kind == "warning":
            self.message(f"Warning: {event['message']}")
        elif kind == "error":
            self.message(f"Error: {event['message']}\n{event.get('traceback', '')}".rstrip())
        elif kind == "stopped":
            self.message(
                f"Stopped ({event['reason']}) after {format_duration(event['wall_s'] * 1000)}"
            )

    def message(self, text: str) -> None:
        self.clear()
        print(f"[{datetime.now():%H:%M:%S}] {text}", file=self._out, flush=True)
        self._draw()

    def clear(self) -> None:
        if self._width:
            self._out.write("\r" + " " * self._width + "\r")
            self._out.flush()
            self._width = 0

    def _draw(self) -> None:
        if not self._live or self._started is None:
            return
        elapsed = format_duration((self._clock() - self._started) * 1000)
        line = (
            f"REC {elapsed}  [{level_bar(self._level_db)}] {self._level_db:6.1f} dBFS  "
            f"segment {self._closed_segments + 1}" + _gap_note(self._gap_ms)
        )
        self._out.write("\r" + line.ljust(self._width))
        self._out.flush()
        self._width = len(line)


# --- formatting --------------------------------------------------------------------------------


def format_duration(ms: float) -> str:
    """Milliseconds -> H:MM:SS."""
    seconds = int(ms // 1000)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def level_bar(db: float, width: int = LEVEL_BAR_WIDTH) -> str:
    """dBFS -> "#####-----" from LEVEL_FLOOR_DB (empty) to 0 dBFS (full)."""
    filled = round((db - LEVEL_FLOOR_DB) / -LEVEL_FLOOR_DB * width)
    filled = min(max(filled, 0), width)
    return "#" * filled + "-" * (width - filled)


def print_recovery(results: list[Recovered], out: TextIO, quiet: bool = False) -> None:
    """One line per recovered session, plus its problems; `quiet`: nothing if there are none."""
    if not results:
        if not quiet:
            print("Recovery: nothing to do.", file=out)
        return
    for r in results:
        what = []
        if r.added:
            what.append(f"segments added {r.added}")
        if r.repaired:
            what.append(f"files repaired {r.repaired}")
        print(
            f"Recovered {r.session_id}: {r.status_before} -> {r.status_after}, "
            f"{format_duration(r.duration_ms)}" + "".join(f", {w}" for w in what),
            file=out,
        )
        for problem in r.problems:
            print(f"  problem: {problem}", file=out)


def print_run_recovery(runs: list[RecoveredRun], out: TextIO) -> None:
    for run in runs:
        percent = 100 * run.done_ms // max(run.audio_ms, 1)
        print(f"Transcription of {run.session_id} interrupted at {percent}% "
              f"({format_duration(run.done_ms)}): run `transcribe` again to resume it.", file=out)


def _gap_note(gap_ms: int) -> str:
    return f", gap {gap_ms / 1000:.1f} s" if gap_ms else ""


def _local_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")
