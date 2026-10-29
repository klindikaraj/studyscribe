"""Command-line entry point: record, sessions, recover."""

from __future__ import annotations

import argparse
import functools
import os
import sqlite3
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from studyscribe import __version__
from studyscribe.audio.capture import CaptureConfig
from studyscribe.config import (
    DATA_DIR_ENV,
    SILENCE_LIMIT_S,
    db_path,
    resolve_data_dir,
    session_dir,
)
from studyscribe.db import repo
from studyscribe.db.connection import open_database
from studyscribe.recording import Event, RecorderHandle
from studyscribe.recovery import Recovered, recover_all, recover_session
from studyscribe.session import SessionSink, new_session

LEVEL_FLOOR_DB = -60.0  # left end of the level bar
LEVEL_BAR_WIDTH = 20


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
    with closing(open_database(db_path(data_dir))) as conn:
        if args.command == "record":
            return cmd_record(conn, data_dir, args, sys.stdout)
        if args.command == "sessions":
            return cmd_sessions(conn, data_dir, sys.stdout)
        return cmd_recover(conn, data_dir, sys.stdout)


# --- commands ----------------------------------------------------------------------------------


def cmd_record(
    conn: sqlite3.Connection,
    data_dir: Path,
    args: argparse.Namespace,
    out: TextIO,
    capture_options: dict[str, Any] | None = None,
    max_seconds: float | None = None,
) -> int:
    """Record until Ctrl+C (or `max_seconds`, used by tests); 0 if the session is 'recorded'.

    `capture_options` override CaptureConfig fields (tests pass fake sources).
    """
    print_recovery(recover_all(conn, data_dir), out, quiet=True)
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


def cmd_sessions(conn: sqlite3.Connection, data_dir: Path, out: TextIO) -> int:
    print_recovery(recover_all(conn, data_dir), out, quiet=True)
    rows = repo.list_session_summaries(conn)
    if not rows:
        print("No sessions yet.", file=out)
        return 0
    table = [("SESSION", "STATUS", "STARTED", "DURATION", "GAP", "COURSE", "TITLE")]
    for row in rows:
        table.append(
            (
                row["id"],
                row["status"],
                _local_time(row["started_at"]),
                format_duration(row["duration_ms"]),
                f"{row['gap_ms'] / 1000:.1f} s" if row["gap_ms"] else "-",
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
    print_recovery(recover_all(conn, data_dir), out)
    return 0


# --- recording loop ----------------------------------------------------------------------------


def _run_capture(config: CaptureConfig, out: TextIO, max_seconds: float | None) -> int | None:
    """Run the capture process, show its events, stop it on Ctrl+C; return its exit code."""
    handle = RecorderHandle(config)
    handle.start()
    print(f"Parent pid {os.getpid()}, capture pid {handle.pid}. Ctrl+C to stop.", file=out)
    status = StatusLine(out)
    deadline = None if max_seconds is None else time.monotonic() + max_seconds
    try:
        while handle.is_alive() and (deadline is None or time.monotonic() < deadline):
            for event in handle.events(timeout=0.2):
                status.show(event)
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

    def __init__(self, out: TextIO, clock: Callable[[], float] = time.monotonic) -> None:
        self._out = out
        self._live = out.isatty()
        self._clock = clock
        self._started: float | None = None
        self._level_db = LEVEL_FLOOR_DB
        self._closed_segments = 0
        self._gap_ms = 0
        self._width = 0  # characters of the live line currently on screen

    def show(self, event: Event) -> None:
        kind = event["type"]
        if kind == "started":
            self._started = self._clock()
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
            self.message(
                f"Recording from {event['device']} again, gap {event['gap_ms']} ms"
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


def _gap_note(gap_ms: int) -> str:
    return f", gap {gap_ms / 1000:.1f} s" if gap_ms else ""


def _local_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")
