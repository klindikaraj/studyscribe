"""Guided manual test of recording (step 1); checklist: docs/manual-test-recording.md.

Usage:
    .venv\\Scripts\\python scripts\\manual_recording_test.py [--data-dir PATH]
    .venv\\Scripts\\python scripts\\manual_recording_test.py --check SESSION_ID [--data-dir PATH]

The guided run records a session and asks you, at suggested times, to press play, pause,
resume, unplug and replug the headphones, check `powercfg /requests` and finally kill the
capture process from Task Manager. It notes when each step happened, recovers the session as
the CLI does and checks the recording against those times. The notes are saved in the session
folder (manual-test.json), so --check can verify a session later, e.g. after killing every
process, or any session recorded with `studyscribe record`.

Default data directory: %TEMP%\\studyscribe-manual-test, to keep test sessions out of the real one.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import tempfile
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from studyscribe import verify
from studyscribe.audio.capture import CaptureConfig
from studyscribe.cli import StatusLine, format_duration, print_recovery
from studyscribe.config import SEGMENT_SECONDS, db_path, resolve_data_dir, session_dir
from studyscribe.db import repo
from studyscribe.db.connection import open_database
from studyscribe.recording import RecorderHandle
from studyscribe.recovery import recover_all, recover_session
from studyscribe.session import SessionSink, new_session

NOTES_FILE = "manual-test.json"
REPORT_FILE = "manual-test-report.txt"
TOLERANCE_S = 1.0  # marks are pressed by hand; the kill is noticed within 0.2 s
SWITCH_WITHIN_S = 5.0  # a device switch must be reported this soon after its mark
GAP_TOLERANCE_MS = 100  # per switch, manifest gap vs reported gap
# Kill in segment 4, so that both device switches fall in segments already closed (the gap_ms
# of the segment open at the kill is lost by design).
KILL_AT_S = 3 * SEGMENT_SECONDS + 10

# (mark, suggested time in seconds, instruction)
STEPS = [
    ("play", 5, "Press PLAY on the video (the recording started in silence)"),
    ("pause", 60, "PAUSE the video"),
    ("resume", 80, "RESUME the video"),
    ("unplug", 100, "UNPLUG the headphones (the video keeps playing on the monitor)"),
    ("replug", 125, "PLUG the headphones back in"),
]


# --- guided run --------------------------------------------------------------------------------


class Notes:
    """Marks (epoch seconds) and capture events, saved after every change."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict = {"marks": {}, "events": [], "powercfg": None}

    def mark(self, name: str, events: list[dict]) -> None:
        self.data["marks"][name] = time.time()
        self.save(events)

    def save(self, events: list[dict]) -> None:
        self.data["events"] = list(events)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def load(path: Path) -> dict | None:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


class EventLog(threading.Thread):
    """Reads the capture events in the background (prompts block the main thread), prints them
    as the CLI does and keeps the ones the checks need."""

    KEEP = {"started", "device_lost", "device_opened", "warning", "error", "stopped"}

    def __init__(self, handle: RecorderHandle) -> None:
        super().__init__(daemon=True)
        self._handle = handle
        self._status = StatusLine(sys.stdout, live=False)
        self._done = threading.Event()
        self.events: list[dict] = []

    def run(self) -> None:
        while not self._done.is_set():
            for event in self._handle.events(timeout=0.2):
                if event["type"] in self.KEEP:
                    self.events.append(event)
                self._status.show(event)

    def first(self, *kinds: str) -> dict | None:
        return next((e for e in self.events if e["type"] in kinds), None)

    def stop(self) -> None:
        self._done.set()
        self.join()


def guided(data_dir: Path) -> int:
    print(
        "\nBefore starting:\n"
        "  - headphones plugged in, so the default output is the Realtek jack;\n"
        "  - a lecture video ready and PAUSED where someone is talking;\n"
        "  - Task Manager open on the Details tab, and an administrator PowerShell.\n"
        "At each prompt do what it says and press Enter at the same moment.\n"
    )
    input("Press Enter to start recording...")
    with closing(open_database(db_path(data_dir))) as conn:
        print_recovery(recover_all(conn, data_dir), sys.stdout, quiet=True)
        title = f"Manual test {datetime.now():%Y-%m-%d %H:%M}"
        session_id, owner_lock = new_session(conn, data_dir, title)
        notes = Notes(session_dir(data_dir, session_id) / NOTES_FILE)
        try:
            sink = functools.partial(SessionSink, str(data_dir), session_id)
            handle = RecorderHandle(CaptureConfig(sink_factory=sink))
            handle.start()
            print(f"Session {session_id}\nThis script: pid {os.getpid()}; "
                  f"capture process: pid {handle.pid}")
            log = EventLog(handle)
            log.start()
            completed = _run_steps(handle, log, notes)
            log.stop()
            notes.save(log.events)
            if not completed:
                print("Test aborted: stopping the recording.")
                handle.stop()
                return 1
            handle.stop()  # already dead: just collects its exit code
            print(f"Capture process exit code: {handle.exitcode}. Recovering the session.")
            result = recover_session(conn, data_dir, session_id, owner_lock=owner_lock)
            print_recovery([result] if result else [], sys.stdout)
        finally:
            owner_lock.release()
        return report(conn, data_dir, session_id, notes.data)


def _run_steps(handle: RecorderHandle, log: EventLog, notes: Notes) -> bool:
    """Prompt the steps; False if aborted with Ctrl+C or if the capture could not start."""
    try:
        deadline = time.time() + 20
        while log.first("started", "error") is None and time.time() < deadline:
            time.sleep(0.1)
        started = log.first("started")
        if started is None:
            print("The capture did not start (see the error above).")
            return False
        t0 = started["time"]
        for mark, at_s, text in STEPS:
            if not _wait_until(handle, t0 + at_s):
                break
            input(f"[{_clock(t0)}] {text}, then press Enter. ")
            notes.mark(mark, log.events)
        if handle.is_alive():
            answer = input(
                f"[{_clock(t0)}] In the administrator PowerShell run `powercfg /requests`. "
                "Is python.exe listed under SYSTEM? [y/n, Enter = skip] "
            ).strip().lower()
            notes.data["powercfg"] = answer[:1] or None
            notes.save(log.events)
        if handle.is_alive() and _wait_until(handle, t0 + KILL_AT_S):
            print(
                f"[{_clock(t0)}] Now kill the capture process, as a crash would: Task Manager > "
                f"Details > python.exe with PID {handle.pid} > End task\n"
                f"    (or: taskkill /F /PID {handle.pid}). Not PID {os.getpid()}: that is "
                "this script. Waiting..."
            )
            while handle.is_alive():
                time.sleep(0.2)
        notes.mark("killed", log.events)
        return True
    except (KeyboardInterrupt, EOFError):
        return False


def _wait_until(handle: RecorderHandle, when: float) -> bool:
    """Sleep until `when`; False if the capture process died meanwhile."""
    while time.time() < when:
        if not handle.is_alive():
            print("The capture process is gone: skipping to the checks.")
            return False
        time.sleep(0.2)
    return handle.is_alive()


def _clock(t0: float) -> str:
    return format_duration((time.time() - t0) * 1000)[2:]  # mm:ss (the test lasts minutes)


# --- checks ------------------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    result: str  # PASS, FAIL, INFO or SKIP
    detail: str


def run_checks(conn, data_dir: Path, session_id: str, notes: dict) -> list[Check]:
    session = repo.get_session(conn, session_id)
    summary = repo.get_session_summary(conn, session_id)
    marks = notes.get("marks", {})
    events = notes.get("events", [])
    started = _epoch(session["started_at"])
    ended = _epoch(session["ended_at"]) if session["ended_at"] else None
    duration = summary["duration_ms"] / 1000
    pos = {name: t - started for name, t in marks.items()}  # marks on the timeline
    checks = []

    killed = "killed" in pos
    expected = ("interrupted",) if killed else ("recorded", "interrupted")
    checks.append(Check("Session status", "PASS" if session["status"] in expected else "FAIL",
                        session["status"]))

    problems = verify.manifest_problems(conn, data_dir, session_id)
    checks.append(Check("Segments and files agree", "FAIL" if problems else "PASS",
                        "; ".join(problems) or f"{summary['segments']} segment(s), contiguous"))

    real_end = marks.get("killed", ended)
    if real_end is None:
        checks.append(Check("Duration matches real time", "SKIP", "no end time"))
    else:
        real = real_end - started
        diff = duration - real
        checks.append(Check(
            "Duration matches real time", _within(diff),
            f"timeline {duration:.2f} s, real {real:.2f} s ({diff:+.2f} s)"
            + ("; + means the device clock ran fast" if session["status"] == "recorded"
               and not killed else ""),
        ))
    if killed and ended is not None:
        diff = ended - marks["killed"]
        checks.append(Check("ended_at (last write) matches the kill", _within(diff),
                            f"{diff:+.2f} s"))

    levels = verify.sound_levels(conn, data_dir, session_id)
    checks.append(_timing("First sound = play pressed (initial silence kept)",
                          verify.first_sound(levels), pos.get("play")))
    if "pause" in pos and "resume" in pos:
        checks.append(_timing(
            "Last sound before the pause = pause pressed",
            verify.last_sound(levels, before_s=pos["resume"] - TOLERANCE_S,
                              after_s=pos.get("play", 0.0)),
            pos["pause"]))
        checks.append(_timing(
            "First sound after the pause = resume pressed",
            verify.first_sound(levels, after_s=pos["pause"] + 2.0), pos["resume"]))
    else:
        checks.append(Check("Pause and resume land on time", "SKIP", "no marks"))

    first_device = next((e["device"] for e in events if e["type"] == "started"), None)
    checks.append(_switch("Unplug followed", events, marks.get("unplug"),
                          lambda opened: opened != first_device))
    checks.append(_switch("Replug followed", events, marks.get("replug"),
                          lambda opened: opened == first_device))
    checks.append(_gaps(conn, session_id, events, started))

    answer = notes.get("powercfg") or ""
    checks.append(Check("Anti-standby listed by powercfg /requests",
                        {"y": "PASS", "n": "FAIL"}.get(answer, "SKIP"), answer or "not asked"))

    checks.append(Check("Short digital silences inside sound", "INFO",
                        f"{verify.short_digital_silences(levels)} (timeline insertions or "
                        "silent spots in the video)"))
    warnings = [e["message"] for e in events if e["type"] in ("warning", "error")]
    checks.append(Check("Warnings and errors from the capture", "INFO",
                        "; ".join(warnings) if warnings else "none"))
    return checks


def _within(diff: float, tolerance: float = TOLERANCE_S) -> str:
    return "PASS" if abs(diff) <= tolerance else "FAIL"


def _timing(name: str, found: float | None, mark: float | None) -> Check:
    if mark is None:
        return Check(name, "SKIP", "no mark")
    if found is None:
        return Check(name, "FAIL", f"no sound found (mark at {mark:.2f} s)")
    return Check(name, _within(found - mark),
                 f"sound at {found:.2f} s, pressed at {mark:.2f} s ({found - mark:+.2f} s)")


def _switch(name: str, events: list[dict], mark: float | None, expected) -> Check:
    if mark is None:
        return Check(name, "SKIP", "no mark")
    soon = [e for e in events if mark - 1.0 <= e["time"] <= mark + SWITCH_WITHIN_S]
    lost = next((e for e in soon if e["type"] == "device_lost"), None)
    opened = next((e for e in soon if e["type"] == "device_opened"), None)
    if lost is None or opened is None:
        return Check(name, "FAIL", f"no device switch reported within {SWITCH_WITHIN_S:g} s")
    detail = (f"lost {lost['device']} ({lost['reason']}), then {opened['device']}, "
              f"gap {opened['gap_ms']} ms")
    return Check(name, "PASS" if expected(opened["device"]) else "FAIL", detail)


def _gaps(conn, session_id: str, events: list[dict], started: float) -> Check:
    opened = [e for e in events if e["type"] == "device_opened"]
    rows = repo.list_audio_segments(conn, session_id)
    manifest = sum(row["gap_ms"] for row in rows)
    reported = sum(e["gap_ms"] for e in opened)
    detail = f"manifest {manifest} ms, reported {reported} ms"
    if not opened:
        return Check("Device gaps in the manifest", "SKIP", detail)
    last_start = rows[-1]["start_ms"] / 1000 if rows else 0.0
    if any(e["time"] - started >= last_start for e in opened):
        return Check("Device gaps in the manifest", "INFO",
                     detail + " (a switch fell in the last segment, whose gap may be lost)")
    tolerance = GAP_TOLERANCE_MS * len(opened)
    if abs(manifest - reported) <= tolerance:
        return Check("Device gaps in the manifest", "PASS", detail)
    if manifest < reported:
        return Check("Device gaps in the manifest", "INFO",
                     detail + " (known limit: a loss noticed late is partly ordinary silence)")
    return Check("Device gaps in the manifest", "FAIL", detail)


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def report(conn, data_dir: Path, session_id: str, notes: dict | None) -> int:
    summary = repo.get_session_summary(conn, session_id)
    checks = run_checks(conn, data_dir, session_id, notes or {})
    lines = [
        f"Checks for session {session_id}: {summary['status']}, "
        f"{format_duration(summary['duration_ms'])} in {summary['segments']} segment(s), "
        f"gap {summary['gap_ms'] / 1000:.1f} s",
    ]
    width = max(len(c.name) for c in checks)
    lines += [f"  {c.result:<4}  {c.name.ljust(width)}  {c.detail}" for c in checks]
    failed = sum(c.result == "FAIL" for c in checks)
    lines.append("All checks passed." if not failed else f"{failed} check(s) FAILED.")
    path = session_dir(data_dir, session_id) / REPORT_FILE
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines) + f"\nReport saved to {path}")
    return 1 if failed else 0


# --- entry point -------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", metavar="PATH",
                        default=str(Path(tempfile.gettempdir()) / "studyscribe-manual-test"),
                        help="data directory (default: %%TEMP%%\\studyscribe-manual-test)")
    parser.add_argument("--check", metavar="SESSION_ID",
                        help="recover and check an existing session instead of recording")
    args = parser.parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    print(f"Data directory: {data_dir}")
    if args.check is None:
        return guided(data_dir)
    with closing(open_database(db_path(data_dir))) as conn:
        print_recovery(recover_all(conn, data_dir), sys.stdout)
        if repo.get_session(conn, args.check) is None:
            print(f"No session {args.check} in this data directory.")
            return 2
        notes = Notes.load(session_dir(data_dir, args.check) / NOTES_FILE)
        return report(conn, data_dir, args.check, notes)


if __name__ == "__main__":
    raise SystemExit(main())
