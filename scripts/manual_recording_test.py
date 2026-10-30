"""Guided manual test of recording (step 1); checklist: docs/manual-test-recording.md.

Usage:
    .venv\\Scripts\\python scripts\\manual_recording_test.py [--manual] [--data-dir PATH]
    .venv\\Scripts\\python scripts\\manual_recording_test.py --check SESSION_ID [--data-dir PATH]

The guided run records a session. At suggested times it plays, pauses and resumes the video
by sending the media Play/Pause key itself (--manual: you do it and press Enter), asks you to
unplug and replug the headphones and to check `powercfg /requests`, and finally to kill the
capture process from Task Manager. It notes when each step was asked for and done, recovers
the session as the CLI does and checks the recording against those times. The notes are saved
in the session folder (manual-test.json), so --check can verify a session later, e.g. after
killing every process, or any session recorded with `studyscribe record`.

Default data directory: %TEMP%\\studyscribe-manual-test, to keep test sessions out of the real one.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import functools
import json
import os
import sys
import tempfile
import threading
import time
from contextlib import closing
from ctypes import wintypes
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
TOLERANCE_S = 1.0  # duration and end time against the kill, which is noticed within 0.2 s
SWITCH_GRACE_S = 5.0  # a device switch may be reported up to this long after Enter
GAP_TOLERANCE_MS = 100  # per switch, manifest gap vs reported gap
KEY_EFFECT_WAIT_S = 3.0  # how long to watch the level after sending a media key
# Kill in segment 4, so that both device switches fall in segments already closed (the gap_ms
# of the segment open at the kill is lost by design).
KILL_AT_S = 3 * SEGMENT_SECONDS + 10


@dataclass(frozen=True)
class StepSpec:
    name: str
    at_s: int  # suggested time from the start of the recording
    text: str
    media: bool  # play/pause: sent as a media key unless --manual


STEPS = [
    StepSpec("play", 5, "PLAY the video (the recording started in silence)", True),
    StepSpec("pause", 60, "PAUSE the video", True),
    StepSpec("resume", 80, "RESUME the video", True),
    StepSpec("unplug", 100, "UNPLUG the headphones (the video keeps playing on the monitor)",
             False),
    StepSpec("replug", 125, "PLUG the headphones back in", False),
]


# --- media key ---------------------------------------------------------------------------------

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
VK_MEDIA_PLAY_PAUSE = 0xB3


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class MOUSEINPUT(ctypes.Structure):  # never sent: it gives INPUT the size SendInput expects
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _InputUnion)]


def send_media_play_pause() -> bool:
    """Press and release the media Play/Pause key, as a keyboard would (SendInput).

    Windows hands it to the current media session (the browser that played the video), even
    with the browser in the background. False if Windows did not take the input. Input meant
    for an elevated window in the foreground is dropped silently (UIPI).
    """
    keys = (INPUT * 2)()
    for key, flags in zip(keys, (KEYEVENTF_EXTENDEDKEY, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP)):
        key.type = INPUT_KEYBOARD
        key.ki = KEYBDINPUT(wVk=VK_MEDIA_PLAY_PAUSE, dwFlags=flags)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    user32.SendInput.restype = wintypes.UINT
    return user32.SendInput(len(keys), keys, ctypes.sizeof(INPUT)) == len(keys)


# --- guided run --------------------------------------------------------------------------------


class Notes:
    """Steps (when asked for and done, epoch seconds) and capture events, saved after every
    change."""

    def __init__(self, path: Path, media_keys: bool) -> None:
        self.path = path
        self.data: dict = {"media_keys": media_keys, "steps": {}, "events": [], "powercfg": None}

    def step(self, name: str, asked: float, done: float, by_key: bool, events: list[dict]):
        self.data["steps"][name] = {"prompt": asked, "done": done, "by_key": by_key}
        self.save(events)

    def save(self, events: list[dict]) -> None:
        self.data["events"] = list(events)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def load_notes(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if "marks" in data:
        data["steps"] = _steps_from_marks(data.pop("marks"), data.get("events", []))
    return data


def _steps_from_marks(marks: dict, events: list[dict]) -> dict:
    """Notes of the first version of this script kept only the Enter times. Its prompts came at
    the suggested times, or right after the previous Enter if that was later."""
    started = next((e["time"] for e in events if e["type"] == "started"), None)
    steps, previous = {}, None
    for spec in STEPS:
        if spec.name not in marks:
            continue
        done = marks[spec.name]
        asked = done if started is None else started + spec.at_s
        if previous is not None:
            asked = max(asked, previous)
        steps[spec.name] = {"prompt": asked, "done": done, "by_key": False}
        previous = done
    if "killed" in marks:
        steps["killed"] = {"prompt": marks["killed"], "done": marks["killed"], "by_key": False}
    return steps


class EventLog(threading.Thread):
    """Reads the capture events in the background (prompts block the main thread), prints them
    as the CLI does, keeps the ones the checks need and the recent levels."""

    KEEP = {"started", "device_lost", "device_opened", "warning", "error", "stopped"}

    def __init__(self, handle: RecorderHandle) -> None:
        super().__init__(daemon=True)
        self._handle = handle
        self._status = StatusLine(sys.stdout, live=False)
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._levels: collections.deque[tuple[float, float]] = collections.deque(maxlen=300)
        self.events: list[dict] = []

    def run(self) -> None:
        while not self._done.is_set():
            for event in self._handle.events(timeout=0.2):
                if event["type"] == "level":
                    with self._lock:
                        self._levels.append((event["time"], event["rms_db"]))
                elif event["type"] in self.KEEP:
                    self.events.append(event)
                self._status.show(event)

    def levels_since(self, since: float) -> list[float]:
        with self._lock:
            return [db for t, db in self._levels if t >= since]

    def first(self, *kinds: str) -> dict | None:
        return next((e for e in self.events if e["type"] in kinds), None)

    def stop(self) -> None:
        self._done.set()
        self.join()


def guided(data_dir: Path, media_keys: bool) -> int:
    if media_keys:
        video = (
            "  - the lecture video open in the browser: PLAY it, then PAUSE it where someone is\n"
            "    talking. The script sends the media Play/Pause key at 0:05, 1:00 and 1:20: do\n"
            "    not touch the video, close other media apps, and keep administrator windows\n"
            "    out of the foreground at those moments (Windows drops keys sent to them);\n"
        )
    else:
        video = "  - a lecture video ready and PAUSED where someone is talking;\n"
    print(
        "\nBefore starting:\n"
        "  - headphones plugged in, so the default output is the Realtek jack;\n"
        f"{video}"
        "  - Task Manager open on the Details tab, and an administrator PowerShell.\n"
        "At each prompt, first do what it says, then press Enter.\n"
    )
    input("Press Enter to start recording...")
    with closing(open_database(db_path(data_dir))) as conn:
        print_recovery(recover_all(conn, data_dir), sys.stdout, quiet=True)
        title = f"Manual test {datetime.now():%Y-%m-%d %H:%M}"
        session_id, owner_lock = new_session(conn, data_dir, title)
        notes = Notes(session_dir(data_dir, session_id) / NOTES_FILE, media_keys)
        try:
            sink = functools.partial(SessionSink, str(data_dir), session_id)
            handle = RecorderHandle(CaptureConfig(sink_factory=sink))
            handle.start()
            print(f"Session {session_id}\nThis script: pid {os.getpid()}; "
                  f"capture process: pid {handle.pid}")
            log = EventLog(handle)
            log.start()
            completed = _run_steps(handle, log, notes, media_keys)
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


def _run_steps(handle: RecorderHandle, log: EventLog, notes: Notes, media_keys: bool) -> bool:
    """Run the steps; False if aborted with Ctrl+C or if the capture could not start."""
    try:
        deadline = time.time() + 20
        while log.first("started", "error") is None and time.time() < deadline:
            time.sleep(0.1)
        started = log.first("started")
        if started is None:
            print("The capture did not start (see the error above).")
            return False
        t0 = started["time"]
        for spec in STEPS:
            if not _wait_until(handle, t0 + spec.at_s):
                break
            if spec.media and media_keys:
                sent = _press_media_key(spec, log, t0)
                notes.step(spec.name, sent, sent, True, log.events)
            else:
                asked = time.time()
                input(f"[{_clock(t0)}] {spec.text}, then press Enter. ")
                notes.step(spec.name, asked, time.time(), False, log.events)
        if handle.is_alive():
            answer = input(
                f"[{_clock(t0)}] In the administrator PowerShell run `powercfg /requests`. "
                "Is python.exe listed under SYSTEM? [y/n, Enter = skip] "
            ).strip().lower()
            notes.data["powercfg"] = answer[:1] or None
            notes.save(log.events)
        asked = time.time()
        if handle.is_alive() and _wait_until(handle, t0 + KILL_AT_S):
            asked = time.time()
            print(
                f"[{_clock(t0)}] Now kill the capture process, as a crash would: Task Manager > "
                f"Details > python.exe with PID {handle.pid} > End task\n"
                f"    (or: taskkill /F /PID {handle.pid}). Not PID {os.getpid()}: that is "
                "this script. Waiting..."
            )
            while handle.is_alive():
                time.sleep(0.2)
        notes.step("killed", asked, time.time(), False, log.events)
        return True
    except (KeyboardInterrupt, EOFError):
        return False


def _press_media_key(spec: StepSpec, log: EventLog, t0: float) -> float:
    """Send the media Play/Pause key; warn if the level does not change as expected.

    Returns the time the key was sent."""
    sent = time.time()
    accepted = send_media_play_pause()
    print(f"[{_clock(t0)}] {spec.text}: media Play/Pause key sent.")
    if not accepted:
        print("    Windows did not accept the key: the timing checks of this step will fail.")
        return sent
    starts = spec.name != "pause"
    time.sleep(KEY_EFFECT_WAIT_S)
    # After a pause the player sends silence (or nothing): look past its last ~second of audio.
    levels = log.levels_since(sent if starts else sent + 1.0)
    loud = any(db > verify.SOUND_DB for db in levels)
    if starts and not loud:
        print("    No sound after the key. Is the video page open, with the video played once and "
              "then paused?\n    Other media apps may have taken the key (see the checklist).")
    elif not starts and loud:
        print("    The sound did not stop after the key (see the checklist).")
    return sent


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
    steps = notes.get("steps", {})
    events = notes.get("events", [])
    started = _epoch(session["started_at"])
    ended = _epoch(session["ended_at"]) if session["ended_at"] else None
    duration = summary["duration_ms"] / 1000
    # The steps on the session timeline.
    timeline = {
        name: verify.Step(s["prompt"] - started, s["done"] - started, s["by_key"])
        for name, s in steps.items()
    }
    checks = []

    killed = "killed" in steps
    expected = ("interrupted",) if killed else ("recorded", "interrupted")
    checks.append(Check("Session status", "PASS" if session["status"] in expected else "FAIL",
                        session["status"]))

    problems = verify.manifest_problems(conn, data_dir, session_id)
    checks.append(Check("Segments and files agree", "FAIL" if problems else "PASS",
                        "; ".join(problems) or f"{summary['segments']} segment(s), contiguous"))

    real_end = steps["killed"]["done"] if killed else ended
    if real_end is None:
        checks.append(Check("Duration matches real time", "SKIP", "no end time"))
    else:
        real = real_end - started
        diff = duration - real
        checks.append(Check(
            "Duration matches real time", _within(diff),
            f"timeline {duration:.2f} s, real {real:.2f} s ({diff:+.2f} s)"
            + ("" if killed else "; + means the device clock ran fast"),
        ))
    if killed and ended is not None:
        diff = ended - steps["killed"]["done"]
        checks.append(Check("ended_at (last write) matches the kill", _within(diff),
                            f"{diff:+.2f} s"))

    levels = verify.sound_levels(conn, data_dir, session_id)
    play, pause, resume = (timeline.get(name) for name in ("play", "pause", "resume"))
    checks.append(_timing("Sound starts at play (initial silence kept)",
                          verify.first_sound(levels), play, starts=True, cold=True))
    if play and pause and resume:
        # The pause is the last sound before the resume was asked for; the resume is the first
        # sound once the pause is over (the player may send silence for a while after a pause).
        checks.append(_timing(
            "Sound stops at pause",
            verify.last_sound(levels, before_s=resume.prompt, after_s=play.done),
            pause, starts=False))
        checks.append(_timing(
            "Sound starts again at resume",
            verify.first_sound(levels, after_s=min(pause.done + 2.0, resume.prompt)),
            resume, starts=True))
    else:
        checks.append(Check("Pause and resume", "SKIP", "no record of these steps"))

    first_device = next((e["device"] for e in events if e["type"] == "started"), None)
    checks.append(_switch("Unplug followed", events, steps.get("unplug"),
                          lambda opened: opened != first_device))
    checks.append(_switch("Replug followed", events, steps.get("replug"),
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


def _timing(
    name: str, sound: float | None, step: verify.Step | None, starts: bool, cold: bool = False
) -> Check:
    if step is None:
        return Check(name, "SKIP", "no record of the step")
    ok, detail = verify.step_timing(sound, step, starts, cold)
    return Check(name, "PASS" if ok else "FAIL", detail)


def _switch(name: str, events: list[dict], step: dict | None, expected) -> Check:
    if step is None:
        return Check(name, "SKIP", "no record of the step")
    # The switch happens after the prompt and is usually reported before the Enter.
    switch = verify.device_switch(events, step["prompt"], step["done"] + SWITCH_GRACE_S)
    if switch is None:
        return Check(name, "FAIL", f"no device switch between the prompt and "
                                   f"{SWITCH_GRACE_S:g} s after Enter")
    lost, opened = switch
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


def report(conn, data_dir: Path, session_id: str, notes: dict) -> int:
    summary = repo.get_session_summary(conn, session_id)
    checks = run_checks(conn, data_dir, session_id, notes)
    how = {True: "media key", False: "by hand"}.get(notes.get("media_keys"), "by hand")
    lines = [
        f"Checks for session {session_id}: {summary['status']}, "
        f"{format_duration(summary['duration_ms'])} in {summary['segments']} segment(s), "
        f"gap {summary['gap_ms'] / 1000:.1f} s; play/pause {how}",
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
    parser.add_argument("--manual", action="store_true",
                        help="play, pause and resume the video by hand (Enter after each) "
                        "instead of the media key sent by the script")
    parser.add_argument("--check", metavar="SESSION_ID",
                        help="recover and check an existing session instead of recording")
    args = parser.parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    print(f"Data directory: {data_dir}")
    if args.check is None:
        return guided(data_dir, media_keys=not args.manual)
    with closing(open_database(db_path(data_dir))) as conn:
        print_recovery(recover_all(conn, data_dir), sys.stdout)
        if repo.get_session(conn, args.check) is None:
            print(f"No session {args.check} in this data directory.")
            return 2
        notes = load_notes(session_dir(data_dir, args.check) / NOTES_FILE)
        return report(conn, data_dir, args.check, notes)


if __name__ == "__main__":
    raise SystemExit(main())
