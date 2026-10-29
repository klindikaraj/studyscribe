import functools
import io
from pathlib import Path

import pytest
from fakes import DyingSource, FakeSource, ScriptedDefaultOutput

from studyscribe import cli
from studyscribe.config import session_dir
from studyscribe.db import repo
from studyscribe.session import new_session

FAKE_CAPTURE = {
    "source_factory": functools.partial(FakeSource, frames_per_drain=2_400),
    "default_output_factory": ScriptedDefaultOutput,
}


def record(db, data_dir: Path, *options: str, capture=None, seconds: float = 1.5):
    args = cli.build_parser().parse_args(["record", *options])
    out = io.StringIO()
    code = cli.cmd_record(
        db, data_dir, args, out, capture_options=capture or FAKE_CAPTURE, max_seconds=seconds
    )
    return code, out.getvalue()


def abandoned_session(db, data_dir: Path, title: str) -> str:
    session_id, owner_lock = new_session(db, data_dir, title)
    owner_lock.release()
    return session_id


# --- commands ------------------------------------------------------------------------------


def test_no_command_prints_the_data_dir_and_help(tmp_path: Path, capsys) -> None:
    assert cli.main(["--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"Data directory: {tmp_path.resolve()}")
    assert "record" in out and "sessions" in out and "recover" in out


def test_record_creates_a_recorded_session(db, data_dir: Path) -> None:
    code, out = record(db, data_dir, "--title", "Lecture 1", "--course", "Calculus")
    assert code == 0
    [session] = repo.list_session_summaries(db)
    assert (session["title"], session["course"], session["status"]) == (
        "Lecture 1", "Calculus", "recorded"
    )
    assert session["segments"] == 1 and session["duration_ms"] > 1000
    assert f"Session {session['id']}" in out
    assert str(session_dir(data_dir, session["id"])) in out
    assert "capture pid" in out and "Recording from Fake Speakers" in out
    assert "Stops by itself after 30 min without sound." in out
    assert f"Session {session['id']}: recorded, 0:00:01 in 1 segment(s)" in out


def test_record_without_silence_limit(db, data_dir: Path) -> None:
    code, out = record(db, data_dir, "--title", "t", "--silence-limit", "0", seconds=0.5)
    assert code == 0
    assert "Stops by itself" not in out


def test_record_recovers_its_session_when_the_capture_process_dies(db, data_dir: Path) -> None:
    capture = dict(FAKE_CAPTURE, source_factory=functools.partial(DyingSource, die_after=30))
    code, out = record(db, data_dir, "--title", "doomed", capture=capture, seconds=20)
    assert code == 1
    assert "The capture process stopped abnormally: recovering the session." in out
    [session] = repo.list_session_summaries(db)
    assert session["status"] == "interrupted"
    assert session["duration_ms"] > 1000  # the audio written before the crash is kept
    assert f"Recovered {session['id']}: recording -> interrupted" in out


def test_record_recovers_abandoned_sessions_first(db, data_dir: Path) -> None:
    old = abandoned_session(db, data_dir, "old")
    code, out = record(db, data_dir, "--title", "new", seconds=0.5)
    assert code == 0
    assert f"Recovered {old}: recording -> failed" in out
    assert out.index(f"Recovered {old}") < out.index("capture pid")


def test_sessions_lists_after_recovering(db, data_dir: Path, start_session) -> None:
    live = start_session("still recording", course="Physics")
    old = abandoned_session(db, data_dir, "abandoned")
    out = io.StringIO()
    assert cli.cmd_sessions(db, data_dir, out) == 0
    text = out.getvalue()
    assert f"Recovered {old}: recording -> failed" in text
    lines = text.splitlines()
    assert lines[1].split() == ["SESSION", "STATUS", "STARTED", "DURATION", "GAP", "COURSE",
                                "TITLE"]
    live_line = next(line for line in lines if line.startswith(live))
    assert "recording" in live_line and "Physics" in live_line
    assert lines[-1] == "2 session(s)"


def test_sessions_when_there_are_none(tmp_path: Path, capsys) -> None:
    assert cli.main(["--data-dir", str(tmp_path), "sessions"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "No sessions yet."


def test_recover_command(tmp_path: Path, capsys) -> None:
    assert cli.main(["--data-dir", str(tmp_path), "recover"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "Recovery: nothing to do."


# --- live status line ------------------------------------------------------------------------


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_status_line_redraws_in_place_on_a_terminal() -> None:
    now = [100.0]
    out = Terminal()
    status = cli.StatusLine(out, clock=lambda: now[0])
    status.show({"type": "started", "device": "Speakers", "rate": 48_000, "channels": 2})
    now[0] = 100.0 + 3725  # 1 h 2 min 5 s later
    status.show({"type": "segment", "seq": 1, "start_ms": 0, "end_ms": 60_000, "gap_ms": 400})
    status.show({"type": "level", "rms_db": -30.0, "peak_db": -12.0})
    text = out.getvalue()
    live = "REC 1:02:05  [##########----------]  -30.0 dBFS  segment 2, gap 0.4 s"
    assert "Recording from Speakers, 48000 Hz, 2 ch\n" in text
    assert text.endswith("\r" + live)
    status.show({"type": "warning", "message": "something"})
    # The live line is blanked before the message, then drawn again below it.
    assert "\r" + " " * len(live) + "\r[" in out.getvalue()
    assert out.getvalue().endswith("Warning: something\n\r" + live)


def test_status_line_prints_only_events_when_not_a_terminal() -> None:
    out = io.StringIO()
    status = cli.StatusLine(out)
    status.show({"type": "started", "device": "Speakers", "rate": 48_000, "channels": 2})
    status.show({"type": "level", "rms_db": -30.0, "peak_db": -12.0})
    status.show({"type": "device_lost", "reason": "stream stopped", "device": "Speakers"})
    status.show({"type": "device_opened", "device": "Monitor", "rate": 48_000, "channels": 2,
                 "gap_ms": 174, "attempts": 1})
    lines = out.getvalue().splitlines()
    assert "\r" not in out.getvalue()
    assert lines[1].endswith("Device lost (stream stopped): Speakers")
    assert lines[2].endswith("Switched to Monitor, gap 174 ms")


def test_device_messages_say_again_only_for_the_same_device() -> None:
    out = io.StringIO()
    status = cli.StatusLine(out)
    opened = {"type": "device_opened", "rate": 48_000, "channels": 2, "attempts": 1}
    status.show({"type": "started", "device": "Speakers", "rate": 48_000, "channels": 2})
    status.show({**opened, "device": "Monitor", "gap_ms": 174})  # unplugged
    status.show({**opened, "device": "Speakers", "gap_ms": 487})  # replugged: back, but a switch
    status.show({**opened, "device": "Speakers", "gap_ms": 40, "attempts": 3})  # same device
    lines = out.getvalue().splitlines()
    assert lines[1].endswith("Switched to Monitor, gap 174 ms")
    assert lines[2].endswith("Switched to Speakers, gap 487 ms")
    assert lines[3].endswith("Recording from Speakers again, gap 40 ms after 3 attempts")


# --- formatting ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ms", "text"), [(0, "0:00:00"), (59_999, "0:00:59"), (90_251, "0:01:30"),
                     (3 * 3600_000 + 5_000, "3:00:05")]
)
def test_format_duration(ms: int, text: str) -> None:
    assert cli.format_duration(ms) == text


@pytest.mark.parametrize(
    ("db", "bar"), [(-96.0, "-" * 20), (-60.0, "-" * 20), (-30.0, "#" * 10 + "-" * 10),
                    (0.0, "#" * 20), (3.0, "#" * 20)]
)
def test_level_bar(db: float, bar: str) -> None:
    assert cli.level_bar(db) == bar
