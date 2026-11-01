"""The report of scripts/manual_recording_test.py --check, with and without a guided run."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from studyscribe.audio.wav import wav_header
from studyscribe.config import session_dir
from studyscribe.db import repo

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "manual_recording_test.py"


@pytest.fixture(scope="module")
def manual_test():
    spec = importlib.util.spec_from_file_location("manual_recording_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses look their module up in sys.modules while the module runs.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


@pytest.fixture
def recorded_session(db, data_dir: Path, start_session) -> str:
    """A finished session of one 2 s segment (a tone), with 120 ms of device gap."""
    sid = start_session("t")
    t = np.arange(2 * 16_000) / 16_000
    pcm = np.round(0.1 * 32767 * np.sin(2 * np.pi * 440 * t)).astype("<i2")
    path = session_dir(data_dir, sid) / "audio" / "000001.wav"
    path.write_bytes(wav_header(pcm.size * 2) + pcm.tobytes())
    repo.add_audio_segment(db, sid, 1, "audio/000001.wav", 0, 2000, 120)
    started = repo.get_session(db, sid)["started_at"]
    repo.finish_session(db, sid, "recorded", started)  # any end time will do
    return sid


def report_lines(manual_test, db, data_dir: Path, sid: str, notes: dict) -> list[str]:
    manual_test.report(db, data_dir, sid, notes)
    path = session_dir(data_dir, sid) / manual_test.REPORT_FILE
    return path.read_text(encoding="utf-8").splitlines()


def line_of(lines: list[str], name: str) -> str:
    return next(line for line in lines if name in line)


def test_without_guided_run_the_report_says_so(manual_test, db, data_dir, recorded_session):
    lines = report_lines(manual_test, db, data_dir, recorded_session, notes={})
    assert "no guided-run notes (manual-test.json)" in lines[0]
    assert "play/pause" not in lines[0]
    gaps = line_of(lines, "Device gaps in the manifest")
    assert "SKIP" in gaps and "manifest 120 ms" in gaps
    assert "no capture events recorded" in gaps
    assert "reported" not in gaps
    warnings = line_of(lines, "Warnings and errors from the capture")
    assert "not recorded" in warnings and "none" not in warnings


def test_guided_run_without_switches_reports_zero(manual_test, db, data_dir, recorded_session):
    notes = {"media_keys": True, "steps": {}, "events": [], "powercfg": None}
    lines = report_lines(manual_test, db, data_dir, recorded_session, notes)
    assert "guided run, play/pause media key" in lines[0]
    gaps = line_of(lines, "Device gaps in the manifest")
    assert "manifest 120 ms, reported 0 ms" in gaps
    assert line_of(lines, "Warnings and errors from the capture").endswith("none")


def test_notes_of_the_first_script_version_were_by_hand(manual_test, db, data_dir,
                                                        recorded_session):
    # The first version saved neither media_keys nor steps, only Enter times ("marks").
    notes = {"marks": {}, "events": []}
    lines = report_lines(manual_test, db, data_dir, recorded_session, notes)
    assert "guided run, play/pause by hand" in lines[0]
