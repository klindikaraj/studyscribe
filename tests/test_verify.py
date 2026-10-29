from pathlib import Path

import numpy as np
import pytest

from studyscribe import verify
from studyscribe.audio.wav import wav_header
from studyscribe.config import session_dir
from studyscribe.db import repo

FRAMES = 8_000  # 0.5 s segments


def add_segments(db, data_dir: Path, session_id: str, timeline: np.ndarray) -> None:
    """Cut `timeline` (int16) into FRAMES-long segments, write them and register them."""
    folder = session_dir(data_dir, session_id) / "audio"
    for i, start in enumerate(range(0, len(timeline), FRAMES)):
        chunk = timeline[start : start + FRAMES].astype("<i2")
        name = f"{i + 1:06d}.wav"
        (folder / name).write_bytes(wav_header(chunk.size * 2) + chunk.tobytes())
        repo.add_audio_segment(
            db, session_id, i + 1, f"audio/{name}", start * 1000 // 16_000,
            (start + chunk.size) * 1000 // 16_000, 0,
        )


def tone(seconds: float, amplitude: float = 0.1) -> np.ndarray:
    t = np.arange(round(seconds * 16_000)) / 16_000
    return np.round(amplitude * 32767 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * 16_000), dtype=np.int16)


def test_consistent_session_has_no_problems(db, data_dir: Path, start_session) -> None:
    sid = start_session("t")
    add_segments(db, data_dir, sid, np.concatenate([silence(0.7), tone(0.6)]))
    assert verify.manifest_problems(db, data_dir, sid, segment_frames=FRAMES) == []


def test_problems_are_found(db, data_dir: Path, start_session) -> None:
    sid = start_session("t")
    add_segments(db, data_dir, sid, tone(1.2))
    folder = session_dir(data_dir, sid) / "audio"
    stale = folder / "000003.wav"
    stale.write_bytes(wav_header(0) + stale.read_bytes()[44:])  # header left behind by a kill
    (folder / "000004.wav").write_bytes(wav_header(0))  # never registered
    problems = "\n".join(verify.manifest_problems(db, data_dir, sid, segment_frames=FRAMES))
    assert "segment 3: header size does not match the file" in problems
    assert "000004.wav is not in the manifest" in problems


def test_short_middle_segment_and_missing_file(db, data_dir: Path, start_session) -> None:
    sid = start_session("t")
    add_segments(db, data_dir, sid, tone(1.2))
    folder = session_dir(data_dir, sid) / "audio"
    (folder / "000001.wav").write_bytes(wav_header(200) + b"\x00" * 200)
    (folder / "000002.wav").unlink()
    problems = "\n".join(verify.manifest_problems(db, data_dir, sid, segment_frames=FRAMES))
    assert "segment 1 is not full (100 samples)" in problems
    assert "segment 2: file audio/000002.wav is missing" in problems


def test_sound_is_located_on_the_timeline(db, data_dir: Path, start_session) -> None:
    # 1.0 s silence, 0.5 s tone, 0.8 s silence, 0.4 s tone, across 0.5 s segments.
    sid = start_session("t")
    add_segments(db, data_dir, sid,
                 np.concatenate([silence(1.0), tone(0.5), silence(0.8), tone(0.4)]))
    levels = verify.sound_levels(db, data_dir, sid)
    assert len(levels) == round(2.7 / verify.WINDOW_S)
    assert verify.first_sound(levels) == pytest.approx(1.0)
    assert verify.last_sound(levels, before_s=2.0) == pytest.approx(1.5)
    assert verify.first_sound(levels, after_s=1.6) == pytest.approx(2.3)
    assert verify.last_sound(levels, before_s=0.9) is None


def test_short_digital_silences_inside_sound_are_counted(db, data_dir: Path, start_session) -> None:
    sid = start_session("t")
    timeline = np.concatenate([
        silence(1.0), tone(0.5), silence(0.1), tone(0.5),  # 100 ms hole: counted
        silence(1.0), tone(0.5),  # a pause: too long to count
    ])
    add_segments(db, data_dir, sid, timeline)
    levels = verify.sound_levels(db, data_dir, sid)
    assert verify.short_digital_silences(levels) == 1
