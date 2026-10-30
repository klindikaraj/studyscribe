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


# --- steps of a manual test ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("delay", "starts", "ok"),
    [
        (0.20, True, True),  # usual browser + audio latency after the key
        (0.95, True, True),
        (-0.10, True, True),
        (-1.10, True, False),  # sound before the key: timeline shifted early (initial silence lost)
        (1.40, True, False),  # timeline shifted late
        (0.20, False, True),
        (-0.80, False, True),  # paused between two words
        (0.90, False, False),
    ],
)
def test_media_key_step_timing(delay: float, starts: bool, ok: bool) -> None:
    step = verify.Step(prompt=60.0, done=60.0, by_key=True)
    result, detail = verify.step_timing(60.0 + delay, step, starts=starts)
    assert result is ok
    assert f"({delay:+.2f} s)" in detail


def test_manual_step_must_fall_between_prompt_and_enter() -> None:
    # The user's guided run of 2026-10-30: the action came first, Enter a few seconds later.
    play = verify.Step(prompt=5.0, done=13.17, by_key=False)
    pause = verify.Step(prompt=60.0, done=66.35, by_key=False)
    assert verify.step_timing(8.15, play, starts=True)[0]
    assert verify.step_timing(63.95, pause, starts=False)[0]
    assert not verify.step_timing(3.9, play, starts=True)[0]  # before it was asked for
    assert not verify.step_timing(14.2, play, starts=True)[0]  # after Enter
    assert verify.step_timing(None, play, starts=True) == (False, "no sound change found")


def events_of_a_test_run() -> list[dict]:
    """Device events as the user's run produced them: unplug at 12:56:37, replug at 12:56:56."""
    t = 1_793_300_000.0
    return [
        {"type": "started", "time": t, "device": "Speakers"},
        {"type": "device_lost", "time": t + 120.0, "device": "Speakers",
         "reason": "stream stopped"},
        {"type": "device_opened", "time": t + 120.1, "device": "Monitor", "gap_ms": 121},
        {"type": "device_lost", "time": t + 139.0, "device": "Monitor",
         "reason": "default output changed"},
        {"type": "device_opened", "time": t + 139.1, "device": "Speakers", "gap_ms": 53},
    ]


def test_device_switch_done_before_enter_is_found() -> None:
    events = events_of_a_test_run()
    t = events[0]["time"]
    # Unplug asked at 100 s, done at 120 s, Enter at 126 s: the switch precedes Enter.
    lost, opened = verify.device_switch(events, t + 100.0, t + 126.0 + 5.0)
    assert (lost["reason"], opened["device"]) == ("stream stopped", "Monitor")
    # Replug asked at 126 s (right after the previous Enter), Enter at 142 s.
    lost, opened = verify.device_switch(events, t + 126.0, t + 142.0 + 5.0)
    assert (lost["reason"], opened["device"]) == ("default output changed", "Speakers")


def test_device_switch_outside_the_interval_is_not_found() -> None:
    events = events_of_a_test_run()
    t = events[0]["time"]
    assert verify.device_switch(events, t + 121.0, t + 130.0) is None  # loss before the prompt
    assert verify.device_switch(events, t + 100.0, t + 110.0) is None  # nothing in time
    # A loss whose reopen comes after the interval is not a complete switch.
    assert verify.device_switch(events, t + 135.0, t + 139.05) is None
