"""Checks on a recorded session: manifest against files, and where the sound is on the timeline.

Used by scripts/manual_recording_test.py to verify a real recording after the fact.
"""

from __future__ import annotations

import re
import sqlite3
import struct
import wave
from pathlib import Path

import numpy as np

from studyscribe.audio.dsp import LEVEL_FLOOR_DB
from studyscribe.audio.wav import HEADER_SIZE
from studyscribe.config import AUDIO_SUBDIR, SAMPLE_RATE, SEGMENT_FRAMES, session_dir
from studyscribe.db import repo

WINDOW_S = 0.05  # level resolution
WINDOW_FRAMES = round(WINDOW_S * SAMPLE_RATE)
SOUND_DB = -50.0  # a window this loud counts as sound (lecture speech is around -20 to -40)
DIGITAL_SILENCE_DB = -95.0  # all zeros, or within a sample step of them

_SEGMENT_NAME = re.compile(r"\d{6}\.wav")


def manifest_problems(
    conn: sqlite3.Connection,
    data_dir: Path,
    session_id: str,
    segment_frames: int = SEGMENT_FRAMES,
) -> list[str]:
    """What is wrong with the session's segments; empty if the manifest and the files agree.

    Checks: numbering 1..N, segment N starting at (N-1) full segments, every segment but the
    last full, header sizes matching the file size, row durations matching the files, no file
    missing from the manifest.
    """
    folder = session_dir(data_dir, session_id)
    rows = repo.list_audio_segments(conn, session_id)
    problems = []
    if not rows:
        problems.append("no segments in the manifest")
    if [row["seq"] for row in rows] != list(range(1, len(rows) + 1)):
        problems.append(f"segment numbers are not 1..{len(rows)}")
    for i, row in enumerate(rows):
        seq, path = row["seq"], folder / row["path"]
        expected_start = (seq - 1) * segment_frames * 1000 // SAMPLE_RATE
        if row["start_ms"] != expected_start:
            problems.append(f"segment {seq} starts at {row['start_ms']} ms, not {expected_start}")
        if not path.is_file():
            problems.append(f"segment {seq}: file {row['path']} is missing")
            continue
        size = path.stat().st_size
        with open(path, "rb") as f:
            header = f.read(HEADER_SIZE)
        data_bytes = size - HEADER_SIZE
        if len(header) < HEADER_SIZE or struct.unpack_from("<I", header, 40)[0] != data_bytes:
            problems.append(f"segment {seq}: header size does not match the file")
        frames = data_bytes // 2
        if frames * 1000 // SAMPLE_RATE != row["end_ms"] - row["start_ms"]:
            problems.append(
                f"segment {seq}: file holds {frames * 1000 // SAMPLE_RATE} ms, manifest says "
                f"{row['end_ms'] - row['start_ms']} ms"
            )
        if i < len(rows) - 1 and frames != segment_frames:
            problems.append(f"segment {seq} is not full ({frames} samples) but is not the last")
    registered = {Path(row["path"]).name for row in rows}
    audio_dir = folder / AUDIO_SUBDIR
    if audio_dir.is_dir():
        for path in sorted(audio_dir.iterdir()):
            if _SEGMENT_NAME.fullmatch(path.name) and path.name not in registered:
                problems.append(f"{path.name} is not in the manifest")
    return problems


def sound_levels(conn: sqlite3.Connection, data_dir: Path, session_id: str) -> np.ndarray:
    """RMS level (dBFS) of every WINDOW_S window of the timeline, read segment by segment."""
    folder = session_dir(data_dir, session_id)
    levels = []
    for row in repo.list_audio_segments(conn, session_id):
        with wave.open(str(folder / row["path"])) as w:
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        n = len(samples) // WINDOW_FRAMES * WINDOW_FRAMES
        windows = samples[:n].reshape(-1, WINDOW_FRAMES).astype(np.float64) / 32768.0
        rms = np.sqrt(np.mean(windows * windows, axis=1))
        with np.errstate(divide="ignore"):
            levels.append(np.maximum(20.0 * np.log10(rms), LEVEL_FLOOR_DB))
    return np.concatenate(levels) if levels else np.zeros(0)


def first_sound(levels: np.ndarray, after_s: float = 0.0, threshold: float = SOUND_DB):
    """Start (s) of the first window at or after `after_s` louder than `threshold`, or None."""
    start = max(int(np.ceil(after_s / WINDOW_S)), 0)
    loud = np.flatnonzero(levels[start:] > threshold)
    return None if loud.size == 0 else (start + int(loud[0])) * WINDOW_S


def last_sound(
    levels: np.ndarray, before_s: float, after_s: float = 0.0, threshold: float = SOUND_DB
):
    """End (s) of the last window louder than `threshold` within [after_s, before_s), or None."""
    start = max(int(np.ceil(after_s / WINDOW_S)), 0)
    stop = min(int(before_s / WINDOW_S), len(levels))
    loud = np.flatnonzero(levels[start:stop] > threshold)
    return None if loud.size == 0 else (start + int(loud[-1]) + 1) * WINDOW_S


def short_digital_silences(levels: np.ndarray, max_s: float = 0.3) -> int:
    """Runs of digital silence no longer than `max_s` with sound right before and after.

    Real lecture audio rarely contains exact zeros in the middle of speech, so these are
    worth a look: silence inserted by the timeline (e.g. a device clock slower than the
    system clock) or a video that is digitally silent between sentences.
    """
    silent = levels <= DIGITAL_SILENCE_DB
    loud = levels > SOUND_DB
    count, run, max_run = 0, 0, round(max_s / WINDOW_S)
    for i in range(1, len(levels)):
        if silent[i]:
            run += 1
            continue
        if run and run <= max_run and loud[i] and loud[i - run - 1]:
            count += 1
        run = 0
    return count
