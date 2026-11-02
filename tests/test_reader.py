import hashlib
from pathlib import Path

import numpy as np
import pytest

from studyscribe.audio.reader import SessionAudio, SessionAudioError
from studyscribe.audio.wav import wav_header

FRAMES = 8_000  # 0.5 s segments


def write_segments(folder: Path, timeline: np.ndarray, frames: int = FRAMES) -> list:
    folder.mkdir(parents=True, exist_ok=True)
    segments = []
    for i, start in enumerate(range(0, timeline.size, frames)):
        chunk = timeline[start : start + frames].astype("<i2")
        name = f"{i + 1:06d}.wav"
        (folder / name).write_bytes(wav_header(chunk.size * 2) + chunk.tobytes())
        segments.append((i + 1, name))
    return segments


def test_reads_across_file_boundaries(tmp_path: Path) -> None:
    timeline = np.arange(20_500, dtype=np.int64) % 30_000  # sample value = its position
    segments = write_segments(tmp_path, timeline.astype(np.int16))
    audio = SessionAudio(tmp_path, segments, segment_frames=FRAMES)
    assert audio.total_frames == 20_500
    assert np.array_equal(audio.read(7_990, 30), timeline[7_990:8_020])  # 1 -> 2
    assert np.array_equal(audio.read(0, 20_500), timeline)
    assert np.array_equal(audio.read(20_400, 1_000), timeline[20_400:])  # partial last file
    assert audio.read(25_000, 10).size == 0
    assert audio.read(5, 0).size == 0


def test_files_are_not_modified(tmp_path: Path) -> None:
    segments = write_segments(tmp_path, np.ones(20_000, dtype=np.int16))
    before = {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
              for p in tmp_path.iterdir()}
    audio = SessionAudio(tmp_path, segments, segment_frames=FRAMES)
    audio.read(0, audio.total_frames)
    after = {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
             for p in tmp_path.iterdir()}
    assert after == before


def test_holes_and_short_segments_are_refused(tmp_path: Path) -> None:
    segments = write_segments(tmp_path, np.ones(20_000, dtype=np.int16))
    with pytest.raises(SessionAudioError, match="not 1..2"):
        SessionAudio(tmp_path, [segments[0], segments[2]], segment_frames=FRAMES)
    (tmp_path / "000001.wav").write_bytes(wav_header(200) + b"\x00" * 200)
    with pytest.raises(SessionAudioError, match="holds 100 samples"):
        SessionAudio(tmp_path, segments, segment_frames=FRAMES)
    (tmp_path / "000002.wav").unlink()
    with pytest.raises(SessionAudioError, match="cannot read"):
        SessionAudio(tmp_path, [(1, "000002.wav")], segment_frames=FRAMES)
