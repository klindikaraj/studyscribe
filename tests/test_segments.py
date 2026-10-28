import wave
from pathlib import Path

import numpy as np
import pytest

from studyscribe.audio.segments import ClosedSegment, SegmentWriter, segment_filename
from studyscribe.config import SEGMENT_FRAMES


def frames_in(path: Path) -> int:
    with wave.open(str(path)) as w:
        return w.getnframes()


def make_writer(tmp_path: Path, segment_frames: int, sample_rate: int = 1000):
    closed: list[ClosedSegment] = []
    writer = SegmentWriter(
        tmp_path / "audio", closed.append, segment_frames=segment_frames, sample_rate=sample_rate
    )
    return writer, closed


def test_filenames() -> None:
    assert segment_filename(1) == "000001.wav"
    assert segment_filename(123456) == "123456.wav"


def test_rotates_at_exact_frame_count_whatever_the_chunk_sizes(tmp_path: Path) -> None:
    writer, closed = make_writer(tmp_path, segment_frames=500)  # 0.5 s at 1 kHz
    rng = np.random.default_rng(1)
    total = 0
    while total < 1_250:
        n = int(rng.integers(1, 180))
        writer.write(np.full(n, 7, dtype=np.int16))
        total += n
    writer.close()
    files = sorted((tmp_path / "audio").iterdir())
    assert [f.name for f in files] == ["000001.wav", "000002.wav", "000003.wav"]
    assert [frames_in(f) for f in files] == [500, 500, total - 1000]
    assert [s.seq for s in closed] == [1, 2, 3]
    assert writer.total_frames == total


def test_segments_are_contiguous_on_the_timeline(tmp_path: Path) -> None:
    writer, closed = make_writer(tmp_path, segment_frames=500)
    writer.write(np.zeros(1_234, dtype=np.int16))
    writer.close()
    assert [(s.start_ms, s.end_ms) for s in closed] == [(0, 500), (500, 1000), (1000, 1234)]


def test_gap_frames_are_split_across_a_boundary(tmp_path: Path) -> None:
    writer, closed = make_writer(tmp_path, segment_frames=500)
    writer.write(np.zeros(400, dtype=np.int16))
    writer.write(np.zeros(300, dtype=np.int16), gap=True)  # 100 in seg 1, 200 in seg 2
    writer.write(np.zeros(50, dtype=np.int16))
    writer.close()
    assert [s.gap_ms for s in closed] == [100, 200]


def test_no_file_until_the_first_sample(tmp_path: Path) -> None:
    writer, closed = make_writer(tmp_path, segment_frames=500)
    writer.write(np.zeros(500, dtype=np.int16))  # exactly one segment, closed at once
    assert len(closed) == 1
    writer.close()
    assert [f.name for f in (tmp_path / "audio").iterdir()] == ["000001.wav"]
    assert len(closed) == 1


def test_close_without_data_leaves_nothing(tmp_path: Path) -> None:
    writer, closed = make_writer(tmp_path, segment_frames=500)
    writer.close()
    assert closed == []
    assert list((tmp_path / "audio").iterdir()) == []


def test_on_close_runs_after_the_file_is_complete(tmp_path: Path) -> None:
    seen = []
    writer = SegmentWriter(
        tmp_path / "audio",
        lambda seg: seen.append(frames_in(seg.path)),
        segment_frames=300,
        sample_rate=1000,
    )
    writer.write(np.zeros(300, dtype=np.int16))
    assert seen == [300]
    writer.close()


@pytest.mark.parametrize("seconds", [150])
def test_real_segment_length_is_60_s(tmp_path: Path, seconds: int) -> None:
    closed: list[ClosedSegment] = []
    writer = SegmentWriter(tmp_path / "audio", closed.append)
    chunk = np.zeros(16_000, dtype=np.int16)
    for _ in range(seconds):
        writer.write(chunk)
    writer.close()
    assert SEGMENT_FRAMES == 960_000
    assert [(s.start_ms, s.end_ms) for s in closed] == [
        (0, 60_000),
        (60_000, 120_000),
        (120_000, 150_000),
    ]
