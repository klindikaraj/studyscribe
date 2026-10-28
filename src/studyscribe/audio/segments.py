"""Split the 16 kHz timeline into fixed-length WAV segments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from studyscribe.audio.wav import WavWriter
from studyscribe.config import SAMPLE_RATE, SEGMENT_FRAMES


@dataclass(frozen=True)
class ClosedSegment:
    seq: int
    path: Path
    start_ms: int
    end_ms: int
    gap_ms: int


def segment_filename(seq: int) -> str:
    return f"{seq:06d}.wav"


class SegmentWriter:
    """Writes the timeline into 000001.wav, 000002.wav, ... of exactly `segment_frames` samples.

    Segment N covers frames [(N-1)*segment_frames, N*segment_frames), so its position on the
    timeline follows from its number; only the last segment can be shorter. Files are created
    on the first sample they receive, so no empty segment is left behind. `on_close` is called
    after each segment is closed and synced.
    """

    def __init__(
        self,
        audio_dir: Path | str,
        on_close: Callable[[ClosedSegment], None],
        segment_frames: int = SEGMENT_FRAMES,
        sample_rate: int = SAMPLE_RATE,
        writer_factory: Callable[[Path, int], WavWriter] = WavWriter,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.total_frames = 0
        self._on_close = on_close
        self._segment_frames = segment_frames
        self._sample_rate = sample_rate
        self._writer_factory = writer_factory
        self._writer: WavWriter | None = None
        self._seq = 0
        self._segment_start = 0  # timeline frame where the open segment starts
        self._gap_frames = 0

    def write(self, pcm: np.ndarray, gap: bool = False) -> None:
        """Append samples; `gap=True` marks them as synthetic (device switch) for gap_ms."""
        offset = 0
        while offset < len(pcm):
            if self._writer is None:
                self._open_next()
            assert self._writer is not None
            room = self._segment_frames - self._writer.frames
            chunk = pcm[offset : offset + room]
            self._writer.write(chunk)
            offset += len(chunk)
            self.total_frames += len(chunk)
            if gap:
                self._gap_frames += len(chunk)
            if self._writer.frames == self._segment_frames:
                self._close_current()

    def close(self) -> None:
        if self._writer is not None:
            self._close_current()

    def _open_next(self) -> None:
        self._seq += 1
        self._writer = self._writer_factory(
            self.audio_dir / segment_filename(self._seq), self._sample_rate
        )
        self._gap_frames = 0

    def _close_current(self) -> None:
        writer, self._writer = self._writer, None
        assert writer is not None
        writer.close()
        start = self._segment_start
        end = start + writer.frames
        self._segment_start = end
        self._on_close(
            ClosedSegment(
                seq=self._seq,
                path=writer.path,
                start_ms=self._ms(start),
                end_ms=self._ms(end),
                gap_ms=self._ms(self._gap_frames),
            )
        )

    def _ms(self, frames: int) -> int:
        return frames * 1000 // self._sample_rate
