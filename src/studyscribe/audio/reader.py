"""The audio of a recorded session as one continuous signal, read across segment files."""

from __future__ import annotations

import sqlite3
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from studyscribe.config import SAMPLE_RATE, SEGMENT_FRAMES, session_dir
from studyscribe.db import repo


class SessionAudioError(ValueError):
    """The segments do not form a continuous timeline (missing file, hole, short segment)."""


class SessionAudio:
    """Read-only view of a session's segments as one int16 signal.

    Segment N starts at sample (N - 1) * segment_frames by construction, so every sample has a
    fixed position on the session timeline whatever file it is in. Files are opened read-only.
    """

    def __init__(
        self,
        folder: Path,
        segments: Sequence[tuple[int, str]],
        segment_frames: int = SEGMENT_FRAMES,
    ) -> None:
        """`segments`: (seq, path relative to `folder`) of every segment, 1..N."""
        self.segment_frames = segment_frames
        self._paths: list[Path] = []
        total = 0
        ordered = sorted(segments)
        for i, (seq, relative) in enumerate(ordered):
            if seq != i + 1:
                raise SessionAudioError(f"segment numbers are not 1..{len(ordered)}")
            path = folder / relative
            try:
                with wave.open(str(path), "rb") as w:
                    params = (w.getnchannels(), w.getsampwidth(), w.getframerate())
                    frames = w.getnframes()
            except (OSError, wave.Error, EOFError) as exc:
                raise SessionAudioError(f"segment {seq}: cannot read {relative}: {exc}") from exc
            if params != (1, 2, SAMPLE_RATE):
                raise SessionAudioError(f"segment {seq}: not mono 16-bit {SAMPLE_RATE} Hz")
            if i < len(ordered) - 1 and frames != segment_frames:
                raise SessionAudioError(
                    f"segment {seq} holds {frames} samples instead of {segment_frames}"
                )
            self._paths.append(path)
            total += frames
        self.total_frames = total

    @classmethod
    def from_session(
        cls,
        conn: sqlite3.Connection,
        data_dir: Path,
        session_id: str,
        segment_frames: int = SEGMENT_FRAMES,
    ) -> SessionAudio:
        rows = repo.list_audio_segments(conn, session_id)
        if not rows:
            raise SessionAudioError("the session has no segments")
        segments = [(row["seq"], row["path"]) for row in rows]
        return cls(session_dir(data_dir, session_id), segments, segment_frames)

    @property
    def duration_s(self) -> float:
        return self.total_frames / SAMPLE_RATE

    def read(self, start: int, count: int) -> np.ndarray:
        """Samples [start, start + count) of the timeline (fewer at the end), int16."""
        start = max(start, 0)
        stop = min(start + max(count, 0), self.total_frames)
        parts = []
        position = start
        while position < stop:
            index = position // self.segment_frames
            offset = position - index * self.segment_frames
            n = min(stop - position, self.segment_frames - offset)
            with wave.open(str(self._paths[index]), "rb") as w:
                w.setpos(offset)
                data = np.frombuffer(w.readframes(n), dtype="<i2")
            if data.size == 0:  # file shorter than its header said: nothing more to read
                break
            parts.append(data)
            position += data.size
        if not parts:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(parts).astype(np.int16, copy=False)
