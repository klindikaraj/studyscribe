"""Test doubles. Kept in a plain module so the capture process can unpickle them too."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np


class FakeSource:
    """Stereo (or n-channel) sine source: `frames_per_drain` frames on every drain() call."""

    device_name = "Fake Speakers [Loopback]"

    def __init__(
        self,
        rate: int = 48_000,
        channels: int = 2,
        frames_per_drain: int = 2_400,
        frequency: float = 440.0,
        amplitude: float = 0.5,
        fail_on_open: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.rate = rate
        self.channels = channels
        self.frames_per_drain = frames_per_drain
        self.frequency = frequency
        self.amplitude = amplitude
        self.fail_on_open = fail_on_open
        self.clock = clock
        self.opened = False
        self.close_calls = 0
        self.produced = 0

    def open(self) -> None:
        if self.fail_on_open:
            raise OSError("fake device unavailable")
        self.opened = True

    def drain(self) -> list[tuple[float, bytes, int]]:
        if not self.opened or self.close_calls:
            return []
        n = self.frames_per_drain
        t = (self.produced + np.arange(n)) / self.rate
        mono = (self.amplitude * np.sin(2 * np.pi * self.frequency * t)).astype(np.float32)
        frames = np.repeat(mono[:, None], self.channels, axis=1)
        self.produced += n
        return [(self.clock(), frames.tobytes(), 0)]

    def is_active(self) -> bool:
        return self.opened and not self.close_calls

    def close(self) -> list[str]:
        self.close_calls += 1
        return []


class ListSink:
    """Keeps every written chunk in memory."""

    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []
        self.closed = False

    def write(self, pcm: np.ndarray) -> None:
        self.chunks.append(pcm.copy())

    def close(self) -> None:
        self.closed = True

    @property
    def samples(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.int16)


class RawFileSink:
    """Appends raw int16 samples to a file, so a test can read what another process wrote."""

    def __init__(self, path: str) -> None:
        self._file = Path(path).open("ab")

    def write(self, pcm: np.ndarray) -> None:
        self._file.write(pcm.tobytes())

    def close(self) -> None:
        self._file.close()


class FakeClock:
    """Manual clock; `sleep` advances it instead of waiting."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
