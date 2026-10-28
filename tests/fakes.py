"""Test doubles. Kept in a plain module so the capture process can unpickle them too."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
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
        self.gap_frames = 0
        self.started = False
        self.close_reason: str | None = None

    def start(self, emit) -> None:
        self.started = True

    def write(self, pcm: np.ndarray, gap: bool = False) -> None:
        self.chunks.append(pcm.copy())
        if gap:
            self.gap_frames += len(pcm)

    def close(self, reason: str) -> None:
        self.close_reason = reason

    @property
    def closed(self) -> bool:
        return self.close_reason is not None

    @property
    def samples(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.int16)

    def sound_starts(self, threshold: int = 1000) -> list[int]:
        """Frames where sound begins after silence (|sample| > threshold)."""
        loud = np.abs(self.samples.astype(np.int32)) > threshold
        edges = np.flatnonzero(loud[1:] & ~loud[:-1]) + 1
        return ([0] if loud.size and loud[0] else []) + edges.tolist()


class RawFileSink:
    """Appends raw int16 samples to a file, so a test can read what another process wrote."""

    def __init__(self, path: str) -> None:
        self._file = Path(path).open("ab")

    def start(self, emit) -> None:
        pass

    def write(self, pcm: np.ndarray, gap: bool = False) -> None:
        self._file.write(pcm.tobytes())

    def close(self, reason: str) -> None:
        self._file.close()


def wait_for(handle, kind: str, timeout: float = 20.0) -> list[dict]:
    """Read a RecorderHandle's events until one of type `kind` arrives; return all seen."""
    seen: list[dict] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        seen += handle.events(timeout=0.2)
        if any(e["type"] == kind for e in seen):
            return seen
        if not handle.is_alive():
            break
    raise AssertionError(f"no {kind!r} event; got {[e['type'] for e in seen]}")


class FakeClock:
    """Manual clock; `sleep` advances it instead of waiting.

    Rounded to the nanosecond, so 1000 + 40 x 0.05 equals 1002.0 and tests can compare times
    with plain literals.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now = round(self.now + seconds, 9)


class ClockedSource:
    """A loopback stream driven by a FakeClock: one packet every `packet_s` of clock time.

    Nothing arrives during the `silent` (start, end) windows (nothing playing). From `dies_at`
    on, the stream is inactive (device unplugged). Opening takes `open_s` of clock time. Like
    the real source, drain() after close() still returns what arrived before the close.
    """

    def __init__(
        self,
        clock: FakeClock,
        name: str = "Fake Speakers [Loopback]",
        rate: int = 48_000,
        channels: int = 2,
        packet_s: float = 0.01,
        level: float = 0.5,
        silent: Sequence[tuple[float, float]] = (),
        dies_at: float | None = None,
        fail_on_open: bool = False,
        open_s: float = 0.0,
    ) -> None:
        self.clock = clock
        self.device_name = name
        self.rate = rate
        self.channels = channels
        self.packet_s = packet_s
        self.silent = silent
        self.dies_at = math.inf if dies_at is None else dies_at
        self.fail_on_open = fail_on_open
        self.open_s = open_s
        self.opened = False
        self.close_calls = 0
        self.closed_at = math.inf
        n = round(rate * packet_s)
        self._block = np.full((n, channels), level, dtype=np.float32).tobytes()
        self._origin = 0.0
        self._count = 0

    def open(self) -> None:
        self.clock.sleep(self.open_s)
        if self.fail_on_open:
            raise OSError("fake device unavailable")
        self.opened = True
        self._origin = self.clock()

    def drain(self) -> list[tuple[float, bytes, int]]:
        if not self.opened:
            return []
        end = min(self.clock(), self.dies_at, self.closed_at)
        packets = []
        while True:
            start = round(self._origin + self._count * self.packet_s, 9)
            arrival = round(start + self.packet_s, 9)
            if arrival > end:
                return packets
            self._count += 1
            if not any(a <= start < b for a, b in self.silent):
                packets.append((arrival, self._block, 0))

    def is_active(self) -> bool:
        return self.opened and not self.close_calls and self.clock() < self.dies_at

    def close(self) -> list[str]:
        self.close_calls += 1
        self.closed_at = min(self.closed_at, self.clock())
        return []


class ScriptedDefaultOutput:
    """Default output whose endpoint ID changes at given clock times.

    `changes` is [(time, id), ...]; `fail_after` makes current_id() raise after that many
    calls (COM failing mid-recording).
    """

    def __init__(
        self,
        changes: Sequence[tuple[float, str | None]] = (),
        initial: str | None = "endpoint-1",
        clock: Callable[[], float] = time.perf_counter,
        fail_after: int | None = None,
    ) -> None:
        self.changes = sorted(changes, key=lambda change: change[0])
        self.initial = initial
        self.clock = clock
        self.fail_after = fail_after
        self.calls = 0
        self.closed = False

    def current_id(self) -> str | None:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise OSError("fake COM failure")
        current = self.initial
        for at, endpoint in self.changes:
            if self.clock() >= at:
                current = endpoint
        return current

    def close(self) -> None:
        self.closed = True
