"""Crash-tolerant WAV writing for mono int16 audio."""

from __future__ import annotations

import os
import struct
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from studyscribe.config import SAMPLE_RATE

HEADER_SIZE = 44


def wav_header(data_bytes: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Canonical 44-byte header for PCM int16 mono."""
    channels, bits = 1, 16
    block_align = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_bytes, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, sample_rate * block_align, block_align, bits,
        b"data", data_bytes,
    )


class WavWriter:
    """Writes samples straight to the OS and keeps the header roughly current.

    - Unbuffered: every write() reaches the OS at once, so a killed process loses nothing that
      was written.
    - The header sizes are patched every `header_interval_s`, so a file left behind by a crash
      is playable and at most that much audio is "hidden" until the header is repaired.
    - fsync every `fsync_interval_s` and on close bounds the loss on a power failure.
    Never overwrites an existing file.
    """

    def __init__(
        self,
        path: Path | str,
        sample_rate: int = SAMPLE_RATE,
        header_interval_s: float = 1.0,
        fsync_interval_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.frames = 0
        self._header_interval = header_interval_s
        self._fsync_interval = fsync_interval_s
        self._clock = clock
        self._file = open(self.path, "xb", buffering=0)
        self._write_all(wav_header(0, sample_rate))
        now = clock()
        self._next_header = now + header_interval_s
        self._next_fsync = now + fsync_interval_s

    @property
    def closed(self) -> bool:
        return self._file.closed

    def write(self, pcm: np.ndarray) -> None:
        if pcm.size == 0:
            return
        self._write_all(np.asarray(pcm, dtype="<i2").tobytes())
        self.frames += pcm.size
        now = self._clock()
        if now >= self._next_header:
            self._patch_header()
            self._next_header = now + self._header_interval
        if now >= self._next_fsync:
            os.fsync(self._file.fileno())
            self._next_fsync = now + self._fsync_interval

    def close(self) -> None:
        if self._file.closed:
            return
        try:
            self._patch_header()
            os.fsync(self._file.fileno())
        finally:
            self._file.close()

    def _write_all(self, data: bytes) -> None:
        # A raw (unbuffered) file may accept only part of the data in one call.
        view = memoryview(data)
        while view:
            written = self._file.write(view)
            view = view[written:]

    def _patch_header(self) -> None:
        data_bytes = self.frames * 2
        self._file.seek(4)
        self._file.write(struct.pack("<I", 36 + data_bytes))
        self._file.seek(40)
        self._file.write(struct.pack("<I", data_bytes))
        self._file.seek(0, os.SEEK_END)
