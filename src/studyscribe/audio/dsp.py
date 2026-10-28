"""Signal processing on the capture path: downmix, resample, int16 conversion, levels."""

from __future__ import annotations

import math

import numpy as np
import soxr

# Roughly the noise floor of 16-bit audio; used instead of -inf so levels stay JSON-friendly.
LEVEL_FLOOR_DB = -96.0


def frames_from_bytes(data: bytes, channels: int) -> np.ndarray:
    """Interleaved float32 bytes -> (frames, channels) array."""
    return np.frombuffer(data, dtype=np.float32).reshape(-1, channels)


def downmix_to_mono(frames: np.ndarray) -> np.ndarray:
    """Average the channels of a (frames, channels) array into a 1-D float32 array.

    On 5.1/7.1 output the voice (centre channel) ends up quieter than in stereo; that is
    acceptable for transcription and not an issue with headphones or stereo speakers.
    """
    if frames.ndim == 1:
        return np.ascontiguousarray(frames, dtype=np.float32)
    if frames.shape[1] == 1:
        return np.ascontiguousarray(frames[:, 0], dtype=np.float32)
    return frames.mean(axis=1, dtype=np.float32)


def float_to_int16(samples: np.ndarray) -> np.ndarray:
    """Float samples in [-1, 1] -> int16, clipping anything outside the range."""
    clipped = np.clip(samples, -1.0, 1.0)
    return np.round(clipped * 32767.0).astype(np.int16)


class StreamingResampler:
    """Mono float32 resampler that keeps its filter state across chunks.

    The filter delays the output by a few milliseconds; `flush()` returns what is left, so
    the total output for n input samples is round(n * out_rate / in_rate).
    """

    def __init__(self, in_rate: int, out_rate: int, quality: str = "HQ") -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._stream = (
            None
            if in_rate == out_rate
            else soxr.ResampleStream(in_rate, out_rate, 1, dtype="float32", quality=quality)
        )

    def process(self, samples: np.ndarray) -> np.ndarray:
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if self._stream is None:
            return samples
        return self._stream.resample_chunk(samples)

    def flush(self) -> np.ndarray:
        """Return the samples still held by the filter. Do not call `process` afterwards."""
        empty = np.zeros(0, dtype=np.float32)
        if self._stream is None:
            return empty
        return self._stream.resample_chunk(empty, last=True)


def to_db(amplitude: float) -> float:
    """Linear amplitude (1.0 = full scale) -> dBFS, floored at LEVEL_FLOOR_DB."""
    if amplitude <= 0.0:
        return LEVEL_FLOOR_DB
    return max(20.0 * math.log10(amplitude), LEVEL_FLOOR_DB)


class LevelMeter:
    """Accumulates int16 samples and reports RMS and peak in dBFS since the last report."""

    def __init__(self) -> None:
        self._sum_sq = 0.0
        self._count = 0
        self._peak = 0

    def add(self, pcm: np.ndarray) -> None:
        if pcm.size == 0:
            return
        x = pcm.astype(np.float64)
        self._sum_sq += float(np.dot(x, x))
        self._count += pcm.size
        self._peak = max(self._peak, int(np.max(np.abs(x))))

    def take(self) -> tuple[float, float]:
        """Return (rms_db, peak_db) and start a new window."""
        rms = math.sqrt(self._sum_sq / self._count) / 32768.0 if self._count else 0.0
        peak = self._peak / 32768.0
        self._sum_sq, self._count, self._peak = 0.0, 0, 0
        return to_db(rms), to_db(peak)
