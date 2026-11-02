"""Level normalisation of the copy of the audio that goes to the VAD and the ASR.

The loopback records after the device volume: the same lecture measured about -20 dBFS on the
headphone output and -40 dBFS on the monitor. A slowly varying gain brings speech towards
TARGET_DB without clipping. The recorded WAV files are never touched: this works on a float
copy.
"""

from __future__ import annotations

import numpy as np

from studyscribe.config import SAMPLE_RATE

FRAME = SAMPLE_RATE // 20  # 50 ms: frame level resolution
BLOCK = 10 * SAMPLE_RATE  # one gain per 10 s block, interpolated between block centres
# A block's level is measured over ±10 s around it. Wider follows a level drop late: the
# percentile stays on the louder side (±30 s left 30 s after a 20 dB drop unboosted).
NEIGHBOURHOOD = 10 * SAMPLE_RATE
LEVEL_PERCENTILE = 90
FLOOR_DB = -70.0  # frames quieter than this do not count towards the level
QUIET_LEVEL_DB = -60.0  # a level below this is noise, not quiet speech: left as it is
TARGET_DB = -20.0
MAX_GAIN_DB = 30.0
PEAK_DB = -1.0  # the gain never pushes a sample above this (it never attenuates either)


def to_float(pcm: np.ndarray) -> np.ndarray:
    """int16 -> float32 in [-1, 1)."""
    return pcm.astype(np.float32) / 32768.0


def _db(values: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(values, 1e-12))


def block_gains(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(block centres in samples, gain in dB of each block) for float audio."""
    n = audio.size
    n_frames = -(-n // FRAME)
    padded = np.zeros(n_frames * FRAME, dtype=np.float64)
    padded[:n] = audio
    frame_db = _db(np.sqrt(np.mean(padded.reshape(n_frames, FRAME) ** 2, axis=1)))

    starts = np.arange(0, n, BLOCK)
    ends = np.minimum(starts + BLOCK, n)
    gains, limits = [], []
    for start, end in zip(starts, ends):
        lo = max(start - NEIGHBOURHOOD, 0) // FRAME
        hi = -(-min(end + NEIGHBOURHOOD, n) // FRAME)
        counted = frame_db[lo:hi][frame_db[lo:hi] > FLOOR_DB]
        level = np.percentile(counted, LEVEL_PERCENTILE) if counted.size else -np.inf
        gain = 0.0 if level < QUIET_LEVEL_DB else float(np.clip(TARGET_DB - level, 0, MAX_GAIN_DB))
        peak = float(np.max(np.abs(audio[start:end])))
        gains.append(gain)
        limits.append(PEAK_DB - _db(np.array([peak]))[0])
    gains_arr = np.array(gains)
    limits_arr = np.array(limits)
    # A sample between two block centres gets a gain between theirs: capping each block by its
    # neighbours' limits too keeps every interpolated gain under the limit of its own block.
    capped = limits_arr.copy()
    capped[1:] = np.minimum(capped[1:], limits_arr[:-1])
    capped[:-1] = np.minimum(capped[:-1], limits_arr[1:])
    gains_arr = np.maximum(np.minimum(gains_arr, capped), 0.0)
    return (starts + ends) / 2.0, gains_arr


def normalize(pcm: np.ndarray) -> np.ndarray:
    """Float32 copy of int16 `pcm` with speech brought towards TARGET_DB; the gain never pushes
    a sample above PEAK_DB."""
    audio = to_float(pcm)
    if audio.size == 0:
        return audio
    centres, gains = block_gains(audio)
    if not gains.any():
        return audio
    gain_db = np.interp(np.arange(audio.size), centres, gains)
    return (audio * np.power(10.0, gain_db / 20.0)).astype(np.float32)
