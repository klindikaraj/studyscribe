"""Where to end a transcription window: in a pause, never at a file boundary (pure)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from studyscribe.config import SAMPLE_RATE

MAX_WINDOW_S = 300.0  # a window is at most this long
SEARCH_S = 30.0  # the cut is searched in the last SEARCH_S of a window
MIN_PAUSE_S = 0.2  # a pause shorter than this is not trusted as a cut point
QUIET_S = 0.5  # fallback: the middle of the quietest stretch this long


@dataclass(frozen=True)
class Cut:
    frame: int  # absolute sample where the window ends and the next one starts
    in_pause: bool  # False: no pause found, cut in the quietest stretch instead


def choose_cut(
    region: np.ndarray, region_start: int, speech: list[tuple[int, int]]
) -> Cut:
    """Cut point inside `region` (audio starting at absolute sample `region_start`).

    `speech`: speech spans relative to the region. The cut is the middle of the longest
    non-speech gap (the region's edges count as gap ends); without a gap of MIN_PAUSE_S, the
    middle of the quietest QUIET_S of the region.
    """
    n = region.size
    best_start, best_len = 0, -1
    previous_end = 0
    for start, end in sorted(speech) + [(n, n)]:
        start, end = max(start, 0), min(end, n)
        if start - previous_end > best_len:
            best_start, best_len = previous_end, start - previous_end
        previous_end = max(previous_end, end)
    if best_len >= MIN_PAUSE_S * SAMPLE_RATE:
        return Cut(region_start + best_start + best_len // 2, True)
    width = min(int(QUIET_S * SAMPLE_RATE), n)
    if width == 0:
        return Cut(region_start + n, False)
    energy = np.concatenate(([0.0], np.cumsum(region.astype(np.float64) ** 2)))
    sums = energy[width:] - energy[:-width]
    quietest = int(np.argmin(sums))
    return Cut(region_start + quietest + width // 2, False)
