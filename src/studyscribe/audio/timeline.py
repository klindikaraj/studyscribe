"""Keep the recorded timeline aligned to real time.

The loopback delivers packets only while something plays, and nothing while the stream is
reopened after a device change. Written as received, the audio would be shorter than the
lecture and every timestamp after a pause would be early. TimelineAligner turns the packets
into a timeline where sample n is n / 16000 s after t0:

- while audio flows, silence is inserted only if the timeline falls more than
  FILL_THRESHOLD_S behind the clock, so ordinary jitter never moves anything;
- after IDLE_FILL_AFTER_S without packets, silence is added up to now - IDLE_FILL_MARGIN_S,
  so files keep growing during pauses without writing into the future;
- the first packet after silence (start of the recording, a pause, a device switch) is placed
  exactly where the clock says;
- real audio is never dropped;
- after SILENCE_LIMIT_S without sound, `limit_reached` turns True and the timeline is never
  filled past last sound + limit, even across a single huge clock jump (waking from sleep).

Silence written inside a device gap (open_gap / close_gap) is marked as synthetic. Pure: all
times are passed in, so tests drive it with made-up clocks and packets.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from studyscribe.audio.dsp import (
    StreamingResampler,
    downmix_to_mono,
    float_to_int16,
    frames_from_bytes,
)
from studyscribe.config import (
    FILL_THRESHOLD_S,
    IDLE_FILL_AFTER_S,
    IDLE_FILL_MARGIN_S,
    SAMPLE_RATE,
    SILENCE_LIMIT_S,
    SOUND_FLOOR_DB,
)

# (time.perf_counter() at arrival, interleaved float32 frames, PortAudio status flags)
Packet = tuple[float, bytes, int]
# write(pcm, gap): append int16 samples to the timeline; gap=True marks them as synthetic.
Write = Callable[[np.ndarray, bool], None]

_FLOAT32_BYTES = 4


def block_starts(packets: Sequence[Packet], rate: int, channels: int) -> list[float]:
    """Estimate when the audio of each packet started.

    A packet's audio ends no later than its arrival and no later than the start of the next
    packet. Usually the arrival is the tighter bound; after a hiccup, packets that waited in
    the driver arrive together, and the later ones show that the earlier ones started sooner
    than their own arrival suggests. So such a burst is not mistaken for a hole.
    """
    starts = [0.0] * len(packets)
    end = math.inf
    for i in range(len(packets) - 1, -1, -1):
        arrival, data, _ = packets[i]
        end = min(end, arrival)
        starts[i] = end - len(data) / (_FLOAT32_BYTES * channels * rate)
        end = starts[i]
    return starts


class TimelineAligner:
    """Turns loopback packets into a gap-free mono 16 kHz timeline aligned to the clock.

    Call start_input() when a stream opens, add() with the packets of each drain, idle() on
    every loop tick, end_input() when the stream closes and finish() when the recording stops.
    Once `limit_reached` is True the recording should stop: add() ignores further packets.
    """

    def __init__(
        self,
        t0: float,
        write: Write,
        sample_rate: int = SAMPLE_RATE,
        fill_threshold_s: float = FILL_THRESHOLD_S,
        idle_fill_after_s: float = IDLE_FILL_AFTER_S,
        idle_fill_margin_s: float = IDLE_FILL_MARGIN_S,
        silence_limit_s: float | None = SILENCE_LIMIT_S,
        sound_floor_db: float = SOUND_FLOOR_DB,
    ) -> None:
        self.t0 = t0
        self.frames = 0  # samples on the timeline so far
        self.silence_frames = 0  # of which inserted silence
        self.gap_frames = 0  # of which inside a device gap
        # Arrival of the last packet with sound; t0, so a recording where nothing ever plays
        # stops too.
        self.last_sound = t0
        self.limit_reached = False
        self._write = write
        self._rate = sample_rate
        self._threshold = round(fill_threshold_s * sample_rate)
        self._idle_after = idle_fill_after_s
        self._margin = idle_fill_margin_s
        self._limit = silence_limit_s  # None: no limit
        self._sound_floor = 10.0 ** (sound_floor_db / 20.0)
        # t0, so that a recording that starts in silence is filled from its first second.
        self._last_packet = t0
        self._input: tuple[int, int] | None = None  # (rate, channels) of the open stream
        self._resampler: StreamingResampler | None = None
        self._pending = False  # the resampler holds samples that only flush() returns
        self._new_run = True  # the next packet follows silence: place it exactly
        self._gaps: list[list[float]] = []  # [start_frame, end_frame], end inf while open

    def start_input(self, rate: int, channels: int) -> None:
        """A stream opened (or reopened) with this rate and channel count."""
        self.end_input()
        self._input = (rate, channels)
        self._resampler = StreamingResampler(rate, self._rate)
        self._new_run = True

    def end_input(self) -> None:
        """The stream closed: write the audio still held by the resampler."""
        self._flush()
        self._input = None
        self._resampler = None

    def add(self, packets: Sequence[Packet]) -> None:
        """Append the packets of one drain, inserting silence first where the clock says so."""
        if not packets or self.limit_reached:
            return
        if self._input is None or self._resampler is None:
            raise RuntimeError("add() called without an open input")
        rate, channels = self._input
        for (arrival, data, _), start in zip(packets, block_starts(packets, rate, channels)):
            behind = self._frame_at(start) - self.frames
            if behind > (0 if self._new_run else self._threshold):
                self._flush()
                self._fill_to(self._frame_at(start))
                if self.limit_reached:
                    return  # the recording ends at the limit; later packets are not part of it
            frames = frames_from_bytes(data, channels)
            if frames.size and float(np.max(np.abs(frames))) > self._sound_floor:
                self.last_sound = max(self.last_sound, arrival)
            self._emit(float_to_int16(self._resampler.process(downmix_to_mono(frames))))
            self._pending = True
            self._new_run = False
            self._last_packet = max(self._last_packet, arrival)
        self._check_limit(self._last_packet)  # packets of digital silence keep arriving

    def idle(self, now: float) -> None:
        """Call on every loop tick, after add(): fills the timeline while no packets arrive."""
        if self.limit_reached or now - self._last_packet < self._idle_after:
            return
        self._flush()
        self._new_run = True
        self._fill_to(self._frame_at(now - self._margin))  # stops at the silence limit

    def open_gap(self, at: float) -> None:
        """From `at` on, inserted silence is synthetic: the device was lost."""
        self._gaps.append([self._frame_at(at), math.inf])

    def close_gap(self, at: float) -> None:
        """The device is back at `at`: silence after that is ordinary silence again."""
        if self._gaps and self._gaps[-1][1] == math.inf:
            self._gaps[-1][1] = self._frame_at(at)

    def finish(self, end: float) -> None:
        """The recording stopped at `end`: write the resampler's tail, then fill up to `end`
        (or up to the silence limit, if that comes first)."""
        self.end_input()
        self._fill_to(self._frame_at(end))

    def _frame_at(self, t: float) -> int:
        return round((t - self.t0) * self._rate)

    def _check_limit(self, t: float) -> None:
        if self._limit is not None and t - self.last_sound >= self._limit:
            self.limit_reached = True

    def _emit(self, pcm: np.ndarray) -> None:
        if pcm.size:
            self._write(pcm, False)
            self.frames += pcm.size

    def _flush(self) -> None:
        if self._pending and self._resampler is not None and self._input is not None:
            self._emit(float_to_int16(self._resampler.flush()))
            self._resampler = StreamingResampler(self._input[0], self._rate)
        self._pending = False

    def _fill_to(self, target: int) -> None:
        if self._limit is not None:
            limit = self._frame_at(self.last_sound + self._limit)
            if target >= limit:
                target = limit
                self.limit_reached = True
        # At most one second per write, so even a long jump of the clock never needs one
        # huge buffer.
        while self.frames < target:
            in_gap, change = self._gap_state(self.frames)
            n = int(min(target, change, self.frames + self._rate)) - self.frames
            self._write(np.zeros(n, dtype=np.int16), in_gap)
            self.frames += n
            self.silence_frames += n
            if in_gap:
                self.gap_frames += n
        self._gaps = [g for g in self._gaps if g[1] > self.frames]

    def _gap_state(self, frame: int) -> tuple[bool, float]:
        """Whether `frame` is inside a device gap, and the frame where that changes."""
        change = math.inf
        for start, end in self._gaps:
            if start <= frame < end:
                return True, end
            if frame < start:
                change = min(change, start)
        return False, change
