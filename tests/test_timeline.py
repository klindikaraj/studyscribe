"""TimelineAligner, driven like the capture loop: packets delivered in batches every 50 ms.

Times start at t0 = 0, so frame = seconds x 16000. Most tests feed 16 kHz mono packets, which
pass through without resampling: each packet then becomes exactly one write.
"""

import numpy as np
import pytest

from studyscribe.audio.timeline import TimelineAligner, block_starts

RATE = 16_000
TICK = 0.05


class Out:
    """Records what the aligner writes: (first frame, samples, gap) per write."""

    def __init__(self) -> None:
        self.writes: list[tuple[int, np.ndarray, bool]] = []
        self.frames = 0

    def __call__(self, pcm: np.ndarray, gap: bool) -> None:
        self.writes.append((self.frames, pcm.copy(), gap))
        self.frames += len(pcm)

    @property
    def samples(self) -> np.ndarray:
        return np.concatenate([w[1] for w in self.writes])

    def audio_starts(self) -> list[int]:
        """First frame of every write of real audio (silence is all zeros)."""
        return [start for start, pcm, _ in self.writes if pcm.any()]

    def gap_ranges(self) -> list[tuple[int, int]]:
        return [(start, start + len(pcm)) for start, pcm, gap in self.writes if gap]


def packet(start: float, duration: float = 0.01, rate: int = RATE, channels: int = 1,
           late: float = 0.0, value: float = 0.5):
    """A packet whose audio starts at `start`; it arrives when its audio ends, plus `late`."""
    data = np.full(round(duration * rate) * channels, value, dtype=np.float32).tobytes()
    return (start + duration + late, data, 0)


def run(start: float, end: float, packet_s: float = 0.01) -> list:
    """Back-to-back packets covering [start, end)."""
    n = round((end - start) / packet_s)
    return [packet(start + k * packet_s, packet_s) for k in range(n)]


def make(rate: int = RATE, channels: int = 1):
    out = Out()
    aligner = TimelineAligner(0.0, out)
    aligner.start_input(rate, channels)
    return aligner, out


def drive(aligner: TimelineAligner, packets: list, start: float, end: float, on_tick=None):
    """Deliver `packets` like the capture loop: every TICK, add() what arrived, then idle()."""
    pending = sorted(packets, key=lambda p: p[0])
    i = 0
    for k in range(1, round((end - start) / TICK) + 1):
        now = start + k * TICK
        batch = []
        while i < len(pending) and pending[i][0] <= now + 1e-9:
            batch.append(pending[i])
            i += 1
        aligner.add(batch)
        aligner.idle(now)
        if on_tick:
            on_tick(now)
    assert i == len(pending), "drive() ended before every packet arrived"


# --- block start estimates -----------------------------------------------------------------


def test_block_starts_come_from_arrival_minus_duration() -> None:
    packets = [packet(1.0), packet(1.01), packet(1.5)]
    assert block_starts(packets, RATE, 1) == pytest.approx([1.0, 1.01, 1.5])


def test_block_starts_see_through_a_burst() -> None:
    # Three 10 ms packets held up by a hiccup arrive together at 1.03: they started at 1.0.
    data = packet(0.0)[1]
    packets = [(1.03, data, 0), (1.03, data, 0), (1.03, data, 0)]
    assert block_starts(packets, RATE, 1) == pytest.approx([1.0, 1.01, 1.02])


# --- continuous audio --------------------------------------------------------------------


def test_jitter_below_the_threshold_moves_nothing() -> None:
    rng = np.random.default_rng(3)
    packets, last = [], 0.0
    for k in range(300):
        start = k * 0.01
        late = 0.0 if k == 0 else rng.uniform(0, 0.03)
        last = max(last, start + 0.01 + late)  # a queue delivers in order
        packets.append((last, packet(start)[1], 0))
    aligner, out = make()
    drive(aligner, packets, 0.0, 3.1)
    assert aligner.silence_frames == 0
    assert out.frames == 3 * RATE


def test_short_hole_mid_stream_is_not_filled() -> None:
    aligner, out = make()
    drive(aligner, run(0.0, 1.0) + run(1.08, 2.0), 0.0, 2.05)
    assert aligner.silence_frames == 0


def test_hole_above_the_threshold_is_filled_exactly() -> None:
    aligner, out = make()
    drive(aligner, run(0.0, 1.0) + run(1.2, 2.0), 0.0, 2.05)
    assert aligner.silence_frames == round(0.2 * RATE)
    assert out.audio_starts()[100] == round(1.2 * RATE)


def test_a_burst_after_a_hiccup_is_not_a_hole() -> None:
    # 150 ms of packets held up by a hiccup, then delivered together: nothing was lost.
    packets = run(0.0, 1.0)
    data = packet(0.0)[1]
    packets += [(1.16 + k * 0.0005, data, 0) for k in range(15)]  # audio 1.00-1.15
    packets += [packet(1.15 + k * 0.01) for k in range(85)]  # audio 1.15-2.00
    packets.sort(key=lambda p: p[0])
    aligner, out = make()
    drive(aligner, packets, 0.0, 2.05)
    assert aligner.silence_frames == 0
    assert out.frames == 2 * RATE


# --- silence at the start, pauses ----------------------------------------------------------


def test_initial_silence_is_filled_from_t0() -> None:
    # Recording started, play pressed 1.1 s later: the timeline must start at t0, not at
    # the first packet.
    aligner, out = make()

    def check(now: float) -> None:
        if 0.5 <= now < 1.1:  # files grow before the first packet arrives
            assert aligner.frames == round((now - 0.1) * RATE)

    drive(aligner, run(1.1, 2.0), 0.0, 2.5, on_tick=check)
    aligner.finish(2.5)
    assert out.audio_starts()[0] == round(1.1 * RATE)
    assert not out.samples[: round(1.1 * RATE)].any()
    assert out.frames == round(2.5 * RATE)


def test_initial_silence_shorter_than_the_idle_delay_is_filled_too() -> None:
    aligner, out = make()
    drive(aligner, run(0.3, 1.0), 0.0, 1.0)
    assert out.audio_starts()[0] == round(0.3 * RATE)
    assert aligner.silence_frames == round(0.3 * RATE)


def test_audio_from_the_first_instant_needs_no_silence() -> None:
    aligner, out = make()
    drive(aligner, run(0.0, 1.0), 0.0, 1.0)
    assert out.audio_starts()[0] == 0
    assert aligner.silence_frames == 0


def test_pause_is_filled_without_writing_into_the_future() -> None:
    aligner, out = make()

    def check(now: float) -> None:
        if 2.5 <= now < 5.0:  # paused for at least the idle delay
            assert aligner.frames == round((now - 0.1) * RATE)

    drive(aligner, run(0.0, 2.0) + run(5.0, 6.0), 0.0, 6.5, on_tick=check)
    aligner.finish(6.5)
    starts = out.audio_starts()
    assert starts[199] == round(1.99 * RATE)  # last packet before the pause
    assert starts[200] == 5 * RATE  # first packet after it, exactly on time
    assert aligner.silence_frames == round(3.5 * RATE)  # the pause, then 6.0-6.5
    assert out.frames == round(6.5 * RATE)


def test_finish_pads_to_the_stop_time() -> None:
    aligner, out = make()
    drive(aligner, run(0.0, 1.0), 0.0, 1.3)  # stopped 300 ms into a pause
    aligner.finish(1.3)
    assert out.frames == round(1.3 * RATE)
    assert aligner.gap_frames == 0


# --- device gaps -----------------------------------------------------------------------------


def test_device_gap_is_marked_from_loss_to_reopen() -> None:
    aligner, out = make()
    drive(aligner, run(0.0, 2.0), 0.0, 2.0)
    aligner.end_input()
    aligner.open_gap(2.0)
    drive(aligner, [], 2.0, 2.5)  # retrying, nothing recorded
    aligner.close_gap(2.5)
    aligner.start_input(RATE, 1)
    drive(aligner, run(2.5, 3.0), 2.5, 3.0)
    aligner.finish(3.0)
    assert out.gap_ranges()[0][0] == 2 * RATE
    assert out.gap_ranges()[-1][1] == round(2.5 * RATE)
    assert aligner.gap_frames == aligner.silence_frames == round(0.5 * RATE)
    assert out.audio_starts()[200] == round(2.5 * RATE)


def test_real_silence_before_the_gap_is_not_counted() -> None:
    aligner, out = make()
    drive(aligner, run(0.0, 1.0), 0.0, 3.0)  # video paused at 1.0 s
    aligner.end_input()
    aligner.open_gap(3.0)  # headphones unplugged during the pause
    drive(aligner, [], 3.0, 3.3)
    aligner.close_gap(3.3)
    aligner.start_input(RATE, 1)
    drive(aligner, run(4.0, 4.5), 3.3, 4.5)  # video resumed at 4.0 s
    aligner.finish(4.5)
    assert aligner.gap_frames == round(0.3 * RATE)
    assert aligner.silence_frames == 3 * RATE
    assert out.audio_starts()[100] == 4 * RATE


def test_stop_during_a_gap_pads_with_gap() -> None:
    aligner, out = make()
    drive(aligner, run(0.0, 1.0), 0.0, 1.0)
    aligner.end_input()
    aligner.open_gap(1.0)
    drive(aligner, [], 1.0, 2.0)
    aligner.finish(2.0)
    assert out.frames == 2 * RATE
    assert aligner.gap_frames == RATE


# --- resampling ------------------------------------------------------------------------------


def test_resampler_tail_is_written_before_the_silence() -> None:
    aligner, out = make(rate=48_000, channels=2)
    packets = [packet(k * 0.01, rate=48_000, channels=2) for k in range(100)]
    packets += [packet(3.0 + k * 0.01, rate=48_000, channels=2) for k in range(100)]
    drive(aligner, packets, 0.0, 4.0)
    aligner.finish(4.0)
    # Both runs of audio arrive whole (1 s each), and the second starts exactly at 3.0 s.
    assert aligner.frames - aligner.silence_frames == 2 * RATE
    assert aligner.silence_frames == 2 * RATE
    loud = np.flatnonzero(np.abs(out.samples.astype(np.int32)) > 1000)
    assert loud[loud > 2 * RATE][0] == pytest.approx(3 * RATE, abs=2)


def test_new_stream_with_another_rate() -> None:
    aligner, out = make(rate=48_000, channels=2)
    drive(aligner, [packet(k * 0.01, rate=48_000, channels=2) for k in range(100)], 0.0, 1.0)
    aligner.end_input()
    aligner.start_input(44_100, 2)
    drive(aligner, [packet(1.0 + k * 0.01, rate=44_100, channels=2) for k in range(100)],
          1.0, 2.0)
    aligner.finish(2.0)
    assert aligner.frames == 2 * RATE
    assert aligner.silence_frames == 0


# --- long run ----------------------------------------------------------------------------------


def test_ten_minutes_with_random_pauses_stays_within_150_ms() -> None:
    rng = np.random.default_rng(7)
    packets, truth = [], []
    t, last = 0.0, 0.0
    while t < 600.0:
        n = int(rng.uniform(0.5, 60.0) / 0.01)  # play for 0.5-60 s
        for k in range(n):
            start = t + k * 0.01
            last = max(last, start + 0.01 + rng.uniform(0.002, 0.03))  # latency + jitter
            packets.append((last, packet(start)[1], 0))
            truth.append(start)
        hole = rng.uniform(0.02, 0.4) if rng.random() < 0.5 else rng.uniform(0.5, 8.0)
        t += n * 0.01 + hole
    end = round(last + 1.0, 1)
    aligner, out = make()
    drive(aligner, packets, 0.0, end)
    aligner.finish(end)
    placed = np.array(out.audio_starts()) / RATE
    assert len(placed) == len(truth)
    errors = placed - np.array(truth)
    assert np.max(np.abs(errors)) <= 0.15
    assert out.frames == round(end * RATE)
