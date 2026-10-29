import functools
import itertools
import math
import time
from pathlib import Path

import numpy as np
import pytest
from fakes import (
    ClockedSource,
    FakeClock,
    FakeSource,
    ListSink,
    RawFileSink,
    ScriptedDefaultOutput,
    wait_for,
)

from studyscribe.audio.capture import CaptureConfig, run_capture
from studyscribe.recording import RecorderHandle


def stop_after(ticks: int):
    calls = itertools.count()
    return lambda: "requested" if next(calls) >= ticks else None


class Recorder:
    """Collects emitted events."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, kind: str, **fields) -> None:
        self.events.append({"type": kind, **fields})

    def of_type(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["type"] == kind]


def run_fake(source: FakeSource, ticks: int, clock: FakeClock):
    sink = ListSink()
    emit = Recorder()
    config = CaptureConfig(
        sink_factory=lambda: sink,
        source_factory=lambda: source,
        default_output_factory=lambda: ScriptedDefaultOutput(clock=clock),
    )
    reason = run_capture(config, stop_after(ticks), emit, clock=clock, sleep=clock.sleep)
    return reason, sink, emit


def run_clocked(clock: FakeClock, sources, ticks: int, default_output=None):
    """Run the loop on a FakeClock; `sources` are opened in order, one per (re)open."""
    sink = ListSink()
    emit = Recorder()
    pending = iter(sources)
    default_output = default_output or ScriptedDefaultOutput(clock=clock)
    config = CaptureConfig(
        sink_factory=lambda: sink,
        source_factory=lambda: next(pending),
        default_output_factory=lambda: default_output,
    )
    run_capture(config, stop_after(ticks), emit, clock=clock, sleep=clock.sleep)
    return sink, emit


# --- capture loop, in-process ------------------------------------------------------------


def test_one_second_of_48k_stereo_becomes_16000_samples() -> None:
    clock = FakeClock()
    source = FakeSource(rate=48_000, channels=2, frames_per_drain=2_400, clock=clock)
    reason, sink, emit = run_fake(source, ticks=20, clock=clock)  # 20 x 50 ms = 1 s
    assert reason == "requested"
    assert len(sink.samples) == 16_000
    assert sink.started and sink.close_reason == "requested"
    assert source.close_calls >= 1
    stopped = emit.of_type("stopped")[-1]
    assert stopped["audio_s"] == pytest.approx(1.0)
    assert stopped["reason"] == "requested"


def test_output_is_the_downmixed_resampled_signal() -> None:
    clock = FakeClock()
    source = FakeSource(
        rate=44_100, channels=2, frames_per_drain=2_205, frequency=1000.0, clock=clock
    )
    _, sink, _ = run_fake(source, ticks=40, clock=clock)
    x = sink.samples.astype(np.float64)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    peak_hz = np.fft.rfftfreq(len(x), 1 / 16_000)[np.argmax(spectrum)]
    assert peak_hz == pytest.approx(1000, abs=2)
    assert np.max(np.abs(x[2000:-2000])) == pytest.approx(0.5 * 32767, rel=0.01)


def test_events_started_levels_stats_stopped() -> None:
    clock = FakeClock()
    source = FakeSource(clock=clock)
    _, _, emit = run_fake(source, ticks=50, clock=clock)  # 2.5 s
    assert emit.events[0]["type"] == "started"
    assert emit.events[0]["device"] == FakeSource.device_name
    assert emit.events[-1]["type"] == "stopped"
    assert len(emit.of_type("level")) >= 20
    stats = emit.of_type("stats")
    assert len(stats) == 2
    assert stats[-1]["stream_active"] is True
    assert stats[-1]["max_gap_ms"] == pytest.approx(50, abs=1)
    # Sine at amplitude 0.5: RMS 0.5/sqrt(2) -> -9.03 dBFS.
    assert emit.of_type("level")[-1]["rms_db"] == pytest.approx(-9.0, abs=0.1)


def test_open_failure_closes_everything_and_propagates() -> None:
    source = FakeSource(fail_on_open=True)
    sink = ListSink()
    default_output = ScriptedDefaultOutput()
    config = CaptureConfig(
        sink_factory=lambda: sink,
        source_factory=lambda: source,
        default_output_factory=lambda: default_output,
    )
    with pytest.raises(OSError, match="fake device unavailable"):
        run_capture(config, stop_after(5), Recorder(), clock=FakeClock(), sleep=lambda s: None)
    assert sink.close_reason == "error" and not sink.started
    assert source.close_calls == 1
    assert default_output.closed


# --- silence filling and device changes, in-process on a fake clock ------------------------
#
# The clock starts at 1000.0 and every tick is 50 ms. ClockedSource packets carry 10 ms of a
# constant level, so `sink.sound_starts()` shows where audio lands on the timeline
# (16 frames = 1 ms).


def test_initial_silence_is_filled_from_t0() -> None:
    # Typical start: recording first, then play. The 1.1 s before the first packet must be on
    # the timeline, so that segment 1 really starts when the recording starts.
    clock = FakeClock()
    source = ClockedSource(clock, silent=[(1000.0, 1001.1)])
    sink, emit = run_clocked(clock, [source], ticks=60)  # 3 s
    assert sink.sound_starts() == [pytest.approx(1.1 * 16_000, abs=16)]
    assert len(sink.samples) == 3 * 16_000
    assert sink.gap_frames == 0
    assert emit.of_type("stopped")[-1]["silence_s"] == pytest.approx(1.1, abs=0.002)


def test_nothing_playing_still_records_the_whole_session() -> None:
    clock = FakeClock()
    source = ClockedSource(clock, silent=[(1000.0, math.inf)])
    sink, emit = run_clocked(clock, [source], ticks=60)
    for stats in emit.of_type("stats"):
        assert stats["wall_s"] - stats["audio_s"] <= 0.16
    assert len(sink.samples) == 3 * 16_000
    assert not sink.samples.any()


def test_pause_keeps_the_timeline_aligned() -> None:
    clock = FakeClock()
    source = ClockedSource(clock, silent=[(1001.0, 1004.0)])
    sink, emit = run_clocked(clock, [source], ticks=120)  # 6 s
    assert sink.sound_starts() == [0, pytest.approx(4 * 16_000, abs=16)]
    assert len(sink.samples) == 6 * 16_000
    # During the pause the timeline keeps growing, 100-150 ms behind the clock.
    for stats in emit.of_type("stats"):
        assert stats["wall_s"] - stats["audio_s"] <= 0.16
    assert sink.gap_frames == 0


def test_vanished_device_is_reopened_and_the_hole_counted_as_gap() -> None:
    # Unplugging the headphones: the Realtek stream stops, the default becomes the monitor.
    clock = FakeClock()
    realtek = ClockedSource(clock, name="Realtek", dies_at=1002.0)
    nvidia = ClockedSource(clock, name="NVIDIA", open_s=0.04)
    default_output = ScriptedDefaultOutput(
        [(1002.0, "endpoint-nvidia")], initial="endpoint-realtek", clock=clock
    )
    sink, emit = run_clocked(clock, [realtek, nvidia], ticks=100, default_output=default_output)
    [lost] = emit.of_type("device_lost")
    assert (lost["reason"], lost["device"], lost["at_s"]) == ("stream stopped", "Realtek", 2.0)
    [opened] = emit.of_type("device_opened")
    assert (opened["device"], opened["gap_ms"], opened["attempts"]) == ("NVIDIA", 40, 1)
    # Gap from the last packet (2.0 s) to the reopen (40 ms later).
    assert sink.gap_frames == round(0.04 * 16_000)
    assert sink.sound_starts() == [0, pytest.approx(2.04 * 16_000, abs=16)]
    assert realtek.close_calls >= 1
    assert len(sink.samples) == round(emit.of_type("stopped")[-1]["wall_s"] * 16_000)


def test_default_output_change_is_followed_while_the_old_stream_stays_active() -> None:
    # Replugging the headphones: Windows switches back to Realtek and the monitor's stream
    # stays active but silent. Only the default output ID (polled every 0.5 s) shows it.
    clock = FakeClock()
    nvidia = ClockedSource(clock, name="NVIDIA", silent=[(1002.2, math.inf)])
    realtek = ClockedSource(clock, name="Realtek", open_s=0.04)
    default_output = ScriptedDefaultOutput(
        [(1002.2, "endpoint-realtek")], initial="endpoint-nvidia", clock=clock
    )
    sink, emit = run_clocked(clock, [nvidia, realtek], ticks=100, default_output=default_output)
    [lost] = emit.of_type("device_lost")
    assert (lost["reason"], lost["at_s"]) == ("default output changed", 2.5)
    assert nvidia.close_calls >= 1
    # Gap from the last packet (2.2 s) to the reopen (2.5 s poll + 40 ms to open).
    assert sink.gap_frames == round(0.34 * 16_000)
    assert sink.sound_starts() == [0, pytest.approx(2.54 * 16_000, abs=16)]
    assert default_output.closed


def test_reopen_is_retried_until_an_output_exists() -> None:
    clock = FakeClock()
    sources = [
        ClockedSource(clock, dies_at=1001.0),
        ClockedSource(clock, fail_on_open=True),
        ClockedSource(clock, fail_on_open=True),
        ClockedSource(clock, open_s=0.04),
    ]
    sink, emit = run_clocked(clock, sources, ticks=80)
    # One warning for the same repeated error, not one per attempt.
    [warning] = emit.of_type("warning")
    assert "fake device unavailable" in warning["message"]
    [opened] = emit.of_type("device_opened")
    assert opened["attempts"] == 3
    # Attempts at 1.0, 1.5 and 2.0 s, open at 2.04 s; the timeline kept growing meanwhile.
    assert sink.gap_frames == round(1.04 * 16_000)
    assert sink.sound_starts() == [0, pytest.approx(2.04 * 16_000, abs=16)]
    for stats in emit.of_type("stats"):
        assert stats["wall_s"] - stats["audio_s"] <= 0.16


def test_stop_during_an_outage_pads_the_timeline_with_gap() -> None:
    clock = FakeClock()
    first = ClockedSource(clock, dies_at=1001.0)
    never = itertools.repeat(ClockedSource(clock, fail_on_open=True))
    sink, emit = run_clocked(clock, itertools.chain([first], never), ticks=60)
    assert len(sink.samples) == 3 * 16_000
    assert sink.gap_frames == 2 * 16_000
    stopped = emit.of_type("stopped")[-1]
    assert (stopped["audio_s"], stopped["gap_s"]) == (3.0, 2.0)
    assert sink.close_reason == "requested"


@pytest.mark.parametrize("fail_after", [0, 3])
def test_default_output_failure_is_a_warning_not_an_error(fail_after: int) -> None:
    clock = FakeClock()
    default_output = ScriptedDefaultOutput(clock=clock, fail_after=fail_after)
    sink, emit = run_clocked(clock, [ClockedSource(clock)], ticks=60, default_output=default_output)
    [warning] = emit.of_type("warning")
    assert "cannot read the default output" in warning["message"]
    assert default_output.closed
    assert emit.of_type("device_lost") == []
    assert len(sink.samples) == 3 * 16_000


def test_default_output_that_cannot_start_is_a_warning() -> None:
    clock = FakeClock()

    def broken():
        raise OSError("no COM here")

    sink = ListSink()
    emit = Recorder()
    config = CaptureConfig(
        sink_factory=lambda: sink,
        source_factory=lambda: ClockedSource(clock),
        default_output_factory=broken,
    )
    assert run_capture(config, stop_after(20), emit, clock=clock, sleep=clock.sleep) == "requested"
    [warning] = emit.of_type("warning")
    assert "no COM here" in warning["message"]
    assert len(sink.samples) == 16_000


# --- silence limit -------------------------------------------------------------------------


def run_with_limit(clock: FakeClock, source, should_stop, limit_s: float):
    sink = ListSink()
    emit = Recorder()
    config = CaptureConfig(
        sink_factory=lambda: sink,
        source_factory=lambda: source,
        default_output_factory=lambda: ScriptedDefaultOutput(clock=clock),
        silence_limit_s=limit_s,
    )
    reason = run_capture(config, should_stop, emit, clock=clock, sleep=clock.sleep)
    return reason, sink, emit


def test_recording_stops_by_itself_after_the_silence_limit() -> None:
    clock = FakeClock()
    source = ClockedSource(clock, silent=[(1001.0, math.inf)])
    reason, sink, emit = run_with_limit(clock, source, stop_after(1000), limit_s=5.0)
    assert reason == "silence_limit"
    assert sink.close_reason == "silence_limit"
    assert len(sink.samples) == 6 * 16_000  # last sound at 1.0 s + 5 s
    [warning] = emit.of_type("warning")
    assert "no sound for 5 s (last at t=1.0 s)" in warning["message"]
    assert "jumped" not in warning["message"]
    assert emit.of_type("stopped")[-1]["reason"] == "silence_limit"
    assert source.close_calls >= 1


def test_silence_limit_warning_when_nothing_ever_played() -> None:
    clock = FakeClock()
    source = ClockedSource(clock, silent=[(1000.0, math.inf)])
    reason, sink, emit = run_with_limit(clock, source, stop_after(1000), limit_s=2.0)
    assert reason == "silence_limit"
    [warning] = emit.of_type("warning")
    assert "no sound since the start (2 s limit)" in warning["message"]
    assert len(sink.samples) == 2 * 16_000


def test_clock_jump_after_sleep_fills_only_up_to_the_limit() -> None:
    clock = FakeClock()
    source = ClockedSource(clock, silent=[(1002.0, math.inf)])
    calls = itertools.count()

    def should_stop():
        if next(calls) == 60:  # at 3.0 s the PC sleeps for 8 hours
            clock.sleep(8 * 3600)
        return None

    reason, sink, emit = run_with_limit(clock, source, should_stop, limit_s=60.0)
    assert reason == "silence_limit"
    assert len(sink.samples) == (2 + 60) * 16_000  # last sound at 2.0 s + 60 s, not 8 h
    [warning] = emit.of_type("warning")
    assert "jumped 480 min" in warning["message"]


# --- separate process --------------------------------------------------------------------


def test_capture_process_starts_writes_and_stops_cleanly(tmp_path: Path) -> None:
    out = tmp_path / "capture.raw"
    config = CaptureConfig(
        sink_factory=functools.partial(RawFileSink, str(out)),
        source_factory=functools.partial(FakeSource, frames_per_drain=2_400),
        default_output_factory=ScriptedDefaultOutput,
    )
    handle = RecorderHandle(config)
    handle.start()
    try:
        started = wait_for(handle, "started")
        assert [e for e in started if e["type"] == "started"][0]["pid"] == handle.pid
        time.sleep(0.5)
    finally:
        exitcode = handle.stop()
    assert exitcode == 0
    events = handle.events()
    stopped = [e for e in events if e["type"] == "stopped"]
    assert stopped and stopped[0]["reason"] == "requested"
    samples = np.fromfile(out, dtype=np.int16)
    assert len(samples) == pytest.approx(stopped[0]["audio_s"] * 16_000, abs=16)
    assert len(samples) > 0


def test_capture_process_reports_errors_and_exits_1(tmp_path: Path) -> None:
    config = CaptureConfig(
        sink_factory=functools.partial(RawFileSink, str(tmp_path / "x.raw")),
        source_factory=functools.partial(FakeSource, fail_on_open=True),
        default_output_factory=ScriptedDefaultOutput,
    )
    with RecorderHandle(config) as handle:
        events = wait_for(handle, "error")
    assert handle.exitcode == 1
    error = [e for e in events if e["type"] == "error"][0]
    assert "fake device unavailable" in error["message"]
    assert "Traceback" in error["traceback"]
