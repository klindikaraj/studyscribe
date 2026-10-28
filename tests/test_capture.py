import functools
import itertools
import time
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeClock, FakeSource, ListSink, RawFileSink

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


def run_fake(source: FakeSource, ticks: int, clock: FakeClock | None = None):
    clock = clock or FakeClock()
    sink = ListSink()
    emit = Recorder()
    config = CaptureConfig(sink_factory=lambda: sink, source_factory=lambda: source)
    reason = run_capture(config, stop_after(ticks), emit, clock=clock, sleep=clock.sleep)
    return reason, sink, emit


# --- capture loop, in-process ------------------------------------------------------------


def test_one_second_of_48k_stereo_becomes_16000_samples() -> None:
    source = FakeSource(rate=48_000, channels=2, frames_per_drain=2_400)
    reason, sink, emit = run_fake(source, ticks=20)  # 20 x 50 ms = 1 s
    assert reason == "requested"
    assert len(sink.samples) == 16_000
    assert sink.closed and source.close_calls >= 1
    stopped = emit.of_type("stopped")[-1]
    assert stopped["audio_s"] == pytest.approx(1.0)
    assert stopped["reason"] == "requested"


def test_output_is_the_downmixed_resampled_signal() -> None:
    source = FakeSource(rate=44_100, channels=2, frames_per_drain=2_205, frequency=1000.0)
    _, sink, _ = run_fake(source, ticks=40)
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
    config = CaptureConfig(sink_factory=lambda: sink, source_factory=lambda: source)
    with pytest.raises(OSError, match="fake device unavailable"):
        run_capture(config, stop_after(5), Recorder(), clock=FakeClock(), sleep=lambda s: None)
    assert sink.closed
    assert source.close_calls == 1


# --- separate process --------------------------------------------------------------------


def wait_for(handle: RecorderHandle, kind: str, timeout: float = 20.0) -> list[dict]:
    seen: list[dict] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        seen += handle.events(timeout=0.2)
        if any(e["type"] == kind for e in seen):
            return seen
        if not handle.is_alive():
            break
    raise AssertionError(f"no {kind!r} event; got {[e['type'] for e in seen]}")


def test_capture_process_starts_writes_and_stops_cleanly(tmp_path: Path) -> None:
    out = tmp_path / "capture.raw"
    config = CaptureConfig(
        sink_factory=functools.partial(RawFileSink, str(out)),
        source_factory=functools.partial(FakeSource, frames_per_drain=2_400),
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
    assert len(samples) == round(stopped[0]["audio_s"] * 16_000)
    assert len(samples) > 0


def test_capture_process_reports_errors_and_exits_1(tmp_path: Path) -> None:
    config = CaptureConfig(
        sink_factory=functools.partial(RawFileSink, str(tmp_path / "x.raw")),
        source_factory=functools.partial(FakeSource, fail_on_open=True),
    )
    with RecorderHandle(config) as handle:
        events = wait_for(handle, "error")
    assert handle.exitcode == 1
    error = [e for e in events if e["type"] == "error"][0]
    assert "fake device unavailable" in error["message"]
    assert "Traceback" in error["traceback"]
