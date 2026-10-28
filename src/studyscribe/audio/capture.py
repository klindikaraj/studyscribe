"""Capture process: WASAPI loopback -> mono 16 kHz int16 -> sink.

Runs in its own process (see `studyscribe.recording.RecorderHandle`). The PortAudio callback
only queues packets; a loop every ~50 ms converts them and hands them to the sink.
"""

from __future__ import annotations

import multiprocessing
import os
import queue
import signal
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from studyscribe.audio.dsp import (
    LevelMeter,
    StreamingResampler,
    downmix_to_mono,
    float_to_int16,
    frames_from_bytes,
)
from studyscribe.config import SAMPLE_RATE
from studyscribe.system.power import keep_system_awake

# (time.perf_counter() at arrival, interleaved float32 frames, PortAudio status flags)
Packet = tuple[float, bytes, int]
Emit = Callable[..., None]


class AudioSource(Protocol):
    device_name: str
    rate: int
    channels: int

    def open(self) -> None: ...

    def drain(self) -> list[Packet]:
        """Return the packets received since the last call, oldest first."""
        ...

    def is_active(self) -> bool: ...

    def close(self) -> list[str]:
        """Release the device. Idempotent, never raises; returns the problems it ignored."""
        ...


class Sink(Protocol):
    def start(self, emit: Emit) -> None:
        """Called once when the timeline starts, before the first write."""
        ...

    def write(self, pcm: np.ndarray, gap: bool = False) -> None:
        """Append samples to the timeline; `gap=True` marks them as synthetic."""
        ...

    def close(self, reason: str) -> None:
        """Always called, even if start() was not. reason: requested, parent_exited, error."""
        ...


class LoopbackSource:
    """Loopback of the current default output through PyAudioWPatch, in callback mode."""

    def __init__(self) -> None:
        self.device_name = ""
        self.rate = 0
        self.channels = 0
        self._pa: Any = None
        self._stream: Any = None
        self._packets: queue.SimpleQueue[Packet] = queue.SimpleQueue()
        self._continue: tuple[None, int] = (None, 0)

    def open(self) -> None:
        # Windows-only dependency, imported here so the rest of the package (and the tests)
        # never need it.
        import pyaudiowpatch as pyaudio

        self._continue = (None, pyaudio.paContinue)
        # PortAudio builds its device list at initialisation, so a fresh instance is needed
        # to see devices that appeared since the last one.
        self._pa = pyaudio.PyAudio()
        try:
            info = self._pa.get_default_wasapi_loopback()
            self.device_name = info["name"]
            self.rate = int(info["defaultSampleRate"])
            self.channels = int(info["maxInputChannels"])
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=info["index"],
                stream_callback=self._on_audio,
            )
        except BaseException:
            self.close()
            raise

    def _on_audio(self, in_data: bytes, frame_count: int, time_info: dict, status: int):
        # Runs on PortAudio's thread: do as little as possible.
        self._packets.put((time.perf_counter(), in_data, status))
        return self._continue

    def drain(self) -> list[Packet]:
        packets: list[Packet] = []
        while True:
            try:
                packets.append(self._packets.get_nowait())
            except queue.Empty:
                return packets

    def is_active(self) -> bool:
        if self._stream is None:
            return False
        try:
            return bool(self._stream.is_active())
        except OSError:
            return False

    def close(self) -> list[str]:
        # The device may already be gone (headphones unplugged): report errors, never raise.
        problems: list[str] = []
        stream, self._stream = self._stream, None
        pa, self._pa = self._pa, None
        steps = []
        if stream is not None:
            steps += [("stop stream", stream.stop_stream), ("close stream", stream.close)]
        if pa is not None:
            steps.append(("terminate PortAudio", pa.terminate))
        for name, step in steps:
            try:
                step()
            except Exception as exc:
                problems.append(f"{name}: {type(exc).__name__}: {exc}")
        return problems


@dataclass(frozen=True)
class CaptureConfig:
    """Everything the capture process needs. Must be picklable (spawn start method)."""

    sink_factory: Callable[[], Sink]
    source_factory: Callable[[], AudioSource] = LoopbackSource
    tick_s: float = 0.05
    level_interval_s: float = 0.1
    stats_interval_s: float = 1.0


class _PacketStats:
    """Packet arrival statistics: show how the loopback behaves, e.g. during silence."""

    def __init__(self, start: float) -> None:
        self.packets = 0
        self.flagged = 0
        self._last = start
        self._window_max_gap = 0.0

    def on_packet(self, arrival: float, status: int) -> None:
        self.packets += 1
        if status:
            self.flagged += 1
        self._window_max_gap = max(self._window_max_gap, arrival - self._last)
        self._last = arrival

    def take_window(self, now: float) -> tuple[float, float]:
        """(longest gap between packets in this window, time since the last packet), seconds."""
        since_last = max(now - self._last, 0.0)
        max_gap = max(self._window_max_gap, since_last)
        self._window_max_gap = 0.0
        return max_gap, since_last


def _next_deadline(deadline: float, interval: float, now: float) -> float:
    # Schedule from the previous deadline so periodic events keep their average rate.
    deadline += interval
    return deadline if deadline > now else now + interval


def _to_pcm16(data: bytes, channels: int, resampler: StreamingResampler) -> np.ndarray:
    mono = downmix_to_mono(frames_from_bytes(data, channels))
    return float_to_int16(resampler.process(mono))


def run_capture(
    config: CaptureConfig,
    should_stop: Callable[[], str | None],
    emit: Emit,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Capture until `should_stop()` returns a reason, then return that reason.

    The source and the sink are always closed, errors included.
    """
    sink = config.sink_factory()
    source = config.source_factory()
    reason = "error"
    try:
        source.open()
        resampler = StreamingResampler(source.rate, SAMPLE_RATE)
        emit(
            "started",
            pid=os.getpid(),
            device=source.device_name,
            rate=source.rate,
            channels=source.channels,
        )
        t0 = clock()
        sink.start(emit)
        stats = _PacketStats(t0)
        meter = LevelMeter()
        written = 0
        next_level = t0 + config.level_interval_s
        next_stats = t0 + config.stats_interval_s

        def write(pcm: np.ndarray) -> None:
            nonlocal written
            if pcm.size:
                sink.write(pcm)
                meter.add(pcm)
                written += pcm.size

        while (stop_reason := should_stop()) is None:
            for arrival, data, status in source.drain():
                stats.on_packet(arrival, status)
                write(_to_pcm16(data, source.channels, resampler))
            now = clock()
            if now >= next_level:
                rms_db, peak_db = meter.take()
                emit("level", rms_db=round(rms_db, 1), peak_db=round(peak_db, 1))
                next_level = _next_deadline(next_level, config.level_interval_s, now)
            if now >= next_stats:
                max_gap, since_last = stats.take_window(now)
                emit(
                    "stats",
                    wall_s=round(now - t0, 3),
                    audio_s=round(written / SAMPLE_RATE, 3),
                    packets=stats.packets,
                    flagged_packets=stats.flagged,
                    max_gap_ms=round(max_gap * 1000),
                    since_last_packet_ms=round(since_last * 1000),
                    stream_active=source.is_active(),
                )
                next_stats = _next_deadline(next_stats, config.stats_interval_s, now)
            sleep(config.tick_s)

        # Close first so no packet arrives after the final drain.
        for problem in source.close():
            emit("warning", message=f"closing the audio stream: {problem}")
        for arrival, data, status in source.drain():
            stats.on_packet(arrival, status)
            write(_to_pcm16(data, source.channels, resampler))
        write(float_to_int16(resampler.flush()))
        reason = stop_reason
        emit(
            "stopped",
            reason=reason,
            wall_s=round(clock() - t0, 3),
            audio_s=round(written / SAMPLE_RATE, 3),
            packets=stats.packets,
        )
        return reason
    finally:
        for problem in source.close():
            emit("warning", message=f"closing the audio stream: {problem}")
        sink.close(reason)


class QueueEmitter:
    """Sends events to the parent without ever blocking; drops them if the queue is full."""

    def __init__(self, events: Any) -> None:
        self._events = events
        self.dropped = 0

    def __call__(self, kind: str, **fields: Any) -> None:
        try:
            self._events.put_nowait({"type": kind, "time": time.time(), **fields})
        except queue.Full:
            self.dropped += 1


def capture_main(config: CaptureConfig, stop_event: Any, events: Any) -> None:
    """Entry point of the capture process."""
    # Ctrl+C reaches every process attached to the console; only the parent decides to stop.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    emit = QueueEmitter(events)
    parent = multiprocessing.parent_process()

    def should_stop() -> str | None:
        if stop_event.is_set():
            return "requested"
        if parent is not None and not parent.is_alive():
            return "parent_exited"
        return None

    try:
        with keep_system_awake() as awake:
            if not awake:
                emit("warning", message="anti-standby request refused by Windows")
            reason = run_capture(config, should_stop, emit)
    except BaseException as exc:
        emit("error", message=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise SystemExit(1) from None
    if reason == "parent_exited":
        # Nobody reads the events any more: do not wait for them to be delivered on exit.
        events.cancel_join_thread()
