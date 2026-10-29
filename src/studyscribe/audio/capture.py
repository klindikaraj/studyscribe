"""Capture process: WASAPI loopback -> mono 16 kHz int16 timeline -> sink.

Runs in its own process (see `studyscribe.recording.RecorderHandle`). The PortAudio callback
only queues packets; a loop every ~50 ms aligns them to the clock (audio/timeline.py), hands
them to the sink and reopens the stream when the output device changes.
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

from studyscribe.audio.devices import DefaultOutputMonitor
from studyscribe.audio.dsp import LevelMeter
from studyscribe.audio.timeline import Packet, TimelineAligner
from studyscribe.config import DEVICE_POLL_S, REOPEN_RETRY_S, SAMPLE_RATE, SILENCE_LIMIT_S
from studyscribe.system.power import keep_system_awake

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


class DefaultOutput(Protocol):
    def current_id(self) -> str | None:
        """Endpoint ID of the default output, None if there is none. Raises if unreadable."""
        ...

    def close(self) -> None: ...


class Sink(Protocol):
    def start(self, emit: Emit) -> None:
        """Called once when the timeline starts, before the first write."""
        ...

    def write(self, pcm: np.ndarray, gap: bool = False) -> None:
        """Append samples to the timeline; `gap=True` marks them as synthetic."""
        ...

    def close(self, reason: str) -> None:
        """Always called, even if start() was not.

        reason: requested, silence_limit (auto-stop), parent_exited, error.
        """
        ...


class LoopbackSource:
    """Loopback of the current default output through PyAudioWPatch, in callback mode.

    One instance per stream: after a device change the capture loop opens a new one.
    """

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
    # None: do not watch the default output (a vanished device is still noticed).
    default_output_factory: Callable[[], DefaultOutput] | None = DefaultOutputMonitor
    tick_s: float = 0.05
    level_interval_s: float = 0.1
    stats_interval_s: float = 1.0
    device_poll_s: float = DEVICE_POLL_S
    reopen_retry_s: float = REOPEN_RETRY_S
    # Stop by itself after this long without sound; None: never.
    silence_limit_s: float | None = SILENCE_LIMIT_S


class _PacketStats:
    """Packet arrival statistics: show how the loopback behaves, e.g. during silence."""

    def __init__(self, start: float) -> None:
        self.packets = 0
        self.flagged = 0
        self.last_arrival = start
        self._window_max_gap = 0.0

    def on_packet(self, arrival: float, status: int) -> None:
        self.packets += 1
        if status:
            self.flagged += 1
        self._window_max_gap = max(self._window_max_gap, arrival - self.last_arrival)
        self.last_arrival = arrival

    def take_window(self, now: float) -> tuple[float, float]:
        """(longest gap between packets in this window, time since the last packet), seconds."""
        since_last = max(now - self.last_arrival, 0.0)
        max_gap = max(self._window_max_gap, since_last)
        self._window_max_gap = 0.0
        return max_gap, since_last


class _DefaultOutputWatch:
    """Reads the default output's ID; on failure warns once and stops watching.

    Without it a vanished device is still noticed (its stream stops), but not a manual change
    of the default output, which leaves the old stream active and silent.
    """

    def __init__(self, factory: Callable[[], DefaultOutput] | None, emit: Emit) -> None:
        self._emit = emit
        self._monitor: DefaultOutput | None = None
        if factory is not None:
            try:
                self._monitor = factory()
            except Exception as exc:
                self._give_up(exc)

    @property
    def available(self) -> bool:
        return self._monitor is not None

    def current(self) -> str | None:
        if self._monitor is None:
            return None
        try:
            return self._monitor.current_id()
        except Exception as exc:
            self._give_up(exc)
            return None

    def close(self) -> None:
        monitor, self._monitor = self._monitor, None
        if monitor is not None:
            try:
                monitor.close()
            except Exception as exc:
                self._emit("warning", message=f"closing the default output watch: {exc}")

    def _give_up(self, exc: Exception) -> None:
        self._emit(
            "warning",
            message=f"cannot read the default output ({type(exc).__name__}: {exc}); "
            "an unplugged device is still followed, a manual switch is not",
        )
        self.close()


def _next_deadline(deadline: float, interval: float, now: float) -> float:
    # Schedule from the previous deadline so periodic events keep their average rate.
    deadline += interval
    return deadline if deadline > now else now + interval


def _duration(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.0f} min"


def _close_source(source: AudioSource, emit: Emit) -> None:
    for problem in source.close():
        emit("warning", message=f"closing the audio stream: {problem}")


def _try_open(factory: Callable[[], AudioSource]) -> tuple[AudioSource | None, str]:
    """Open a new source; on failure return (None, error message) instead of raising."""
    source = factory()
    try:
        source.open()
    except Exception as exc:
        source.close()
        return None, f"{type(exc).__name__}: {exc}"
    return source, ""


def run_capture(
    config: CaptureConfig,
    should_stop: Callable[[], str | None],
    emit: Emit,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Capture until `should_stop()` returns a reason, then return that reason.

    Also returns "silence_limit" if nothing was heard for `silence_limit_s`. The first stream
    must open, or the error propagates. After that a lost device never stops the recording:
    the stream is reopened (retrying every `reopen_retry_s`) and the hole is filled with
    silence counted as gap. The source, the default output watch and the sink are always
    closed, errors included.
    """
    sink = config.sink_factory()
    watch: _DefaultOutputWatch | None = None
    source: AudioSource | None = None
    reason = "error"
    try:
        watch = _DefaultOutputWatch(config.default_output_factory, emit)
        source_id = watch.current()
        source = config.source_factory()
        source.open()
        emit(
            "started",
            pid=os.getpid(),
            device=source.device_name,
            rate=source.rate,
            channels=source.channels,
        )
        t0 = clock()
        sink.start(emit)
        meter = LevelMeter()

        def write(pcm: np.ndarray, gap: bool) -> None:
            sink.write(pcm, gap=gap)
            meter.add(pcm)

        timeline = TimelineAligner(t0, write, silence_limit_s=config.silence_limit_s)
        timeline.start_input(source.rate, source.channels)
        stats = _PacketStats(t0)
        next_level = t0 + config.level_interval_s
        next_stats = t0 + config.stats_interval_s
        next_poll = t0 + config.device_poll_s
        next_retry = t0
        lost_at = t0  # start of the current device gap
        failures = 0
        last_failure = ""
        previous_tick = t0

        def take(packets: list[Packet]) -> None:
            for arrival, _, status in packets:
                stats.on_packet(arrival, status)
            timeline.add(packets)

        while (stop_reason := should_stop()) is None:
            if source is not None:
                take(source.drain())
            now = clock()
            clock_jump = now - previous_tick
            previous_tick = now

            default_changed = False
            if now >= next_poll:
                next_poll = _next_deadline(next_poll, config.device_poll_s, now)
                default_id = watch.current()
                default_changed = watch.available and default_id != source_id

            if source is not None and (default_changed or not source.is_active()):
                why = "default output changed" if source.is_active() else "stream stopped"
                _close_source(source, emit)
                take(source.drain())
                timeline.end_input()
                # The change happened at most one poll interval ago; real silence before that
                # (e.g. a paused video) is not counted as gap.
                lost_at = max(stats.last_arrival, now - config.device_poll_s)
                timeline.open_gap(lost_at)
                emit(
                    "device_lost",
                    reason=why,
                    device=source.device_name,
                    at_s=round(now - t0, 3),
                )
                source = None
                next_retry = now

            if source is None and now >= next_retry:
                new_id = watch.current()
                new_source, error = _try_open(config.source_factory)
                if new_source is None:
                    failures += 1
                    if error != last_failure:
                        emit(
                            "warning",
                            message=f"reopening the audio stream failed, retrying every "
                            f"{config.reopen_retry_s:g} s: {error}",
                        )
                        last_failure = error
                    next_retry = now + config.reopen_retry_s
                else:
                    opened_at = clock()
                    timeline.close_gap(opened_at)
                    timeline.start_input(new_source.rate, new_source.channels)
                    emit(
                        "device_opened",
                        device=new_source.device_name,
                        rate=new_source.rate,
                        channels=new_source.channels,
                        gap_ms=round((opened_at - lost_at) * 1000),
                        attempts=failures + 1,
                    )
                    source, source_id = new_source, new_id
                    failures, last_failure = 0, ""

            # Closing and reopening a stream takes tens of milliseconds, and its leftover
            # packets were written meanwhile: read the clock again for what follows.
            now = clock()
            timeline.idle(now)
            if timeline.limit_reached:
                stop_reason = "silence_limit"
                limit = _duration(config.silence_limit_s or 0)
                if timeline.last_sound > t0:
                    since = f"for {limit} (last at t={timeline.last_sound - t0:.1f} s)"
                else:
                    since = f"since the start ({limit} limit)"
                message = f"no sound {since}: recording stopped automatically"
                if clock_jump > 60:
                    message += f"; the clock jumped {_duration(clock_jump)} at once (sleep?)"
                emit("warning", message=message)
                break
            if now >= next_level:
                rms_db, peak_db = meter.take()
                emit("level", rms_db=round(rms_db, 1), peak_db=round(peak_db, 1))
                next_level = _next_deadline(next_level, config.level_interval_s, now)
            if now >= next_stats:
                max_gap, since_last = stats.take_window(now)
                emit(
                    "stats",
                    wall_s=round(now - t0, 3),
                    audio_s=round(timeline.frames / SAMPLE_RATE, 3),
                    silence_s=round(timeline.silence_frames / SAMPLE_RATE, 3),
                    gap_s=round(timeline.gap_frames / SAMPLE_RATE, 3),
                    packets=stats.packets,
                    flagged_packets=stats.flagged,
                    max_gap_ms=round(max_gap * 1000),
                    since_last_packet_ms=round(since_last * 1000),
                    stream_active=source is not None and source.is_active(),
                )
                next_stats = _next_deadline(next_stats, config.stats_interval_s, now)
            sleep(config.tick_s)

        stop_time = clock()
        if source is not None:
            # Close first so no packet arrives after the final drain.
            _close_source(source, emit)
            take(source.drain())
        timeline.finish(stop_time)
        reason = stop_reason
        emit(
            "stopped",
            reason=reason,
            wall_s=round(stop_time - t0, 3),
            audio_s=round(timeline.frames / SAMPLE_RATE, 3),
            silence_s=round(timeline.silence_frames / SAMPLE_RATE, 3),
            gap_s=round(timeline.gap_frames / SAMPLE_RATE, 3),
            packets=stats.packets,
        )
        return reason
    finally:
        if source is not None:
            _close_source(source, emit)
        if watch is not None:
            watch.close()
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
