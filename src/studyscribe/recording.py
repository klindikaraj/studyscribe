"""Parent side of a recording: start the capture process, collect its events, stop it."""

from __future__ import annotations

import multiprocessing
import queue
import time
from typing import Any

from studyscribe.audio.capture import CaptureConfig, capture_main

Event = dict[str, Any]


class RecorderHandle:
    """Owns one capture process. As a context manager it always stops the process on exit."""

    def __init__(self, config: CaptureConfig, max_queued_events: int = 1000) -> None:
        ctx = multiprocessing.get_context("spawn")
        self._stop = ctx.Event()
        self._events = ctx.Queue(maxsize=max_queued_events)
        self._process = ctx.Process(
            target=capture_main,
            args=(config, self._stop, self._events),
            name="studyscribe-capture",
        )
        self._pending: list[Event] = []

    def start(self) -> None:
        self._process.start()

    @property
    def pid(self) -> int | None:
        return self._process.pid

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode

    def is_alive(self) -> bool:
        return self._process.is_alive()

    def events(self, timeout: float = 0.0) -> list[Event]:
        """Events received so far. If there are none, wait up to `timeout` s for the first."""
        received, self._pending = self._pending, []
        if not received and timeout > 0:
            try:
                received.append(self._events.get(timeout=timeout))
            except queue.Empty:
                return received
        received.extend(self._drain())
        return received

    def stop(self, timeout: float = 10.0) -> int | None:
        """Ask the capture process to finish, wait for it and return its exit code.

        Events that arrive meanwhile are kept for `events()`. If the process does not exit
        within `timeout` it is terminated; what it already wrote stays on disk.
        """
        if self._process.pid is None:
            return None
        self._stop.set()
        deadline = time.monotonic() + timeout
        while self._process.is_alive() and time.monotonic() < deadline:
            # Keep reading: the child cannot exit while its last events are still in flight.
            self._pending.extend(self._drain())
            self._process.join(0.1)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(5)
        self._pending.extend(self._drain())
        return self._process.exitcode

    def _drain(self) -> list[Event]:
        drained: list[Event] = []
        while True:
            try:
                drained.append(self._events.get_nowait())
            except queue.Empty:
                return drained

    def __enter__(self) -> RecorderHandle:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
