"""Smoke tests of the COM code on the machine running the tests (no device changes)."""

import re
import sys

import pytest

from studyscribe.audio.devices import DefaultOutputMonitor

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="COM is Windows-only")
RENDER_ENDPOINT = re.compile(r"\{0\.0\.0\.[0-9a-f]{8}\}\.\{[0-9a-f-]{36}\}")


@windows_only
def test_reads_the_default_output_id_or_none() -> None:
    monitor = DefaultOutputMonitor()
    try:
        first = monitor.current_id()  # None on a machine without any output device
        assert first is None or RENDER_ENDPOINT.fullmatch(first)
        assert monitor.current_id() == first
    finally:
        monitor.close()
    monitor.close()  # idempotent
    with pytest.raises(OSError, match="closed"):
        monitor.current_id()


@windows_only
def test_monitors_can_be_created_and_closed_repeatedly() -> None:
    # Every monitor initialises and releases COM on this thread: they must stay balanced.
    for _ in range(20):
        monitor = DefaultOutputMonitor()
        monitor.current_id()
        monitor.close()


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows behaviour")
def test_needs_windows() -> None:
    with pytest.raises(OSError):
        DefaultOutputMonitor()
