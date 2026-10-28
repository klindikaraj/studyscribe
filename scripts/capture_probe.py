"""Capture probe: record the default output's loopback into one WAV and show what the stream does.

Used to check, on real hardware, the open assumptions of the capture design:
- does the loopback stop delivering packets while nothing plays? (audio falls behind wall time)
- what happens when the recorded device disappears (e.g. headphones unplugged)?

Usage:
    .venv\\Scripts\\python scripts\\capture_probe.py [--seconds N] [--out FILE.wav] [--data-dir PATH]

Stop early with Ctrl+C. Default output: <data dir>\\probe\\probe-<timestamp>.wav
"""

from __future__ import annotations

import argparse
import functools
import os
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from studyscribe.audio.capture import CaptureConfig
from studyscribe.config import SAMPLE_RATE, resolve_data_dir
from studyscribe.recording import RecorderHandle


class WaveFileSink:
    """Writes the whole capture to a single WAV file."""

    def __init__(self, path: str) -> None:
        self._wav = wave.open(path, "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(SAMPLE_RATE)

    def write(self, pcm: np.ndarray) -> None:
        self._wav.writeframes(pcm.tobytes())

    def close(self) -> None:
        self._wav.close()


class Printer:
    def __init__(self) -> None:
        self.level_db: float | None = None
        self.last_packets = 0
        self.last_active: bool | None = None

    def __call__(self, event: dict) -> None:
        kind = event["type"]
        clock = datetime.fromtimestamp(event["time"]).strftime("%H:%M:%S")
        if kind == "started":
            print(
                f"[{clock}] capture started: {event['device']}, {event['rate']} Hz, "
                f"{event['channels']} ch (pid {event['pid']})"
            )
        elif kind == "level":
            self.level_db = event["rms_db"]
        elif kind == "stats":
            new_packets = event["packets"] - self.last_packets
            self.last_packets = event["packets"]
            missing = event["wall_s"] - event["audio_s"]
            level = f"{self.level_db:6.1f}" if self.level_db is not None else "     -"
            print(
                f"[{clock}] t={event['wall_s']:7.1f}s audio={event['audio_s']:7.1f}s "
                f"missing={missing:6.2f}s packets=+{new_packets:<4} "
                f"max_gap={event['max_gap_ms']:5}ms idle={event['since_last_packet_ms']:6}ms "
                f"flagged={event['flagged_packets']:<3} active={event['stream_active']!s:5} "
                f"rms={level} dBFS"
            )
            if self.last_active is not None and event["stream_active"] != self.last_active:
                print(f"[{clock}] >>> stream_active changed to {event['stream_active']}")
            self.last_active = event["stream_active"]
        elif kind == "stopped":
            missing = event["wall_s"] - event["audio_s"]
            print(
                f"[{clock}] stopped ({event['reason']}): wall {event['wall_s']:.2f}s, "
                f"audio {event['audio_s']:.2f}s, missing {missing:.2f}s, "
                f"{event['packets']} packets"
            )
        elif kind == "error":
            print(f"[{clock}] ERROR: {event['message']}\n{event.get('traceback', '')}")
        else:
            print(f"[{clock}] {kind.upper()}: {event.get('message', event)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=120.0, help="duration (default 120)")
    parser.add_argument("--out", help="output WAV path")
    parser.add_argument("--data-dir", help="data directory for the default output path")
    args = parser.parse_args()

    if args.out:
        out = Path(args.out).resolve()
    else:
        data_dir = resolve_data_dir(args.data_dir)
        print(f"Data directory: {data_dir}")
        out = data_dir / "probe" / f"probe-{datetime.now():%Y%m%d-%H%M%S}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}")

    handle = RecorderHandle(CaptureConfig(sink_factory=functools.partial(WaveFileSink, str(out))))
    handle.start()
    print(f"Parent pid {os.getpid()}, capture pid {handle.pid}. Ctrl+C to stop.")
    show = Printer()
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline and handle.is_alive():
            for event in handle.events(timeout=0.2):
                show(event)
    except KeyboardInterrupt:
        print("\nStopping...")
    exitcode = handle.stop()
    for event in handle.events():
        show(event)
    print(f"Capture process exit code: {exitcode}")
    return 0 if exitcode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
