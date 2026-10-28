"""Capture probe: record the default output's loopback into one WAV and show what the stream does.

Used to check the capture design on real hardware: the timeline must follow wall time through
silence and pauses (`missing` stays around 0.1 s at most), and unplugging or replugging the
headphones must reopen the stream, with the hole counted as gap.

Usage:
    .venv\\Scripts\\python scripts\\capture_probe.py [--seconds N] [--out FILE.wav] [--data-dir PATH]
    .venv\\Scripts\\python scripts\\capture_probe.py --session "Title" [--seconds N]
    .venv\\Scripts\\python scripts\\capture_probe.py --recover [--data-dir PATH]

Stop early with Ctrl+C. Default output: <data dir>\\probe\\probe-<timestamp>.wav
With --session the probe records into a real session instead (60 s segments + database rows)
and prints the manifest at the end; if the capture process dies (e.g. killed from Task Manager)
the probe recovers the session itself, as the owner. --recover repairs every abandoned session
(e.g. after killing both processes) and prints what it did.
"""

from __future__ import annotations

import argparse
import functools
import os
import time
import wave
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np

from studyscribe.audio.capture import CaptureConfig
from studyscribe.config import (
    SAMPLE_RATE,
    SILENCE_LIMIT_S,
    db_path,
    resolve_data_dir,
    session_dir,
)
from studyscribe.db import repo
from studyscribe.db.connection import open_database
from studyscribe.recording import RecorderHandle
from studyscribe.recovery import Recovered, recover_all, recover_session
from studyscribe.session import SessionSink, new_session


class WaveFileSink:
    """Writes the whole capture to a single WAV file."""

    def __init__(self, path: str) -> None:
        self._wav = wave.open(path, "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(SAMPLE_RATE)

    def start(self, emit) -> None:
        pass

    def write(self, pcm: np.ndarray, gap: bool = False) -> None:
        self._wav.writeframes(pcm.tobytes())

    def close(self, reason: str) -> None:
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
                f"missing={missing:5.2f}s silence={event['silence_s']:6.1f}s "
                f"gap={event['gap_s']:5.2f}s packets=+{new_packets:<4} "
                f"max_gap={event['max_gap_ms']:5}ms idle={event['since_last_packet_ms']:6}ms "
                f"flagged={event['flagged_packets']:<3} active={event['stream_active']!s:5} "
                f"rms={level} dBFS"
            )
            if self.last_active is not None and event["stream_active"] != self.last_active:
                print(f"[{clock}] >>> stream_active changed to {event['stream_active']}")
            self.last_active = event["stream_active"]
        elif kind == "segment":
            print(
                f"[{clock}] segment {event['seq']:06d} closed: "
                f"{event['start_ms'] / 1000:.3f}-{event['end_ms'] / 1000:.3f} s, "
                f"gap {event['gap_ms']} ms"
            )
        elif kind == "device_lost":
            print(
                f"[{clock}] >>> device lost at t={event['at_s']:.2f}s ({event['reason']}): "
                f"{event['device']}"
            )
        elif kind == "device_opened":
            print(
                f"[{clock}] >>> reopened on {event['device']}, {event['rate']} Hz, "
                f"{event['channels']} ch: gap {event['gap_ms']} ms, "
                f"{event['attempts']} attempt(s)"
            )
        elif kind == "stopped":
            missing = event["wall_s"] - event["audio_s"]
            print(
                f"[{clock}] stopped ({event['reason']}): wall {event['wall_s']:.2f}s, "
                f"audio {event['audio_s']:.2f}s, missing {missing:.2f}s, "
                f"silence {event['silence_s']:.2f}s, gap {event['gap_s']:.2f}s, "
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
    parser.add_argument("--data-dir", help="data directory (default output path, sessions)")
    parser.add_argument("--session", metavar="TITLE", help="record into a real session")
    parser.add_argument("--recover", action="store_true", help="recover abandoned sessions")
    parser.add_argument(
        "--silence-limit",
        type=float,
        default=SILENCE_LIMIT_S,
        metavar="SECONDS",
        help=f"stop after this long without sound (default {SILENCE_LIMIT_S:g})",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    if args.recover:
        print(f"Data directory: {data_dir}")
        with closing(open_database(db_path(data_dir))) as conn:
            print_recovery(recover_all(conn, data_dir))
        return 0
    session_id = owner_lock = None
    if args.session:
        print(f"Data directory: {data_dir}")
        with closing(open_database(db_path(data_dir))) as conn:
            session_id, owner_lock = new_session(conn, data_dir, title=args.session)
        print(f"Session: {session_id}\nFolder: {session_dir(data_dir, session_id)}")
        sink_factory = functools.partial(SessionSink, str(data_dir), session_id)
    else:
        if args.out:
            out = Path(args.out).resolve()
        else:
            print(f"Data directory: {data_dir}")
            out = data_dir / "probe" / f"probe-{datetime.now():%Y%m%d-%H%M%S}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Output: {out}")
        sink_factory = functools.partial(WaveFileSink, str(out))

    handle = RecorderHandle(
        CaptureConfig(sink_factory=sink_factory, silence_limit_s=args.silence_limit)
    )
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
    if session_id and owner_lock:
        try:
            if exitcode != 0:
                print("The capture process did not exit cleanly: recovering the session.")
                with closing(open_database(db_path(data_dir))) as conn:
                    result = recover_session(conn, data_dir, session_id, owner_lock=owner_lock)
                print_recovery([result] if result else [])
        finally:
            owner_lock.release()
        print_manifest(data_dir, session_id)
    return 0 if exitcode == 0 else 1


def print_recovery(results: list[Recovered]) -> None:
    if not results:
        print("Recovery: nothing to do.")
    for r in results:
        print(
            f"Recovered {r.session_id}: {r.status_before} -> {r.status_after}, "
            f"{r.duration_ms / 1000:.3f} s, segments added {r.added or '-'}, "
            f"files repaired {r.repaired or '-'}"
        )
        for problem in r.problems:
            print(f"  PROBLEM: {problem}")


def print_manifest(data_dir: Path, session_id: str) -> None:
    with closing(open_database(db_path(data_dir))) as conn:
        session = repo.get_session(conn, session_id)
        segments = repo.list_audio_segments(conn, session_id)
    print(f"\nSession {session_id}: status={session['status']}")
    print(f"  started_at={session['started_at']}  ended_at={session['ended_at']}")
    total_ms = gap_ms = 0
    for seg in segments:
        file = session_dir(data_dir, session_id) / seg["path"]
        with wave.open(str(file)) as w:
            file_ms = w.getnframes() * 1000 // w.getframerate()
        total_ms += seg["end_ms"] - seg["start_ms"]
        gap_ms += seg["gap_ms"]
        print(
            f"  {seg['seq']:>4}  {seg['path']}  {seg['start_ms']:>9}-{seg['end_ms']:<9} ms"
            f"  gap {seg['gap_ms']:>5} ms  file {file_ms} ms"
        )
    print(f"  total {total_ms / 1000:.3f} s in {len(segments)} segments, gap {gap_ms} ms")


if __name__ == "__main__":
    raise SystemExit(main())
