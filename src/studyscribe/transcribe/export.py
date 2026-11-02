"""Exports of a session's transcript (<session>/exports/): SRT, Markdown, the dropped segments,
and the session audio joined into one FLAC file with the same timeline.

Times everywhere are positions on the session timeline: 0 is the first sample of 000001.wav,
the point at t is in segment floor(t / 60 s) + 1, and at t in audio.flac. Every file is written
to a temporary name first, then renamed, so a reader never sees half a file.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

from studyscribe.audio.reader import SessionAudio
from studyscribe.config import SAMPLE_RATE, SEGMENT_SECONDS, session_dir
from studyscribe.db import repo

EXPORTS_SUBDIR = "exports"
SRT_FILE = "transcript.srt"
MD_FILE = "transcript.md"
DROPPED_FILE = "transcript-dropped.txt"
AUDIO_FILE = "audio.flac"

# Subtitle cues: at most two lines of LINE_CHARS and CUE_MAX_MS on screen (common subtitle
# practice), so long segments become several cues.
LINE_CHARS = 42
CUE_MAX_CHARS = 2 * LINE_CHARS
CUE_MAX_MS = 7_000
# Markdown paragraphs: a new one after a pause this long, or once a paragraph lasts this long.
PARAGRAPH_PAUSE_MS = 3_000
PARAGRAPH_MAX_MS = 60_000
AUDIO_BLOCK = SEGMENT_SECONDS * SAMPLE_RATE  # samples read and encoded at a time


class NoTranscript(RuntimeError):
    """The session has no complete transcript to export."""


# --- formatting (pure) ---------------------------------------------------------------------------


def clock(ms: int) -> str:
    """H:MM:SS."""
    s = ms // 1000
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def srt_time(ms: int) -> str:
    s, milli = divmod(ms, 1000)
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d},{milli:03d}"


def wrap_cue(text: str) -> str:
    """One line, or two split at the space nearest the middle if longer than LINE_CHARS."""
    if len(text) <= LINE_CHARS or " " not in text:
        return text
    middle = len(text) // 2
    spaces = [i for i, c in enumerate(text) if c == " "]
    cut = min(spaces, key=lambda i: abs(i - middle))
    return text[:cut] + "\n" + text[cut + 1 :]


def cues(segment: sqlite3.Row | dict) -> list[tuple[int, int, str]]:
    """Subtitle cues of a segment: whole if it is short enough, else split at words."""
    text = segment["text"].strip()
    start, end = segment["start_ms"], segment["end_ms"]
    words = [w for w in json.loads(segment["words_json"] or "[]") if w[2].strip()]
    if (len(text) <= CUE_MAX_CHARS and end - start <= CUE_MAX_MS) or len(words) < 2:
        return [(start, end, text)]
    result, current = [], []
    for word in words:
        candidate = current + [word]
        joined = "".join(w[2] for w in candidate).strip()
        if current and (len(joined) > CUE_MAX_CHARS or word[1] - current[0][0] > CUE_MAX_MS):
            result.append((current[0][0], current[-1][1], "".join(w[2] for w in current).strip()))
            current = [word]
        else:
            current = candidate
    result.append((current[0][0], current[-1][1], "".join(w[2] for w in current).strip()))
    return result


def to_srt(segments: Iterable) -> str:
    blocks = []
    for start, end, text in (cue for s in segments for cue in cues(s)):
        blocks.append(f"{len(blocks) + 1}\n{srt_time(start)} --> {srt_time(end)}\n"
                      f"{wrap_cue(text)}\n")
    return "\n".join(blocks)


def paragraphs(segments: Sequence) -> list[tuple[int, str]]:
    """(start ms, text) of each paragraph."""
    result: list[tuple[int, list[str]]] = []
    last_end = None
    for s in segments:
        if (last_end is None or s["start_ms"] - last_end >= PARAGRAPH_PAUSE_MS
                or s["start_ms"] - result[-1][0] >= PARAGRAPH_MAX_MS):
            result.append((s["start_ms"], []))
        result[-1][1].append(s["text"].strip())
        last_end = s["end_ms"]
    return [(start, " ".join(texts)) for start, texts in result]


def to_markdown(header: dict, segments: Sequence) -> str:
    lines = [f"# {header['title']}", ""]
    for label in ("Course", "Recorded", "Duration", "Language", "Model", "Transcribed"):
        if header.get(label):
            lines.append(f"- {label}: {header[label]}")
    lines += [
        "",
        f"Times are positions in the recording: the point at `H:MM:SS` is at the same time in "
        f"`exports/{AUDIO_FILE}` (`studyscribe export --audio`), or in `audio/NNNNNN.wav` with "
        f"NNNNNN = minutes + 1, at the remaining seconds.",
        "",
    ]
    if not segments:
        lines.append("*No speech was found in this recording.*")
    for start, text in paragraphs(segments):
        lines += [f"**[{clock(start)}]** {text}", ""]
    return "\n".join(lines).rstrip() + "\n"


def to_dropped(segments: Sequence) -> str:
    dropped = [s for s in segments if s["dropped"]]
    lines = [f"Segments dropped by the anti-hallucination filters: {len(dropped)}", ""]
    for s in dropped:
        words = json.loads(s["words_json"] or "[]")
        mean_p = sum(w[3] for w in words) / len(words) if words else None
        metrics = "  ".join(
            f"{name} {value:.2f}" for name, value in (
                ("no_speech", s["no_speech_prob"]), ("avg_logprob", s["avg_logprob"]),
                ("compression", s["compression_ratio"]), ("mean_word_p", mean_p),
            ) if value is not None
        )
        lines.append(f"[{clock(s['start_ms'])} -> {clock(s['end_ms'])}] {s['dropped']}  "
                     f"({metrics})\n    {s['text']}")
    return "\n".join(lines).rstrip() + "\n"


# --- files ---------------------------------------------------------------------------------------


def exports_dir(data_dir: Path, session_id: str) -> Path:
    return session_dir(data_dir, session_id) / EXPORTS_SUBDIR


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def export_text(conn: sqlite3.Connection, data_dir: Path, session_id: str) -> list[Path]:
    """Write transcript.srt, transcript.md and transcript-dropped.txt; return their paths."""
    run = repo.get_done_run(conn, session_id)
    if run is None:
        raise NoTranscript(f"session {session_id} has no complete transcript: run `transcribe`")
    all_segments = repo.list_run_segments(conn, run["id"])
    kept = [s for s in all_segments if s["dropped"] is None]
    folder = exports_dir(data_dir, session_id)
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        SRT_FILE: to_srt(kept),
        MD_FILE: to_markdown(_header(conn, session_id, run), kept),
        DROPPED_FILE: to_dropped(all_segments),
    }
    for name, text in files.items():
        _write_text(folder / name, text)
    return [folder / name for name in files]


def _header(conn: sqlite3.Connection, session_id: str, run: sqlite3.Row) -> dict:
    session = repo.get_session(conn, session_id)
    summary = repo.get_session_summary(conn, session_id)
    if run["language"] is None:
        language = "none (no speech)"
    else:
        language = f"{run['language']} ({run['language_source']})"
        if run["language_probs"]:
            top = json.loads(run["language_probs"])
            language += " - " + ", ".join(f"{code} {p:.2f}" for code, p in top)
    return {
        "title": session["title"] or session_id,
        "Course": summary["course"],
        "Recorded": _local(session["started_at"]) + f" (session {session_id})",
        "Duration": clock(summary["duration_ms"]),
        "Language": language,
        "Model": f"{run['engine']} {run['model']} ({run['compute_type']})",
        "Transcribed": _local(run["finished_at"]) if run["finished_at"] else None,
    }


def _local(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")


def export_audio(conn: sqlite3.Connection, data_dir: Path, session_id: str) -> Path:
    """Join the session's segments into exports/audio.flac: lossless, so it holds exactly the
    samples of the segments and time t of the transcript is at t in the file."""
    import av

    audio = SessionAudio.from_session(conn, data_dir, session_id)
    folder = exports_dir(data_dir, session_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / AUDIO_FILE
    tmp = path.with_name(path.name + ".tmp")
    with av.open(str(tmp), "w", format="flac") as container:
        stream = container.add_stream("flac", rate=SAMPLE_RATE, layout="mono")
        stream.format = "s16"
        for start in range(0, audio.total_frames, AUDIO_BLOCK):
            pcm = audio.read(start, AUDIO_BLOCK)
            frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = SAMPLE_RATE
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    os.replace(tmp, path)
    return path
