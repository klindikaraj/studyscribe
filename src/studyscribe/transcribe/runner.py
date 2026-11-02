"""Transcribe a session into the database: checks, lock, run lifecycle, resume.

The run is committed window by window, so an interruption (Ctrl+C, out of GPU memory, a killed
process) loses at most the window in progress; the same call resumes it. The session's
previous transcript stays visible until the new run is complete (see db/migrations.py, V2).
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from studyscribe import verify
from studyscribe.audio.reader import SessionAudio, SessionAudioError
from studyscribe.config import SAMPLE_RATE
from studyscribe.db import repo
from studyscribe.db.connection import transaction
from studyscribe.session import TRANSCRIBE_LOCK_BYTE, session_lock
from studyscribe.transcribe import filters
from studyscribe.transcribe import language as lang
from studyscribe.transcribe import pipeline, segments, windows
from studyscribe.transcribe.engine import Transcriber
from studyscribe.transcribe.segments import TranscriptSegment
from studyscribe.transcribe.vad import Vad, speech_spans

Emit = Callable[..., None]  # emit(kind, **info)
TRANSCRIBABLE = ("recorded", "interrupted")


class CannotTranscribe(RuntimeError):
    """The session cannot be transcribed now (unknown, still recording, broken manifest,
    already being transcribed). Nothing was changed."""


@dataclass
class Result:
    run_id: int
    language: str | None  # None: no speech in the session
    language_source: str
    guess: lang.LanguageGuess | None  # the detection, when there was one in this call
    kept: int
    dropped: int
    audio_ms: int
    resumed_from_ms: int  # 0 for a fresh run
    processing_s: float  # summed over every attempt of the run
    warnings: list[str] = field(default_factory=list)


def _ignore(*args: Any, **kwargs: Any) -> None:
    pass


def run_options(transcriber: Transcriber, language: str, hints: str | None,
                normalize_audio: bool) -> dict:
    """Everything that shapes the output of a run: a run is resumed only if these match."""
    return {
        "language": language,
        "hints": hints,
        "normalize": normalize_audio,
        "engine": transcriber.options(),
        "windows": {"max_s": windows.MAX_WINDOW_S, "search_s": windows.SEARCH_S},
        "max_segment_ms": segments.MAX_SEGMENT_MS,
        "filters": {
            "no_speech_prob": filters.NO_SPEECH_PROB, "log_prob": filters.LOG_PROB,
            "compression_ratio": filters.COMPRESSION_RATIO,
            "duplicate_run": filters.DUPLICATE_RUN, "isolation_ms": filters.ISOLATION_MS,
            "low_word_probability": filters.LOW_WORD_PROBABILITY, "short_ms": filters.SHORT_MS,
            "known_phrases": sorted(filters.KNOWN_PHRASES),
        },
    }


def transcribe_session(
    conn: sqlite3.Connection,
    data_dir: Path,
    session_id: str,
    load: Callable[[], Transcriber],
    language: str = "auto",
    hints: str | None = None,
    restart: bool = False,
    emit: Emit = _ignore,
    vad: Vad = speech_spans,
    normalize_audio: bool = True,
    window_options: dict | None = None,
) -> Result:
    """Transcribe (or resume transcribing) a recorded session; return what was done.

    `load` builds the engine: it is called only after the checks, so a session that cannot be
    transcribed does not cost a model load. Raises CannotTranscribe, TranscriberError (engine
    failure: the run stays interrupted and resumable) or KeyboardInterrupt (likewise).
    """
    session = repo.get_session(conn, session_id)
    if session is None:
        raise CannotTranscribe(f"no session {session_id} in this data directory")
    if session["status"] not in TRANSCRIBABLE:
        why = {"recording": "it is still being recorded (live transcription comes later)",
               "failed": "its recording failed: there is no audio"}
        raise CannotTranscribe(why.get(session["status"], f"its status is {session['status']}"))
    problems = verify.manifest_problems(conn, data_dir, session_id)
    if problems:
        raise CannotTranscribe("the segments do not match the manifest: " + "; ".join(problems))
    try:
        audio = SessionAudio.from_session(conn, data_dir, session_id)
    except SessionAudioError as exc:
        raise CannotTranscribe(str(exc)) from exc

    lock = session_lock(data_dir, session_id, TRANSCRIBE_LOCK_BYTE)
    if not lock.acquire():
        raise CannotTranscribe("another process is transcribing this session")
    run_id = None
    try:
        transcriber = load()
        options_json = json.dumps(run_options(transcriber, language, hints, normalize_audio),
                                  sort_keys=True)
        audio_ms = audio.total_frames * 1000 // SAMPLE_RATE
        run_id, resumed_from = _open_run(conn, session_id, transcriber, options_json, audio_ms,
                                         restart, emit)
        result = _run(conn, run_id, audio, transcriber, language, hints, emit, vad,
                      normalize_audio, window_options or {})
        result.resumed_from_ms = resumed_from
        return result
    except BaseException as exc:
        if run_id is not None:
            error = "interrupted by the user" if isinstance(exc, KeyboardInterrupt) else str(exc)
            with transaction(conn):
                repo.set_run_status(conn, run_id, "interrupted", error or type(exc).__name__)
        raise
    finally:
        lock.release()


def _open_run(conn, session_id, transcriber, options_json, audio_ms, restart, emit):
    """(run id, done_ms it resumes from): the interrupted run if it matches, else a new one."""
    with transaction(conn):
        run = repo.get_open_run(conn, session_id)
        if run is not None:
            same = (run["engine"], run["model"], run["compute_type"], run["options_json"],
                    run["audio_ms"]) == (transcriber.engine, transcriber.model,
                                         transcriber.compute_type, options_json, audio_ms)
            if same and not restart:
                repo.set_run_status(conn, run["id"], "running")
                emit("resumed", run_id=run["id"], done_ms=run["done_ms"], audio_ms=audio_ms)
                return run["id"], run["done_ms"]
            repo.delete_run(conn, run["id"])
            emit("discarded", run_id=run["id"], done_ms=run["done_ms"],
                 reason="--restart" if restart else "different options")
        run_id = repo.create_run(conn, session_id, transcriber.engine, transcriber.model,
                                 transcriber.compute_type, options_json, audio_ms)
    emit("started", run_id=run_id, audio_ms=audio_ms)
    return run_id, 0


def _run(conn, run_id, audio, transcriber, language, hints, emit, vad, normalize_audio,
         window_options) -> Result:
    run = repo.get_run(conn, run_id)
    warnings: list[str] = []
    guess = None
    started = time.perf_counter()
    if run["language_source"] is None:  # not chosen yet in this run
        if language != "auto":
            chosen, source, probs = language, "forced", None
        else:
            guess = pipeline.detect_language(audio, transcriber, vad=vad,
                                             normalize_audio=normalize_audio)
            chosen, source = (guess.language if guess else None), "detected"
            probs = None if guess is None else json.dumps(guess.top)
            if guess is not None and lang.doubt(guess):
                warnings.append(lang.doubt(guess))
        with transaction(conn):
            repo.set_run_language(conn, run_id, chosen, source, probs)
        run = repo.get_run(conn, run_id)
    emit("language", language=run["language"], source=run["language_source"], guess=guess,
         warnings=warnings)

    pending_s = time.perf_counter() - started  # detection time, counted with the first window
    if run["language"] is not None:
        start = run["done_ms"] * SAMPLE_RATE // 1000
        before = [_segment(row) for row in repo.list_run_segments(conn, run_id)][-5:]
        mark = time.perf_counter()
        for window in pipeline.transcribe_windows(
            audio, transcriber, run["language"], start=start, hints=hints, vad=vad,
            normalize_audio=normalize_audio, before=before,
            on_segment=lambda s: emit("segment", end_ms=s.end_ms), **window_options,
        ):
            now = time.perf_counter()
            done_ms = window.end_frame * 1000 // SAMPLE_RATE
            with transaction(conn):
                repo.add_transcript_window(conn, run_id, [_row(s) for s in window.segments],
                                           done_ms, now - mark + pending_s)
            pending_s, mark = 0.0, now
            if not window.cut_in_pause:
                warnings.append(f"no pause found to end the window at "
                                f"{window.end_frame / SAMPLE_RATE:.1f} s: a word may be split")
            emit("window", start_ms=window.start_frame * 1000 // SAMPLE_RATE, end_ms=done_ms,
                 kept=len(window.kept), dropped=len(window.segments) - len(window.kept))
    with transaction(conn):
        if pending_s:
            repo.add_transcript_window(conn, run_id, [], run["audio_ms"], pending_s)
        repo.finish_run(conn, run_id)
    run = repo.get_run(conn, run_id)
    rows = repo.list_run_segments(conn, run_id)
    dropped = sum(row["dropped"] is not None for row in rows)
    return Result(run_id, run["language"], run["language_source"], guess, len(rows) - dropped,
                  dropped, run["audio_ms"], 0, run["processing_s"], warnings)


def _row(s: TranscriptSegment) -> tuple:
    return (s.start_ms, s.end_ms, s.text, json.dumps(s.words, ensure_ascii=False),
            s.avg_logprob, s.no_speech_prob, s.compression_ratio, s.dropped)


def _segment(row: sqlite3.Row) -> TranscriptSegment:
    words = [tuple(w) for w in json.loads(row["words_json"] or "[]")]
    return TranscriptSegment(row["start_ms"], row["end_ms"], row["text"], words,
                             row["avg_logprob"], row["no_speech_prob"], row["compression_ratio"],
                             row["dropped"])
