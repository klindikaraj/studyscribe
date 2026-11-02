"""Transcribing a session into the database: lifecycle, atomic replacement, resume, recovery,
CLI. Fake engine and energy VAD: no GPU, no model."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeTranscriber, loud_spans

from studyscribe import cli
from studyscribe.audio.wav import wav_header
from studyscribe.config import session_dir
from studyscribe.db import repo
from studyscribe.db.connection import transaction
from studyscribe.recovery import recover_transcriptions
from studyscribe.session import TRANSCRIBE_LOCK_BYTE, session_lock
from studyscribe.transcribe.engine import TranscriberError
from studyscribe.transcribe.runner import CannotTranscribe, transcribe_session

RATE = 16_000
SMALL_WINDOWS = {"max_window_s": 4.0, "search_s": 1.5}
BURSTS = [(0.5, 1.7), (2.5, 3.5), (4.2, 5.8), (6.4, 6.9), (8.0, 9.3)]


def timeline(bursts=BURSTS, seconds: float = 10.0) -> np.ndarray:
    pcm = np.zeros(round(seconds * RATE), dtype=np.int16)
    for start, end in bursts:
        t = np.arange(round((end - start) * RATE)) / RATE
        pcm[round(start * RATE) : round(start * RATE) + t.size] = np.round(
            0.14 * 32767 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
    return pcm


@pytest.fixture
def recorded(db, data_dir: Path, start_session):
    """A finished session of one 10 s segment (shorter than 60 s: it is the last one)."""
    def make(pcm: np.ndarray | None = None, status: str = "recorded") -> str:
        pcm = timeline() if pcm is None else pcm
        sid = start_session("Lecture")
        path = session_dir(data_dir, sid) / "audio" / "000001.wav"
        path.write_bytes(wav_header(pcm.size * 2) + pcm.astype("<i2").tobytes())
        with transaction(db):
            repo.add_audio_segment(db, sid, 1, "audio/000001.wav", 0, pcm.size * 1000 // RATE, 0)
            repo.finish_session(db, sid, status, repo.utc_now_iso())
        return sid
    return make


def run(db, data_dir, sid, engine=None, **kwargs):
    engine = engine or FakeTranscriber()
    options = {"vad": loud_spans, "window_options": SMALL_WINDOWS}
    options.update(kwargs)
    return transcribe_session(db, data_dir, sid, lambda: engine, **options)


class FailingTranscriber(FakeTranscriber):
    """Raises `error` when asked for window number `at` (0-based)."""

    def __init__(self, at: int, error: BaseException) -> None:
        super().__init__()
        self._at, self._error = at, error

    def transcribe(self, audio, language, hints=None):
        if len(self.windows) == self._at:
            self.windows.append((audio.size, language, hints))
            raise self._error
        yield from super().transcribe(audio, language, hints)


def spans(rows) -> list[tuple[int, int]]:
    return [(r["start_ms"], r["end_ms"]) for r in rows]


def test_transcript_is_stored_on_the_session_timeline(db, data_dir, recorded) -> None:
    sid = recorded()
    result = run(db, data_dir, sid)
    assert (result.language, result.language_source, result.kept, result.dropped) == (
        "en", "detected", 5, 0)
    rows = repo.list_transcript(db, sid)
    assert len(rows) == 5
    for row, (a, b) in zip(rows, BURSTS):
        assert abs(row["start_ms"] - a * 1000) <= 50 and abs(row["end_ms"] - b * 1000) <= 50
        assert json.loads(row["words_json"])[0][2] == row["text"]
    run_row = repo.get_done_run(db, sid)
    assert (run_row["status"], run_row["done_ms"], run_row["audio_ms"]) == ("done", 10_000, 10_000)
    assert json.loads(run_row["language_probs"])[0] == ["en", 0.9]
    session = repo.get_session(db, sid)
    assert (session["asr_engine"], session["asr_model"], session["language"]) == (
        "fake", "fake-model", "en")
    assert session["status"] == "recorded"  # the recording status is left alone


def test_sessions_that_cannot_be_transcribed(db, data_dir, recorded, start_session) -> None:
    loads = []

    def load():
        loads.append(1)
        return FakeTranscriber()

    live = start_session("still recording")
    for sid, why in [(live, "still being recorded"), ("nope", "no session nope"),
                     (recorded(status="failed"), "no audio")]:
        with pytest.raises(CannotTranscribe, match=why):
            transcribe_session(db, data_dir, sid, load)
    assert loads == []  # no model loaded for nothing
    sid = recorded()
    (session_dir(data_dir, sid) / "audio" / "000001.wav").unlink()
    with pytest.raises(CannotTranscribe, match="missing"):
        transcribe_session(db, data_dir, sid, load)


def test_a_session_being_transcribed_is_refused(db, data_dir, recorded) -> None:
    sid = recorded()
    lock = session_lock(data_dir, sid, TRANSCRIBE_LOCK_BYTE)
    assert lock.acquire()
    try:
        with pytest.raises(CannotTranscribe, match="another process"):
            run(db, data_dir, sid)
    finally:
        lock.release()


def test_retranscribing_replaces_the_transcript_atomically(db, data_dir, recorded) -> None:
    sid = recorded()
    run(db, data_dir, sid)
    first = [r["text"] for r in repo.list_transcript(db, sid)]
    seen_during = []

    def emit(kind, **info):
        if kind == "window":  # new rows committed, old transcript still the visible one
            seen_during.append([r["text"] for r in repo.list_transcript(db, sid)])

    engine = FakeTranscriber(text=lambda i: f"new {i}")
    run(db, data_dir, sid, engine, emit=emit, language="it")
    assert seen_during and all(texts == first for texts in seen_during)
    assert [r["text"] for r in repo.list_transcript(db, sid)] == [f"new {i}" for i in range(5)]
    runs = db.execute("SELECT COUNT(*) FROM transcription_runs").fetchone()[0]
    segments = db.execute("SELECT COUNT(*) FROM transcript_segments").fetchone()[0]
    assert (runs, segments) == (1, 5)  # the old run and its segments are gone
    assert repo.get_session(db, sid)["language"] == "it"


@pytest.mark.parametrize("error", [TranscriberError("out of memory"), KeyboardInterrupt()])
def test_interrupted_run_keeps_old_transcript_and_resumes(db, data_dir, recorded, error) -> None:
    sid = recorded()
    run(db, data_dir, sid, language="en")
    old = spans(repo.list_transcript(db, sid))
    with pytest.raises(type(error)):
        run(db, data_dir, sid, FailingTranscriber(at=1, error=error), language="en")
    open_run = repo.get_open_run(db, sid)
    assert open_run["status"] == "interrupted" and 0 < open_run["done_ms"] < 10_000
    assert open_run["error"] in ("out of memory", "interrupted by the user")
    assert spans(repo.list_transcript(db, sid)) == old  # still the previous transcript
    committed = spans(repo.list_run_segments(db, open_run["id"]))

    engine = FakeTranscriber()
    result = run(db, data_dir, sid, engine, language="en")
    assert result.resumed_from_ms == open_run["done_ms"]
    assert result.run_id == open_run["id"]
    final = spans(repo.list_transcript(db, sid))
    assert final[: len(committed)] == committed
    assert len(final) == len(old) and all(abs(a - c) <= 50 and abs(b - d) <= 50
                                          for (a, b), (c, d) in zip(final, old))
    assert sum(size for size, _, _ in engine.windows) < 10 * RATE  # only the rest was redone


def test_different_options_or_restart_start_again(db, data_dir, recorded) -> None:
    sid = recorded()
    with pytest.raises(TranscriberError):
        run(db, data_dir, sid, FailingTranscriber(at=1, error=TranscriberError("x")))
    events = []
    result = run(db, data_dir, sid, hints="JavaScript",
                 emit=lambda kind, **info: events.append((kind, info.get("reason"))))
    assert ("discarded", "different options") in events and result.resumed_from_ms == 0
    with pytest.raises(TranscriberError):
        run(db, data_dir, sid, FailingTranscriber(at=1, error=TranscriberError("x")))
    events.clear()
    run(db, data_dir, sid, restart=True,
        emit=lambda kind, **info: events.append((kind, info.get("reason"))))
    assert ("discarded", "--restart") in events


def test_no_speech_means_no_language_and_no_segments(db, data_dir, recorded) -> None:
    sid = recorded(np.zeros(5 * RATE, dtype=np.int16))
    engine = FakeTranscriber()
    result = run(db, data_dir, sid, engine)
    assert (result.language, result.kept) == (None, 0)
    assert engine.detected == [] and engine.windows == []
    assert repo.get_done_run(db, sid)["language"] is None
    assert repo.transcript_states(db)[sid] == "no speech"


def test_long_segments_are_split_before_storing(db, data_dir, recorded) -> None:
    # One 30 s burst: the fake engine returns one segment with one word per second.
    sid = recorded(timeline([(1.0, 31.0)], seconds=32.0))

    class Wordy(FakeTranscriber):
        def transcribe(self, audio, language, hints=None):
            from studyscribe.transcribe.engine import AsrSegment, AsrWord
            for seg in super().transcribe(audio, language, hints):
                words = tuple(AsrWord(t, t + 0.8, f" w{int(t)}." if int(t) % 5 == 0 else
                                      f" w{int(t)}", 0.9)
                              for t in np.arange(seg.start, seg.end - 0.5, 1.0))
                yield AsrSegment(seg.start, seg.end, "".join(w.text for w in words), words,
                                 -0.2, 0.01, 1.3)

    run(db, data_dir, sid, Wordy(), window_options={"max_window_s": 60.0, "search_s": 5.0})
    rows = repo.list_transcript(db, sid)
    assert len(rows) >= 3
    assert all(r["end_ms"] - r["start_ms"] <= 12_000 for r in rows)
    assert all(r["text"].endswith(".") for r in rows[:-1])


def test_orphaned_runs_are_recovered(db, data_dir, recorded) -> None:
    sid = recorded()
    with transaction(db):
        run_id = repo.create_run(db, sid, "fake", "fake-model", "none", "{}", 10_000)
    lock = session_lock(data_dir, sid, TRANSCRIBE_LOCK_BYTE)
    assert lock.acquire()
    try:
        assert recover_transcriptions(db, data_dir) == []  # its process still runs
    finally:
        lock.release()
    [recovered] = recover_transcriptions(db, data_dir)
    assert (recovered.run_id, recovered.session_id) == (run_id, sid)
    assert repo.get_run(db, run_id)["status"] == "interrupted"
    assert repo.transcript_states(db)[sid] == "interrupted 0%"


# --- CLI -----------------------------------------------------------------------------------


def transcribe_cli(db, data_dir, *argv, engine=None):
    args = cli.build_parser().parse_args(["transcribe", *argv])
    out = io.StringIO()
    code = cli.cmd_transcribe(db, data_dir, args, out, load=lambda: engine or FakeTranscriber(),
                              vad=loud_spans, window_options=SMALL_WINDOWS)
    return code, out.getvalue()


def test_cli_transcribes_and_lists_the_transcript(db, data_dir, recorded) -> None:
    sid = recorded()
    code, out = transcribe_cli(db, data_dir, sid)
    assert code == 0
    assert "Language: en (detected on" in out and "en 0.90, it 0.10" in out
    assert "Transcribed up to 0:00:10 of 0:00:10" in out  # not a terminal: one line per window
    assert "Done: 0:00:10 in" in out and "5 segments kept, 0 dropped" in out
    listing = io.StringIO()
    cli.cmd_sessions(db, data_dir, listing)
    line = next(l for l in listing.getvalue().splitlines() if l.startswith(sid))
    assert line.split()[6] == "en"


def test_cli_errors_and_resume_messages(db, data_dir, recorded) -> None:
    sid = recorded()
    assert transcribe_cli(db, data_dir, sid, "--language", "xx")[0] == 2
    code, out = transcribe_cli(db, data_dir, "missing")
    assert code == 2 and "Cannot transcribe missing: no session" in out
    code, out = transcribe_cli(db, data_dir, sid, engine=FailingTranscriber(
        at=1, error=TranscriberError("the GPU ran out of memory")))
    assert code == 1 and "Error: the GPU ran out of memory" in out
    code, out = transcribe_cli(db, data_dir, sid)
    assert code == 0 and "Resuming an interrupted transcription at" in out
    assert "Done: 0:00:10 (0:00:0" in out and "of processing in all" in out


def test_cli_ctrl_c_says_how_to_resume(db, data_dir, recorded) -> None:
    sid = recorded()
    code, out = transcribe_cli(db, data_dir, sid, "--language", "en",
                               engine=FailingTranscriber(at=1, error=KeyboardInterrupt()))
    assert code == 1 and "Interrupted at 0:00:0" in out and "same command to resume" in out
    recover_out = io.StringIO()
    cli.cmd_sessions(db, data_dir, recover_out)
    assert "interrupted" in recover_out.getvalue()


def test_progress_line_on_a_terminal() -> None:
    out = io.StringIO()
    now = [0.0]
    progress = cli.TranscribeProgress(out, clock=lambda: now[0], live=True)
    progress.show("started", run_id=1, audio_ms=3_600_000)
    now[0] = 60.0
    progress.show("segment", end_ms=1_800_000)
    line = out.getvalue().split("\r")[-1]
    assert "50%" in line and "0:30:00 / 1:00:00" in line and "elapsed 0:01:00" in line
    assert "left ~0:01:00" in line and "30x" in line
