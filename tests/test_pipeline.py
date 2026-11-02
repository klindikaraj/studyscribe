"""The transcription pipeline with a fake engine and an energy VAD (no GPU, no model)."""

from pathlib import Path

import numpy as np
import pytest
from fakes import FakeTranscriber, loud_spans
from test_reader import write_segments

from studyscribe.audio.reader import SessionAudio
from studyscribe.transcribe import pipeline

RATE = 16_000
SEGMENT = RATE  # 1 s files in these tests
WINDOW = {"max_window_s": 4.0, "search_s": 1.5, "vad": loud_spans}


def burst(seconds: float, db: float = -20.0) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    return np.round(10 ** (db / 20) * np.sqrt(2) * 32767 * np.sin(2 * np.pi * 300 * t)).astype(
        np.int16)


def quiet(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * RATE), dtype=np.int16)


def session(tmp_path: Path, *parts: np.ndarray) -> SessionAudio:
    timeline = np.concatenate(parts)
    return SessionAudio(tmp_path, write_segments(tmp_path, timeline, SEGMENT), SEGMENT)


# Bursts ("words") at 0.5-1.7 s (across the 1 s file boundary), 2.5-3.5 s, 4.2-5.8 s (across
# 5 s), 6.4-6.9 s, 8.0-9.3 s; 10 s in all.
BURSTS = [(0.5, 1.7), (2.5, 3.5), (4.2, 5.8), (6.4, 6.9), (8.0, 9.3)]


def lecture(tmp_path: Path, db: float = -20.0) -> SessionAudio:
    parts, position = [], 0.0
    for start, end in BURSTS:
        parts += [quiet(start - position), burst(end - start, db)]
        position = end
    parts.append(quiet(10.0 - position))
    return session(tmp_path, *parts)


def test_segments_land_at_their_session_times(tmp_path: Path) -> None:
    audio = lecture(tmp_path)
    windows = list(pipeline.transcribe_windows(audio, FakeTranscriber(), "en", **WINDOW))
    segments = [s for w in windows for s in w.segments]
    # The fake VAD works on 50 ms frames counted from each window's start.
    assert len(segments) == len(BURSTS)
    for s, (a, b) in zip(segments, BURSTS):
        assert abs(s.start_ms - a * 1000) <= 50 and abs(s.end_ms - b * 1000) <= 50
    assert segments[0].words == [(500, 1700, "words 0", 0.9)]


def test_windows_cover_the_session_and_are_cut_in_pauses(tmp_path: Path) -> None:
    audio = lecture(tmp_path)
    windows = list(pipeline.transcribe_windows(audio, FakeTranscriber(), "en", **WINDOW))
    assert windows[0].start_frame == 0 and windows[-1].end_frame == audio.total_frames
    for previous, current in zip(windows, windows[1:]):
        assert previous.end_frame == current.start_frame
    for w in windows[:-1]:
        cut_s = w.end_frame / RATE
        assert w.cut_in_pause
        assert not any(a < cut_s < b for a, b in BURSTS), f"cut at {cut_s} s inside a word"
        assert w.end_frame - w.start_frame <= 4 * RATE
    assert len(windows) >= 3


def test_resume_from_a_window_end_gives_the_same_rest(tmp_path: Path) -> None:
    audio = lecture(tmp_path)
    full = list(pipeline.transcribe_windows(audio, FakeTranscriber(), "en", **WINDOW))
    resumed = list(pipeline.transcribe_windows(
        audio, FakeTranscriber(), "en", start=full[1].start_frame,
        before=full[0].segments, **WINDOW))
    assert [(w.start_frame, w.end_frame) for w in resumed] == [
        (w.start_frame, w.end_frame) for w in full[1:]]
    assert [(s.start_ms, s.end_ms) for w in resumed for s in w.segments] == [
        (s.start_ms, s.end_ms) for w in full[1:] for s in w.segments]


def test_engine_gets_language_hints_and_the_window_audio(tmp_path: Path) -> None:
    audio = lecture(tmp_path)
    engine = FakeTranscriber()
    windows = list(pipeline.transcribe_windows(audio, engine, "it", hints="JavaScript", **WINDOW))
    assert [size for size, _, _ in engine.windows] == [w.end_frame - w.start_frame
                                                        for w in windows]
    assert {(language, hints) for _, language, hints in engine.windows} == {("it", "JavaScript")}


def test_quiet_audio_is_normalised_before_the_vad_and_the_engine(tmp_path: Path) -> None:
    audio = lecture(tmp_path, db=-45.0)  # below the fake VAD's -40 dBFS without normalisation
    raw = list(pipeline.transcribe_windows(audio, FakeTranscriber(), "en",
                                           normalize_audio=False, **WINDOW))
    assert not [s for w in raw for s in w.segments]
    normalised = list(pipeline.transcribe_windows(audio, FakeTranscriber(), "en", **WINDOW))
    assert len([s for w in normalised for s in w.segments]) == len(BURSTS)


def test_filters_run_on_each_window(tmp_path: Path) -> None:
    audio = lecture(tmp_path)
    engine = FakeTranscriber(text=lambda i: "Thanks for watching" if i == 3 else f"words {i}")
    windows = list(pipeline.transcribe_windows(audio, engine, "en", **WINDOW))
    segments = [s for w in windows for s in w.segments]
    # Burst 3 (6.4-6.9 s) has 0.6 s and 1.1 s of quiet around it: not isolated, kept.
    assert [s.dropped for s in segments] == [None] * 5


def test_language_is_detected_on_speech_only(tmp_path: Path) -> None:
    audio = lecture(tmp_path)
    engine = FakeTranscriber({"en": 0.8, "it": 0.2})
    guess = pipeline.detect_language(audio, engine, vad=loud_spans)
    assert guess.language == "en" and guess.probability == pytest.approx(0.8)
    speech = sum(round((b - a) * RATE) for a, b in BURSTS)
    assert sum(engine.detected) == pytest.approx(speech, abs=5 * 800)  # frame rounding
    assert guess.speech_s == pytest.approx(speech / RATE, abs=0.3)


def test_no_speech_means_no_language(tmp_path: Path) -> None:
    audio = session(tmp_path, quiet(5.0))
    engine = FakeTranscriber()
    assert pipeline.detect_language(audio, engine, vad=loud_spans) is None
    assert engine.detected == []  # never asked to guess from silence
