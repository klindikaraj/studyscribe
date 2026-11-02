"""GPU probe for transcription: one 60 s segment of a session through faster-whisper.

Usage:
    .venv\\Scripts\\python scripts\\transcribe_probe.py --data-dir PATH --session ID [--segment N]
        [--model large-v3-turbo] [--compute-type float16] [--device cuda] [--language en]
    .venv\\Scripts\\python scripts\\transcribe_probe.py --data-dir PATH --session ID
        --from 0:09:00 [--to 0:11:00] [--no-normalize] [--language en]

The second form runs the whole step 2 pipeline on a stretch of the session instead: continuous
reading across segments, normalisation, language detection on speech, windows cut in pauses,
filters (dropped segments are shown as such).

Prints library versions and what CUDA offers, downloads the model if needed (and checks
model.bin against the hash published for the pinned revision), then times the model load and
two transcriptions of the segment (the first includes CUDA warm-up), reports VRAM in use
(nvidia-smi, sampled every 0.1 s), the detected language and the text with times on the
session timeline. Options as planned for step 2: vad_filter, word_timestamps, no conditioning
on previous text. The WAV file is only read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

from studyscribe.config import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_MODEL,
    PINNED_MODELS,
    SEGMENT_SECONDS,
    resolve_data_dir,
    session_dir,
)
from studyscribe.transcribe.cuda import preload_cublas


class VramMonitor(threading.Thread):
    """Total VRAM in use (MiB) from nvidia-smi: per-process figures are N/A under WDDM."""

    def __init__(self, interval: float = 0.1) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.last = 0
        self._done = threading.Event()

    @staticmethod
    def read() -> int:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
        return int(out.split()[0])

    def run(self) -> None:
        while not self._done.is_set():
            self.last = self.read()
            self.peak = max(self.peak, self.last)
            time.sleep(self.interval)

    def stop(self) -> None:
        self._done.set()
        self.join()


def published_sha256(repo: str, revision: str, filename: str) -> tuple[int, str]:
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}"
    with urllib.request.urlopen(url, timeout=30) as response:
        for entry in json.load(response):
            if entry["path"] == filename:
                return entry["size"], entry["lfs"]["oid"]
    raise LookupError(f"{filename} not listed in {repo}@{revision}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_segment(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16_000)
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def transcribe(model, audio: np.ndarray, language: str | None):
    started = time.perf_counter()
    segments, info = model.transcribe(
        audio,
        language=language,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
        beam_size=5,
    )
    segments = list(segments)  # decoding happens while iterating
    return segments, info, time.perf_counter() - started


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}.{int(seconds * 100) % 100:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, metavar="PATH")
    parser.add_argument("--session", required=True, metavar="ID")
    parser.add_argument("--segment", type=int, default=10, metavar="N",
                        help="segment number (default 10: 0:09:00-0:10:00)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--compute-type", default=DEFAULT_COMPUTE_TYPE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default=None, help="default: detected")
    parser.add_argument("--from", dest="start", metavar="H:MM:SS",
                        help="pipeline mode: transcribe the session from here (continuous "
                        "reading, normalisation, windows, filters) instead of one segment")
    parser.add_argument("--to", dest="end", metavar="H:MM:SS", help="pipeline mode: up to here")
    parser.add_argument("--hints", metavar="TEXT", help="pipeline mode: hints for the model")
    parser.add_argument("--no-normalize", action="store_true",
                        help="pipeline mode: feed the audio as recorded")
    args = parser.parse_args()
    if args.start is not None:
        return pipeline_mode(args)

    wav = session_dir(resolve_data_dir(args.data_dir), args.session) / "audio" / (
        f"{args.segment:06d}.wav"
    )
    print(f"Segment: {wav}")
    audio = read_segment(wav)
    rms = float(np.sqrt(np.mean(audio * audio)))
    print(f"  {len(audio) / 16_000:.2f} s, RMS {20 * np.log10(max(rms, 1e-9)):.1f} dBFS")

    if args.device == "cuda":
        for path in preload_cublas():
            print(f"Preloaded {path}")
    import ctranslate2
    import faster_whisper
    from faster_whisper.utils import download_model

    print(f"faster-whisper {faster_whisper.__version__}, ctranslate2 {ctranslate2.__version__}")
    if args.device == "cuda":
        print(f"CUDA devices: {ctranslate2.get_cuda_device_count()}, compute types: "
              f"{sorted(ctranslate2.get_supported_compute_types('cuda'))}")

    repo, revision = PINNED_MODELS.get(args.model, (args.model, None))
    started = time.perf_counter()
    model_dir = Path(download_model(repo, revision=revision))
    print(f"Model {repo}@{(revision or 'default')[:7]}: {model_dir} "
          f"(download/lookup {time.perf_counter() - started:.1f} s)")
    if revision is not None:
        size, expected = published_sha256(repo, revision, "model.bin")
        local = model_dir / "model.bin"
        actual = sha256_of(local)
        same = actual == expected and local.stat().st_size == size
        print(f"  model.bin {local.stat().st_size} bytes, sha256 {actual}\n"
              f"  published {size} bytes, sha256 {expected}: {'MATCH' if same else 'MISMATCH'}")

    vram = VramMonitor()
    baseline = vram.read()
    vram.start()
    try:
        started = time.perf_counter()
        model = faster_whisper.WhisperModel(
            str(model_dir), device=args.device, compute_type=args.compute_type
        )
        load_s = time.perf_counter() - started
        after_load = vram.read()
        print(f"Model loaded in {load_s:.2f} s ({args.device}, {args.compute_type}); "
              f"VRAM {baseline} -> {after_load} MiB (+{after_load - baseline})")

        segments, info, cold_s = transcribe(model, audio, args.language)
        _, _, warm_s = transcribe(model, audio, args.language)
    finally:
        vram.stop()
    seconds = len(audio) / 16_000
    print(f"Transcription of {seconds:.0f} s: first {cold_s:.2f} s ({seconds / cold_s:.0f}x), "
          f"second {warm_s:.2f} s ({seconds / warm_s:.0f}x)")
    print(f"VRAM peak {vram.peak} MiB (+{vram.peak - baseline} over the {baseline} MiB before "
          "loading)")
    top = sorted(info.all_language_probs or [], key=lambda p: -p[1])[:3]
    print(f"Language: {info.language} ({info.language_probability:.2f}); top 3: "
          + ", ".join(f"{lang} {p:.2f}" for lang, p in top))
    print(f"Speech after VAD: {info.duration_after_vad:.1f} of {info.duration:.1f} s\n")

    offset = (args.segment - 1) * SEGMENT_SECONDS
    for s in segments:
        low = min((w.probability for w in s.words), default=1.0)
        print(f"[{clock(offset + s.start)} -> {clock(offset + s.end)}] {s.text.strip()}")
        print(f"    no_speech {s.no_speech_prob:.2f}  avg_logprob {s.avg_logprob:.2f}  "
              f"compression {s.compression_ratio:.2f}  words {len(s.words)}, lowest p {low:.2f}")
    return 0


def parse_clock(text: str) -> float:
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def pipeline_mode(args) -> int:
    """The step 2 pipeline on [--from, --to) of the session, with the real model and VAD."""
    from contextlib import closing

    from studyscribe.audio.reader import SessionAudio
    from studyscribe.config import db_path
    from studyscribe.db.connection import connect
    from studyscribe.transcribe import language as lang
    from studyscribe.transcribe import pipeline
    from studyscribe.transcribe.whisper import FasterWhisperTranscriber

    data_dir = resolve_data_dir(args.data_dir)
    with closing(connect(db_path(data_dir))) as conn:  # read only: no migration, no recovery
        audio = SessionAudio.from_session(conn, data_dir, args.session)
    start = int(parse_clock(args.start) * 16_000)
    end = audio.total_frames if args.end is None else int(parse_clock(args.end) * 16_000)
    normalize_audio = not args.no_normalize
    started = time.perf_counter()
    engine = FasterWhisperTranscriber.load(args.model, args.compute_type, args.device)
    print(f"Model {engine.model} loaded in {time.perf_counter() - started:.1f} s; "
          f"normalisation {'on' if normalize_audio else 'off'}; hints {args.hints!r}")
    language = args.language
    if language is None:
        started = time.perf_counter()
        guess = pipeline.detect_language(audio, engine, start, end,
                                         normalize_audio=normalize_audio)
        took = time.perf_counter() - started
        if guess is None:
            print(f"Language: none (no speech in the range) [{took:.1f} s]")
            return 0
        print(f"Language: {guess.language} on {guess.speech_s:.0f} s of speech "
              f"({lang.describe(guess)}) [{took:.1f} s]")
        if lang.doubt(guess):
            print(f"  warning: {lang.doubt(guess)}")
        language = guess.language
    started = time.perf_counter()
    for window in pipeline.transcribe_windows(audio, engine, language, start, end,
                                              normalize_audio=normalize_audio, hints=args.hints):
        print(f"--- window {clock(window.start_frame / 16_000)} -> "
              f"{clock(window.end_frame / 16_000)}"
              + ("" if window.cut_in_pause else " (cut outside a pause)"))
        for s in window.segments:
            mark = f"  DROPPED ({s.dropped})" if s.dropped else ""
            p = s.mean_word_probability
            print(f"[{clock(s.start_ms / 1000)} -> {clock(s.end_ms / 1000)}] {s.text}{mark}")
            print(f"    no_speech {s.no_speech_prob:.2f}  avg_logprob {s.avg_logprob:.2f}  "
                  f"words {len(s.words)}, mean p {p if p is None else round(p, 2)}")
    took = time.perf_counter() - started
    seconds = (end - start) / 16_000
    print(f"Transcribed {seconds:.0f} s in {took:.1f} s ({seconds / took:.0f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
