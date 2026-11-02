"""faster-whisper engine: the only module that imports faster_whisper and ctranslate2."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from studyscribe.config import DEFAULT_COMPUTE_TYPE, DEFAULT_MODEL, PINNED_MODELS
from studyscribe.transcribe.engine import AsrSegment, AsrWord, TranscriberError
from studyscribe.transcribe.vad import SPEECH_VAD

ENGINE = "faster-whisper"
# Decoding options (see docs/plan-transcription.md, Decoding). The thresholds are the ones
# OpenAI's reference implementation uses for long-form audio.
DECODING = {
    "beam_size": 5,
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "no_speech_threshold": 0.6,
    "log_prob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
    "condition_on_previous_text": False,
    "word_timestamps": True,
    "vad_filter": True,
}

SMALLER = "--compute-type int8_float16, or a smaller model (--model medium or small)"


def describe_error(exc: BaseException) -> str:
    """One-line explanation and remedy for an error of ctranslate2 / CUDA."""
    text = str(exc)
    lowered = text.lower()
    if "out of memory" in lowered:
        return (f"the GPU ran out of memory ({text.strip()}). Close applications that use the "
                f"GPU, or use {SMALLER}.")
    if "cublas" in lowered and ("not found" in lowered or "cannot be loaded" in lowered):
        return f'cuBLAS 12 is missing ({text.strip()}): install it with `pip install -e ".[cuda]"`.'
    if "no cuda" in lowered or "cuda driver" in lowered or "insufficient" in lowered:
        return (f"CUDA is not usable ({text.strip()}). Update the NVIDIA driver, or run on the "
                "CPU (slow): --device cpu --model small --compute-type int8.")
    if "compute type" in lowered or "compute_type" in lowered:
        return f"{text.strip()}. Try {SMALLER}."
    return text.strip() or type(exc).__name__


class FasterWhisperTranscriber:
    """Transcriber backed by a faster-whisper model, loaded once."""

    engine = ENGINE

    def __init__(self, model, model_name: str, compute_type: str, device: str) -> None:
        self._model = model
        self.model = model_name
        self.compute_type = compute_type
        self.device = device

    @classmethod
    def load(
        cls,
        model: str = DEFAULT_MODEL,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        device: str = "cuda",
    ) -> FasterWhisperTranscriber:
        """Download (first time) and load the model. TranscriberError with a remedy on failure."""
        if device == "cuda":
            from studyscribe.transcribe.cuda import CublasNotFound, preload_cublas

            try:
                preload_cublas()
            except (CublasNotFound, OSError) as exc:
                raise TranscriberError(describe_error(exc)) from exc
        try:
            import ctranslate2
            from faster_whisper import WhisperModel
            from faster_whisper.utils import download_model
        except ImportError as exc:
            raise TranscriberError(f"faster-whisper is not installed: {exc}") from exc

        if device == "cuda":
            if ctranslate2.get_cuda_device_count() == 0:
                raise TranscriberError(
                    "no CUDA GPU found. Check the NVIDIA driver, or run on the CPU (slow): "
                    "--device cpu --model small --compute-type int8."
                )
            supported = ctranslate2.get_supported_compute_types("cuda")
            if compute_type not in supported:
                raise TranscriberError(
                    f"compute type {compute_type} is not supported by this GPU "
                    f"(supported: {', '.join(sorted(supported))})"
                )
        repo, revision = PINNED_MODELS.get(model, (model, None))
        name = repo if revision is None else f"{repo}@{revision[:7]}"
        try:
            path = download_model(repo, revision=revision)
        except Exception as exc:  # network, unknown name, disk: all reported the same way
            raise TranscriberError(f"cannot get the model {name}: {exc}") from exc
        try:
            whisper = WhisperModel(path, device=device, compute_type=compute_type)
        except Exception as exc:
            raise TranscriberError(describe_error(exc)) from exc
        return cls(whisper, name, compute_type, device)

    def options(self) -> dict:
        return {"engine": ENGINE, "model": self.model, "compute_type": self.compute_type,
                "device": self.device, "decoding": DECODING, "vad": SPEECH_VAD}

    def language_probabilities(self, speech: np.ndarray) -> dict[str, float]:
        try:
            _, _, probabilities = self._model.detect_language(
                audio=speech, language_detection_segments=1
            )
        except RuntimeError as exc:
            raise TranscriberError(describe_error(exc)) from exc
        return dict(probabilities)

    def transcribe(
        self, audio: np.ndarray, language: str, hints: str | None = None
    ) -> Iterator[AsrSegment]:
        from faster_whisper.vad import VadOptions

        try:
            segments, _ = self._model.transcribe(
                audio, language=language, hotwords=hints,
                vad_parameters=VadOptions(**SPEECH_VAD), **DECODING,
            )
            for s in segments:  # decoding happens while iterating
                yield AsrSegment(
                    start=s.start,
                    end=s.end,
                    text=s.text,
                    words=tuple(AsrWord(w.start, w.end, w.word, w.probability)
                                for w in (s.words or ())),
                    avg_logprob=s.avg_logprob,
                    no_speech_prob=s.no_speech_prob,
                    compression_ratio=s.compression_ratio,
                )
        except RuntimeError as exc:
            raise TranscriberError(describe_error(exc)) from exc
