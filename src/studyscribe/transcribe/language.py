"""Language of a session, from the first minutes of its speech (pure parts).

faster-whisper's own detection looks at the first 30 s only, or stops at the first 30 s it is
confident about: an intro song decides the language. Here the probabilities of up to
MAX_CHUNKS chunks of speech (after the VAD) are averaged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from studyscribe.config import SAMPLE_RATE

DETECT_SPEECH_S = 180.0  # speech collected for the detection
CHUNK_S = 30.0  # Whisper's window
MAX_CHUNKS = int(DETECT_SPEECH_S // CHUNK_S)
MIN_LAST_CHUNK_S = 5.0  # a shorter remainder is left out, unless it is all there is
CONFIDENT = 0.6
EXPECTED = ("en", "it")


@dataclass(frozen=True)
class LanguageGuess:
    language: str
    probability: float
    top: tuple[tuple[str, float], ...]  # best three, most likely first
    speech_s: float  # seconds of speech the guess is based on


def chunks(speech: np.ndarray) -> list[np.ndarray]:
    """Cut speech into Whisper-sized chunks (at most MAX_CHUNKS)."""
    size = int(CHUNK_S * SAMPLE_RATE)
    parts = [speech[i : i + size] for i in range(0, speech.size, size)][:MAX_CHUNKS]
    if len(parts) > 1 and parts[-1].size < MIN_LAST_CHUNK_S * SAMPLE_RATE:
        parts.pop()
    return parts


def average(probabilities: list[dict[str, float]], speech_s: float) -> LanguageGuess:
    """The language with the highest mean probability over the chunks."""
    if not probabilities:
        raise ValueError("no chunk to decide on")
    languages = set().union(*probabilities)
    mean = {lang: sum(p.get(lang, 0.0) for p in probabilities) / len(probabilities)
            for lang in languages}
    top = tuple(sorted(mean.items(), key=lambda item: (-item[1], item[0]))[:3])
    return LanguageGuess(top[0][0], top[0][1], top, speech_s)


def doubt(guess: LanguageGuess) -> str | None:
    """A warning when the guess deserves a look, else None."""
    if guess.probability < CONFIDENT:
        return (f"language uncertain ({describe(guess)}): use --language to force it")
    if guess.language not in EXPECTED:
        return (f"detected {guess.language}, neither English nor Italian ({describe(guess)}): "
                "use --language to force it")
    return None


def describe(guess: LanguageGuess) -> str:
    return ", ".join(f"{lang} {p:.2f}" for lang, p in guess.top)
