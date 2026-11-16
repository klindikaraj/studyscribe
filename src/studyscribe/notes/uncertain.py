"""Words Whisper was unsure about, for the appendix "Words to check by listening".

No LLM and no claim that a word is wrong: the list only points the reader to the places worth
listening to. The rule was tuned in the probe of 2026-11-16 (docs/plan-notes.md): a word is
listed when Whisper's probability is below 0.4, it has at least 3 letters (shorter ones are
mostly function words: "the", "di") and it is said at most twice in the lecture (a word said
often is taken as known). On the test lectures it lists "TEC" (a real error, `detach`) among
3 Italian words, and 11 English words of which about 7 are recognition errors.

It misses confident errors: "PyTorch.neuralNetwork" had a probability of 0.82-0.87.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from studyscribe.notes.transcript import Line

LOW_PROBABILITY = 0.4
MIN_LETTERS = 3
RARE_MAX_COUNT = 2
_WORD = re.compile(r"\w+")
_EDGE_PUNCTUATION = " .,;:!?\"'()[]"


@dataclass(frozen=True)
class UncertainWord:
    word: str  # as Whisper wrote it, without the surrounding spaces and punctuation
    n: int  # S-number of the line
    start_ms: int  # start of the word on the session timeline
    probability: float


def uncertain_words(lines: Sequence[Line], *, threshold: float = LOW_PROBABILITY,
                    min_letters: int = MIN_LETTERS,
                    max_count: int = RARE_MAX_COUNT) -> list[UncertainWord]:
    """The words to check, in transcript order. Lines without Whisper's words give none."""
    counts = Counter(normalize(word.text) for line in lines for word in line.words)
    found = []
    for line in lines:
        for word in line.words:
            key = normalize(word.text)
            if (word.probability < threshold and sum(c.isalpha() for c in key) >= min_letters
                    and counts[key] <= max_count):
                found.append(UncertainWord(word.text.strip(_EDGE_PUNCTUATION), line.n,
                                           word.start_ms, word.probability))
    return found


def normalize(text: str) -> str:
    """Case and punctuation removed: "Keyboard," and "keyboard" count as the same word."""
    return "".join(_WORD.findall(text)).lower()
