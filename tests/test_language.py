import numpy as np
import pytest

from studyscribe.transcribe import language as lang

RATE = 16_000


def test_speech_is_cut_into_whisper_chunks() -> None:
    sizes = [c.size for c in lang.chunks(np.zeros(int(95 * RATE)))]
    assert sizes == [30 * RATE, 30 * RATE, 30 * RATE, 5 * RATE]
    assert [c.size for c in lang.chunks(np.zeros(int(62 * RATE)))] == [30 * RATE] * 2
    assert [c.size for c in lang.chunks(np.zeros(3 * RATE))] == [3 * RATE]  # all there is
    assert len(lang.chunks(np.zeros(400 * RATE))) == lang.MAX_CHUNKS


def test_average_beats_one_confident_chunk() -> None:
    # A sung English intro, then Italian speech: the first chunk alone would say "en".
    chunks = [{"en": 0.95, "it": 0.03}] + [{"it": 0.9, "en": 0.05}] * 5
    guess = lang.average(chunks, speech_s=180)
    assert guess.language == "it"
    assert guess.probability == pytest.approx((0.03 + 0.9 * 5) / 6)
    assert [code for code, _ in guess.top][:2] == ["it", "en"]
    assert lang.doubt(guess) is None


def test_doubts() -> None:
    uncertain = lang.average([{"en": 0.5, "it": 0.45}], speech_s=30)
    assert "uncertain" in lang.doubt(uncertain)
    spanish = lang.average([{"es": 0.9, "en": 0.1}], speech_s=30)
    assert "neither English nor Italian" in lang.doubt(spanish)
    with pytest.raises(ValueError):
        lang.average([], speech_s=0)
