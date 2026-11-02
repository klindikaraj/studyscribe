import numpy as np
import pytest

from studyscribe.transcribe.whisper import describe_error


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("CUDA failed with error out of memory", "int8_float16"),
        ("Library cublas64_12.dll is not found or cannot be loaded", 'pip install -e ".[cuda]"'),
        ("CUDA driver version is insufficient for CUDA runtime version", "--device cpu"),
        ("Requested float16 compute type, but the target device or backend do not support "
         "efficient float16 computation.", "int8_float16"),
    ],
)
def test_gpu_errors_get_a_remedy(message: str, expected: str) -> None:
    text = describe_error(RuntimeError(message))
    assert expected in text and message.split()[0] in text
    assert "\n" not in text


def test_other_errors_are_passed_through() -> None:
    assert describe_error(RuntimeError("something else")) == "something else"
    assert describe_error(ValueError()) == "ValueError"


@pytest.mark.gpu
def test_real_model_on_silence_and_noise() -> None:
    """Needs the GPU and the downloaded model: pytest -m gpu."""
    from studyscribe.transcribe.whisper import FasterWhisperTranscriber

    engine = FasterWhisperTranscriber.load()
    assert list(engine.transcribe(np.zeros(16_000 * 10, dtype=np.float32), "en")) == []
    probabilities = engine.language_probabilities(
        np.random.default_rng(0).standard_normal(16_000 * 5).astype(np.float32) * 0.01)
    assert abs(sum(probabilities.values()) - 1.0) < 0.05
