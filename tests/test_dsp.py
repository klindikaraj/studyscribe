import numpy as np
import pytest

from studyscribe.audio.dsp import (
    LEVEL_FLOOR_DB,
    LevelMeter,
    StreamingResampler,
    downmix_to_mono,
    float_to_int16,
    frames_from_bytes,
)


def sine(freq: float, rate: int, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def resample_in_chunks(x: np.ndarray, in_rate: int, out_rate: int, chunks: int) -> np.ndarray:
    resampler = StreamingResampler(in_rate, out_rate)
    parts = [resampler.process(c) for c in np.array_split(x, chunks)]
    parts.append(resampler.flush())
    return np.concatenate(parts)


def dominant_frequency(x: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / rate)[np.argmax(spectrum)])


# --- downmix -----------------------------------------------------------------------------


def test_downmix_averages_channels() -> None:
    frames = np.array([[0.5, -0.5], [0.2, 0.4], [1.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(downmix_to_mono(frames), [0.0, 0.3, 1.0], atol=1e-7)


def test_downmix_six_channels() -> None:
    frames = np.tile(np.arange(6, dtype=np.float32), (4, 1))
    np.testing.assert_allclose(downmix_to_mono(frames), np.full(4, 2.5))


def test_downmix_mono_passthrough_is_contiguous_float32() -> None:
    frames = np.arange(5, dtype=np.float64).reshape(5, 1)
    out = downmix_to_mono(frames)
    assert out.dtype == np.float32 and out.ndim == 1 and out.flags.c_contiguous
    np.testing.assert_array_equal(out, np.arange(5))


def test_frames_from_interleaved_bytes() -> None:
    interleaved = np.array([1, -1, 2, -2, 3, -3], dtype=np.float32)
    frames = frames_from_bytes(interleaved.tobytes(), channels=2)
    np.testing.assert_array_equal(frames, [[1, -1], [2, -2], [3, -3]])


# --- resample ----------------------------------------------------------------------------


@pytest.mark.parametrize("in_rate", [48_000, 44_100, 96_000])
def test_resampled_length_matches_rate_ratio(in_rate: int) -> None:
    x = sine(1000, in_rate, 2.0)
    out = resample_in_chunks(x, in_rate, 16_000, chunks=73)
    assert abs(len(out) - len(x) * 16_000 / in_rate) <= 1


@pytest.mark.parametrize("in_rate", [48_000, 44_100])
def test_resample_preserves_frequency_and_amplitude(in_rate: int) -> None:
    out = resample_in_chunks(sine(1000, in_rate, 1.0, amplitude=0.5), in_rate, 16_000, chunks=40)
    assert dominant_frequency(out, 16_000) == pytest.approx(1000, abs=2)
    middle = out[2000:-2000]  # skip the filter's edges
    assert np.max(np.abs(middle)) == pytest.approx(0.5, rel=0.01)


def test_chunked_resampling_equals_one_shot() -> None:
    x = sine(440, 48_000, 0.5)
    one_shot = resample_in_chunks(x, 48_000, 16_000, chunks=1)
    chunked = resample_in_chunks(x, 48_000, 16_000, chunks=97)
    np.testing.assert_allclose(chunked, one_shot, atol=1e-6)


def test_resample_removes_content_above_new_nyquist() -> None:
    out = resample_in_chunks(sine(12_000, 48_000, 1.0), 48_000, 16_000, chunks=10)
    assert np.max(np.abs(out[2000:-2000])) < 0.01


def test_same_rate_is_passthrough() -> None:
    x = sine(440, 16_000, 0.1)
    resampler = StreamingResampler(16_000, 16_000)
    np.testing.assert_array_equal(resampler.process(x), x)
    assert resampler.flush().size == 0


# --- int16 and levels --------------------------------------------------------------------


def test_float_to_int16_scales_and_clips() -> None:
    x = np.array([0.0, 1.0, -1.0, 1.5, -1.5, 0.25], dtype=np.float32)
    np.testing.assert_array_equal(
        float_to_int16(x), [0, 32767, -32767, 32767, -32767, 8192]
    )
    assert float_to_int16(x).dtype == np.int16


def test_level_meter_full_scale_sine_is_about_minus_3_db() -> None:
    meter = LevelMeter()
    meter.add(float_to_int16(sine(1000, 16_000, 1.0, amplitude=1.0)))
    rms_db, peak_db = meter.take()
    assert rms_db == pytest.approx(-3.01, abs=0.05)
    assert peak_db == pytest.approx(0.0, abs=0.01)


def test_level_meter_silence_and_reset() -> None:
    meter = LevelMeter()
    assert meter.take() == (LEVEL_FLOOR_DB, LEVEL_FLOOR_DB)
    meter.add(np.zeros(160, dtype=np.int16))
    assert meter.take() == (LEVEL_FLOOR_DB, LEVEL_FLOOR_DB)
    meter.add(np.full(160, 16384, dtype=np.int16))
    assert meter.take()[0] == pytest.approx(-6.02, abs=0.01)
    assert meter.take() == (LEVEL_FLOOR_DB, LEVEL_FLOOR_DB)  # window was reset
