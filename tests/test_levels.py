import numpy as np
import pytest

from studyscribe.transcribe import levels

RATE = 16_000


def tone(seconds: float, db: float, frequency: float = 440.0) -> np.ndarray:
    """int16 sine whose RMS is `db` dBFS."""
    t = np.arange(round(seconds * RATE)) / RATE
    amplitude = 10 ** (db / 20) * np.sqrt(2)
    return np.round(amplitude * 32767 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def test_quiet_speech_is_brought_to_the_target() -> None:
    out = levels.normalize(tone(60, -40))
    assert rms_db(out) == pytest.approx(levels.TARGET_DB, abs=0.5)
    assert out.dtype == np.float32


def test_loud_audio_and_noise_are_left_alone() -> None:
    loud = tone(30, -15)
    assert np.allclose(levels.normalize(loud), levels.to_float(loud))
    noise = tone(30, -65)  # below QUIET_LEVEL_DB: not boosted into "speech"
    assert np.allclose(levels.normalize(noise), levels.to_float(noise))
    silence = np.zeros(RATE * 20, dtype=np.int16)
    assert not levels.normalize(silence).any()
    assert levels.normalize(np.zeros(0, dtype=np.int16)).size == 0


def test_gain_is_capped() -> None:
    out = levels.normalize(tone(30, -55))
    assert rms_db(out) == pytest.approx(-55 + levels.MAX_GAIN_DB, abs=0.5)


def test_no_sample_is_pushed_above_the_peak_limit() -> None:
    # Quiet speech with sharp clicks: the clicks cap the gain of their blocks and neighbours.
    pcm = tone(90, -40)
    pcm[RATE * 15 :: RATE * 20] = 20_000  # -4.3 dBFS: room for 3.3 dB of gain at most
    out = levels.normalize(pcm)
    assert np.max(np.abs(out)) <= 10 ** (levels.PEAK_DB / 20) + 1e-6
    assert rms_db(out) > -40 + 3  # still raised


def test_a_level_step_is_followed() -> None:
    # The device switch of the 3 h test: speech drops from -20 to -40 dBFS.
    pcm = np.concatenate([tone(120, -20), tone(120, -40)])
    out = levels.normalize(pcm)
    first, second = out[: RATE * 80], out[RATE * 160 :]
    assert rms_db(first) == pytest.approx(-20, abs=0.5)
    assert rms_db(second) == pytest.approx(-20, abs=0.5)
    centres, gains = levels.block_gains(levels.to_float(pcm))
    assert np.all(np.diff(gains) >= 0)  # the gain only rises after the drop: no steps down
    assert gains[0] == pytest.approx(0, abs=0.01) and gains[-1] == pytest.approx(20, abs=0.5)
    # Boosted within 20 s of the drop (at 120 s).
    assert np.interp(140 * RATE, centres, gains) == pytest.approx(20, abs=0.5)


def test_wav_samples_are_not_changed() -> None:
    pcm = tone(30, -40)
    copy = pcm.copy()
    levels.normalize(pcm)
    assert np.array_equal(pcm, copy)
