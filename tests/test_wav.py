import struct
import wave
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeClock

from studyscribe.audio import wav as wav_module
from studyscribe.audio.wav import HEADER_SIZE, WavWriter, wav_header


def header_sizes(path: Path) -> tuple[int, int]:
    """(RIFF size, data size) as currently written on disk."""
    raw = path.read_bytes()[:HEADER_SIZE]
    return struct.unpack_from("<I", raw, 4)[0], struct.unpack_from("<I", raw, 40)[0]


def test_header_matches_stdlib_wave(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    with wave.open(str(reference), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(b"\x00\x00" * 100)
    assert reference.read_bytes()[:HEADER_SIZE] == wav_header(200)


def test_written_file_is_a_valid_wav(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    pcm = (np.arange(16_000) % 2000 - 1000).astype(np.int16)
    writer = WavWriter(path)
    writer.write(pcm[:5000])
    writer.write(pcm[5000:])
    writer.close()
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16_000)
        assert w.getnframes() == 16_000
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    np.testing.assert_array_equal(data, pcm)
    assert path.stat().st_size == HEADER_SIZE + 32_000


def test_header_is_patched_periodically_while_open(tmp_path: Path) -> None:
    clock = FakeClock()
    path = tmp_path / "b.wav"
    writer = WavWriter(path, header_interval_s=1.0, clock=clock)
    writer.write(np.zeros(1000, dtype=np.int16))
    assert header_sizes(path) == (36, 0)  # not due yet; but the samples are already on disk
    assert path.stat().st_size == HEADER_SIZE + 2000
    clock.sleep(1.0)
    writer.write(np.zeros(500, dtype=np.int16))
    assert header_sizes(path) == (36 + 3000, 3000)
    writer.close()


def test_fsync_is_periodic_and_on_close(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(wav_module.os, "fsync", lambda fd: calls.append(fd))
    clock = FakeClock()
    writer = WavWriter(tmp_path / "c.wav", fsync_interval_s=10.0, clock=clock)
    writer.write(np.zeros(10, dtype=np.int16))
    assert calls == []
    clock.sleep(10.0)
    writer.write(np.zeros(10, dtype=np.int16))
    assert len(calls) == 1
    writer.close()
    assert len(calls) == 2


def test_never_overwrites_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "d.wav"
    path.write_bytes(b"precious")
    with pytest.raises(FileExistsError):
        WavWriter(path)
    assert path.read_bytes() == b"precious"


def test_close_is_idempotent(tmp_path: Path) -> None:
    writer = WavWriter(tmp_path / "e.wav")
    writer.close()
    writer.close()
    assert writer.closed
