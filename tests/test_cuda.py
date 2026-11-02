import sys

import pytest

from studyscribe.transcribe import cuda

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="cuBLAS preload is Windows only")


@windows_only
def test_cublas_dlls_are_found_in_the_wheel() -> None:
    folder = cuda.cublas_dir()
    if folder is None:
        pytest.skip('the "cuda" extra is not installed')
    assert folder.name == "bin" and folder.parent.name == "cublas"
    loaded = cuda.preload_cublas()
    assert [path.name for path in loaded] == list(cuda.CUBLAS_DLLS)
    assert cuda.preload_cublas() == loaded  # once


def test_missing_wheel_gives_a_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(cuda.sys, "platform", "win32")
    monkeypatch.setattr(cuda, "_loaded", [])
    monkeypatch.setattr(cuda, "cublas_dir", lambda: None)
    with pytest.raises(cuda.CublasNotFound, match=r'pip install -e "\.\[cuda\]"'):
        cuda.preload_cublas()
