"""cuBLAS for ctranslate2 on Windows, from the nvidia-cublas-cu12 wheel (the `cuda` extra).

ctranslate2 loads cuBLAS on first use with LoadLibraryA("cublas64_12.dll"): the plain DLL search
order, which ignores os.add_dll_directory and does not include the wheel's folder. Loading the
DLLs here by full path first makes that call return the module already in the process, so no
PATH change or system-wide CUDA install is needed. Call preload_cublas() before the first model
is created.
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path

# cublasLt first: cublas64_12.dll imports it.
CUBLAS_DLLS = ("cublasLt64_12.dll", "cublas64_12.dll")
_loaded: list[Path] = []


class CublasNotFound(RuntimeError):
    """The nvidia-cublas-cu12 wheel is not installed (or its DLLs are missing)."""


def cublas_dir() -> Path | None:
    """Folder of the wheel's DLLs (<site-packages>/nvidia/cublas/bin), or None."""
    try:
        spec = importlib.util.find_spec("nvidia.cublas")
    except ModuleNotFoundError:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        folder = Path(location) / "bin"
        if all((folder / name).is_file() for name in CUBLAS_DLLS):
            return folder
    return None


def preload_cublas() -> list[Path]:
    """Load the cuBLAS DLLs once; return their paths. No-op outside Windows.

    Raises CublasNotFound if the wheel is missing.
    """
    if sys.platform != "win32" or _loaded:
        return list(_loaded)
    folder = cublas_dir()
    if folder is None:
        raise CublasNotFound(
            'cuBLAS 12 not found: install it with `pip install -e ".[cuda]"`'
        )
    for name in CUBLAS_DLLS:
        ctypes.WinDLL(str(folder / name))
        _loaded.append(folder / name)
    return list(_loaded)
