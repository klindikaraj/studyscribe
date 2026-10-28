"""Default output device (render endpoint), read through COM with ctypes.

Needed to notice a manual switch of the default output: the loopback stream of the old device
stays active but receives nothing, so only the endpoint ID shows that something changed.
An unplugged device is noticed anyway, because its stream stops.
"""

from __future__ import annotations

import ctypes
import sys
import uuid
from typing import Any

_CLSID_MMDEVICE_ENUMERATOR = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
_IID_IMMDEVICE_ENUMERATOR = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"
_CLSCTX_INPROC_SERVER = 0x1
_COINIT_APARTMENTTHREADED = 0x2
_RPC_E_CHANGED_MODE = 0x80010106
_E_NOTFOUND = 0x80070490  # HRESULT_FROM_WIN32(ERROR_NOT_FOUND): no output device at all
_E_RENDER = 0
_E_MULTIMEDIA = 1  # the role PortAudio uses for its default output

# Vtable slots; IUnknown takes 0-2.
_RELEASE = 2
_ENUMERATOR_GET_DEFAULT_AUDIO_ENDPOINT = 4
_DEVICE_GET_ID = 5


class _GUID(ctypes.Structure):
    _fields_ = [
        ("data1", ctypes.c_uint32),
        ("data2", ctypes.c_uint16),
        ("data3", ctypes.c_uint16),
        ("data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def parse(cls, text: str) -> _GUID:
        return cls.from_buffer_copy(uuid.UUID(text).bytes_le)


def _unsigned(hresult: int) -> int:
    return hresult & 0xFFFFFFFF


def _com_method(obj: ctypes.c_void_p, slot: int, restype: Any, *argtypes: Any) -> Any:
    """Method `slot` of a COM interface pointer, callable as f(obj, *args).

    With restype ctypes.HRESULT, ctypes raises OSError when the call fails.
    """
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return prototype(vtable[slot])


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        _com_method(obj, _RELEASE, ctypes.c_ulong)(obj)


class DefaultOutputMonitor:
    """Reads the endpoint ID of the default output, e.g. "{0.0.0.00000000}.{<guid>}".

    Initialises COM on the calling thread, so create, use and close it on one thread (the
    capture loop's). Raises OSError if COM or the device enumerator are not available.
    """

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("reading the default output needs Windows")
        ole32 = ctypes.WinDLL("ole32")
        ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        ole32.CoInitializeEx.restype = ctypes.c_long
        ole32.CoUninitialize.argtypes = []
        ole32.CoUninitialize.restype = None
        ole32.CoCreateInstance.argtypes = [
            ctypes.POINTER(_GUID),
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(_GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        ole32.CoCreateInstance.restype = ctypes.HRESULT
        ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        ole32.CoTaskMemFree.restype = None
        self._ole32 = ole32
        self._enumerator = ctypes.c_void_p()
        hr = ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED)
        if hr < 0 and _unsigned(hr) != _RPC_E_CHANGED_MODE:
            raise ctypes.WinError(hr)
        # S_OK and S_FALSE must be balanced by CoUninitialize. RPC_E_CHANGED_MODE means COM is
        # already initialised on this thread in the other mode, which works for these calls.
        self._uninitialize = hr >= 0
        try:
            ole32.CoCreateInstance(
                ctypes.byref(_GUID.parse(_CLSID_MMDEVICE_ENUMERATOR)),
                None,
                _CLSCTX_INPROC_SERVER,
                ctypes.byref(_GUID.parse(_IID_IMMDEVICE_ENUMERATOR)),
                ctypes.byref(self._enumerator),
            )
        except BaseException:
            self.close()
            raise

    def current_id(self) -> str | None:
        """Endpoint ID of the default output, or None if there is no output device at all."""
        if not self._enumerator:
            raise OSError("the default output monitor is closed")
        get_default = _com_method(
            self._enumerator,
            _ENUMERATOR_GET_DEFAULT_AUDIO_ENDPOINT,
            ctypes.HRESULT,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        )
        device = ctypes.c_void_p()
        try:
            get_default(self._enumerator, _E_RENDER, _E_MULTIMEDIA, ctypes.byref(device))
        except OSError as exc:
            if exc.winerror is not None and _unsigned(exc.winerror) == _E_NOTFOUND:
                return None
            raise
        try:
            get_id = _com_method(
                device, _DEVICE_GET_ID, ctypes.HRESULT, ctypes.POINTER(ctypes.c_void_p)
            )
            text = ctypes.c_void_p()
            get_id(device, ctypes.byref(text))
            try:
                return ctypes.wstring_at(text.value)
            finally:
                self._ole32.CoTaskMemFree(text)
        finally:
            _release(device)

    def close(self) -> None:
        """Release the enumerator and COM. Idempotent."""
        enumerator, self._enumerator = self._enumerator, ctypes.c_void_p()
        _release(enumerator)
        if self._uninitialize:
            self._uninitialize = False
            self._ole32.CoUninitialize()
