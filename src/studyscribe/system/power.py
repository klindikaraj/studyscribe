"""Keep Windows from sleeping while a recording runs."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

SetStateFn = Callable[[int], int]


def _set_thread_execution_state(flags: int) -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = kernel32.SetThreadExecutionState
    fn.argtypes = [ctypes.c_uint32]
    fn.restype = ctypes.c_uint32
    return fn(flags)


@contextmanager
def keep_system_awake(set_state: SetStateFn | None = None) -> Iterator[bool]:
    """Ask Windows not to sleep because of inactivity until the block exits.

    The request belongs to the calling thread, so use this in the thread that lives for the
    whole recording. It is released on exit, errors included; if the process is killed,
    Windows drops it together with the thread. It does not keep the display on and does not
    block a manual sleep. Yields whether Windows accepted the request.
    """
    if set_state is None:
        if sys.platform != "win32":
            yield False
            return
        set_state = _set_thread_execution_state
    accepted = set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0
    try:
        yield accepted
    finally:
        set_state(ES_CONTINUOUS)
