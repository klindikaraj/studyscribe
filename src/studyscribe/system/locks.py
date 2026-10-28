"""Byte locks on a file that the OS releases when the holding process dies."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _lock(fd: int, offset: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, offset, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, offset, os.SEEK_SET)


def _unlock(fd: int, offset: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, offset, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.lockf(fd, fcntl.LOCK_UN, 1, offset, os.SEEK_SET)


class ByteLock:
    """Exclusive lock on one byte of a file, taken without waiting.

    The OS drops the lock when the process exits or is killed, so a lock that can be taken
    means that nobody holds it any more. On Windows (msvcrt.locking) locks belong to the open
    file handle: a second ByteLock on the same byte fails even within one process. On other
    systems (fcntl) they belong to the process, so that case is not detected there.
    """

    def __init__(self, path: Path | str, offset: int) -> None:
        self.path = Path(path)
        self.offset = offset
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> bool:
        """Take the lock; False if someone else holds it. OSError if the file cannot be opened."""
        if self._fd is not None:
            return True
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
        try:
            _lock(fd, self.offset)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        """Release the lock. Idempotent."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock(fd, self.offset)
        except OSError:
            pass  # closing the file releases it anyway
        finally:
            os.close(fd)
