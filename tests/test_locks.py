import subprocess
import sys
from pathlib import Path

import pytest

from studyscribe.system.locks import ByteLock

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="locks belong to file handles only on Windows"
)

# Takes a lock in another process, reports it and waits to be killed.
HOLDER = """
import sys, time
from studyscribe.system.locks import ByteLock
lock = ByteLock(sys.argv[1], int(sys.argv[2]))
assert lock.acquire()
print("locked", flush=True)
time.sleep(60)
"""


def test_acquire_and_release(tmp_path: Path) -> None:
    lock = ByteLock(tmp_path / "session.lock", 0)
    assert not lock.held
    assert lock.acquire()
    assert lock.held and lock.acquire()  # already ours
    lock.release()
    lock.release()  # idempotent
    assert not lock.held


@windows_only
def test_second_handle_cannot_take_a_held_byte(tmp_path: Path) -> None:
    path = tmp_path / "session.lock"
    first, second, other_byte = ByteLock(path, 1), ByteLock(path, 1), ByteLock(path, 0)
    assert first.acquire()
    try:
        assert not second.acquire()
        assert other_byte.acquire()  # bytes are independent
        other_byte.release()
    finally:
        first.release()
    assert second.acquire()
    second.release()


def test_missing_directory_is_an_error_not_a_held_lock(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        ByteLock(tmp_path / "nope" / "session.lock", 0).acquire()


def test_lock_is_released_when_its_process_is_killed(tmp_path: Path) -> None:
    path = tmp_path / "session.lock"
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(path), "1"], stdout=subprocess.PIPE, text=True
    )
    probe = ByteLock(path, 1)
    try:
        assert holder.stdout.readline().strip() == "locked"
        assert not probe.acquire()
    finally:
        holder.kill()  # TerminateProcess on Windows, as Task Manager does
        holder.wait()
        holder.stdout.close()
    assert probe.acquire()
    probe.release()
