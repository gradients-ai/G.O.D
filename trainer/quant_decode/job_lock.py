"""One decode or upload attempt at a time per job, across processes.

The trainer and the re-run CLI are separate processes that work in the same per-job directory. Whoever changes that
directory (a decode, a round of the upload retry, the startup sweep) first takes the job's lock: an exclusive `flock`
on `<work root>/.locks/<work dir name>.lock`. The kernel drops the lock when its holder exits, however it exits, so
a crashed process never leaves a job locked.

Two locks on the same job conflict inside one process as well, so a new decode in the trainer also waits for a
background retry of the same job to let go.
"""

import asyncio
import fcntl
import os
import time
from collections.abc import Awaitable
from collections.abc import Callable
from pathlib import Path

import trainer.constants as cst


LOCK_DIR_NAME = ".locks"
LOCK_SUFFIX = ".lock"
OWNER_RWX = 0o700
OWNER_RW = 0o600
LOCK_POLL_SECONDS = 1.0

LockReporter = Callable[[str], Awaitable[None]]


class JobLock:
    """The lock of one work dir name under a work root. Not re-entrant."""

    def __init__(self, work_root: Path, name: str) -> None:
        self.path = work_root / LOCK_DIR_NAME / f"{name}{LOCK_SUFFIX}"
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def try_acquire(self) -> bool:
        """Take the lock if nobody holds it. Raises OSError when the work root is missing or unusable."""
        if self.held:
            raise RuntimeError(f"{self.path} is already held by this lock")
        self.path.parent.mkdir(mode=OWNER_RWX, exist_ok=True)
        while True:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, OWNER_RW)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                return False
            except BaseException:
                os.close(fd)
                raise
            if self._is_current(fd):
                self._fd = fd
                return True
            os.close(fd)  # the file was removed between open and lock: lock the new one

    def _is_current(self, fd: int) -> bool:
        try:
            return os.stat(self.path).st_ino == os.fstat(fd).st_ino
        except FileNotFoundError:
            return False

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def remove_and_release(self) -> None:
        """Delete the lock file while holding it; a waiter that had it open locks a fresh file instead."""
        if self._fd is not None:
            self.path.unlink(missing_ok=True)
            self.release()


async def acquire(lock: JobLock, report: LockReporter, timeout: float | None = None) -> bool:
    """Wait for the lock, at most `timeout` seconds (QUANT_DECODE_JOB_LOCK_WAIT_SECONDS by default).

    False when the wait ran out. A wait is reported once. Raises OSError like `try_acquire`.
    """
    timeout = cst.QUANT_DECODE_JOB_LOCK_WAIT_SECONDS if timeout is None else timeout
    deadline = time.monotonic() + timeout
    reported = False
    while not lock.try_acquire():  # a few non-blocking syscalls: no thread hop, so decodes keep their arrival order
        if time.monotonic() >= deadline:
            return False
        if not reported:
            reported = True
            await report("decode: waiting for another decode or upload of this job to finish")
        await asyncio.sleep(LOCK_POLL_SECONDS)
    return True
