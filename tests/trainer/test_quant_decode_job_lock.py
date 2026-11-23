"""The per-job lock of the freeform decode (trainer/quant_decode/job_lock.py): one holder at a time, across processes."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from trainer.quant_decode import job_lock
from trainer.quant_decode.job_lock import JobLock


NAME = "task-1_hk"
HOLD_IN_ANOTHER_PROCESS = """
import sys
from pathlib import Path
from trainer.quant_decode.job_lock import JobLock
lock = JobLock(Path(sys.argv[1]), sys.argv[2])
assert lock.try_acquire()
print("held", flush=True)
sys.stdin.read()
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


async def no_report(message: str) -> None:
    return None


def test_a_held_lock_refuses_a_second_holder_until_it_is_released(root: Path):
    first, second = JobLock(root, NAME), JobLock(root, NAME)

    assert first.try_acquire()
    assert not second.try_acquire()
    first.release()
    assert second.try_acquire()
    second.release()


def test_locks_of_different_jobs_do_not_conflict(root: Path):
    first, other = JobLock(root, NAME), JobLock(root, "task-2_hk")

    assert first.try_acquire() and other.try_acquire()
    first.release()
    other.release()


def test_a_lock_held_by_another_process_is_refused_and_freed_when_that_process_dies(root: Path):
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD_IN_ANOTHER_PROCESS, str(root), NAME], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        lock = JobLock(root, NAME)
        assert not lock.try_acquire()
        holder.kill()
        holder.wait(timeout=10)
        assert lock.try_acquire()  # no stale lock after a crash
        lock.release()
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_lock_dir_and_file_are_owner_only(root: Path):
    lock = JobLock(root, NAME)
    assert lock.try_acquire()
    lock.release()

    assert (root / job_lock.LOCK_DIR_NAME).stat().st_mode & 0o777 == 0o700
    assert lock.path.stat().st_mode & 0o777 == 0o600


def test_missing_work_root_is_an_oserror(root: Path):
    with pytest.raises(OSError):
        JobLock(root / "missing", NAME).try_acquire()


def test_a_removed_lock_file_is_not_shared_with_its_next_holder(root: Path):
    first = JobLock(root, NAME)
    assert first.try_acquire()
    first.remove_and_release()
    assert not first.path.exists()

    second, third = JobLock(root, NAME), JobLock(root, NAME)
    assert second.try_acquire()
    assert not third.try_acquire()
    second.release()


async def test_acquire_waits_for_the_holder_and_reports_the_wait_once(root: Path, monkeypatch):
    monkeypatch.setattr(job_lock, "LOCK_POLL_SECONDS", 0.01)
    holder, waiter = JobLock(root, NAME), JobLock(root, NAME)
    assert holder.try_acquire()
    lines: list[str] = []

    async def report(message: str) -> None:
        lines.append(message)

    waiting = asyncio.create_task(job_lock.acquire(waiter, report, timeout=10))
    await asyncio.sleep(0.1)
    assert not waiting.done()
    holder.release()

    assert await waiting is True
    assert lines == ["decode: waiting for another decode or upload of this job to finish"]
    waiter.release()


async def test_acquire_gives_up_after_the_timeout(root: Path, monkeypatch):
    monkeypatch.setattr(job_lock, "LOCK_POLL_SECONDS", 0.01)
    holder, waiter = JobLock(root, NAME), JobLock(root, NAME)
    assert holder.try_acquire()

    assert await job_lock.acquire(waiter, no_report, timeout=0.05) is False
    assert not waiter.held
    holder.release()
