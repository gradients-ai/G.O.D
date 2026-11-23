"""Limits on how many heavy transfers one trainer runs at once.

Jobs on one box share its network link. Image builds (base image and package pulls), uploads and freeform decodes
each take a slot from a small pool, so a few run at full speed while the rest queue, instead of all of them crawling
into their timeouts. Slots are handed out in arrival order.

A step's own timeout must start inside its slot: time spent queued is the trainer's, never the miner's. A job that
is queued, or has been, is also kept clear of the stale-job deadline in cleanup.py through `is_waiting`,
`waiting_seconds` and `waited_seconds`, up to a cap.

The pools belong to one trainer process and one event loop. A separate process, such as the decode re-run CLI, has
pools of its own and does not queue behind the running trainer.
"""

import asyncio
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager

from trainer import constants as cst
from trainer.telemetry import logger


SlotReporter = Callable[[str], Awaitable[None]]
JobKey = tuple[str, str]  # (task_id, hotkey)

# Read by the cleanup loop from its own thread.
_waits_lock = threading.Lock()
_waiting_jobs: Counter[JobKey] = Counter()
_waiting_since: dict[JobKey, float] = {}
_waited_seconds: dict[JobKey, float] = {}


async def _no_report(message: str) -> None:
    return None


def is_waiting(job: JobKey) -> bool:
    """True while the job is queued for a slot in any pool."""
    with _waits_lock:
        return _waiting_jobs[job] > 0


def waiting_seconds(job: JobKey) -> float:
    """Length of the job's current unbroken wait; 0 when it is not queued."""
    with _waits_lock:
        since = _waiting_since.get(job)
    return 0.0 if since is None else time.monotonic() - since


def waited_seconds(job: JobKey) -> float:
    """Time the job has spent in finished waits for slots since the last `forget_waits`."""
    with _waits_lock:
        return _waited_seconds.get(job, 0.0)


def forget_waits(job: JobKey) -> None:
    with _waits_lock:
        _waited_seconds.pop(job, None)


def _wait_started(job: JobKey | None) -> None:
    if job is not None:
        with _waits_lock:
            _waiting_jobs[job] += 1
            _waiting_since.setdefault(job, time.monotonic())


def _wait_ended(job: JobKey | None, seconds: float) -> None:
    if job is not None:
        with _waits_lock:
            _waiting_jobs[job] -= 1
            if _waiting_jobs[job] <= 0:
                del _waiting_jobs[job]
                _waiting_since.pop(job, None)
            _waited_seconds[job] = _waited_seconds.get(job, 0.0) + seconds


class SlotLimiter:
    """At most `limit` holders at a time; the rest wait in arrival order."""

    def __init__(self, name: str, limit: int):
        if limit < 1:
            raise ValueError(f"{name} slot limit must be at least 1, got {limit}")
        self.name = name
        self.limit = limit
        self._semaphore = asyncio.Semaphore(limit)

    @asynccontextmanager
    async def slot(self, report: SlotReporter = _no_report, job: JobKey | None = None) -> AsyncIterator[float]:
        """Hold one slot for the body; yields the seconds spent queued for it.

        A wait is logged and sent to `report` (the job log), once when it starts and once when the slot is granted.
        """
        waited = 0.0
        if self._semaphore.locked():
            started = time.monotonic()
            _wait_started(job)
            try:
                message = f"Waiting for {self.name} slot ({self.limit} at a time on this trainer)"
                logger.info(f"{message} | job={job}")
                await report(message)
                await self._semaphore.acquire()
            finally:
                waited = time.monotonic() - started
                _wait_ended(job, waited)
        else:
            await self._semaphore.acquire()
        try:
            if waited:
                message = f"Got {self.name} slot after waiting {waited:.0f}s"
                logger.info(f"{message} | job={job}")
                await report(message)
            yield waited
        finally:
            self._semaphore.release()


image_builds = SlotLimiter("image build", cst.MAX_CONCURRENT_IMAGE_BUILDS)
# One pool for everything the trainer sends out: model uploads to Hugging Face and decoded files to object storage.
# A decoded-file upload that runs out of time gives its slot back while its thread is still sending (the storage client
# has no cancel), until the request in flight fails or returns. For that short while one more transfer than the limit
# can be on the link.
uploads = SlotLimiter("upload", cst.MAX_CONCURRENT_UPLOADS)
quant_decodes = SlotLimiter("freeform decode", cst.MAX_CONCURRENT_QUANT_DECODES)
