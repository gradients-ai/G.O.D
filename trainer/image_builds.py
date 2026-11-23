"""Image builds in flight on this trainer, and the host-wide build cleanup that must wait for them.

Pruning dangling images and the builder cache is host-wide: it also removes the intermediate state of a build that is
still running, which then fails. Jobs on one box build concurrently, so the prune runs only while no build is in
flight, and holds off new builds until it is done. A skipped prune is retried by the periodic cleanup loop.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import docker

from trainer import constants as cst
from trainer.telemetry import logger


_lock = threading.Lock()
_builds_in_flight = 0


class BuildCleanupBusy(Exception):
    """A host build cleanup kept a build waiting for longer than it was willing to wait."""


@contextmanager
def image_build_in_flight(cleanup_wait_seconds: float | None = None) -> Iterator[None]:
    """Count one build as in flight. Waits for a cleanup in progress: `cleanup_wait_seconds` at most, else for good."""
    global _builds_in_flight
    if not _lock.acquire(timeout=-1 if cleanup_wait_seconds is None else cleanup_wait_seconds):
        raise BuildCleanupBusy(f"host build cleanup still running after {cleanup_wait_seconds:.0f}s")
    try:
        _builds_in_flight += 1
    finally:
        _lock.release()
    try:
        yield
    finally:
        with _lock:
            _builds_in_flight -= 1


def prune_build_leftovers() -> None:
    """Remove dangling images (a failed build's leftover layers) and, if configured, the host build cache.

    Skipped while any image build is in flight. The client has no read timeout: a prune the client gave up on would
    keep running on the daemon after the lock is released.
    """
    with _lock:
        if _builds_in_flight:
            logger.info(f"Build cleanup deferred: {_builds_in_flight} image build(s) in progress.")
            return
        client = docker.from_env(timeout=None)
        client.images.prune(filters={"dangling": True})
        if cst.PRUNE_HOST_BUILD_CACHE:
            client.api.prune_builds()
            logger.info("Cleaned up dangling images and build cache.")
        else:
            logger.info("Cleaned up dangling images; host build cache kept (TRAINER_PRUNE_BUILD_CACHE=false).")
