"""Image builds in flight on this trainer, the layer cache they share, and the host-wide cleanup that empties it.

The layer cache is the untagged images that builds leave behind: the finished steps of a failed or timed-out build, and
the layers under a finished job's removed image. A later build reuses a layer only where it has the same parent layer
and the same instruction (for COPY and ADD, the same file contents), so a retry resumes after its last finished step
and a build that starts with the same steps skips them.

A build never keeps a layer made more than IMAGE_BUILD_CACHE_MAX_AGE_HOURS ago, so a step whose result changes over
time, such as an unpinned install, is not reused for longer: a build that takes such a layer from the cache is stopped
and run again without the cache (ReusedLayers, layer_too_old, runtime.BuildStream). Emptying the cache cannot give
that limit by itself, because layers under the image of a running job outlive it.

The cache is emptied, never trimmed, to give its disk back: every dangling image goes at once, when the cache has not
been emptied for IMAGE_BUILD_CACHE_MAX_AGE_HOURS, when the disk under Docker's data directory is
IMAGE_BUILD_CACHE_MAX_DISK_USED_PERCENT full, or after every job with IMAGE_BUILD_NO_CACHE.

Pruning dangling images and the builder cache is host-wide: it also removes the intermediate state of a build that is
still running, which then fails. Jobs on one box build concurrently, so the prune runs only while no build is in
flight, and holds off new builds until it is done. A skipped prune is retried by the periodic cleanup loop.
"""

import os
import re
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import docker
from dateutil.parser import isoparse
from docker.errors import APIError
from pydantic import BaseModel
from pydantic import ValidationError
from pydantic import field_validator

from trainer import constants as cst
from trainer.telemetry import logger


_lock = threading.Lock()
_builds_in_flight = 0


@contextmanager
def image_build_in_flight() -> Iterator[None]:
    global _builds_in_flight
    with _lock:
        _builds_in_flight += 1
    try:
        yield
    finally:
        with _lock:
            _builds_in_flight -= 1


# A step the classic builder takes from the cache prints two lines and nothing else:
#    ---> Using cache
#    ---> 0123456789ab
_CACHE_HIT_LINE = re.compile(r"^\s*--->\s+Using cache\s*$")
_LAYER_LINE = re.compile(r"^\s*--->\s+((?:sha256:)?[0-9a-f]{12,64})\s*$")
_STREAM_LINE_MAX_CHARS = 64 * 1024


class ReusedLayers:
    """Follows one build's stream for the layers it takes from the cache.

    Every one of them has to be looked at: within one stage each reused layer sits on the one before, but every FROM
    starts a new stage whose layers can be older than the previous stage's. The id line counts only directly after the
    cache line. The stream also carries the steps' own output; a step that prints these lines can only make its own
    build run again without the cache.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._after_cache_hit = False

    def feed(self, chunk: dict) -> list[str]:
        """The ids of the reused layers named in this chunk."""
        text = chunk.get("stream") if isinstance(chunk, dict) else None
        if not isinstance(text, str):
            return []
        layers = []
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            layer = _LAYER_LINE.match(line) if self._after_cache_hit else None
            self._after_cache_hit = _CACHE_HIT_LINE.match(line) is not None
            if layer is not None:
                layers.append(layer.group(1))
        if len(self._buffer) > _STREAM_LINE_MAX_CHARS:
            self._buffer = ""
        return layers


def layer_too_old(client: docker.DockerClient, layer_id: str) -> bool:
    """Whether a cached layer was made more than IMAGE_BUILD_CACHE_MAX_AGE_HOURS ago.

    A layer whose age cannot be read, for any reason, is not: the stream may have named something that is not a layer,
    and a daemon that does not answer must not end the build that is being followed.
    """
    try:
        created = isoparse(client.api.inspect_image(layer_id)["Created"])
    except Exception as e:
        logger.info(f"Could not read the age of cached layer {layer_id}: {e}")
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - created > timedelta(hours=cst.IMAGE_BUILD_CACHE_MAX_AGE_HOURS)


class BuildCacheState(BaseModel):
    """Kept on disk so a trainer restart does not restart the cache's age."""

    emptied_at: datetime

    @field_validator("emptied_at")
    @classmethod
    def _in_utc(cls, value: datetime) -> datetime:
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


# Used while the state file cannot be read or written, so the age rule still comes due within this process.
_emptied_at_in_memory: datetime | None = None


def _cache_emptied_at() -> datetime:
    """When the layer cache was last emptied. With no record at all the cache counts as emptied now."""
    try:
        return BuildCacheState.model_validate_json(Path(cst.IMAGE_BUILD_CACHE_STATE_PATH).read_text()).emptied_at
    except (OSError, ValidationError):
        return _emptied_at_in_memory or _record_cache_emptied()


def _record_cache_emptied() -> datetime:
    global _emptied_at_in_memory
    state = BuildCacheState(emptied_at=datetime.now(timezone.utc))
    _emptied_at_in_memory = state.emptied_at
    path = Path(cst.IMAGE_BUILD_CACHE_STATE_PATH)
    partial = path.with_name(f"{path.name}.tmp")
    try:
        partial.write_text(state.model_dump_json())
        os.replace(partial, path)
    except OSError as e:
        logger.warning(f"Could not record the build cache state, keeping it in memory: {e}")
    return state.emptied_at


def _docker_disk_used_percent(client: docker.DockerClient) -> float | None:
    """How full the disk under Docker's data directory is; None when the trainer cannot see it."""
    try:
        usage = shutil.disk_usage(client.info()["DockerRootDir"])
    except (OSError, KeyError, APIError) as e:
        logger.warning(f"Could not read the disk usage of Docker's data directory: {e}")
        return None
    return 100 * usage.used / usage.total


def reason_to_empty_cache(client: docker.DockerClient) -> str | None:
    """Why the layer cache must be emptied now, or None to keep it."""
    if cst.IMAGE_BUILD_NO_CACHE:
        return "the layer cache is off (TRAINER_IMAGE_BUILD_NO_CACHE=true)"
    used_percent = _docker_disk_used_percent(client)
    if used_percent is not None and used_percent >= cst.IMAGE_BUILD_CACHE_MAX_DISK_USED_PERCENT:
        return f"the Docker disk is {used_percent:.0f}% full (limit {cst.IMAGE_BUILD_CACHE_MAX_DISK_USED_PERCENT:.0f}%)"
    age = datetime.now(timezone.utc) - _cache_emptied_at()
    if age >= timedelta(hours=cst.IMAGE_BUILD_CACHE_MAX_AGE_HOURS):
        return f"it was last emptied {age.total_seconds() / 3600:.1f} h ago (limit {cst.IMAGE_BUILD_CACHE_MAX_AGE_HOURS:g} h)"
    return None


def prune_build_leftovers() -> None:
    """Empty the layer cache when it is due: remove every dangling image and, if configured, the host build cache.

    Skipped while any image build is in flight. The client has no read timeout: a prune the client gave up on would
    keep running on the daemon after the lock is released.
    """
    with _lock:
        if _builds_in_flight:
            logger.info(f"Build cleanup deferred: {_builds_in_flight} image build(s) in progress.")
            return
        client = docker.from_env(timeout=None)
        reason = reason_to_empty_cache(client)
        if reason is None:
            logger.info("Image layer cache kept.")
            return
        client.images.prune(filters={"dangling": True})
        if cst.PRUNE_HOST_BUILD_CACHE:
            client.api.prune_builds()
        _record_cache_emptied()
        used_percent = _docker_disk_used_percent(client)
        if used_percent is not None and used_percent >= cst.IMAGE_BUILD_CACHE_MAX_DISK_USED_PERCENT:
            logger.warning(
                f"The Docker disk is still {used_percent:.0f}% full after the image layer cache was emptied (limit "
                f"{cst.IMAGE_BUILD_CACHE_MAX_DISK_USED_PERCENT:.0f}%): the cache is emptied on every cleanup and builds "
                "reuse little until other data on that disk is removed or the limit is raised."
            )
        host_cache = "and the host build cache" if cst.PRUNE_HOST_BUILD_CACHE else "(host build cache kept)"
        logger.info(f"Removed dangling images {host_cache}: {reason}.")
