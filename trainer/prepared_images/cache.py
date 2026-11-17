"""Host-side cache of prepared images: last-use tracking and eviction.

Prepared images are kept after training so later rounds on the same trainer start without a pull. They are
evicted when unused for PREPARED_IMAGE_TTL_HOURS, or least-recently-used first while free disk is below
PREPARED_IMAGE_MIN_FREE_DISK_GB. Images used by any container are never removed.
"""

import json
import os
import shutil
import tempfile
import threading
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import docker

from core.logging import get_logger
from trainer import constants as cst


logger = get_logger(__name__)
_state_lock = threading.Lock()


def _state_path() -> Path:
    return Path(cst.PREPARED_IMAGES_STATE_PATH)


def _load() -> dict[str, str]:
    try:
        return json.loads(_state_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(state: dict[str, str]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".prepared-images-", text=True)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def touch_prepared_image(ref: str) -> None:
    with _state_lock:
        state = _load()
        state[ref] = datetime.now(timezone.utc).isoformat()
        _save(state)


def _prepared_refs(image) -> list[str]:
    prefix = f"127.0.0.1:{cst.LOCAL_REGISTRY_PORT}/"
    refs = list(image.attrs.get("RepoDigests") or []) + list(image.tags or [])
    return [ref for ref in refs if ref.startswith(prefix)]


def select_evictions(
    images: list[tuple[str, list[str]]],
    last_used: dict[str, str],
    in_use: set[str],
    now: datetime,
    free_disk_gb: float,
    image_sizes_gb: dict[str, float],
) -> list[str]:
    """Pure eviction policy. images: (image_id, refs). Returns image ids to remove."""
    ttl = timedelta(hours=cst.PREPARED_IMAGE_TTL_HOURS)

    def last(refs: list[str]) -> datetime:
        stamps = [datetime.fromisoformat(last_used[r]) for r in refs if r in last_used]
        return max(stamps) if stamps else datetime.min.replace(tzinfo=timezone.utc)

    candidates = sorted(((last(refs), image_id) for image_id, refs in images if image_id not in in_use))
    evict = [image_id for used, image_id in candidates if now - used > ttl]
    for used, image_id in candidates:
        if free_disk_gb >= cst.PREPARED_IMAGE_MIN_FREE_DISK_GB:
            break
        if image_id not in evict:
            evict.append(image_id)
            free_disk_gb += image_sizes_gb.get(image_id, 0.0)
    return evict


def evict_prepared_images() -> None:
    client = docker.from_env()
    images = [(image.id, _prepared_refs(image), image) for image in client.images.list()]
    images = [entry for entry in images if entry[1]]
    if not images:
        return
    in_use = set()
    for container in client.containers.list(all=True):
        image_id = container.attrs.get("Image")
        if image_id:
            in_use.add(image_id)
    free_disk_gb = shutil.disk_usage("/").free / 1e9
    sizes = {image_id: (image.attrs.get("Size") or 0) / 1e9 for image_id, _, image in images}
    with _state_lock:
        state = _load()
    evict = select_evictions(
        [(image_id, refs) for image_id, refs, _ in images], state, in_use, datetime.now(timezone.utc), free_disk_gb, sizes
    )
    removed_refs = set()
    for image_id, refs, _ in images:
        if image_id not in evict:
            continue
        # Remove by reference (not id): an image with several references cannot be removed by id without force.
        for ref in refs:
            try:
                client.images.remove(ref, force=False)
                removed_refs.add(ref)
            except docker.errors.ImageNotFound:
                removed_refs.add(ref)
            except docker.errors.APIError as exc:
                logger.warning(f"Could not evict prepared image {ref}: {exc}")
        logger.info(f"Evicted prepared image {refs[0]}")
    if removed_refs:
        with _state_lock:
            state = _load()
            for ref in removed_refs:
                state.pop(ref, None)
            _save(state)
    try:
        client.api.prune_builds(filters={"until": cst.IMAGE_BUILD_CACHE_MAX_AGE})
    except docker.errors.APIError as exc:
        logger.warning(f"Build cache prune failed: {exc}")
