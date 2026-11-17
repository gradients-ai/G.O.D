"""Host-side, digest-keyed image prefetch queue for prepared training images."""

import asyncio
import json
import os
import shutil
import tempfile
import threading
from datetime import datetime
from datetime import timezone
from pathlib import Path

import docker

from core.logging import get_logger
from core.models.payload_models import PreparedTrainingImage
from core.models.payload_models import TrainerImageInventoryItem
from core.models.payload_models import TrainerImagesResponse
from core.models.payload_models import TrainerPrefetchEntry


logger = get_logger(__name__)
STATE_PATH = Path(os.getenv("IMAGE_PREFETCH_STATE_PATH", "trainer/image_prefetch_state.json"))
_state_lock = threading.Lock()
_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
_workers: list[asyncio.Task] = []
_events: dict[str, asyncio.Event] = {}


def _load() -> dict[str, dict]:
    try:
        payload = json.loads(STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    for item in payload.values():
        if item["status"] == "downloading":
            item["status"] = "queued"
    return payload


_state = _load()


def _save() -> None:
    with _state_lock:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=STATE_PATH.parent, prefix=".prefetch-", text=True)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(_state, handle, indent=2, sort_keys=True)
            os.replace(name, STATE_PATH)
        finally:
            if os.path.exists(name):
                os.unlink(name)


def _exact_ref(image: PreparedTrainingImage) -> str:
    return image.pull_ref if "@" in image.pull_ref else f"{image.pull_ref}@{image.digest}"


def _pull(entry: TrainerPrefetchEntry) -> None:
    client = docker.from_env()
    exact_ref = _exact_ref(entry)
    image = client.images.pull(exact_ref)
    repo_digests = set(image.attrs.get("RepoDigests") or [])
    if not any(value.endswith(f"@{entry.digest}") for value in repo_digests):
        client.images.remove(exact_ref, force=True)
        raise RuntimeError(f"Pulled image did not expose expected digest {entry.digest}")


def _evict_if_needed() -> None:
    usage = shutil.disk_usage("/")
    high = float(os.getenv("IMAGE_CACHE_HIGH_WATERMARK", "0.85"))
    low = float(os.getenv("IMAGE_CACHE_LOW_WATERMARK", "0.75"))
    if usage.used / usage.total < high:
        return
    client = docker.from_env()
    running_refs = {container.attrs.get("Config", {}).get("Image") for container in client.containers.list()}
    candidates = sorted(
        (
            (digest, item)
            for digest, item in _state.items()
            if item["status"] == "ready" and _exact_ref(PreparedTrainingImage.model_validate(item["entry"])) not in running_refs
        ),
        key=lambda pair: pair[1].get("last_used_at") or "",
    )
    for digest, item in candidates:
        image = PreparedTrainingImage.model_validate(item["entry"])
        try:
            client.images.remove(_exact_ref(image))
        except docker.errors.ImageNotFound:
            pass
        _state.pop(digest, None)
        usage = shutil.disk_usage("/")
        if usage.used / usage.total <= low:
            break
    _save()


async def _worker() -> None:
    while True:
        priority, digest = await _queue.get()
        item = _state.get(digest)
        if not item or item["status"] == "ready":
            _queue.task_done()
            continue
        item["status"] = "downloading"
        item["last_error"] = None
        _save()
        try:
            entry = TrainerPrefetchEntry.model_validate(item["entry"])
            await asyncio.to_thread(_evict_if_needed)
            await asyncio.to_thread(_pull, entry)
            item["status"] = "ready"
            item["last_used_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"Prepared image ready: {digest}")
        except Exception as exc:
            item["status"] = "failed"
            item["last_error"] = str(exc)[-1000:]
            logger.exception(f"Prepared image pull failed: {digest}")
        finally:
            _save()
            _events.setdefault(digest, asyncio.Event()).set()
            _queue.task_done()


async def start_workers() -> None:
    if _workers:
        return
    concurrency = int(os.getenv("IMAGE_PREFETCH_CONCURRENCY", "2"))
    for digest, item in _state.items():
        if item["status"] == "queued":
            await _queue.put((item["entry"].get("priority", 100), digest))
    _workers.extend(asyncio.create_task(_worker()) for _ in range(concurrency))


async def enqueue(entries: list[TrainerPrefetchEntry]) -> TrainerImagesResponse:
    await start_workers()
    for entry in entries:
        existing = _state.get(entry.digest)
        if existing and existing["status"] in {"queued", "downloading", "ready"}:
            continue
        _events[entry.digest] = asyncio.Event()
        _state[entry.digest] = {"status": "queued", "entry": entry.model_dump(mode="json"), "last_error": None}
        await _queue.put((entry.priority, entry.digest))
    _save()
    return inventory()


async def ensure_ready(image: PreparedTrainingImage, timeout: float | None = None) -> str:
    item = _state.get(image.digest)
    if not item or item["status"] not in {"queued", "downloading", "ready"}:
        await enqueue([TrainerPrefetchEntry(**image.model_dump(), priority=0)])
        item = _state[image.digest]
    if item["status"] != "ready":
        event = _events.setdefault(image.digest, asyncio.Event())
        await asyncio.wait_for(event.wait(), timeout=timeout)
        item = _state[image.digest]
    if item["status"] != "ready":
        raise RuntimeError(item.get("last_error") or f"Image {image.digest} is not ready")
    item["last_used_at"] = datetime.now(timezone.utc).isoformat()
    _save()
    return _exact_ref(image)


def inventory() -> TrainerImagesResponse:
    images = []
    for digest, item in _state.items():
        entry = item["entry"]
        images.append(
            TrainerImageInventoryItem(
                **entry,
                status=item["status"],
                last_error=item.get("last_error"),
                last_used_at=item.get("last_used_at"),
            )
        )
    return TrainerImagesResponse(
        images=images,
        queue_depth=sum(item["status"] in {"queued", "downloading"} for item in _state.values()),
    )
