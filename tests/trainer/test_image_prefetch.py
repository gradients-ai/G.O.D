import asyncio

import pytest

from core.models.payload_models import PreparedTrainingImage
from core.models.payload_models import TrainerPrefetchEntry
from trainer import image_prefetch


@pytest.mark.asyncio
async def test_prefetch_coalesces_by_digest_and_becomes_ready(monkeypatch, tmp_path):
    image_prefetch._state.clear()
    image_prefetch._workers.clear()
    image_prefetch._events.clear()
    image_prefetch._queue = asyncio.PriorityQueue()
    monkeypatch.setattr(image_prefetch, "STATE_PATH", tmp_path / "prefetch.json")
    monkeypatch.setattr(image_prefetch, "_pull", lambda entry: None)
    monkeypatch.setattr(image_prefetch, "_evict_if_needed", lambda: None)
    image = PreparedTrainingImage(
        build_id="build-1",
        pull_ref="registry.depot.dev/project:tag",
        digest="sha256:" + "a" * 64,
        variant="text",
    )
    entry = TrainerPrefetchEntry(**image.model_dump(), priority=1)

    await image_prefetch.enqueue([entry, entry])
    exact_ref = await image_prefetch.ensure_ready(image, timeout=2)

    assert exact_ref == f"{image.pull_ref}@{image.digest}"
    assert image_prefetch.inventory().images[0].status == "ready"
    for worker in image_prefetch._workers:
        worker.cancel()
    await asyncio.gather(*image_prefetch._workers, return_exceptions=True)
    image_prefetch._workers.clear()
