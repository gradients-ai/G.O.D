import asyncio

import httpx

from core.models.payload_models import PreparedTrainingImage
from core.models.payload_models import TrainerImagesResponse
from core.models.payload_models import TrainerPrefetchEntry
from core.models.payload_models import TrainerPrefetchRequest
from validator.infrastructure.service_constants import TRAINER_IMAGES_INVENTORY_ENDPOINT
from validator.infrastructure.service_constants import TRAINER_PREFETCH_IMAGES_ENDPOINT


def _base_url(trainer_ip: str) -> str:
    return f"http://{trainer_ip if ':' in trainer_ip else f'{trainer_ip}:8001'}"


async def get_inventory(trainer_ip: str) -> TrainerImagesResponse:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{_base_url(trainer_ip)}{TRAINER_IMAGES_INVENTORY_ENDPOINT}")
        response.raise_for_status()
        return TrainerImagesResponse.model_validate(response.json())


async def ensure_prefetched(
    trainer_ip: str,
    image: PreparedTrainingImage,
    timeout_seconds: float,
) -> None:
    request = TrainerPrefetchRequest(entries=[TrainerPrefetchEntry(**image.model_dump(), priority=0)])
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{_base_url(trainer_ip)}{TRAINER_PREFETCH_IMAGES_ENDPOINT}",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        inventory = await get_inventory(trainer_ip)
        item = next((entry for entry in inventory.images if entry.digest == image.digest), None)
        if item and item.status == "ready":
            return
        if item and item.status == "failed":
            raise RuntimeError(item.last_error or f"Image prefetch failed: {image.digest}")
        await asyncio.sleep(5)
    raise TimeoutError(f"Image prefetch timed out: {image.digest}")


async def request_prefetch(
    trainer_ip: str,
    images: list[PreparedTrainingImage],
    priority: int = 100,
) -> None:
    if not images:
        return
    request = TrainerPrefetchRequest(entries=[TrainerPrefetchEntry(**image.model_dump(), priority=priority) for image in images])
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{_base_url(trainer_ip)}{TRAINER_PREFETCH_IMAGES_ENDPOINT}",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()
