"""HTTP client for trainer image-builder endpoints (trainer/prepared_images)."""

import httpx

from core.models.payload_models import ImageBuilderStatus
from core.models.payload_models import ImageBuildRequest
from core.models.payload_models import ImageBuildStatusResponse
from validator.infrastructure.service_constants import IMAGE_BUILD_ENDPOINT
from validator.infrastructure.service_constants import IMAGE_BUILD_STATUS_ENDPOINT
from validator.infrastructure.service_constants import IMAGE_BUILDER_STATUS_ENDPOINT
from validator.tournament.image_builds import constants as cst


class BuilderRejected(Exception):
    """The builder definitely did not start the job (refused or unreachable before sending)."""


class BuilderOutcomeUnknown(Exception):
    """The request may or may not have reached the builder (timeout, 5xx)."""


def base_url(address: str) -> str:
    return f"http://{address if ':' in address else f'{address}:{cst.DEFAULT_TRAINER_PORT}'}"


async def fetch_status(address: str) -> ImageBuilderStatus | None:
    try:
        async with httpx.AsyncClient(timeout=cst.BUILDER_HTTP_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{base_url(address)}{IMAGE_BUILDER_STATUS_ENDPOINT}")
        if response.status_code != 200:
            return None
        return ImageBuilderStatus.model_validate(response.json())
    except (httpx.HTTPError, ValueError):
        return None


async def submit(address: str, request: ImageBuildRequest) -> ImageBuildStatusResponse:
    try:
        async with httpx.AsyncClient(timeout=cst.BUILDER_HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{base_url(address)}{IMAGE_BUILD_ENDPOINT}", json=request.model_dump(mode="json"))
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise BuilderRejected(f"cannot connect: {exc}") from exc
    except httpx.HTTPError as exc:
        raise BuilderOutcomeUnknown(str(exc)) from exc
    if response.status_code in (400, 403, 404, 409, 422, 429, 503):
        raise BuilderRejected(f"HTTP {response.status_code}: {response.text[:300]}")
    if response.status_code != 200:
        raise BuilderOutcomeUnknown(f"HTTP {response.status_code}: {response.text[:300]}")
    return ImageBuildStatusResponse.model_validate(response.json())


async def poll(address: str, build_id: str, attempt: int) -> ImageBuildStatusResponse | None:
    """Job status; None when the builder does not know the job (it restarted). Raises httpx errors if unreachable."""
    path = IMAGE_BUILD_STATUS_ENDPOINT.format(build_id=build_id, attempt=attempt)
    async with httpx.AsyncClient(timeout=cst.BUILDER_HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{base_url(address)}{path}")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return ImageBuildStatusResponse.model_validate(response.json())
