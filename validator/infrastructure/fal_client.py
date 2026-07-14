import asyncio
import mimetypes
import tempfile
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

import validator.tasks.synthetics.constants as synth_cst
from core.constants.credentials import BUCKET_NAME
from core.logging import get_logger
from core.models.dataset_models import ImageTextPair
from validator.tasks.datasets.constants import TEMP_PATH_FOR_IMAGES
from validator.tasks.details import retry_http_with_backoff
from validator.tasks.details import upload_file_to_minio


logger = get_logger(__name__)


def _fal_headers() -> dict[str, str]:
    if not synth_cst.FAL_KEY:
        raise ValueError("FAL_KEY is not set")
    return {
        "Authorization": f"Key {synth_cst.FAL_KEY}",
        "Content-Type": "application/json",
    }


def _fal_url(model_id: str) -> str:
    return f"https://fal.run/{model_id.strip('/')}"


@retry_http_with_backoff
async def post_to_fal(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    timeout = httpx.Timeout(synth_cst.FAL_TIMEOUT_SECONDS, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout, headers=_fal_headers()) as client:
        response = await client.post(_fal_url(model_id), json=payload)
        if response.status_code != 200:
            logger.error(f"FAL {model_id} error: {response.status_code} - {response.text}")
            response.raise_for_status()
        return response.json()


def extract_image_urls(result: dict[str, Any]) -> list[str]:
    images = result.get("images")
    if isinstance(images, list):
        urls = []
        for image in images:
            if isinstance(image, str):
                urls.append(image)
            elif isinstance(image, dict) and isinstance(image.get("url"), str):
                urls.append(image["url"])
        if urls:
            return urls

    data = result.get("data")
    if isinstance(data, dict):
        return extract_image_urls(data)

    image = result.get("image")
    if isinstance(image, dict) and isinstance(image.get("url"), str):
        return [image["url"]]
    if isinstance(image, str):
        return [image]

    raise RuntimeError(f"FAL result did not include image URLs: {result}")


def extract_text(result: dict[str, Any]) -> str:
    data = result.get("data")
    if isinstance(data, dict):
        try:
            return extract_text(data)
        except RuntimeError:
            pass

    choices = result.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first_choice.get("text"), str):
                return first_choice["text"]

    for key in ("output", "text", "content", "response", "message"):
        value = result.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("content"), str):
            return value["content"]

    raise RuntimeError(f"FAL result did not include text content: {result}")


def _extension_from_response(url: str, response: httpx.Response) -> str:
    content_type = response.headers.get("content-type", "").split(";")[0].strip()
    extension = mimetypes.guess_extension(content_type) if content_type else None
    if extension:
        return ".jpg" if extension == ".jpe" else extension

    path_suffix = Path(urlparse(url).path).suffix
    if path_suffix:
        return path_suffix
    return ".png"


def _ensure_image_response(url: str, response: httpx.Response) -> None:
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type.startswith("image/"):
        return

    raise httpx.HTTPStatusError(
        f"URL did not return an image: {url} (content-type={content_type or 'unknown'})",
        request=response.request,
        response=response,
    )


@retry_http_with_backoff
async def download_url(url: str, destination_without_suffix: Path) -> Path:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(url, headers={"Accept": "image/*"})
        response.raise_for_status()
        _ensure_image_response(url, response)

    destination = destination_without_suffix.with_suffix(_extension_from_response(url, response))
    destination.write_bytes(response.content)
    return destination


async def upload_local_file(file_path: Path, prefix: str | None = None) -> str:
    if not BUCKET_NAME:
        raise ValueError("S3_BUCKET_NAME is not set")

    object_prefix = f"{prefix.strip('/')}/" if prefix else ""
    object_name = f"{object_prefix}{uuid.uuid4()}{file_path.suffix}"
    url = await upload_file_to_minio(str(file_path), BUCKET_NAME, object_name)
    if not url:
        raise RuntimeError(f"Failed to upload {file_path} to S3")
    return url


async def persist_image_text_pair(image_url: str, prompt: str, work_dir: Path, index: int) -> ImageTextPair:
    logger.info(f"Persisting image-text pair {index + 1}")
    image_path = await download_url(image_url, work_dir / f"{index}")

    text_path = work_dir / f"{index}.txt"
    text_path.write_text(prompt)

    image_s3_url, text_s3_url = await asyncio.gather(
        upload_local_file(image_path, "image_synth/images"),
        upload_local_file(text_path, "image_synth/prompts"),
    )
    logger.info(f"Persisted image-text pair {index + 1}")
    return ImageTextPair(image_url=image_s3_url, text_url=text_s3_url)


async def persist_image_text_pairs(image_prompt_pairs: list[tuple[str, str]]) -> list[ImageTextPair]:
    Path(TEMP_PATH_FOR_IMAGES).mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(synth_cst.FAL_IMAGE_GENERATION_CONCURRENCY)

    async def persist_one(index: int, image_url: str, prompt: str, work_dir: Path) -> ImageTextPair:
        async with semaphore:
            return await persist_image_text_pair(image_url, prompt, work_dir, index)

    with tempfile.TemporaryDirectory(dir=TEMP_PATH_FOR_IMAGES) as tmp_dir:
        work_dir = Path(tmp_dir)
        tasks = [persist_one(index, image_url, prompt, work_dir) for index, (image_url, prompt) in enumerate(image_prompt_pairs)]
        return await asyncio.gather(*tasks)
