"""Re-sign stored presigned URLs before handing them out.

Tasks store their data as presigned GET URLs on our S3 endpoint. Those stop working when they expire, and
all at once when the access key that signed them is rotated. Read endpoints call resign_presigned_urls on
what they return, so every URL on our endpoint carries a fresh signature from the current key.
"""

import asyncio
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from core.logging import get_logger
from validator.infrastructure.minio_client import AsyncMinioClient
from validator.infrastructure.minio_client import async_minio_client


logger = get_logger(__name__)


def _is_our_presigned_url(value: str, client: AsyncMinioClient) -> bool:
    if "X-Amz-Signature=" not in value or not value.startswith(("http://", "https://")):
        return False
    return urlparse(value).netloc.lower() == client.endpoint.lower()


async def _resign(value: str, client: AsyncMinioClient) -> str:
    # Not get_new_presigned_url: that logs every URL, and a task list holds thousands.
    try:
        bucket_name, object_name = client.parse_s3_url(value)
        return await client.get_presigned_url(bucket_name, object_name)
    except Exception as e:  # noqa: BLE001 - keep the stored URL; no worse than before
        logger.warning(f"Could not re-sign a presigned URL: {type(e).__name__}")
        return value


async def resign_presigned_urls(obj: Any, client: AsyncMinioClient = async_minio_client) -> Any:
    """Re-sign, in place where possible, every presigned URL on our endpoint found in obj.

    Walks pydantic models, dicts and lists; strings are replaced in their container. URLs on other hosts are
    left as they are. Returns obj (or the replacement, when obj is itself a string).
    """
    if isinstance(obj, str):
        return await _resign(obj, client) if _is_our_presigned_url(obj, client) else obj

    if isinstance(obj, BaseModel):
        names = [name for name in type(obj).model_fields if getattr(obj, name, None) is not None]
        values = await asyncio.gather(*(resign_presigned_urls(getattr(obj, name), client) for name in names))
        for name, value in zip(names, values):
            if value is not getattr(obj, name):
                object.__setattr__(obj, name, value)
        return obj

    if isinstance(obj, list):
        obj[:] = await asyncio.gather(*(resign_presigned_urls(item, client) for item in obj))
        return obj

    if isinstance(obj, dict):
        keys = list(obj)
        values = await asyncio.gather(*(resign_presigned_urls(obj[key], client) for key in keys))
        obj.update(zip(keys, values))
        return obj

    return obj
