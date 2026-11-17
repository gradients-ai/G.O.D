"""Make a prepared image available in the local Docker image store before training.

Fast path: the image is already present (this trainer built it, or a previous round pulled it). Otherwise the
blobs are fetched from R2 with parallel ranged GETs into the localhost registry storage (sha256-verified) and
``docker pull`` runs against localhost. Measured on an H100 host: ~2x faster than a plain registry pull, whose
single connection per layer tops out around 50 MB/s from R2.
"""

import asyncio
import subprocess

from core.image_registry import load_registry_config
from core.logging import get_logger
from core.models.payload_models import PreparedTrainingImage
from trainer import constants as cst
from trainer.prepared_images.cache import touch_prepared_image
from trainer.prepared_images.local_registry import LocalRegistry
from trainer.prepared_images.local_registry import local_registry
from trainer.prepared_images.r2_transfer import R2ImageStore
from trainer.prepared_images.r2_transfer import fetch_image


logger = get_logger(__name__)
_pull_locks: dict[str, asyncio.Lock] = {}


class PreparedImageError(RuntimeError):
    pass


def _image_present(ref: str) -> bool:
    result = subprocess.run(["docker", "image", "inspect", ref], capture_output=True, timeout=60)
    return result.returncode == 0


def _docker_pull(ref: str) -> tuple[int, str]:
    try:
        result = subprocess.run(
            ["docker", "pull", ref], capture_output=True, text=True, timeout=cst.IMAGE_PULL_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        return 124, f"docker pull timed out after {cst.IMAGE_PULL_TIMEOUT_SECONDS}s"
    return result.returncode, (result.stdout + result.stderr)[-2000:]


async def ensure_prepared_image(
    image: PreparedTrainingImage,
    log_labels: dict | None = None,
    registry: LocalRegistry = local_registry,
) -> str:
    """Return a local image reference (repo@digest) for ``image``, pulling it if needed."""
    ref = registry.ref(image.repository, image.digest)
    if await asyncio.to_thread(_image_present, ref):
        touch_prepared_image(ref)
        logger.info(f"Prepared image already present: {ref}", extra=log_labels)
        return ref

    config = load_registry_config()
    if config is None:
        raise PreparedImageError("IMAGE_REGISTRY_* settings are missing on this trainer")

    lock = _pull_locks.setdefault(image.digest, asyncio.Lock())
    async with lock:
        if await asyncio.to_thread(_image_present, ref):
            touch_prepared_image(ref)
            return ref
        last_error: Exception | None = None
        for attempt in range(1, cst.IMAGE_PULL_ATTEMPTS + 1):
            try:
                with registry.lease():
                    await asyncio.to_thread(registry.ensure_running)
                    loop = asyncio.get_running_loop()
                    started = loop.time()
                    fetched = await asyncio.to_thread(fetch_image, R2ImageStore(config), registry, image.repository, image.digest)
                    fetch_seconds = loop.time() - started
                    rc, output = await asyncio.to_thread(_docker_pull, ref)
                    if rc != 0:
                        raise PreparedImageError(f"docker pull failed (rc={rc}): {output}")
                if not await asyncio.to_thread(_image_present, ref):
                    raise PreparedImageError(f"docker pull reported success but {ref} is not present")
                touch_prepared_image(ref)
                logger.info(
                    f"Pulled prepared image {ref}: {fetched / 1e9:.1f} GB from R2 in {fetch_seconds:.0f}s, "
                    f"total {loop.time() - started:.0f}s",
                    extra=log_labels,
                )
                return ref
            except Exception as exc:
                last_error = exc
                logger.warning(f"Prepared image pull attempt {attempt} failed: {exc}", extra=log_labels)
                if attempt < cst.IMAGE_PULL_ATTEMPTS:
                    await asyncio.sleep(10 * attempt)
        raise PreparedImageError(f"Could not pull prepared image {ref}: {last_error}")
