"""Parallel blob transfer between the localhost staging registry and the R2 image registry.

Publish order makes a half-finished publish invisible: blobs first, then repository link files, then the
manifest revision, and the tag last. Readers always address images by manifest digest and verify every blob.
"""

import hashlib
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import urllib3
from minio.error import S3Error

from core.image_registry import ImageRegistryConfig
from core.image_registry import blob_key
from core.image_registry import layer_link_key
from core.image_registry import make_minio_client
from core.image_registry import manifest_child_digests
from core.image_registry import revision_link_key
from core.image_registry import sha256_digest
from core.image_registry import tag_link_keys
from core.image_registry import validate_digest
from core.logging import get_logger
from trainer import constants as cst
from trainer.prepared_images.local_registry import LocalRegistry


logger = get_logger(__name__)

_blob_locks: dict[str, threading.Lock] = {}
_blob_locks_guard = threading.Lock()


def _blob_lock(digest: str) -> threading.Lock:
    with _blob_locks_guard:
        return _blob_locks.setdefault(digest, threading.Lock())


def _retry(action, what: str, attempts: int = cst.R2_TRANSFER_RETRIES):
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as exc:
            if isinstance(exc, S3Error) and exc.code in ("NoSuchKey", "NoSuchBucket", "AccessDenied"):
                raise
            if attempt == attempts:
                raise
            logger.warning(f"R2 {what} failed (attempt {attempt}/{attempts}): {exc}; retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2


def file_sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


@dataclass
class PublishResult:
    size_bytes: int
    uploaded_bytes: int


class R2ImageStore:
    def __init__(self, config: ImageRegistryConfig):
        self.config = config
        pool = urllib3.PoolManager(
            num_pools=4,
            maxsize=cst.R2_PARALLEL_PARTS * cst.R2_PARALLEL_BLOBS + 8,
            timeout=urllib3.Timeout(connect=15, read=180),
            retries=urllib3.Retry(total=3, backoff_factor=1.0, status_forcelist=(500, 502, 503, 504)),
        )
        self.client = make_minio_client(config, http_client=pool)

    # --- small objects -------------------------------------------------------------------------------------
    def stat_size(self, key: str) -> int | None:
        try:
            return _retry(lambda: self.client.stat_object(self.config.bucket, key).size, f"stat {key}")
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchObject", "NotFound"):
                return None
            raise

    def put_bytes(self, key: str, data: bytes) -> None:
        _retry(
            lambda: self.client.put_object(self.config.bucket, key, io.BytesIO(data), len(data)),
            f"put {key}",
        )

    def get_bytes(self, key: str) -> bytes:
        def fetch():
            response = self.client.get_object(self.config.bucket, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        return _retry(fetch, f"get {key}")

    # --- blobs ---------------------------------------------------------------------------------------------
    def upload_blob(self, digest: str, path: Path) -> int:
        """Upload one blob unless an object of the same size is already there. Returns bytes uploaded."""
        key = blob_key(self.config, digest)
        size = path.stat().st_size
        with _blob_lock(digest):
            if self.stat_size(key) == size:
                return 0
            _retry(
                lambda: self.client.fput_object(
                    self.config.bucket,
                    key,
                    str(path),
                    part_size=cst.R2_PART_SIZE_BYTES,
                    num_parallel_uploads=cst.R2_PARALLEL_PARTS,
                ),
                f"upload {digest}",
            )
            remote = self.stat_size(key)
            if remote != size:
                raise RuntimeError(f"R2 blob {digest} has size {remote} after upload, expected {size}")
            return size

    def download_blob(self, digest: str, dest: Path, size: int | None = None) -> int:
        """Download a blob with parallel ranged GETs, verify its sha256, and move it into place atomically."""
        validate_digest(digest)
        key = blob_key(self.config, digest)
        if size is None:
            size = self.stat_size(key)
            if size is None:
                raise FileNotFoundError(f"Blob {digest} is missing from the image registry")
        if dest.exists() and dest.stat().st_size == size:
            return 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{threading.get_ident()}.part")
        part = max(cst.R2_PART_SIZE_BYTES, 1)
        ranges = [(offset, min(part, size - offset)) for offset in range(0, size, part)] or [(0, 0)]
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, size)

            def fetch_range(offset: int, length: int) -> None:
                if length == 0:
                    return

                def once():
                    response = self.client.get_object(self.config.bucket, key, offset=offset, length=length)
                    try:
                        position = offset
                        for chunk in response.stream(1024 * 1024):
                            os.pwrite(fd, chunk, position)
                            position += len(chunk)
                        if position != offset + length:
                            raise IOError(f"short read for {digest} range {offset}+{length}: got {position - offset}")
                    finally:
                        response.close()
                        response.release_conn()

                _retry(once, f"download {digest}[{offset}]")

            with ThreadPoolExecutor(max_workers=cst.R2_PARALLEL_PARTS) as pool:
                for future in [pool.submit(fetch_range, off, length) for off, length in ranges]:
                    future.result()
        finally:
            os.close(fd)
        try:
            actual = file_sha256(tmp)
            if actual != digest:
                raise IOError(f"Blob digest mismatch: expected {digest}, got {actual}")
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        return size


def _read_manifest(path: Path, digest: str) -> tuple[bytes, dict]:
    data = path.read_bytes()
    if sha256_digest(data) != digest:
        raise IOError(f"Manifest {digest} does not match its content")
    return data, json.loads(data)


def publish_image(store: R2ImageStore, registry: LocalRegistry, repository: str, tag: str, digest: str) -> PublishResult:
    """Copy one image (by manifest digest) from the localhost registry storage to R2."""
    manifests: list[str] = []
    blobs: list[str] = []
    pending = [digest]
    while pending:
        manifest_digest = pending.pop()
        _, manifest = _read_manifest(registry.blob_path(manifest_digest), manifest_digest)
        manifests.append(manifest_digest)
        children = manifest_child_digests(manifest)
        if manifest.get("manifests"):
            pending.extend(children)
        else:
            blobs.extend(children)
    blobs = list(dict.fromkeys(blobs))

    missing = [d for d in blobs + manifests if not registry.blob_path(d).exists()]
    if missing:
        raise FileNotFoundError(f"Local registry is missing blobs for {repository}@{digest}: {missing[:3]}")

    with ThreadPoolExecutor(max_workers=cst.R2_PARALLEL_BLOBS) as pool:
        uploaded = sum(pool.map(lambda d: store.upload_blob(d, registry.blob_path(d)), blobs))
    # Child manifests before the parent so the parent never references something unreadable.
    for manifest_digest in reversed(manifests):
        uploaded += store.upload_blob(manifest_digest, registry.blob_path(manifest_digest))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda d: store.put_bytes(layer_link_key(store.config, repository, d), d.encode()), blobs))
    for manifest_digest in reversed(manifests):
        store.put_bytes(revision_link_key(store.config, repository, manifest_digest), manifest_digest.encode())
    index_key, current_key = tag_link_keys(store.config, repository, tag, digest)
    store.put_bytes(index_key, digest.encode())
    store.put_bytes(current_key, digest.encode())

    size = sum(registry.blob_path(d).stat().st_size for d in blobs + manifests)
    return PublishResult(size_bytes=size, uploaded_bytes=uploaded)


def fetch_image(store: R2ImageStore, registry: LocalRegistry, repository: str, digest: str) -> int:
    """Download one image (by manifest digest) from R2 into the localhost registry storage. Returns bytes fetched."""
    fetched = 0
    pending = [digest]
    layers: list[str] = []
    manifests: list[str] = []
    while pending:
        manifest_digest = pending.pop()
        data = store.get_bytes(blob_key(store.config, manifest_digest))
        if sha256_digest(data) != manifest_digest:
            raise IOError(f"Manifest {manifest_digest} in the image registry does not match its digest")
        manifest = json.loads(data)
        path = registry.blob_path(manifest_digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifests.append(manifest_digest)
        children = manifest_child_digests(manifest)
        if manifest.get("manifests"):
            # Only follow linux/amd64 children of an index; attestation manifests are not needed to run.
            amd64 = [
                entry["digest"]
                for entry in manifest["manifests"]
                if (entry.get("platform") or {}).get("architecture") in (None, "amd64")
                and (entry.get("platform") or {}).get("os") in (None, "linux")
            ]
            pending.extend(amd64 or children)
        else:
            layers.extend(children)
    layers = list(dict.fromkeys(layers))
    with ThreadPoolExecutor(max_workers=cst.R2_PARALLEL_BLOBS) as pool:
        fetched += sum(pool.map(lambda d: store.download_blob(d, registry.blob_path(d)), layers))
    for layer in layers:
        registry.write_layer_link(repository, layer)
    for manifest_digest in manifests:
        registry.write_revision_link(repository, manifest_digest)
    return fetched
