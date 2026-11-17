"""Shared contract for prepared tournament training images stored in an S3-compatible bucket (Cloudflare R2).

Images are stored in the on-disk layout of the CNCF distribution registry (``registry:3``) under
``<prefix>/docker/registry/v2``. Trainers write blobs and link files directly with parallel multipart
uploads (the registry's own S3 driver uploads one part at a time, ~30 MB/s on R2) and read them back with
parallel ranged GETs. Because the layout is the registry's own, a stock read-only ``registry:3`` pointed at
the same bucket can serve the images too (useful for debugging and as a fallback).

Writers: trainer builders (``trainer/prepared_images``). Pruner: validator (``validator/tournament/image_builds``).
"""

import hashlib
import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse


REGISTRY_ROOT = "docker/registry/v2"
REPOSITORY_NAMESPACE = "god"
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

ENV_ENDPOINT = "IMAGE_REGISTRY_ENDPOINT"
ENV_BUCKET = "IMAGE_REGISTRY_BUCKET"
ENV_ACCESS_KEY = "IMAGE_REGISTRY_ACCESS_KEY"
ENV_SECRET_KEY = "IMAGE_REGISTRY_SECRET_KEY"
ENV_PREFIX = "IMAGE_REGISTRY_PREFIX"
ENV_REGION = "IMAGE_REGISTRY_REGION"

OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
OCI_INDEX = "application/vnd.oci.image.index.v1+json"
DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
DOCKER_MANIFEST_LIST = "application/vnd.docker.distribution.manifest.list.v2+json"
MANIFEST_ACCEPT = ", ".join((OCI_MANIFEST, DOCKER_MANIFEST, OCI_INDEX, DOCKER_MANIFEST_LIST))


@dataclass(frozen=True)
class ImageRegistryConfig:
    endpoint: str
    bucket: str
    access_key: str
    secret_key: str
    prefix: str = "registry"
    region: str = "auto"

    @property
    def host(self) -> str:
        parsed = urlparse(self.endpoint if "://" in self.endpoint else f"https://{self.endpoint}")
        return parsed.netloc

    @property
    def secure(self) -> bool:
        return not self.endpoint.startswith("http://")

    @property
    def root(self) -> str:
        prefix = self.prefix.strip("/")
        return f"{prefix}/{REGISTRY_ROOT}" if prefix else REGISTRY_ROOT


def load_registry_config() -> ImageRegistryConfig | None:
    """Read registry settings from the environment; None when not configured."""
    values = {name: os.getenv(name, "").strip() for name in (ENV_ENDPOINT, ENV_BUCKET, ENV_ACCESS_KEY, ENV_SECRET_KEY)}
    if not all(values.values()):
        return None
    return ImageRegistryConfig(
        endpoint=values[ENV_ENDPOINT],
        bucket=values[ENV_BUCKET],
        access_key=values[ENV_ACCESS_KEY],
        secret_key=values[ENV_SECRET_KEY],
        prefix=os.getenv(ENV_PREFIX, "registry").strip(),
        region=os.getenv(ENV_REGION, "auto").strip() or "auto",
    )


def make_minio_client(config: ImageRegistryConfig, http_client=None):
    from minio import Minio

    return Minio(
        config.host,
        access_key=config.access_key,
        secret_key=config.secret_key,
        secure=config.secure,
        region=config.region,
        http_client=http_client,
    )


def validate_digest(digest: str) -> str:
    if not DIGEST_RE.match(digest or ""):
        raise ValueError(f"Invalid image digest: {digest!r}")
    return digest


def sha256_digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _slug(value: str, max_len: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:max_len].strip("-") or "x"


def tournament_repository_prefix(tournament_id: str) -> str:
    return f"{REPOSITORY_NAMESPACE}/{_slug(tournament_id, 64)}"


def repository_name(tournament_id: str, hotkey: str, variant: str) -> str:
    """Registry repository for one participant/variant. Lowercase-only, so the hotkey gets a hash suffix."""
    hotkey_hash = hashlib.sha256(hotkey.encode()).hexdigest()[:10]
    return f"{tournament_repository_prefix(tournament_id)}/{_slug(hotkey, 16)}-{hotkey_hash}-{_slug(variant, 24)}"


def blob_key(config: ImageRegistryConfig, digest: str) -> str:
    hex_digest = validate_digest(digest).split(":", 1)[1]
    return f"{config.root}/blobs/sha256/{hex_digest[:2]}/{hex_digest}/data"


def repository_key(config: ImageRegistryConfig, repository: str) -> str:
    return f"{config.root}/repositories/{repository.strip('/')}/"


def layer_link_key(config: ImageRegistryConfig, repository: str, digest: str) -> str:
    hex_digest = validate_digest(digest).split(":", 1)[1]
    return f"{repository_key(config, repository)}_layers/sha256/{hex_digest}/link"


def revision_link_key(config: ImageRegistryConfig, repository: str, digest: str) -> str:
    hex_digest = validate_digest(digest).split(":", 1)[1]
    return f"{repository_key(config, repository)}_manifests/revisions/sha256/{hex_digest}/link"


def tag_link_keys(config: ImageRegistryConfig, repository: str, tag: str, digest: str) -> tuple[str, str]:
    hex_digest = validate_digest(digest).split(":", 1)[1]
    base = f"{repository_key(config, repository)}_manifests/tags/{tag}"
    return f"{base}/index/sha256/{hex_digest}/link", f"{base}/current/link"


def blob_digest_from_key(key: str) -> str | None:
    """Inverse of ``blob_key``: return the digest for a blob data key, else None."""
    parts = key.rstrip("/").split("/")
    if len(parts) >= 4 and parts[-1] == "data" and parts[-4] == "sha256" and len(parts[-2]) == 64:
        return f"sha256:{parts[-2]}"
    return None


def manifest_child_digests(manifest: dict) -> list[str]:
    """Digests a manifest references: config + layers, or child manifests for an index."""
    digests = [entry["digest"] for entry in manifest.get("manifests", [])]
    if "config" in manifest:
        digests.append(manifest["config"]["digest"])
    digests.extend(layer["digest"] for layer in manifest.get("layers", []))
    return [validate_digest(d) for d in digests]
