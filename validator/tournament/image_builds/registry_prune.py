"""Delete completed tournaments' images from the R2 image registry and garbage-collect unreferenced blobs.

Deleting a repository removes only its link files; blobs can be shared with other repositories (identical layers
across miners and tournaments), so a mark-and-sweep removes blobs no remaining repository links to. Blobs younger
than ORPHAN_BLOB_MIN_AGE_HOURS are kept: a build in flight uploads blobs before writing its links.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

from minio.deleteobjects import DeleteObject

from core.image_registry import ImageRegistryConfig
from core.image_registry import blob_digest_from_key
from core.image_registry import make_minio_client
from core.image_registry import repository_key
from core.image_registry import tournament_repository_prefix
from core.logging import get_logger
from validator.tournament.image_builds import constants as cst


logger = get_logger(__name__)


def _delete_keys(client, bucket: str, keys: list[str]) -> int:
    deleted = 0
    for start in range(0, len(keys), 1000):
        batch = [DeleteObject(key) for key in keys[start : start + 1000]]
        errors = list(client.remove_objects(bucket, batch))
        for error in errors:
            logger.warning(f"Could not delete {error.name}: {error.code} {error.message}")
        deleted += len(batch) - len(errors)
    return deleted


def delete_tournament_repositories(config: ImageRegistryConfig, tournament_id: str) -> int:
    client = make_minio_client(config)
    prefix = repository_key(config, tournament_repository_prefix(tournament_id))
    keys = [obj.object_name for obj in client.list_objects(config.bucket, prefix=prefix, recursive=True)]
    return _delete_keys(client, config.bucket, keys)


def referenced_digests_from_keys(keys: list[str]) -> set[str]:
    """Digests linked by any repository: .../_layers/sha256/<hex>/link and .../revisions/sha256/<hex>/link."""
    referenced = set()
    for key in keys:
        parts = key.split("/")
        if len(parts) >= 4 and parts[-1] == "link" and parts[-3] == "sha256" and len(parts[-2]) == 64:
            referenced.add(f"sha256:{parts[-2]}")
    return referenced


def orphan_blob_keys(blobs: list[tuple[str, datetime]], referenced: set[str], now: datetime) -> list[str]:
    cutoff = now - timedelta(hours=cst.ORPHAN_BLOB_MIN_AGE_HOURS)
    orphans = []
    for key, modified in blobs:
        digest = blob_digest_from_key(key)
        if digest and digest not in referenced and modified < cutoff:
            orphans.append(key)
    return orphans


def garbage_collect_blobs(config: ImageRegistryConfig) -> tuple[int, int]:
    """Returns (orphans deleted, bytes freed)."""
    client = make_minio_client(config)
    link_keys = [
        obj.object_name
        for obj in client.list_objects(config.bucket, prefix=f"{config.root}/repositories/", recursive=True)
        if obj.object_name.endswith("/link")
    ]
    referenced = referenced_digests_from_keys(link_keys)
    blobs = list(client.list_objects(config.bucket, prefix=f"{config.root}/blobs/", recursive=True))
    sizes = {obj.object_name: obj.size or 0 for obj in blobs}
    orphans = orphan_blob_keys(
        [(obj.object_name, obj.last_modified) for obj in blobs if obj.last_modified], referenced, datetime.now(timezone.utc)
    )
    deleted = _delete_keys(client, config.bucket, orphans)
    return deleted, sum(sizes[key] for key in orphans)
