"""Quant artifacts: counted bytes on HF and the decoded freeform model in object storage.

Stdlib + huggingface_hub + pydantic only, so it imports both on the validator and inside the quant
evaluator image. Nothing here executes downloaded content: the decoded model directory is bf16
safetensors written by the trainer's decode job plus the base model's config and tokenizer.

Object-storage layout written by the trainer decode job, one prefix per (task, hotkey):

    quant-decoded/<task_id>/<hotkey>/index.json          DecodedModelIndex, written last
    quant-decoded/<task_id>/<hotkey>/<shard>.safetensors every file listed in the index
"""

import hashlib
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from pydantic import TypeAdapter
from pydantic import ValidationError

import core.constants.docker as docker_cst
from validator.scoring.models import HubArtifactFile


DECODED_MODEL_ROOT = "quant-decoded"
DECODED_WEIGHT_SUFFIX = ".safetensors"
# Weights and weight indexes come from the decoded prefix only, never from the base snapshot.
BASE_WEIGHT_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf", "*.ckpt", "*.index.json", "*.msgpack", "*.h5"]
STALE_INDEX_PATTERN = "*.index.json"
MAX_INDEX_BYTES = 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60
HUB_TIMEOUT_SECONDS = 30.0

_DECODED_MODEL_URLS = TypeAdapter(dict[str, dict[str, str]])


class DecodedModelCorrupt(Exception):
    """The decoded model in object storage does not match the index the trainer wrote."""


class DecodedFile(BaseModel):
    name: str
    bytes: int
    sha256: str


SHA256_KIND_LFS = "sha256"
SHA256_KIND_GIT_BLOB = "git-blob-sha1"


class HubArtifactListing(BaseModel):
    """One hub listing of an artifact repo: the counted bytes, the per-file hashes and the revision they were read at."""

    bytes: int
    files: list[HubArtifactFile]
    revision: str | None = None


class DecodedModelIndex(BaseModel):
    files: list[DecodedFile]
    n_tensors: int
    decoder_sha256: str | None = None
    artifact_bytes: int | None = None


def decoded_model_prefix(task_id, hotkey: str) -> str:
    return f"{DECODED_MODEL_ROOT}/{task_id}/{hotkey}"


def _hub_model_info(repo: str, token: str | None, timeout: float):
    return HfApi(token=token).model_info(repo, files_metadata=True, timeout=timeout)


def _sized_siblings(repo: str, model_info) -> list:
    siblings = model_info.siblings or []
    if not siblings:
        raise ValueError(f"{repo} lists no files")
    missing = [sibling.rfilename for sibling in siblings if sibling.size is None]
    if missing:
        raise ValueError(f"{repo} has no size for {missing}")
    return siblings


def artifact_bytes_from_hub(repo: str, token: str | None = None, timeout: float = HUB_TIMEOUT_SECONDS) -> int:
    """Total size of every file in the repo; a miner can carry bytes in any file, so none is exempt.

    The default timeout bounds the scoring call site (_hub_artifact_bytes), which passes none.
    """
    return sum(sibling.size for sibling in _sized_siblings(repo, _hub_model_info(repo, token, timeout)))


def hub_file_from_sibling(repo: str, sibling) -> HubArtifactFile:
    lfs = getattr(sibling, "lfs", None)
    if lfs is not None and getattr(lfs, "sha256", None):
        sha256, kind = lfs.sha256, SHA256_KIND_LFS
    elif getattr(sibling, "blob_id", None):
        sha256, kind = sibling.blob_id, SHA256_KIND_GIT_BLOB
    else:
        raise ValueError(f"{repo} has no hash for {sibling.rfilename}")
    return HubArtifactFile(name=sibling.rfilename, bytes=sibling.size, sha256=sha256, sha256_kind=kind)


def artifact_listing_from_model_info(repo: str, model_info) -> HubArtifactListing:
    """Counted bytes (every sibling, .gitattributes included), file hashes and the commit sha."""
    siblings = _sized_siblings(repo, model_info)
    files = sorted((hub_file_from_sibling(repo, sibling) for sibling in siblings), key=lambda f: f.name)
    return HubArtifactListing(bytes=sum(f.bytes for f in files), files=files, revision=getattr(model_info, "sha", None))


def artifact_listing_from_hub(repo: str, token: str | None = None, timeout: float = HUB_TIMEOUT_SECONDS) -> HubArtifactListing:
    """One hub call giving artifact_bytes, the per-file hashes and the revision (freeform scoring)."""
    return artifact_listing_from_model_info(repo, _hub_model_info(repo, token, timeout))


def parse_decoded_model_urls(env_value: str | None) -> dict[str, dict[str, str]]:
    """repo -> {filename -> presigned GET url}; empty when the env var is unset (fixed track)."""
    if not env_value:
        return {}
    return _DECODED_MODEL_URLS.validate_json(env_value)


def _stream_to_file(url: str, dest: Path, max_bytes: int) -> str:
    """Download url to dest, refusing more than max_bytes; returns the sha256 hex digest."""
    digest = hashlib.sha256()
    written = 0
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, dest.open("wb") as out:
        while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
            written += len(chunk)
            if written > max_bytes:
                raise DecodedModelCorrupt(f"{dest.name} is larger than {max_bytes} bytes")
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def _load_index(urls: dict[str, str], dest: Path) -> DecodedModelIndex:
    index_url = urls.get(docker_cst.DECODED_MODEL_INDEX_NAME)
    if index_url is None:
        raise DecodedModelCorrupt(f"no {docker_cst.DECODED_MODEL_INDEX_NAME} among the decoded model urls")
    index_path = dest / docker_cst.DECODED_MODEL_INDEX_NAME
    _stream_to_file(index_url, index_path, MAX_INDEX_BYTES)
    try:
        return DecodedModelIndex.model_validate_json(index_path.read_bytes())
    except ValidationError as e:
        raise DecodedModelCorrupt(f"invalid {docker_cst.DECODED_MODEL_INDEX_NAME}: {e}") from e
    finally:
        index_path.unlink()


def _check_listed_name(name: str) -> None:
    if "/" in name or "\\" in name or not name.endswith(DECODED_WEIGHT_SUFFIX):
        raise DecodedModelCorrupt(f"index lists {name!r}; only flat *{DECODED_WEIGHT_SUFFIX} names are allowed")


def _download_listed_file(entry: DecodedFile, urls: dict[str, str], dest: Path) -> None:
    _check_listed_name(entry.name)
    if entry.name not in urls:
        raise DecodedModelCorrupt(f"index lists {entry.name} but no url was provided for it")
    path = dest / entry.name
    sha256 = _stream_to_file(urls[entry.name], path, entry.bytes)
    if path.stat().st_size != entry.bytes or sha256 != entry.sha256:
        raise DecodedModelCorrupt(f"{entry.name} does not match its index entry (bytes or sha256)")


def materialise_decoded_model(original_model: str, urls: dict[str, str], dest: Path, hf_token: str | None) -> Path:
    """Build a loadable model dir: base-model metadata from HF plus the sha256-verified decoded shards.

    Only files the index lists are downloaded; any other url is ignored.
    """
    snapshot_download(original_model, local_dir=dest, ignore_patterns=BASE_WEIGHT_PATTERNS, token=hf_token)
    index = _load_index(urls, dest)
    for entry in index.files:
        _download_listed_file(entry, urls, dest)
    for stale_index in dest.glob(STALE_INDEX_PATTERN):
        stale_index.unlink()
    return dest
