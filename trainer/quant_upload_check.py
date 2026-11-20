"""Pre-upload check of a quant job's output dir: upload exactly what the miner's quant_manifest.json declares complete.

For quant tasks the uploaded bytes are the score, so the trainer refuses an output without a valid manifest (the
pipeline did not finish, e.g. a timed-out partial checkpoint), checks every listed file against its declared size,
caps the total, deletes everything unlisted, and rejects symlinks, exec bits and code-like suffixes in what remains.
Every rejection is a training failure.

The training container writes the output dir as root, so a non-root trainer must first own it to prune it: runtime.py
runs `output_reclaim.reclaim_output_dir` before this check. Should a file still be out of the trainer's reach, the
OSError becomes a rejection, never an unchecked upload or a crash.
"""

import math
import shutil
from pathlib import Path
from pathlib import PurePosixPath

from pydantic import BaseModel
from pydantic import ValidationError

import trainer.constants as cst
from core.constants.quant import QUANT_ARTIFACT_MAX_BYTES_MULTIPLIER
from core.constants.quant import QUANT_MANIFEST_NAME
from core.logging import get_logger
from core.models.quant_artifact_models import QuantManifest
from core.models.quant_artifact_models import track_for
from core.models.task_models import TaskType
from trainer.quant_decode.job import check_artifact_files
from trainer.quant_decode.output_check import BF16_BYTES
from trainer.quant_decode.output_check import ExpectedTensors


logger = get_logger(__name__)


class QuantUploadVerdict(BaseModel):
    ok: bool
    failure: str | None = None
    total_bytes: int = 0
    n_files: int = 0


class _Rejected(Exception):
    """The output is not uploadable; the message is the reason without the prefix."""


def _read_manifest(output_dir: Path) -> tuple[QuantManifest, int]:
    """Return the parsed manifest and its own size in bytes (it is uploaded, so it counts)."""
    path = output_dir / QUANT_MANIFEST_NAME
    if path.is_symlink() or not path.exists():
        raise _Rejected(f"{QUANT_MANIFEST_NAME} missing: the pipeline did not finish")
    if not path.is_file():
        raise _Rejected(f"{QUANT_MANIFEST_NAME} is not a regular file")
    try:
        manifest = QuantManifest.model_validate_json(path.read_bytes())
    except ValidationError as e:
        raise _Rejected(f"{QUANT_MANIFEST_NAME} invalid: {str(e)[: cst.QUANT_MANIFEST_ERROR_MAX_CHARS]}") from None
    return manifest, path.stat().st_size


def _check_identity(manifest: QuantManifest, task_id: str, task_type: TaskType) -> None:
    if manifest.task_id != task_id:
        raise _Rejected(f"manifest task_id {manifest.task_id!r} is not this task")
    if manifest.track != track_for(task_type):
        raise _Rejected(f"manifest track {manifest.track!r} does not match {task_type}")


def _check_listed_files(output_dir: Path, manifest: QuantManifest) -> None:
    root = output_dir.resolve()
    for entry in manifest.files:
        path = output_dir / entry.name
        if not path.exists() and not path.is_symlink():
            raise _Rejected(f"listed file {entry.name} is missing")
        if path.is_symlink() or path.resolve() != root / PurePosixPath(entry.name):
            raise _Rejected(f"listed file {entry.name} is a symlink or under one")
        if not path.is_file():
            raise _Rejected(f"listed file {entry.name} is not a regular file")
        size = path.stat().st_size
        if size != entry.bytes:
            raise _Rejected(f"listed file {entry.name} is {size} bytes, manifest says {entry.bytes}")


def _check_caps(manifest: QuantManifest, manifest_bytes: int, max_bytes: int | None) -> int:
    """Return the uploaded total: every listed file plus the manifest."""
    total = sum(entry.bytes for entry in manifest.files) + manifest_bytes
    if max_bytes is not None and total > max_bytes:
        raise _Rejected(f"output is {total} bytes, cap is {max_bytes}")
    return total


def _prune_dir(root: Path, directory: Path, keep: set[PurePosixPath], keep_dirs: set[PurePosixPath]) -> None:
    for entry in directory.iterdir():
        rel = PurePosixPath(entry.relative_to(root).as_posix())
        if rel in keep:
            continue
        is_real_dir = entry.is_dir() and not entry.is_symlink()
        if is_real_dir and rel in keep_dirs:
            _prune_dir(root, entry, keep, keep_dirs)
        elif is_real_dir:
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _prune_unlisted(output_dir: Path, manifest: QuantManifest) -> None:
    """Delete every file and directory that is neither listed nor the manifest, so nothing uncounted rides along."""
    keep = {PurePosixPath(entry.name) for entry in manifest.files} | {PurePosixPath(QUANT_MANIFEST_NAME)}
    keep_dirs = {parent for name in keep for parent in name.parents}
    _prune_dir(output_dir, output_dir, keep, keep_dirs)


def _check_quant_output(output_dir: Path, task_id: str, task_type: TaskType, max_bytes: int | None) -> QuantUploadVerdict:
    if output_dir.is_symlink():
        raise _Rejected("output dir is a symlink")
    manifest, manifest_bytes = _read_manifest(output_dir)
    _check_identity(manifest, task_id, task_type)
    _check_listed_files(output_dir, manifest)
    total_bytes = _check_caps(manifest, manifest_bytes, max_bytes)
    _prune_unlisted(output_dir, manifest)
    artifact_failure = check_artifact_files(output_dir)
    if artifact_failure is not None:
        raise _Rejected(artifact_failure)
    return QuantUploadVerdict(ok=True, total_bytes=total_bytes, n_files=len(manifest.files))


def check_quant_output(output_dir: Path, *, task_id: str, task_type: TaskType, max_bytes: int | None) -> QuantUploadVerdict:
    """Check and prune a quant output dir in place. max_bytes None skips the size cap (base model unreadable)."""
    try:
        return _check_quant_output(output_dir, task_id, task_type, max_bytes)
    except _Rejected as e:
        return QuantUploadVerdict(ok=False, failure=f"{cst.QUANT_OUTPUT_REJECTED_PREFIX}{e}")
    except OSError as e:  # e.g. a root-owned file the reclaim did not reach; never upload what was not checked
        failure = f"{cst.QUANT_OUTPUT_REJECTED_PREFIX}output not accessible to the trainer: {e}"
        return QuantUploadVerdict(ok=False, failure=failure)


def _bf16_bytes_of(base_model_dir: Path) -> int:
    expected = ExpectedTensors.from_safetensors_dir(base_model_dir)
    return sum(math.prod(spec.shape) * BF16_BYTES for spec in expected.tensors.values())


def quant_output_max_bytes(base_model_dir: Path) -> int | None:
    """QUANT_ARTIFACT_MAX_BYTES_MULTIPLIER x the base model's bf16 bytes; None when the base model is unreadable,
    which is ours (infra), so only the cap is skipped."""
    try:
        bf16_bytes = _bf16_bytes_of(base_model_dir)
    except (OSError, ValueError) as e:
        logger.warning(f"Quant upload cap skipped: base model {base_model_dir} unreadable: {e}")
        return None
    if bf16_bytes == 0:
        logger.warning(f"Quant upload cap skipped: no safetensors in base model {base_model_dir}")
        return None
    return int(bf16_bytes * QUANT_ARTIFACT_MAX_BYTES_MULTIPLIER)
