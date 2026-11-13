"""Trainer-side decode of a freeform quant submission, run after its HF upload while the job still holds its GPU slot.

Steps: fetch the decoder snapshot the validator wrote to MinIO, extract it, download the artifact from HF and reject
any executable in it, build and run the decoder in the sandbox, check the output against the base model, and upload
the checked bf16 shards plus index.json (last, as the completion marker) to MinIO.

A miner's fault ends in `quant_failure` (a scored fail); ours (image missing, MinIO/HF down, snapshot sha mismatch)
ends in `infra_error`, which is never scored. Neither ever fails the training job.
"""

import asyncio
import hashlib
import math
import os
import shutil
import tarfile
import time
import urllib.request
import uuid
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterator
from pathlib import Path

import docker
from huggingface_hub import snapshot_download
from pydantic import BaseModel

import trainer.constants as cst
from core.constants.quant import DECODE_TIMEOUT_SECONDS
from core.constants.quant import DECODER_DIR_NAME
from core.constants.quant import DECODER_ENTRYPOINT
from core.constants.quant import DECODER_MAX_EXTRACTED_BYTES
from core.constants.quant import DECODER_MAX_XZ_BYTES
from core.models.quant_artifact_models import QuantDecodedFile
from core.models.quant_artifact_models import QuantDecodedIndex
from core.models.quant_artifact_models import QuantDecodeRequest
from core.models.quant_artifact_models import QuantDecodeResult
from trainer.quant_decode.output_check import BF16_BYTES
from trainer.quant_decode.output_check import SAFETENSORS_INDEX_NAME
from trainer.quant_decode.output_check import SAFETENSORS_SUFFIX
from trainer.quant_decode.output_check import ExpectedTensors
from trainer.quant_decode.output_check import check_decoded_output
from trainer.quant_decode.sandbox import DecodeOutcome
from trainer.quant_decode.sandbox import build_phase_spec
from trainer.quant_decode.sandbox import build_phase_warning
from trainer.quant_decode.sandbox import run_phase
from trainer.quant_decode.sandbox import run_phase_spec
from trainer.quant_decode.sandbox import sanitize_failure
from trainer.telemetry import logger
from validator.infrastructure.minio_client import async_minio_client


DECODER_XZ_NAME = "decoder.tar.xz"
DECODED_INDEX_NAME = "index.json"
HF_LOCAL_CACHE_DIR = ".cache"
CONTAINER_ERROR_PREFIX = "container error:"
TIMEOUT_FAILURE = "timeout"
UNSAFE_MEMBERS_FAILURE = "decoder archive contains unsafe members"
DOWNLOAD_CHUNK_BYTES = 1 << 20
EXEC_BITS = 0o111
OWNER_RWX = 0o700
OWNER_RW = 0o600
BYTES_PER_MB = 1_000_000
# A work dir that survives removal is renamed to <name><STALE_WORK_DIR_MARKER><hex> so a re-run starts clean.
STALE_WORK_DIR_MARKER = ".stale-"
# Proves the work root was created (or adopted while empty) by the decode job, so prepare_work_root never chmods a
# shared directory it did not make.
WORK_ROOT_MARKER = ".god-quant-decode-root"


# Progress lines for the operator: the job log in task_history.json (the trainer logger may have no live sink).
ProgressReporter = Callable[[str], Awaitable[None]]


async def _no_report(message: str) -> None:
    return None


class _ScoredFail(Exception):
    """The miner's fault: becomes quant_failure."""


class _InfraFault(Exception):
    """Ours: becomes infra_error, which the orchestrator retries or alerts on and never scores."""


class _DecodeProgress(BaseModel):
    """Everything measured so far; carried into the result whatever the outcome."""

    artifact_bytes: int | None = None
    decoder_bytes: int | None = None
    decoder_sha256: str | None = None
    n_tensors: int = 0
    uploaded_files: list[str] = []
    build_seconds: float = 0.0
    run_seconds: float = 0.0
    index_object: str | None = None
    build_warning: str | None = None
    artifact_files: list[QuantDecodedFile] = []
    decoded_files: list[QuantDecodedFile] = []


class _WorkDirs(BaseModel):
    root: Path

    @property
    def decoder_xz(self) -> Path:
        return self.root / DECODER_XZ_NAME

    @property
    def decoder(self) -> Path:
        return self.root / "decoder"

    @property
    def build(self) -> Path:
        return self.root / "build"

    @property
    def artifact(self) -> Path:
        return self.root / "artifact"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def index(self) -> Path:
        return self.root / DECODED_INDEX_NAME

    def create(self) -> None:
        """Fresh, empty, owner-only work dirs, whatever an earlier attempt for the same task and hotkey left behind.

        The work root is private (0o700, off the cache volume that every training container mounts) and so is every
        dir below it, so the decoder snapshot and the artifact are unreadable to co-tenant containers.

        The sandbox runs as the trainer's own uid (sandbox.sandbox_user), so the trainer owns everything in here and
        `remove_work_tree` empties it. Anything that still survives (a leftover from a sandbox that ran as root) is
        renamed aside, which only needs write access to the parent, so a re-run never fails on it.
        """
        try:
            prepare_work_root(self.root.parent)
        except OSError as e:
            raise _InfraFault(f"decode work dir unusable: {e}") from None
        remove_work_tree(self.root)
        try:
            if self.root.exists() or self.root.is_symlink():
                stale = self.root.with_name(f"{self.root.name}{STALE_WORK_DIR_MARKER}{uuid.uuid4().hex[:8]}")
                self.root.rename(stale)
                logger.warning(f"Decode work dir {self.root} could not be removed; moved it aside to {stale}")
            self.root.mkdir()
            self.root.chmod(OWNER_RWX)
            for path in (self.decoder, self.build, self.artifact, self.output):
                path.mkdir()
                path.chmod(OWNER_RWX)  # explicit: independent of the umask
        except OSError as e:
            raise _InfraFault(f"decode work dir unusable: {e}") from None


def prepare_work_root(root: Path) -> None:
    """Make the work root a private (0o700) directory that is ours, and prove it is writable.

    Missing ancestors are created with default permissions: a 0o700 root alone blocks traversal, so they need nothing and
    a shared parent is never narrowed. A missing root is created and marked (WORK_ROOT_MARKER). An existing root is
    chmodded only when it carries the marker or is empty (adopted and marked); a non-empty directory we did not create,
    such as an existing /data, is refused untouched. Raises OSError when the root is unusable; the caller turns that
    into infra_error.
    """
    marker = root / WORK_ROOT_MARKER
    if root.exists() or root.is_symlink():
        if not marker.exists() and any(root.iterdir()):
            raise OSError(
                f"{root} exists, is not empty and was not created by the decode job (no {WORK_ROOT_MARKER}); "
                "point QUANT_DECODE_WORK_ROOT at a directory of its own"
            )
    else:
        root.mkdir(parents=True)
    root.chmod(OWNER_RWX)
    marker.touch(mode=OWNER_RW, exist_ok=True)
    probe = root / f".probe-{uuid.uuid4().hex[:8]}"
    probe.touch(mode=OWNER_RW, exist_ok=False)
    probe.unlink()


def _make_owner_removable(directory: Path) -> None:
    """Give every dir under `directory` that we own owner rwx, so rmtree can list and empty it. The decoder can leave a
    0o000 dir behind; it owns nothing the trainer does not, so this is enough. Symlinks are never followed."""
    try:
        directory.chmod(OWNER_RWX)
        children = list(os.scandir(directory))
    except OSError:
        return
    for child in children:
        if child.is_dir(follow_symlinks=False):
            _make_owner_removable(Path(child.path))


def remove_work_tree(root: Path) -> None:
    """Best-effort removal of a decode work dir; never raises."""
    if root.is_symlink():
        root.unlink(missing_ok=True)
        return
    if not root.is_dir():
        return
    _make_owner_removable(root)
    shutil.rmtree(root, ignore_errors=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_decoder(request: QuantDecodeRequest, dest: Path) -> int:
    """Stream the decoder snapshot to dest under a hard size cap; return its size. We wrote that object, so any
    failure here, including a sha mismatch, is ours or tampering, never the miner's."""
    cap = DECODER_MAX_XZ_BYTES + cst.QUANT_DECODER_FETCH_SLACK_BYTES
    digest = hashlib.sha256()
    written = 0
    try:
        with urllib.request.urlopen(request.decoder_snapshot_url, timeout=cst.QUANT_DECODER_FETCH_TIMEOUT_SECONDS) as response:
            dest.touch(mode=OWNER_RW)
            dest.chmod(OWNER_RW)
            with dest.open("wb") as f:
                while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                    written += len(chunk)
                    if written > cap:
                        raise _InfraFault(f"decoder snapshot exceeds {cap} bytes")
                    digest.update(chunk)
                    f.write(chunk)
    except (OSError, ValueError) as e:
        # The type only: the message can carry the presigned URL.
        raise _InfraFault(f"decoder snapshot download failed ({type(e).__name__})") from None
    if digest.hexdigest() != request.decoder_sha256:
        raise _InfraFault("decoder snapshot sha mismatch")
    return dest.stat().st_size


def _decoder_members(tar: tarfile.TarFile) -> Iterator[tarfile.TarInfo]:
    """Yield members with the git-archive `decoder/` prefix stripped. Lazy, so a zip bomb is stopped at the first
    header past the cap rather than after decompressing the whole stream."""
    prefix = f"{DECODER_DIR_NAME}/"
    extracted_bytes = 0
    for member in tar:
        if member.isdir() and member.name == DECODER_DIR_NAME:
            continue
        if member.issym() or member.islnk() or not member.name.startswith(prefix):
            raise _ScoredFail(UNSAFE_MEMBERS_FAILURE)
        extracted_bytes += member.size
        if extracted_bytes > DECODER_MAX_EXTRACTED_BYTES:
            raise _ScoredFail("decoder expands past cap")
        yield member.replace(name=member.name.removeprefix(prefix), deep=False)


def _extract_decoder(xz_path: Path, decoder_dir: Path) -> None:
    try:
        with tarfile.open(xz_path, mode="r:xz") as tar:
            tar.extractall(decoder_dir, members=_decoder_members(tar), filter="data")
    except tarfile.FilterError:
        raise _ScoredFail(UNSAFE_MEMBERS_FAILURE) from None
    except (tarfile.TarError, EOFError) as e:
        raise _InfraFault(f"decoder snapshot unreadable: {e}") from None
    if not (decoder_dir / DECODER_ENTRYPOINT).is_file():
        raise _ScoredFail(f"decoder has no {DECODER_ENTRYPOINT}")


async def _fetch_artifact(artifact_repo: str, artifact_dir: Path, hf_token: str | None) -> None:
    try:
        await asyncio.to_thread(snapshot_download, repo_id=artifact_repo, local_dir=artifact_dir, token=hf_token)
    except Exception as e:  # HF raises many unrelated types; every one of them is an outage on our side
        raise _InfraFault(f"artifact download failed: {e}") from None
    shutil.rmtree(artifact_dir / HF_LOCAL_CACHE_DIR, ignore_errors=True)


def _lock_down_artifact(artifact_dir: Path) -> None:
    """Owner-only perms on what we downloaded: dirs 0o700, files 0o600. Run only after check_artifact_files, which
    needs the exec bits (and symlinks) intact; the sandbox runs as this uid, so the ro mount still reads."""
    artifact_dir.chmod(OWNER_RWX)
    for path in artifact_dir.rglob("*"):
        if path.is_symlink():
            continue
        path.chmod(OWNER_RWX if path.is_dir() else OWNER_RW)


def check_artifact_files(artifact_dir: Path) -> str | None:
    """Reject any executable in the artifact: no symlinks, exec bits or code-like suffixes."""
    for path in sorted(artifact_dir.rglob("*")):
        name = path.relative_to(artifact_dir)
        if path.is_symlink():
            return f"artifact file {name} is a symlink"
        if path.is_dir():
            continue
        if not path.is_file():
            return f"artifact file {name} is not a regular file"
        if path.suffix.lower() in cst.QUANT_ARTIFACT_FORBIDDEN_SUFFIXES:
            return f"artifact file {name} has forbidden suffix {path.suffix}"
        if path.stat().st_mode & EXEC_BITS:
            return f"artifact file {name} is executable"
    return None


def artifact_size_bytes(artifact_dir: Path) -> int:
    """Every regular file counts, .gitattributes included: the same rule as the validator's hub count."""
    return sum(path.stat().st_size for path in artifact_dir.rglob("*") if path.is_file())


def artifact_file_records(artifact_dir: Path) -> list[QuantDecodedFile]:
    """name (relative posix path), bytes and sha256 of every regular file artifact_size_bytes counts, by name."""
    records = [
        QuantDecodedFile(name=path.relative_to(artifact_dir).as_posix(), bytes=path.stat().st_size, sha256=_sha256_file(path))
        for path in artifact_dir.rglob("*")
        if path.is_file()
    ]
    return sorted(records, key=lambda record: record.name)


def _expected_tensors(base_model_dir: Path) -> ExpectedTensors:
    try:
        expected = ExpectedTensors.from_safetensors_dir(base_model_dir)
    except (OSError, ValueError) as e:
        raise _InfraFault(f"base model safetensors unreadable: {e}") from None
    if not expected.tensors:
        raise _InfraFault("base model has no safetensors")
    return expected


def _bf16_bytes(expected: ExpectedTensors) -> int:
    return sum(math.prod(spec.shape) * BF16_BYTES for spec in expected.tensors.values())


def _raise_for_phase(outcome: DecodeOutcome, timeout_failure: str, failure_prefix: str) -> None:
    if outcome.ok:
        return
    if outcome.failure.startswith(CONTAINER_ERROR_PREFIX):  # image missing, daemon down: ours
        raise _InfraFault(outcome.failure)
    if outcome.failure == TIMEOUT_FAILURE:
        raise _ScoredFail(timeout_failure)
    raise _ScoredFail(f"{failure_prefix}: {outcome.failure}")


async def _build(docker_client: docker.DockerClient, work: _WorkDirs, progress: _DecodeProgress) -> None:
    outcome = await run_phase(docker_client, build_phase_spec(work.decoder, work.build))
    progress.build_seconds = outcome.wall_seconds
    _raise_for_phase(outcome, "decoder build timed out", "decoder build")
    ran_build_script = (work.decoder / cst.DECODE_BUILD_SCRIPT).exists()
    progress.build_warning = await asyncio.to_thread(build_phase_warning, outcome, work.build, ran_build_script)
    if progress.build_warning is not None:
        logger.warning(f"!!! QUANT DECODER BUILD WARNING (not a scored fail): {progress.build_warning} !!!")


async def _run(docker_client: docker.DockerClient, work: _WorkDirs, bf16_bytes: int, progress: _DecodeProgress) -> None:
    spec = run_phase_spec(work.artifact, work.decoder, work.output, bf16_bytes, build_dir=work.build)
    outcome = await run_phase(docker_client, spec)
    progress.run_seconds = outcome.wall_seconds
    _raise_for_phase(outcome, f"decoder timed out after {DECODE_TIMEOUT_SECONDS}s", "decoder")


def _check_output(output_dir: Path, expected: ExpectedTensors) -> int:
    """Check the decoded shards and drop any index: vLLM globs the shards, so only checked files reach it."""
    check = check_decoded_output(output_dir, expected)
    if not check.ok:
        raise _ScoredFail(f"decoded output: {check.failure}")
    (output_dir / SAFETENSORS_INDEX_NAME).unlink(missing_ok=True)
    return check.n_tensors


def upload_timeout_seconds(size_bytes: int) -> float:
    """Deadline for one decoded-file upload: a fixed floor plus the file at the slowest rate we accept."""
    return cst.QUANT_DECODE_UPLOAD_MIN_TIMEOUT_SECONDS + size_bytes / cst.QUANT_DECODE_UPLOAD_MIN_BYTES_PER_SECOND


async def _upload(bucket: str, object_name: str, path: Path, report: ProgressReporter = _no_report) -> None:
    """Multipart upload with large parts in parallel. A stalled upload is ours, so it ends as an infra_error.

    On timeout the executor thread is not killed (minio has no cancel); it finishes or fails on its own while the
    job moves on, and the index is never written, so the validator never sees a partial decode as complete.
    """
    size = path.stat().st_size
    timeout = upload_timeout_seconds(size)
    started = time.monotonic()
    try:
        uploaded = await asyncio.wait_for(
            async_minio_client.upload_file(
                bucket,
                object_name,
                str(path),
                part_size=cst.QUANT_DECODE_UPLOAD_PART_BYTES,
                num_parallel_uploads=cst.QUANT_DECODE_UPLOAD_PARALLEL,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise _InfraFault(f"upload of {object_name} timed out after {timeout:.0f}s") from None
    if not uploaded:
        raise _InfraFault(f"upload of {object_name} failed")
    elapsed = time.monotonic() - started
    rate = size / BYTES_PER_MB / max(elapsed, 1e-3)
    await report(f"decode: uploaded {path.name} ({size} bytes) in {elapsed:.0f}s ({rate:.1f} MB/s)")


async def _upload_shards(
    request: QuantDecodeRequest, output_dir: Path, progress: _DecodeProgress, report: ProgressReporter = _no_report
) -> list[QuantDecodedFile]:
    files: list[QuantDecodedFile] = []
    shards = sorted(output_dir.glob(f"*{SAFETENSORS_SUFFIX}"))
    total = sum(shard.stat().st_size for shard in shards)
    await report(f"decode: uploading {len(shards)} decoded shards ({total} bytes)")
    for shard in shards:
        await _upload(request.bucket, f"{request.object_prefix}/{shard.name}", shard, report)
        sha256 = await asyncio.to_thread(_sha256_file, shard)
        record = QuantDecodedFile(name=shard.name, bytes=shard.stat().st_size, sha256=sha256)
        files.append(record)
        progress.decoded_files.append(record)
        progress.uploaded_files.append(shard.name)
    return files


async def _upload_index(
    request: QuantDecodeRequest,
    index: QuantDecodedIndex,
    index_path: Path,
    progress: _DecodeProgress,
    report: ProgressReporter = _no_report,
) -> None:
    index_path.touch(mode=OWNER_RW)
    index_path.chmod(OWNER_RW)
    index_path.write_text(index.model_dump_json())
    object_name = f"{request.object_prefix}/{DECODED_INDEX_NAME}"
    await _upload(request.bucket, object_name, index_path, report)
    progress.uploaded_files.append(DECODED_INDEX_NAME)
    progress.index_object = object_name


async def _decode(
    request: QuantDecodeRequest,
    *,
    task_id: str,
    hotkey: str,
    artifact_repo: str,
    base_model_dir: Path,
    hf_token: str | None,
    work: _WorkDirs,
    docker_client: docker.DockerClient,
    progress: _DecodeProgress,
    report: ProgressReporter,
) -> None:
    work.create()
    progress.decoder_bytes = await asyncio.to_thread(_fetch_decoder, request, work.decoder_xz)
    progress.decoder_sha256 = request.decoder_sha256
    await asyncio.to_thread(_extract_decoder, work.decoder_xz, work.decoder)

    await _fetch_artifact(artifact_repo, work.artifact, hf_token)
    artifact_failure = check_artifact_files(work.artifact)
    if artifact_failure is not None:
        raise _ScoredFail(artifact_failure)
    try:
        await asyncio.to_thread(_lock_down_artifact, work.artifact)
        progress.artifact_bytes = artifact_size_bytes(work.artifact)
        progress.artifact_files = await asyncio.to_thread(artifact_file_records, work.artifact)
    except OSError as e:
        raise _InfraFault(f"artifact files unreadable: {e}") from None
    await report(f"decode: artifact fetched ({progress.artifact_bytes} bytes)")

    expected = await asyncio.to_thread(_expected_tensors, base_model_dir)
    await _build(docker_client, work, progress)
    await report(f"decode: decoder built in {progress.build_seconds:.0f}s")
    await _run(docker_client, work, _bf16_bytes(expected), progress)
    await report(f"decode: decoder ran in {progress.run_seconds:.0f}s")
    progress.n_tensors = await asyncio.to_thread(_check_output, work.output, expected)
    await report(f"decode: output checked ({progress.n_tensors} tensors)")

    files = await _upload_shards(request, work.output, progress, report)
    index = QuantDecodedIndex(
        files=files,
        n_tensors=progress.n_tensors,
        decoder_sha256=request.decoder_sha256,
        artifact_bytes=progress.artifact_bytes,
        task_id=task_id,
        hotkey=hotkey,
        artifact_files=progress.artifact_files,
    )
    await _upload_index(request, index, work.index, progress, report)


async def run_freeform_decode(
    request: QuantDecodeRequest,
    *,
    task_id: str,
    hotkey: str,
    artifact_repo: str,
    base_model_dir: Path,
    hf_token: str | None,
    work_root: Path,
    docker_client: docker.DockerClient,
    report: ProgressReporter = _no_report,
) -> QuantDecodeResult:
    """Decode one freeform submission end to end. Never raises for a miner's fault; the work dir is always removed.

    `report` gets one line per stage (and per uploaded shard) so a long decode is visible on the job log.
    """
    work = _WorkDirs(root=work_root / f"{task_id}_{hotkey[:8]}")
    progress = _DecodeProgress()
    try:
        await _decode(
            request,
            task_id=task_id,
            hotkey=hotkey,
            artifact_repo=artifact_repo,
            base_model_dir=base_model_dir,
            hf_token=hf_token,
            work=work,
            docker_client=docker_client,
            progress=progress,
            report=report,
        )
        return QuantDecodeResult(ok=True, **progress.model_dump())
    except _ScoredFail as e:
        return QuantDecodeResult(quant_failure=sanitize_failure(str(e)), **progress.model_dump())
    except _InfraFault as e:
        return QuantDecodeResult(infra_error=sanitize_failure(str(e)), **progress.model_dump())
    finally:
        await asyncio.to_thread(remove_work_tree, work.root)
