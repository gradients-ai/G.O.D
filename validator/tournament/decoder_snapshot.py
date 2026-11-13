"""
Canonical snapshot of a quant artifact repo's decoder/ directory.

The decoder counts toward a freeform submission's size, so its byte count must be identical on every
machine for the same commit. `git archive <commit> decoder/` gives a canonical tar for the tree: entries
in tree order, owner/group root, every mtime set to the commit time, and nothing from the clone path.
Git runs with global/system config disabled so local settings (tar.umask, filters) cannot change the
bytes. Submodules under decoder/ archive as empty directories and are counted as-is. A symlink (or
hardlink) in the archive raises DecoderUnsafeMembers: the trainer refuses to extract one
(`_decoder_members` in trainer/quant_decode/job.py), which would be a scored fail on every freeform
task, so registration rejects it up front. The tar is compressed with single-threaded xz (multi-threaded xz changes the block
layout and therefore the bytes) and the counted .tar.xz is the exact file uploaded to MinIO.

git archive honours in-tree .gitattributes (export-ignore, export-subst), so the uploaded tar is the
only faithful copy of what was counted: anything that runs the decoder must build it from the MinIO
tar, never from a fresh clone. The tar embeds the commit id and commit time, so the snapshot is
canonical per commit rather than per tree; key any dedup or cache on tar_sha256.

The tree-ish form `git archive --prefix=decoder/ <commit>:decoder` is not used: git stamps mtime=now for a
tree (as opposed to a commit) archive, so only `<commit> decoder/` is byte-stable.
"""

import asyncio
import hashlib
import os
import subprocess
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from core.constants.quant import DECODER_DIR_NAME
from core.constants.quant import DECODER_ENTRYPOINT
from core.constants.quant import DECODER_MAX_EXTRACTED_BYTES
from core.constants.quant import DECODER_MAX_XZ_BYTES
from core.git import build_authenticated_git_url
from core.git import sanitize_git_text
from core.logging import get_logger
from validator.infrastructure.minio_client import async_minio_client


logger = get_logger(__name__)

XZ_ARGS = ("xz", "-9", "-T1", "--format=xz", "--check=crc64", "-c")
GIT_ARCHIVE_FORMAT = "tar"
# Bounds git and xz. Single-threaded xz -9 runs at a few MB/s, so a decoder/ of a few hundred MB times
# out (RuntimeError) instead of raising DecoderTooLarge; callers should reject registration on both.
SUBPROCESS_TIMEOUT_SECONDS = 120
# A clone fetches the whole repo history (miner repos can carry large training assets), so it gets its own bound.
CLONE_TIMEOUT_SECONDS = 600
# Registration snapshots run in parallel, each bounded by the clone timeout; this caps concurrent clones.
DECODER_SNAPSHOT_CONCURRENCY = 4
DECODER_OBJECT_PREFIX = "quant-decoders"
_ISOLATED_GIT_ENV = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
_MISSING_PATH_MARKER = "did not match"
UNSAFE_MEMBERS_SHOWN = 10


class DecoderMissing(ValueError):
    """The pinned commit has no decoder/ directory."""


class DecoderNotRunnable(ValueError):
    """decoder/ exists but the trainer's freeform decode would reject it on every task: no decode.py, or it extracts
    past DECODER_MAX_EXTRACTED_BYTES."""


class DecoderTooLarge(ValueError):
    """The decoder's xz-compressed archive exceeds the size cap."""

    def __init__(self, xz_bytes: int, cap: int):
        super().__init__(f"Decoder is {xz_bytes} bytes after xz, cap is {cap}")
        self.xz_bytes = xz_bytes
        self.cap = cap


class DecoderUnsafeMembers(ValueError):
    """decoder/ holds symlinks or hardlinks, which the trainer's decode rejects on every freeform task."""

    def __init__(self, members: list[str]):
        shown = ", ".join(members[:UNSAFE_MEMBERS_SHOWN])
        more = len(members) - UNSAFE_MEMBERS_SHOWN
        super().__init__(
            f"decoder/ contains symlinks or hardlinks, which the trainer rejects on every freeform task: {shown}"
            + (f" and {more} more" if more > 0 else "")
        )
        self.members = members


class DecoderSnapshot(BaseModel):
    repo_url: str
    commit_hash: str
    tar_sha256: str
    xz_bytes: int
    xz_sha256: str
    xz_version: str
    tree_sha: str
    object_name: str | None = None


def _run(args: list[str], *, timeout: float = SUBPROCESS_TIMEOUT_SECONDS, **kwargs) -> subprocess.CompletedProcess:
    # Never pass check= with a token-bearing argv: CalledProcessError carries argv. xz_version is the only check=True caller.
    try:
        return subprocess.run(args, timeout=timeout, **kwargs)
    except FileNotFoundError as e:
        raise RuntimeError(f"Required binary '{args[0]}' not found on PATH") from e
    except subprocess.TimeoutExpired:
        pass
    # Raised outside the except block so no chained TimeoutExpired carries argv (and any token) along.
    raise RuntimeError(f"'{args[0]}' timed out after {timeout}s")


def _git_env() -> dict[str, str]:
    return {**os.environ, **_ISOLATED_GIT_ENV}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_at_commit(repo_url: str, commit_hash: str, github_token: str | None, dest: Path) -> None:
    clone_url = build_authenticated_git_url(repo_url, github_token)
    clone_proc = _run(
        ["git", "clone", "--quiet", clone_url, str(dest)],
        timeout=CLONE_TIMEOUT_SECONDS,
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    if clone_proc.returncode != 0:
        raise RuntimeError(f"Failed to clone {repo_url}: {sanitize_git_text(clone_proc.stderr, github_token)}")

    checkout_proc = _run(
        ["git", "-C", str(dest), "checkout", "--quiet", commit_hash], capture_output=True, text=True, env=_git_env()
    )
    if checkout_proc.returncode != 0:
        raise RuntimeError(
            f"Commit {commit_hash} not found in {repo_url}: {sanitize_git_text(checkout_proc.stderr, github_token)}"
        )


def archive_decoder_dir(repo_dir: Path, commit_hash: str, out_tar: Path) -> None:
    """Write the canonical tar of decoder/ at commit_hash to out_tar."""
    args = ["git", "-C", str(repo_dir), "archive", f"--format={GIT_ARCHIVE_FORMAT}", commit_hash, f"{DECODER_DIR_NAME}/"]
    with open(out_tar, "wb") as f:
        proc = _run(args, stdout=f, stderr=subprocess.PIPE, env=_git_env())
    if proc.returncode == 0:
        return

    stderr = sanitize_git_text(proc.stderr.decode(errors="replace"))
    if _MISSING_PATH_MARKER in stderr:
        raise DecoderMissing(f"No {DECODER_DIR_NAME}/ directory at commit {commit_hash}")
    raise RuntimeError(f"git archive failed for commit {commit_hash}: {stderr}")


def decoder_tree_sha(repo_dir: Path, commit_hash: str) -> str:
    """Git tree id of decoder/ at commit_hash: content only, so it survives a squash into an orphan commit."""
    proc = _run(
        ["git", "-C", str(repo_dir), "rev-parse", f"{commit_hash}:{DECODER_DIR_NAME}"],
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git rev-parse failed for {DECODER_DIR_NAME}/ at commit {commit_hash}: {sanitize_git_text(proc.stderr)}"
        )
    return proc.stdout.strip()


def unsafe_decoder_members(tar_path: Path) -> list[str]:
    """Symlink and hardlink entries in a decoder tar, the members the trainer's extraction refuses."""
    with tarfile.open(tar_path, mode="r:") as tar:
        return [member.name for member in tar if member.issym() or member.islnk()]


def xz_bytes_of(tar_path: Path, out_xz: Path) -> tuple[int, str]:
    """Compress tar_path into out_xz with the pinned xz settings; return (byte count, sha256)."""
    with open(tar_path, "rb") as src, open(out_xz, "wb") as dst:
        proc = _run(list(XZ_ARGS), stdin=src, stdout=dst, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"xz failed: {proc.stderr.decode(errors='replace')}")
    return out_xz.stat().st_size, sha256_file(out_xz)


def xz_version() -> str:
    proc = _run([XZ_ARGS[0], "--version"], capture_output=True, text=True, check=True)
    return proc.stdout.splitlines()[0]


def check_decoder_runnable(tar_path: Path) -> None:
    """Raise DecoderNotRunnable where the trainer's extraction (`_extract_decoder` in trainer/quant_decode/job.py) would
    scored-fail this tar: members totalling more than DECODER_MAX_EXTRACTED_BYTES, or no regular decoder/decode.py."""
    entrypoint = f"{DECODER_DIR_NAME}/{DECODER_ENTRYPOINT}"
    extracted_bytes = 0
    has_entrypoint = False
    with tarfile.open(tar_path, mode="r:") as tar:
        for member in tar:
            extracted_bytes += member.size
            has_entrypoint = has_entrypoint or (member.isfile() and member.name == entrypoint)
    if extracted_bytes > DECODER_MAX_EXTRACTED_BYTES:
        raise DecoderNotRunnable(
            f"{DECODER_DIR_NAME}/ extracts to {extracted_bytes} bytes, over the trainer's cap of {DECODER_MAX_EXTRACTED_BYTES}"
        )
    if not has_entrypoint:
        raise DecoderNotRunnable(f"{DECODER_DIR_NAME}/ has no {DECODER_ENTRYPOINT}, the entrypoint the trainer runs")


def decoder_object_name(commit_hash: str, xz_sha256: str) -> str:
    return f"{DECODER_OBJECT_PREFIX}/{commit_hash}/{xz_sha256}.tar.xz"


RepoCheck = Callable[[Path], object]


def build_decoder_snapshot(
    repo_url: str,
    commit_hash: str,
    github_token: str | None,
    work_dir: Path,
    repo_check: RepoCheck | None = None,
) -> tuple[DecoderSnapshot, Path]:
    """Clone, archive and compress decoder/ inside work_dir; return the snapshot and the .tar.xz path.

    repo_check runs on the checked-out clone before anything is archived and rejects by raising; registration
    passes its build lint here so the repo is cloned once for both.
    """
    repo_dir = work_dir / "repo"
    tar_path = work_dir / "decoder.tar"
    xz_path = work_dir / "decoder.tar.xz"

    clone_at_commit(repo_url, commit_hash, github_token, repo_dir)
    if repo_check is not None:
        repo_check(repo_dir)
    archive_decoder_dir(repo_dir, commit_hash, tar_path)
    unsafe = unsafe_decoder_members(tar_path)
    if unsafe:
        raise DecoderUnsafeMembers(unsafe)
    xz_bytes, xz_sha256 = xz_bytes_of(tar_path, xz_path)
    if xz_bytes > DECODER_MAX_XZ_BYTES:
        raise DecoderTooLarge(xz_bytes, DECODER_MAX_XZ_BYTES)
    check_decoder_runnable(tar_path)

    snapshot = DecoderSnapshot(
        repo_url=repo_url,
        commit_hash=commit_hash,
        tar_sha256=sha256_file(tar_path),
        xz_bytes=xz_bytes,
        xz_sha256=xz_sha256,
        xz_version=xz_version(),
        tree_sha=decoder_tree_sha(repo_dir, commit_hash),
    )
    return snapshot, xz_path


async def snapshot_decoder(
    repo_url: str,
    commit_hash: str,
    github_token: str | None,
    bucket: str,
    upload: bool = True,
    repo_check: RepoCheck | None = None,
) -> DecoderSnapshot:
    with tempfile.TemporaryDirectory() as temp_dir:
        snapshot, xz_path = await asyncio.to_thread(
            build_decoder_snapshot, repo_url, commit_hash, github_token, Path(temp_dir), repo_check
        )
        logger.info(f"Decoder snapshot for {sanitize_git_text(repo_url, github_token)}@{commit_hash}: {snapshot.xz_bytes} bytes")
        if not upload:
            return snapshot

        object_name = decoder_object_name(commit_hash, snapshot.xz_sha256)
        if not await async_minio_client.upload_file(bucket, object_name, str(xz_path)):
            raise RuntimeError(f"Failed to upload decoder snapshot to {bucket}/{object_name}")
        return snapshot.model_copy(update={"object_name": object_name})
