import os
import shutil
import subprocess
from pathlib import Path

import pytest

from validator.tournament import decoder_snapshot
from validator.tournament.decoder_snapshot import DecoderMissing
from validator.tournament.decoder_snapshot import DecoderSnapshot
from validator.tournament.decoder_snapshot import DecoderTooLarge
from validator.tournament.decoder_snapshot import DecoderUnsafeMembers
from validator.tournament.decoder_snapshot import archive_decoder_dir
from validator.tournament.decoder_snapshot import clone_at_commit
from validator.tournament.decoder_snapshot import sha256_file
from validator.tournament.decoder_snapshot import snapshot_decoder
from validator.tournament.registration_lint import RegistrationLintFailed
from validator.tournament.registration_lint import check_registration_lint


_MISSING = [binary for binary in ("git", "tar", "xz") if shutil.which(binary) is None]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=f"decoder snapshot tests need {', '.join(_MISSING)} on PATH")

_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], env=_GIT_ENV, check=True, capture_output=True, text=True
    ).stdout.strip()


def _make_repo(path: Path, files: dict[str, bytes]) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "--quiet", "--initial-branch=main")
    for name, content in files.items():
        _write_file(path, name, content)
    _git(path, "add", "--all")
    _git(path, "commit", "--quiet", "-m", "init")
    return _git(path, "rev-parse", "HEAD")


def _write_file(repo: Path, name: str, content: bytes) -> None:
    file_path = repo / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(content)


def _commit_file(repo: Path, name: str, content: bytes) -> str:
    _write_file(repo, name, content)
    _git(repo, "add", "--all")
    _git(repo, "commit", "--quiet", "-m", f"update {name}")
    return _git(repo, "rev-parse", "HEAD")


_DECODER_FILES = {
    "README.md": b"artifact\n",
    "decoder/decode.py": b"from x import decode\n",
    "decoder/x.py": b"def decode():\n    return 1\n",
    "decoder/lib/util.py": b"VALUE = 2\n",
}


@pytest.fixture
def uploads(monkeypatch) -> list[tuple[str, str, int]]:
    calls: list[tuple[str, str, int]] = []

    async def fake_upload(bucket, object_name, file_path):
        calls.append((bucket, object_name, Path(file_path).stat().st_size))
        return True

    monkeypatch.setattr(decoder_snapshot.async_minio_client, "upload_file", fake_upload)
    return calls


async def test_independent_clones_give_identical_snapshots(tmp_path):
    origin = tmp_path / "origin"
    commit = _make_repo(origin, _DECODER_FILES)

    first = await snapshot_decoder(str(origin), commit, None, "bucket", upload=False)
    second = await snapshot_decoder(str(origin), commit, None, "bucket", upload=False)

    assert first == second
    assert first.object_name is None
    assert first.xz_version.startswith("xz")


def test_clone_path_and_mtimes_do_not_leak_into_tar(tmp_path):
    origin = tmp_path / "origin"
    commit = _make_repo(origin, _DECODER_FILES)
    clone_a = tmp_path / "a"
    clone_b = tmp_path / "deeper" / "path" / "b"
    clone_at_commit(str(origin), commit, None, clone_a)
    clone_at_commit(str(origin), commit, None, clone_b)
    for file_path in (clone_b / "decoder").rglob("*"):
        os.utime(file_path, (0, 0))

    archive_decoder_dir(clone_a, commit, tmp_path / "a.tar")
    archive_decoder_dir(clone_b, commit, tmp_path / "b.tar")

    assert (tmp_path / "a.tar").read_bytes() == (tmp_path / "b.tar").read_bytes()


async def test_file_creation_order_does_not_change_snapshot(tmp_path):
    forward = _make_repo(tmp_path / "forward", _DECODER_FILES)
    reverse = _make_repo(tmp_path / "reverse", dict(reversed(list(_DECODER_FILES.items()))))

    first = await snapshot_decoder(str(tmp_path / "forward"), forward, None, "bucket", upload=False)
    second = await snapshot_decoder(str(tmp_path / "reverse"), reverse, None, "bucket", upload=False)

    assert (first.tar_sha256, first.xz_bytes, first.xz_sha256) == (second.tar_sha256, second.xz_bytes, second.xz_sha256)


async def test_one_byte_change_changes_hashes(tmp_path):
    origin = tmp_path / "origin"
    before_commit = _make_repo(origin, _DECODER_FILES)
    after_commit = _commit_file(origin, "decoder/x.py", b"def decode():\n    return 2\n")

    before = await snapshot_decoder(str(origin), before_commit, None, "bucket", upload=False)
    after = await snapshot_decoder(str(origin), after_commit, None, "bucket", upload=False)

    assert before.tar_sha256 != after.tar_sha256
    assert before.xz_sha256 != after.xz_sha256


async def test_empty_decoder_file_is_a_valid_snapshot(tmp_path):
    origin = tmp_path / "origin"
    commit = _make_repo(origin, {"decoder/decode.py": b""})

    snapshot = await snapshot_decoder(str(origin), commit, None, "bucket", upload=False)

    assert 0 < snapshot.xz_bytes < 1024


@pytest.mark.parametrize("files", [{"README.md": b"no decoder\n"}, {"decoder": b"a file, not a directory\n"}])
async def test_missing_decoder_dir_raises(tmp_path, uploads, files):
    origin = tmp_path / "origin"
    commit = _make_repo(origin, files)

    with pytest.raises(DecoderMissing):
        await snapshot_decoder(str(origin), commit, None, "bucket")
    assert uploads == []


async def test_oversized_decoder_raises_before_upload(tmp_path, uploads):
    origin = tmp_path / "origin"
    commit = _make_repo(origin, {"decoder/blob.bin": os.urandom(2 * 1024 * 1024)})

    with pytest.raises(DecoderTooLarge) as exc_info:
        await snapshot_decoder(str(origin), commit, None, "bucket")

    assert exc_info.value.xz_bytes > exc_info.value.cap
    assert uploads == []


def _commit_symlink(repo: Path, name: str, target: str) -> str:
    link = repo / name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    _git(repo, "add", "--all")
    _git(repo, "commit", "--quiet", "-m", f"link {name}")
    return _git(repo, "rev-parse", "HEAD")


async def test_symlink_in_decoder_raises_before_upload(tmp_path, uploads):
    """The trainer refuses symlinks in the decoder tar (a scored fail on every freeform task), so registration does."""
    origin = tmp_path / "origin"
    _make_repo(origin, _DECODER_FILES)
    commit = _commit_symlink(origin, "decoder/alias.py", "x.py")

    with pytest.raises(DecoderUnsafeMembers) as exc_info:
        await snapshot_decoder(str(origin), commit, None, "bucket")

    assert exc_info.value.members == ["decoder/alias.py"]
    assert "every freeform task" in str(exc_info.value)
    assert uploads == []


async def test_symlink_outside_decoder_is_not_archived_and_passes(tmp_path):
    origin = tmp_path / "origin"
    _make_repo(origin, _DECODER_FILES)
    commit = _commit_symlink(origin, "docs/readme-link", "../README.md")

    snapshot = await snapshot_decoder(str(origin), commit, None, "bucket", upload=False)

    assert snapshot.commit_hash == commit


async def test_repo_check_runs_on_the_clone_and_its_rejection_stops_the_snapshot(tmp_path, uploads):
    origin = tmp_path / "origin"
    commit = _make_repo(origin, _DECODER_FILES)
    seen: list[Path] = []

    def reject(repo_dir: Path) -> None:
        seen.append(repo_dir)
        assert (repo_dir / "decoder" / "x.py").is_file()
        raise ValueError("lint says no")

    with pytest.raises(ValueError, match="lint says no"):
        await snapshot_decoder(str(origin), commit, None, "bucket", repo_check=reject)

    assert len(seen) == 1 and seen[0].name == "repo"
    assert uploads == []


@pytest.mark.parametrize("enforce", [True, False])
async def test_registration_lint_on_a_real_clone(tmp_path, uploads, monkeypatch, enforce):
    """A miner-owned base image fails the build lint end to end; warn-only snapshots and uploads anyway."""
    monkeypatch.setattr("validator.tournament.constants.QUANT_REGISTRATION_LINT_ENFORCE", enforce)
    origin = tmp_path / "origin"
    files = {**_DECODER_FILES, "dockerfiles/standalone-text-trainer.dockerfile": b"FROM miner/base:latest\n"}
    commit = _make_repo(origin, files)

    if enforce:
        with pytest.raises(RegistrationLintFailed, match="unpinned_base_image"):
            await snapshot_decoder(str(origin), commit, None, "bucket", repo_check=check_registration_lint)
        assert uploads == []
    else:
        snapshot = await snapshot_decoder(str(origin), commit, None, "bucket", repo_check=check_registration_lint)
        assert snapshot.object_name is not None
        assert len(uploads) == 1


async def test_unknown_commit_raises_runtime_error(tmp_path):
    origin = tmp_path / "origin"
    _make_repo(origin, _DECODER_FILES)

    with pytest.raises(RuntimeError, match="not found"):
        await snapshot_decoder(str(origin), "0" * 40, None, "bucket", upload=False)


def test_missing_xz_binary_raises_runtime_error(tmp_path, monkeypatch):
    tar_path = tmp_path / "decoder.tar"
    tar_path.write_bytes(b"")
    monkeypatch.setattr(decoder_snapshot, "XZ_ARGS", ("xz-not-installed", "-c"))

    with pytest.raises(RuntimeError, match="xz-not-installed"):
        decoder_snapshot.xz_bytes_of(tar_path, tmp_path / "decoder.tar.xz")


def test_missing_git_binary_raises_runtime_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(RuntimeError, match="'git' not found"):
        clone_at_commit(str(tmp_path / "origin"), "0" * 40, None, tmp_path / "clone")


def test_timeout_message_does_not_leak_token(tmp_path, monkeypatch):
    fake_git = tmp_path / "bin" / "git"
    fake_git.parent.mkdir()
    fake_git.write_text("#!/bin/sh\nexec sleep 5\n")
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_git.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(decoder_snapshot, "CLONE_TIMEOUT_SECONDS", 0.01)
    token = "SECRET-TOKEN-123"

    with pytest.raises(RuntimeError, match="timed out") as exc_info:
        clone_at_commit("https://github.com/x/y", "0" * 40, token, tmp_path / "clone")

    assert token not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.fixture
def run_timeouts(monkeypatch) -> list[float | None]:
    timeouts: list[float | None] = []

    def fake_run(args, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(decoder_snapshot, "_run", fake_run)
    return timeouts


def test_clone_uses_clone_timeout(tmp_path, run_timeouts):
    clone_at_commit("https://github.com/x/y", "0" * 40, None, tmp_path / "clone")

    assert run_timeouts == [decoder_snapshot.CLONE_TIMEOUT_SECONDS, None]


def test_archive_and_xz_use_default_timeout(tmp_path, run_timeouts):
    tar_path = tmp_path / "decoder.tar"
    tar_path.write_bytes(b"")
    xz_path = tmp_path / "decoder.tar.xz"
    xz_path.write_bytes(b"")

    archive_decoder_dir(tmp_path, "0" * 40, tmp_path / "out.tar")
    decoder_snapshot.xz_bytes_of(tar_path, xz_path)

    assert run_timeouts == [None, None]


async def test_failed_upload_raises_runtime_error(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    commit = _make_repo(origin, _DECODER_FILES)

    async def failing_upload(bucket, object_name, file_path):
        return False

    monkeypatch.setattr(decoder_snapshot.async_minio_client, "upload_file", failing_upload)

    with pytest.raises(RuntimeError, match="Failed to upload"):
        await snapshot_decoder(str(origin), commit, None, "bucket")


async def test_happy_path_uploads_counted_bytes_once(tmp_path, uploads):
    origin = tmp_path / "origin"
    commit = _make_repo(origin, _DECODER_FILES)

    snapshot = await snapshot_decoder(str(origin), commit, None, "bucket")

    assert isinstance(snapshot, DecoderSnapshot)
    assert len(uploads) == 1
    bucket, object_name, uploaded_size = uploads[0]
    assert bucket == "bucket"
    assert object_name == f"quant-decoders/{commit}/{snapshot.xz_sha256}.tar.xz"
    assert snapshot.object_name == object_name
    assert snapshot.xz_bytes == uploaded_size


def test_sha256_file_matches_known_digest(tmp_path):
    file_path = tmp_path / "empty"
    file_path.write_bytes(b"")

    assert sha256_file(file_path) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


async def test_tree_sha_is_content_only_and_equals_an_orphan_squash(tmp_path):
    origin = tmp_path / "origin"
    _make_repo(origin, _DECODER_FILES)
    # A later commit that only touches non-decoder files changes the commit id, never the decoder tree.
    commit = _commit_file(origin, "README.md", b"changed artifact notes\n")

    snapshot = await snapshot_decoder(str(origin), commit, None, "bucket", upload=False)

    # The handover push squashes into an orphan commit: same decoder/ content -> same tree id.
    squashed = tmp_path / "squashed"
    squashed.mkdir()
    _git(squashed, "init", "--quiet", "--initial-branch=main")
    for name, content in _DECODER_FILES.items():
        if name.startswith("decoder/"):
            _write_file(squashed, name, content)
    _git(squashed, "add", "--all")
    _git(squashed, "commit", "--quiet", "-m", "orphan squash")
    assert snapshot.tree_sha == _git(squashed, "rev-parse", "HEAD:decoder")
    assert len(snapshot.tree_sha) == 40


async def test_tree_sha_changes_with_decoder_content_but_not_with_other_files(tmp_path):
    origin = tmp_path / "origin"
    first = _make_repo(origin, _DECODER_FILES)
    other_files = _commit_file(origin, "README.md", b"x\n")
    decoder_change = _commit_file(origin, "decoder/x.py", b"def decode():\n    return 3\n")

    shas = [
        (await snapshot_decoder(str(origin), c, None, "bucket", upload=False)).tree_sha
        for c in (first, other_files, decoder_change)
    ]

    assert shas[0] == shas[1]
    assert shas[2] != shas[0]
