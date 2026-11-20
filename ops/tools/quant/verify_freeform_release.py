"""Third-party verifier for a released freeform quant result, using only public inputs.

Inputs: the public scorecard (`/auditing/tasks/{task_id}`, one `quant_scorecard` per hotkey), the released decoder
snapshot (`decoder-<sha>.tar.xz`, optionally `release.json`) from the public `<org>/freeform-releases` dataset, a local
clone of the released (squashed) champion repo, and the artifact downloaded from the public HF repo.

Checks (each is one CheckResult):
  a. decoder .tar.xz sha256 and size == scorecard decoder.xz_sha256 / xz_bytes (and release.json)
  b. `git rev-parse HEAD:decoder` in the released repo == scorecard decoder.tree_sha
  c. the files inside the .tar.xz == the released repo's decoder/ (compared by git blob id)
  d. artifact files: sha256 (LFS) or git blob sha1 (plain file) and size == scorecard artifact.files; byte sum ==
     artifact_bytes; no extra files
  e. total_bytes == artifact_bytes + decoder_xz_bytes and effective_bytes == total_bytes x exp(ln_penalty)
  f. optional decode: run the decoder as `python /decoder/decode.py /artifact /output` (the trainer's contract, after
     `/bin/sh /decoder/build.sh` when the decoder has one) and compare the shard sha256s with scorecard decoded.files

The library functions take local paths only and make no network call, so they are testable offline; only the CLI
fetches (urllib for public URLs, huggingface_hub.snapshot_download for the artifact).
Needs pydantic, huggingface_hub, `git`, and `docker` for a real decode.

The sandbox contract uses fixed absolute paths (/decoder, /artifact, /output, /build) inside a container, so a faithful
--decode needs docker (DockerDecodeRunner, the CLI default). HostDecodeRunner runs the decode phase on the host with the
same argv remapped to local directories; it cannot run a build.sh and is meant for tests and build-less decoders.

Docker memory follows the trainer's rule so a correct release is not OOM-killed here: the build phase gets the floor
(8 GiB) and the run phase max(floor, ceil(2.5 x the scorecard's decoded bf16 bytes)). `--memory 32g` (any docker
--memory value) overrides the run phase, for a decoder that needs more than the trainer's allowance would give it.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from pathlib import PurePosixPath
from typing import Protocol

from huggingface_hub import snapshot_download
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import ValidationError


DECODER_DIR_NAME = "decoder"
DECODER_ENTRYPOINT = "decode.py"
DECODER_BUILD_SCRIPT = "build.sh"
SHA256_KIND_LFS = "sha256"
SHA256_KIND_GIT_BLOB = "git-blob-sha1"
HASH_CHUNK_BYTES = 1 << 20
SHARD_SUFFIX = ".safetensors"
RELEASE_JSON_NAME = "release.json"
# Mirrors trainer/constants.py and core/constants/quant.py; kept here so the tool needs nothing from the repo.
BUILD_IMAGE = "gradientsio/quant-decoder-build:pinned"
RUN_IMAGE = "gradientsio/quant-decoder-run:pinned"
DECODE_TIMEOUT_SECONDS = 900
DECODE_CPUS = 16
DECODE_PIDS_LIMIT = 256
# Mirror core/constants/quant.py DECODE_MEMORY_MULTIPLIER and trainer/constants.py QUANT_DECODE_MIN_MEMORY_GB (a repo test
# fails if they drift).
DECODE_MEMORY_MULTIPLIER = 2.5
DECODE_MIN_MEMORY_GB = 8
EFFECTIVE_BYTES_REL_TOL = 1e-9
IGNORED_ARTIFACT_PARTS = {".cache"}
GIT_ENV = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


class VerifierError(Exception):
    """An input is unusable (not a verification failure of the release itself)."""


class _Public(BaseModel):
    model_config = ConfigDict(extra="ignore")


class FileHash(_Public):
    name: str
    bytes: int
    sha256: str
    sha256_kind: str = SHA256_KIND_LFS


class ScorecardBytes(_Public):
    artifact_bytes: int | None = None
    decoder_xz_bytes: int | None = None
    total_bytes: int | None = None
    ln_penalty: float | None = None
    effective_bytes: float | None = None


class ScorecardArtifact(_Public):
    repo: str | None = None
    revision: str | None = None
    files: list[FileHash] = []


class ScorecardDecoder(_Public):
    xz_bytes: int | None = None
    xz_sha256: str | None = None
    tar_sha256: str | None = None
    tree_sha: str | None = None


class ScorecardDecoded(_Public):
    files: list[FileHash] = []


class ScorecardRelease(_Public):
    decoder_snapshot_url: str | None = None
    training_repo: str | None = None
    training_commit_hash: str | None = None


class Scorecard(_Public):
    """The slice of QuantFreeformScorecard the verifier reads."""

    bytes: ScorecardBytes
    artifact: ScorecardArtifact
    decoder: ScorecardDecoder
    decoded: ScorecardDecoded
    release: ScorecardRelease | None = None


class ReleaseDecoder(_Public):
    xz_bytes: int
    xz_sha256: str
    tree_sha: str | None = None


class ReleaseTask(_Public):
    task_id: str
    repo: str | None = None
    revision: str | None = None
    artifact_files: list[FileHash] = []
    decoded_files: list[FileHash] = []


class ReleaseJson(_Public):
    """release.json next to the released decoder."""

    tournament_id: str
    hotkey: str
    training_commit_hash: str | None = None
    training_repo: str | None = None
    decoder: ReleaseDecoder
    tasks: list[ReleaseTask] = []


class CheckResult(BaseModel):
    name: str
    ok: bool
    detail: str


class VerificationReport(BaseModel):
    checks: list[CheckResult] = []

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(check.ok for check in self.checks)

    def failures(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.ok]

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(CheckResult(name=name, ok=ok, detail=detail))


class VerifyInputs(BaseModel):
    scorecard: Scorecard
    decoder_xz: Path
    repo_dir: Path
    artifact_dir: Path
    release: ReleaseJson | None = None


def load_scorecard(raw: str, hotkey: str | None = None) -> Scorecard:
    """Parse a scorecard JSON: the scorecard itself, or a task response (`hotkey_details`) with `hotkey` selecting one."""
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise VerifierError(f"scorecard is not JSON: {e}") from None
    if isinstance(data, dict) and "hotkey_details" in data:
        matches = [d for d in data["hotkey_details"] if d.get("hotkey") == hotkey and d.get("quant_scorecard")]
        if not matches:
            raise VerifierError(f"no quant_scorecard for hotkey {hotkey!r} in the task response")
        data = matches[0]["quant_scorecard"]
    try:
        return Scorecard.model_validate(data)
    except ValidationError as e:
        raise VerifierError(f"not a freeform scorecard: {e}") from None


def load_release_json(raw: str) -> ReleaseJson:
    try:
        return ReleaseJson.model_validate_json(raw)
    except ValidationError as e:
        raise VerifierError(f"not a release.json: {e}") from None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Git blob id: sha1 over `blob <len>\\0` + content."""
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        while chunk := f.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_dir: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, env={**os.environ, **GIT_ENV}, check=False
    )
    if proc.returncode != 0:
        raise VerifierError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def check_decoder_xz(inputs: VerifyInputs, report: VerificationReport) -> None:
    """a. the released snapshot is the exact file that was counted and run."""
    decoder = inputs.scorecard.decoder
    actual_sha, actual_bytes = sha256_file(inputs.decoder_xz), inputs.decoder_xz.stat().st_size
    report.add(
        "decoder_xz_sha256",
        actual_sha == decoder.xz_sha256,
        f"released {actual_sha}, scorecard {decoder.xz_sha256}",
    )
    expected_bytes = decoder.xz_bytes if decoder.xz_bytes is not None else inputs.scorecard.bytes.decoder_xz_bytes
    report.add("decoder_xz_bytes", actual_bytes == expected_bytes, f"released {actual_bytes}, scorecard {expected_bytes}")
    if inputs.release is not None:
        rel = inputs.release.decoder
        report.add(
            "release_json_decoder",
            (rel.xz_sha256, rel.xz_bytes) == (actual_sha, actual_bytes) and rel.tree_sha in (None, decoder.tree_sha),
            f"release.json says sha256 {rel.xz_sha256}, {rel.xz_bytes} bytes, tree {rel.tree_sha}",
        )


def check_tree_sha(inputs: VerifyInputs, report: VerificationReport) -> None:
    """b. the released repo's decoder/ tree is the registered one (survives the squash into an orphan commit)."""
    try:
        actual = _git(inputs.repo_dir, "rev-parse", f"HEAD:{DECODER_DIR_NAME}").strip()
    except VerifierError as e:
        report.add("tree_sha", False, str(e))
        return
    expected = inputs.scorecard.decoder.tree_sha
    report.add("tree_sha", actual == expected, f"git rev-parse HEAD:decoder = {actual}, scorecard {expected}")


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise VerifierError(f"unsafe path in decoder archive: {name!r}")
    return path


def extract_decoder(xz_path: Path, dest: Path) -> None:
    """Extract a git-archive `decoder/` tar.xz into dest with the prefix stripped; refuses links and path escapes."""
    try:
        with tarfile.open(xz_path, mode="r:xz") as tar:
            for member in tar:
                path = _safe_member_path(member.name)
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise VerifierError(f"decoder archive member {member.name!r} is not a plain file or directory")
                if path.parts[:1] != (DECODER_DIR_NAME,):
                    raise VerifierError(f"decoder archive member {member.name!r} is outside {DECODER_DIR_NAME}/")
                relative = PurePosixPath(*path.parts[1:])
                if not relative.parts:
                    continue
                target = dest.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise VerifierError(f"cannot read {member.name!r}")
                with source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
    except (tarfile.TarError, EOFError) as e:
        raise VerifierError(f"decoder archive unreadable: {e}") from None


def _repo_decoder_blobs(repo_dir: Path) -> dict[str, str]:
    """relative path (below decoder/) -> blob id of every file in the repo's HEAD:decoder tree."""
    listing = _git(repo_dir, "ls-tree", "-r", "-z", "HEAD", f"{DECODER_DIR_NAME}/")
    blobs: dict[str, str] = {}
    for entry in filter(None, listing.split("\0")):
        meta, path = entry.split("\t", 1)
        _mode, kind, blob_id = meta.split()
        if kind == "blob":
            blobs[path.removeprefix(f"{DECODER_DIR_NAME}/")] = blob_id
    return blobs


def check_tar_matches_repo(inputs: VerifyInputs, report: VerificationReport) -> None:
    """c. the snapshot's files are exactly the released repo's decoder/ (by git blob id, so mode-independent)."""
    with tempfile.TemporaryDirectory() as work:
        extracted = Path(work) / "decoder"
        extracted.mkdir()
        try:
            extract_decoder(inputs.decoder_xz, extracted)
            repo_blobs = _repo_decoder_blobs(inputs.repo_dir)
        except VerifierError as e:
            report.add("tar_matches_repo", False, str(e))
            return
        tar_blobs = {p.relative_to(extracted).as_posix(): git_blob_sha1(p) for p in extracted.rglob("*") if p.is_file()}
    differing = sorted(name for name in tar_blobs.keys() & repo_blobs.keys() if tar_blobs[name] != repo_blobs[name])
    only_tar, only_repo = sorted(tar_blobs.keys() - repo_blobs.keys()), sorted(repo_blobs.keys() - tar_blobs.keys())
    ok = not (differing or only_tar or only_repo)
    detail = f"{len(tar_blobs)} files compared" if ok else f"differ {differing}, only in tar {only_tar}, only in repo {only_repo}"
    report.add("tar_matches_repo", ok, detail)


def _artifact_files_on_disk(artifact_dir: Path) -> dict[str, Path]:
    return {
        p.relative_to(artifact_dir).as_posix(): p
        for p in artifact_dir.rglob("*")
        if p.is_file() and not IGNORED_ARTIFACT_PARTS & set(p.relative_to(artifact_dir).parts)
    }


def check_artifact_files(inputs: VerifyInputs, report: VerificationReport) -> None:
    """d. every downloaded artifact file matches the hub hashes in the scorecard; the sizes add up to artifact_bytes."""
    listed = inputs.scorecard.artifact.files
    on_disk = _artifact_files_on_disk(inputs.artifact_dir)
    problems: list[str] = []
    for entry in listed:
        path = on_disk.get(entry.name)
        if path is None:
            problems.append(f"{entry.name} missing")
            continue
        if path.stat().st_size != entry.bytes:
            problems.append(f"{entry.name} is {path.stat().st_size} bytes, scorecard {entry.bytes}")
            continue
        actual = git_blob_sha1(path) if entry.sha256_kind == SHA256_KIND_GIT_BLOB else sha256_file(path)
        if actual != entry.sha256:
            problems.append(f"{entry.name} {entry.sha256_kind} {actual} != scorecard {entry.sha256}")
    problems.extend(f"{name} is not in the scorecard" for name in sorted(on_disk.keys() - {e.name for e in listed}))
    report.add("artifact_file_hashes", bool(listed) and not problems, "; ".join(problems) or f"{len(listed)} files match")

    listed_sum = sum(entry.bytes for entry in listed)
    report.add(
        "artifact_bytes",
        listed_sum == inputs.scorecard.bytes.artifact_bytes,
        f"files sum to {listed_sum}, scorecard artifact_bytes {inputs.scorecard.bytes.artifact_bytes}",
    )


def check_byte_arithmetic(inputs: VerifyInputs, report: VerificationReport) -> None:
    """e. total = artifact + decoder xz; S = total x exp(ln_penalty)."""
    b = inputs.scorecard.bytes
    if b.artifact_bytes is None or b.decoder_xz_bytes is None or b.total_bytes is None:
        report.add("total_bytes", False, "scorecard lacks artifact_bytes, decoder_xz_bytes or total_bytes")
        return
    report.add(
        "total_bytes",
        b.total_bytes == b.artifact_bytes + b.decoder_xz_bytes,
        f"{b.artifact_bytes} + {b.decoder_xz_bytes} vs total {b.total_bytes}",
    )
    try:
        expected = b.total_bytes * math.exp(b.ln_penalty or 0.0)
    except OverflowError:
        report.add("effective_bytes", False, f"total x exp({b.ln_penalty}) overflows, scorecard {b.effective_bytes}")
        return
    ok = b.effective_bytes is not None and math.isclose(b.effective_bytes, expected, rel_tol=EFFECTIVE_BYTES_REL_TOL)
    report.add("effective_bytes", ok, f"total x exp(ln_penalty) = {expected}, scorecard {b.effective_bytes}")


def run_phase_memory(decoded_bytes: int) -> str:
    """Docker --memory (bytes) for the run phase: the multiplier x the decoded bf16 size, never below the floor."""
    floor_bytes = DECODE_MIN_MEMORY_GB * 1024**3
    return str(max(floor_bytes, math.ceil(DECODE_MEMORY_MULTIPLIER * decoded_bytes)))


class DecodeInvocation(BaseModel):
    """Local stand-ins for the sandbox's /decoder, /artifact, /output and /build. decoded_bytes is the scorecard's
    bf16 size, which sizes the docker run phase."""

    decoder_dir: Path
    artifact_dir: Path
    output_dir: Path
    build_dir: Path
    decoded_bytes: int = 0


class DecodeRunner(Protocol):
    def __call__(self, invocation: DecodeInvocation) -> None: ...


class HostDecodeRunner:
    """Run `python decode.py <artifact> <output>` on the host (the sandbox's argv with the paths remapped). It cannot
    honour a build.sh, which writes to the absolute /build, so a decoder with one is refused."""

    def __init__(self, python: str = sys.executable) -> None:
        self.python = python

    def __call__(self, invocation: DecodeInvocation) -> None:
        if (invocation.decoder_dir / DECODER_BUILD_SCRIPT).exists():
            raise VerifierError("decoder has a build.sh; use DockerDecodeRunner (--runner docker)")
        proc = subprocess.run(
            [
                self.python,
                str(invocation.decoder_dir / DECODER_ENTRYPOINT),
                str(invocation.artifact_dir),
                str(invocation.output_dir),
            ],
            cwd=invocation.decoder_dir,
            env={"HOME": str(invocation.output_dir.parent), "PATH": os.environ.get("PATH", "")},
            capture_output=True,
            text=True,
            timeout=DECODE_TIMEOUT_SECONDS,
            check=False,
        )
        if proc.returncode != 0:
            raise VerifierError(f"decoder exited {proc.returncode}: {proc.stderr.strip()[-300:]}")


class DockerDecodeRunner:
    """The trainer's two phases through the docker CLI with the sandbox's isolation: build (only when build.sh exists)
    then run, no network, read-only root, all caps dropped, the pinned images."""

    def __init__(self, docker: str = "docker", memory: str | None = None) -> None:
        self.docker = docker
        self.memory = memory  # run-phase override of the size-derived limit

    def _run(self, image: str, command: list[str], mounts: list[tuple[Path, str, str]], memory: str) -> None:
        args = [
            self.docker, "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--user", f"{os.geteuid()}:{os.getegid()}",
            "--tmpfs", "/tmp:size=1g", "-e", "HOME=/tmp", "--pids-limit", str(DECODE_PIDS_LIMIT),
            "--cpus", str(DECODE_CPUS), "--memory", memory,
        ]  # fmt: skip
        for host, container, mode in mounts:
            args += ["-v", f"{host}:{container}:{mode}"]
        proc = subprocess.run(
            [*args, image, *command], capture_output=True, text=True, timeout=DECODE_TIMEOUT_SECONDS, check=False
        )
        if proc.returncode != 0:
            raise VerifierError(f"docker {command[0]} exited {proc.returncode}: {proc.stderr.strip()[-300:]}")

    def __call__(self, invocation: DecodeInvocation) -> None:
        decoder_mount = (invocation.decoder_dir, "/decoder", "ro")
        if (invocation.decoder_dir / DECODER_BUILD_SCRIPT).exists():
            self._run(
                BUILD_IMAGE,
                ["/bin/sh", f"/decoder/{DECODER_BUILD_SCRIPT}"],
                [decoder_mount, (invocation.build_dir, "/build", "rw")],
                f"{DECODE_MIN_MEMORY_GB}g",
            )
        mounts = [
            (invocation.artifact_dir, "/artifact", "ro"),
            decoder_mount,
            (invocation.output_dir, "/output", "rw"),
            (invocation.build_dir, "/build", "ro"),
        ]
        self._run(
            RUN_IMAGE,
            ["python", f"/decoder/{DECODER_ENTRYPOINT}", "/artifact", "/output"],
            mounts,
            self.memory or run_phase_memory(invocation.decoded_bytes),
        )


def check_decode(inputs: VerifyInputs, report: VerificationReport, runner: DecodeRunner) -> None:
    """f. re-run the released decoder on the public artifact and compare the shard hashes with decoded.files."""
    expected = {f.name: f for f in inputs.scorecard.decoded.files}
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        invocation = DecodeInvocation(
            decoder_dir=root / "decoder",
            artifact_dir=inputs.artifact_dir,
            output_dir=root / "output",
            build_dir=root / "build",
            decoded_bytes=sum(f.bytes for f in inputs.scorecard.decoded.files),
        )
        for directory in (invocation.decoder_dir, invocation.output_dir, invocation.build_dir):
            directory.mkdir()
        try:
            extract_decoder(inputs.decoder_xz, invocation.decoder_dir)
            if not (invocation.decoder_dir / DECODER_ENTRYPOINT).is_file():
                raise VerifierError(f"decoder has no {DECODER_ENTRYPOINT}")
            runner(invocation)
        except (VerifierError, subprocess.SubprocessError, OSError) as e:
            report.add("decode_reproduces", False, f"decode failed: {e}")
            return
        produced = {p.name: p for p in invocation.output_dir.iterdir() if p.is_file() and p.suffix == SHARD_SUFFIX}
        problems = [f"{name} not produced" for name in sorted(expected.keys() - produced.keys())]
        problems += [f"{name} is not in the scorecard" for name in sorted(produced.keys() - expected.keys())]
        for name in sorted(produced.keys() & expected.keys()):
            path = produced[name]
            if path.stat().st_size != expected[name].bytes or sha256_file(path) != expected[name].sha256:
                problems.append(f"{name} differs from the scorecard (bytes or sha256)")
    report.add("decode_reproduces", bool(expected) and not problems, "; ".join(problems) or f"{len(expected)} shards reproduced")


def check_release_json_tasks(inputs: VerifyInputs, report: VerificationReport) -> None:
    """release.json's entry for this artifact repo carries the same hashes as the scorecard."""
    if inputs.release is None:
        return
    card = inputs.scorecard
    entries = [t for t in inputs.release.tasks if t.repo == card.artifact.repo and t.revision == card.artifact.revision]
    ok = any(
        {(f.name, f.bytes, f.sha256) for f in t.artifact_files} == {(f.name, f.bytes, f.sha256) for f in card.artifact.files}
        and {(f.name, f.bytes, f.sha256) for f in t.decoded_files} == {(f.name, f.bytes, f.sha256) for f in card.decoded.files}
        for t in entries
    )
    report.add(
        "release_json_task", ok, f"{len(entries)} release.json task entries for {card.artifact.repo}@{card.artifact.revision}"
    )


def verify_release(inputs: VerifyInputs, runner: DecodeRunner | None = None) -> VerificationReport:
    """Run every check; the decode check only when a runner is given."""
    report = VerificationReport()
    check_decoder_xz(inputs, report)
    check_tree_sha(inputs, report)
    check_tar_matches_repo(inputs, report)
    check_artifact_files(inputs, report)
    check_byte_arithmetic(inputs, report)
    check_release_json_tasks(inputs, report)
    if runner is not None:
        check_decode(inputs, report, runner)
    return report


def _read(source: str) -> str:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as response:  # public URLs only
            return response.read().decode()
    return Path(source).read_text()


def _download(url: str, dest: Path) -> Path:
    with urllib.request.urlopen(url, timeout=300) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)
    return dest


def _clone(url: str, commit: str | None, dest: Path) -> Path:
    subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True, env={**os.environ, **GIT_ENV})
    if commit:
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True, env={**os.environ, **GIT_ENV})
    return dest


def _fetch_artifact(repo: str, revision: str | None, dest: Path) -> Path:
    return Path(snapshot_download(repo_id=repo, revision=revision, local_dir=dest))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a released freeform quant result from public inputs only.")
    parser.add_argument("--scorecard", required=True, help="scorecard JSON or /auditing/tasks/<id> response: file path or URL")
    parser.add_argument("--hotkey", help="which hotkey's scorecard, when --scorecard is a whole task response")
    parser.add_argument("--decoder-xz", type=Path, help="released decoder-<sha>.tar.xz (default: download scorecard release url)")
    parser.add_argument("--release-json", help="release.json path or URL (default: next to the decoder snapshot url)")
    parser.add_argument(
        "--repo-dir", type=Path, help="local clone of the released champion repo (default: clone release.training_repo)"
    )
    parser.add_argument(
        "--artifact-dir", type=Path, help="downloaded artifact (default: snapshot_download of artifact.repo@revision)"
    )
    parser.add_argument(
        "--decode", action="store_true", help="re-run the released decoder on the artifact and compare shard hashes"
    )
    parser.add_argument(
        "--runner", choices=["docker", "host"], default="docker", help="decode runner (docker needs the pinned images)"
    )
    parser.add_argument(
        "--memory",
        help="docker --memory for the decode run phase, e.g. 32g (default: max(8 GiB, 2.5 x decoded bytes), as the trainer)",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def _resolve_inputs(args: argparse.Namespace, work: Path) -> VerifyInputs:
    scorecard = load_scorecard(_read(args.scorecard), args.hotkey)
    release_info = scorecard.release
    snapshot_url = release_info.decoder_snapshot_url if release_info else None
    decoder_xz = args.decoder_xz
    if decoder_xz is None:
        if not snapshot_url:
            raise VerifierError("scorecard has no release section; pass --decoder-xz")
        decoder_xz = _download(snapshot_url, work / "decoder.tar.xz")
    release_source = args.release_json or (snapshot_url.rsplit("/", 1)[0] + f"/{RELEASE_JSON_NAME}" if snapshot_url else None)
    release = None
    if release_source:
        try:
            release = load_release_json(_read(release_source))
        except OSError as e:
            if args.release_json:
                raise VerifierError(f"cannot read release.json: {e}") from None
    repo_dir = args.repo_dir
    if repo_dir is None:
        if release_info is None or not release_info.training_repo:
            raise VerifierError("scorecard has no public training_repo; pass --repo-dir")
        repo_dir = _clone(release_info.training_repo, None, work / "repo")
    artifact_dir = args.artifact_dir
    if artifact_dir is None:
        if not scorecard.artifact.repo:
            raise VerifierError("scorecard has no artifact.repo; pass --artifact-dir")
        artifact_dir = _fetch_artifact(scorecard.artifact.repo, scorecard.artifact.revision, work / "artifact")
    return VerifyInputs(scorecard=scorecard, decoder_xz=decoder_xz, repo_dir=repo_dir, artifact_dir=artifact_dir, release=release)


def main(argv: list[str] | None = None, runner_factory: Callable[[str], DecodeRunner] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with tempfile.TemporaryDirectory() as work:
            inputs = _resolve_inputs(args, Path(work))
            runner = None
            if args.decode:
                runner = (
                    runner_factory(args.runner)
                    if runner_factory
                    else (DockerDecodeRunner(memory=args.memory) if args.runner == "docker" else HostDecodeRunner())
                )
            report = verify_release(inputs, runner)
    except (VerifierError, subprocess.CalledProcessError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        for check in report.checks:
            print(f"{'PASS' if check.ok else 'FAIL'}  {check.name}: {check.detail}")
        print("VERIFIED" if report.ok else "VERIFICATION FAILED")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
