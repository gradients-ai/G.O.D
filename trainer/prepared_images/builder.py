"""Build miner repositories into prepared images and publish them to the R2 image registry.

The validator dispatches one build per (tournament, hotkey, variant) to idle trainers (or a fallback CPU builder).
A build is: clone -> pick Dockerfile -> `docker buildx build --push` to the localhost registry (zstd) -> parallel
upload to R2. Jobs are in-memory and idempotent on (build_id, attempt); a trainer restart loses them and the
validator's lease expires and retries elsewhere.
"""

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

from core.git import sanitize_git_text
from core.image_registry import load_registry_config
from core.logging import get_logger
from core.models.payload_models import ImageBuilderStatus
from core.models.payload_models import ImageBuildRequest
from core.models.payload_models import ImageBuildStatusResponse
from trainer import constants as cst
from trainer.host import clone_repo
from trainer.job_state import get_running_jobs
from trainer.prepared_images.cache import touch_prepared_image
from trainer.prepared_images.local_registry import LocalRegistry
from trainer.prepared_images.local_registry import local_registry
from trainer.prepared_images.r2_transfer import R2ImageStore
from trainer.prepared_images.r2_transfer import publish_image


logger = get_logger(__name__)

VARIANT_DOCKERFILES = {
    "text": cst.TEXT_DOCKERFILE_PATHS,
    "image-toolkit": cst.IMAGE_TOOLKIT_DOCKERFILE_PATHS,
    "image-flux": cst.IMAGE_DOCKERFILE_PATHS,
}

# Output markers of transient network/infrastructure trouble. Seen in practice: DNS failures inside RUN steps
# under concurrent builds, corrupt git packs ("unable to read sha1 file"), registry/Docker Hub rate limits.
_INFRA_MARKERS = (
    "could not resolve host",
    "temporary failure in name resolution",
    "name or service not known",
    "no such host",
    "connection reset",
    "connection refused",
    "connection timed out",
    "operation timed out",
    "i/o timeout",
    "tls handshake timeout",
    "unexpected eof",
    "early eof",
    "rpc failed",
    "unable to read sha1 file",
    "the remote end hung up",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway",
    "429 too many requests",
    "toomanyrequests",
    "rate limit",
    "no space left on device",
    "dial tcp",
    "cannot connect to the docker daemon",
    "max retries exceeded",
    "network is unreachable",
    "readtimeouterror",
    "read timed out",
    "failed to fetch oauth token",
    "failed to do request",
)
_DOCKERFILE_MARKERS = (
    "failed to read dockerfile",
    "dockerfile parse error",
    "failed to parse dockerfile",
    "unknown instruction",
    "failed to compute cache key",
)
_CLONE_DETERMINISTIC_MARKERS = (
    "invalid commit hash",
    "repository not found",
    "authentication failed",
    "could not read username",
    "terminal prompts disabled",
    "not found",
    "403",
)


class BuilderUnavailable(Exception):
    pass


def classify_build_output(output: str) -> str:
    lowered = output.lower()
    if any(marker in lowered for marker in _INFRA_MARKERS):
        return "infrastructure"
    if any(marker in lowered for marker in _DOCKERFILE_MARKERS):
        return "dockerfile"
    return "build"


def classify_clone_error(message: str) -> str:
    lowered = message.lower()
    if any(marker in lowered for marker in _INFRA_MARKERS):
        return "infrastructure"
    if any(marker in lowered for marker in _CLONE_DETERMINISTIC_MARKERS):
        return "clone"
    return "infrastructure"


def compute_slots(free_disk_gb: float, vcpus: int, ram_gb: float, active: int, max_slots: int) -> tuple[int, int]:
    """Return (slots_total, free_slots). Disk is the binding resource; it already reflects active builds."""
    static_total = max(0, min(vcpus // cst.IMAGE_BUILD_VCPUS_PER_SLOT, int(ram_gb // cst.IMAGE_BUILD_RAM_GB_PER_SLOT), max_slots))
    disk_slots = max(0, int(free_disk_gb * cst.IMAGE_BUILD_DISK_HEADROOM // cst.IMAGE_BUILD_DISK_GB_PER_SLOT))
    free = max(0, min(static_total - active, disk_slots))
    return static_total, free


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class BuildJob:
    request: ImageBuildRequest
    state: str = "queued"
    started_at: datetime = field(default_factory=_now)
    last_progress_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = None
    digest: str | None = None
    resolved_commit: str | None = None
    dockerfile_path: str | None = None
    size_bytes: int | None = None
    failure_class: str | None = None
    error: str | None = None
    task: asyncio.Task | None = None

    def to_response(self) -> ImageBuildStatusResponse:
        return ImageBuildStatusResponse(
            build_id=self.request.build_id,
            attempt=self.request.attempt,
            state=self.state,
            digest=self.digest,
            resolved_commit=self.resolved_commit,
            dockerfile_path=self.dockerfile_path,
            size_bytes=self.size_bytes,
            failure_class=self.failure_class,
            error=self.error,
            started_at=self.started_at,
            last_progress_at=self.last_progress_at,
            finished_at=self.finished_at,
        )


class BuildFailure(Exception):
    def __init__(self, failure_class: str, message: str):
        super().__init__(message)
        self.failure_class = failure_class


async def run_streaming(
    cmd: list[str],
    *,
    log_path: Path,
    timeout_seconds: float,
    stall_seconds: float,
    on_progress,
    env: dict | None = None,
    cwd: str | None = None,
) -> tuple[int, str, str | None]:
    """Run a command, stream output to a log, and kill its process group on timeout or stall.

    Returns (returncode, output tail, abort reason or None).
    """
    tail: deque[str] = deque()
    tail_len = 0
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
        env=env,
        cwd=cwd,
    )
    reason = None
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                reason = "timeout"
                break
            try:
                chunk = await asyncio.wait_for(proc.stdout.read(65536), timeout=min(stall_seconds, remaining))
            except asyncio.TimeoutError:
                reason = "timeout" if deadline - loop.time() <= 0 else "stalled"
                break
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            log.write(text)
            tail.append(text)
            tail_len += len(text)
            while tail_len > cst.IMAGE_BUILD_LOG_TAIL_CHARS * 2 and len(tail) > 1:
                tail_len -= len(tail.popleft())
            on_progress()
    if reason:
        _kill_process_group(proc.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=15)
        except asyncio.TimeoutError:
            _kill_process_group(proc.pid, signal.SIGKILL)
            await proc.wait()
        return proc.returncode if proc.returncode is not None else -1, "".join(tail), reason
    return await proc.wait(), "".join(tail), None


def _kill_process_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def detect_build_compression(driver_status: list) -> str:
    """zstd on the containerd image store; gzip on graph drivers (overlay2), which silently ignore zstd.

    Every trainer can pull both (Docker >= 23), so images built on different hosts can be mixed.
    """
    return "zstd" if "io.containerd.snapshotter" in json.dumps(driver_status or []) else "gzip"


def _docker_preflight() -> tuple[str | None, str]:
    """Return (reason this host cannot build or None, layer compression to use)."""
    try:
        info = json.loads(
            subprocess.run(
                ["docker", "info", "--format", "{{json .}}"], capture_output=True, text=True, timeout=30, check=True
            ).stdout
        )
        subprocess.run(["docker", "buildx", "version"], capture_output=True, timeout=30, check=True)
    except Exception as exc:
        return f"docker/buildx unavailable: {exc}", "gzip"
    return None, detect_build_compression(info.get("DriverStatus"))


class ImageBuilder:
    def __init__(self, registry: LocalRegistry = local_registry):
        self.registry = registry
        self.jobs: dict[tuple[str, int], BuildJob] = {}
        self._lock = asyncio.Lock()
        self._preflight_reason: str | None = None
        self._preflight_at: datetime | None = None
        self.compression = "gzip"

    def _enabled_reason(self) -> str | None:
        if os.getenv(cst.IMAGE_BUILDER_ENABLED_ENV, "true").lower() == "false":
            return f"{cst.IMAGE_BUILDER_ENABLED_ENV}=false"
        if load_registry_config() is None:
            return "IMAGE_REGISTRY_* settings are missing"
        # Re-check periodically: a Docker restart or daemon.json change should not need a trainer restart.
        if self._preflight_at is None or _now() - self._preflight_at > timedelta(minutes=10):
            self._preflight_reason, self.compression = _docker_preflight()
            self._preflight_at = _now()
        return self._preflight_reason

    def _active(self) -> list[BuildJob]:
        return [job for job in self.jobs.values() if job.state not in ("ready", "failed")]

    def status(self) -> ImageBuilderStatus:
        reason = self._enabled_reason()
        free_disk_gb = _free_disk_gb()
        active = len(self._active())
        max_slots = int(os.getenv("IMAGE_BUILD_MAX_CONCURRENCY", str(cst.IMAGE_BUILD_MAX_SLOTS)))
        total, free = compute_slots(free_disk_gb, os.cpu_count() or 1, _ram_gb(), active, max_slots)
        return ImageBuilderStatus(
            enabled=reason is None,
            reason=reason,
            compression=self.compression if reason is None else None,
            slots_total=total,
            active_builds=active,
            free_slots=free if reason is None else 0,
            gpu_jobs_running=len(get_running_jobs()),
            free_disk_gb=round(free_disk_gb, 1),
        )

    async def submit(self, request: ImageBuildRequest) -> BuildJob:
        async with self._lock:
            self._forget_old_jobs()
            key = (request.build_id, request.attempt)
            if key in self.jobs:
                return self.jobs[key]
            status = await asyncio.to_thread(self.status)
            if not status.enabled:
                raise BuilderUnavailable(status.reason or "builder disabled")
            if status.free_slots <= 0:
                raise BuilderUnavailable("no free build slots")
            job = BuildJob(request=request)
            self.jobs[key] = job
            job.task = asyncio.create_task(self._run(job))
            return job

    def get(self, build_id: str, attempt: int) -> BuildJob | None:
        return self.jobs.get((build_id, attempt))

    def _forget_old_jobs(self) -> None:
        cutoff = _now() - timedelta(hours=cst.IMAGE_BUILD_JOB_RETENTION_HOURS)
        for key, job in list(self.jobs.items()):
            if job.finished_at and job.finished_at < cutoff:
                self.jobs.pop(key, None)

    async def _run(self, job: BuildJob) -> None:
        request = job.request
        labels = {"build_id": request.build_id, "attempt": str(request.attempt), "variant": request.variant}
        work_dir = Path(cst.IMAGE_BUILD_WORK_PATH) / f"{request.build_id}-{request.attempt}"
        log_path = Path(cst.IMAGE_BUILD_WORK_PATH) / "logs" / f"{request.build_id}-{request.attempt}.log"
        tokens = (request.github_token,)

        def progress() -> None:
            job.last_progress_at = _now()

        try:
            with self.registry.lease():
                await asyncio.to_thread(self.registry.ensure_running)
                job.state = "cloning"
                repo_dir = await self._clone(request, work_dir)
                progress()
                job.resolved_commit = await asyncio.to_thread(_git_head, repo_dir)
                job.dockerfile_path = _pick_dockerfile(repo_dir, request.variant)

                job.state = "building"
                compression = self.compression
                metadata_path = work_dir / "metadata.json"
                target = self.registry.tag_ref(request.repository, request.tag)
                cmd = [
                    "docker", "buildx", "build", "--builder", "default", "--progress=plain",
                    "--pull", "--provenance=false", "--sbom=false",
                    "-f", str(Path(repo_dir) / job.dockerfile_path),
                    "--output",
                    f"type=image,name={target},push=true,compression={compression},"
                    "force-compression=true,oci-mediatypes=true",
                    "--metadata-file", str(metadata_path),
                    repo_dir,
                ]  # fmt: skip
                timeout = float(os.getenv("IMAGE_BUILD_TIMEOUT_SECONDS", str(cst.IMAGE_BUILD_TIMEOUT_SECONDS)))
                stall = float(os.getenv("IMAGE_BUILD_STALL_SECONDS", str(cst.IMAGE_BUILD_STALL_SECONDS)))
                rc, output, abort = await run_streaming(
                    cmd, log_path=log_path, timeout_seconds=timeout, stall_seconds=stall, on_progress=progress
                )
                if abort == "stalled":
                    raise BuildFailure("stalled", f"No build output for {stall:.0f}s; build killed.\n{output}")
                if abort == "timeout":
                    raise BuildFailure("timeout", f"Build exceeded {timeout:.0f}s; build killed.\n{output}")
                if rc != 0:
                    raise BuildFailure(classify_build_output(output), output)
                digest = json.loads(metadata_path.read_text()).get("containerimage.digest")
                if not digest:
                    raise BuildFailure("infrastructure", "Build succeeded but reported no image digest")

                job.state = "publishing"
                progress()
                config = load_registry_config()
                if config is None:
                    raise BuildFailure("infrastructure", "IMAGE_REGISTRY_* settings are missing")
                try:
                    result = await asyncio.to_thread(
                        publish_image, R2ImageStore(config), self.registry, request.repository, request.tag, digest
                    )
                except Exception as exc:
                    raise BuildFailure("infrastructure", f"Publishing to the image registry failed: {exc}") from exc
                job.digest = digest
                job.size_bytes = result.size_bytes
                touch_prepared_image(self.registry.ref(request.repository, digest))
                job.state = "ready"
                logger.info(
                    f"Prepared image ready: {request.repository}@{digest} ({compression}, "
                    f"{result.size_bytes / 1e9:.1f} GB, uploaded {result.uploaded_bytes / 1e9:.1f} GB)",
                    extra=labels,
                )
        except asyncio.CancelledError:
            job.failure_class, job.error = "cancelled", "Build cancelled on the trainer"
            job.state = "failed"
            raise
        except BuildFailure as exc:
            job.failure_class = exc.failure_class
            job.error = sanitize_git_text(str(exc), *tokens)[-cst.IMAGE_BUILD_LOG_TAIL_CHARS :]
            job.state = "failed"
            logger.warning(f"Image build failed ({exc.failure_class})", extra=labels)
        except Exception as exc:
            job.failure_class = "infrastructure"
            job.error = sanitize_git_text(f"{type(exc).__name__}: {exc}", *tokens)[-cst.IMAGE_BUILD_LOG_TAIL_CHARS :]
            job.state = "failed"
            logger.exception("Image build crashed", extra=labels)
        finally:
            job.finished_at = _now()
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _clone(self, request: ImageBuildRequest, work_dir: Path) -> str:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    clone_repo,
                    repo_url=request.repo,
                    parent_dir=str(work_dir),
                    commit_hash=request.commit or "HEAD",
                    github_token=request.github_token,
                ),
                timeout=cst.IMAGE_CLONE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise BuildFailure("infrastructure", "Repository clone timed out") from exc
        except Exception as exc:
            message = sanitize_git_text(str(exc), request.github_token)
            raise BuildFailure(classify_clone_error(message), message) from exc


def _pick_dockerfile(repo_dir: str, variant: str) -> str:
    candidates = VARIANT_DOCKERFILES[variant]
    for candidate in candidates:
        if (Path(repo_dir) / candidate).is_file():
            return candidate
    raise BuildFailure("dockerfile", f"Training repository is missing a supported Dockerfile. Checked: {', '.join(candidates)}")


def _git_head(repo_dir: str) -> str | None:
    result = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    head = result.stdout.strip()
    return head if re.fullmatch(r"[0-9a-f]{40}", head) else None


def _free_disk_gb() -> float:
    paths = [cst.LOCAL_REGISTRY_STORAGE_PATH, "/var/lib/docker", "/"]
    free = []
    for path in paths:
        probe = Path(path)
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            free.append(shutil.disk_usage(probe).free / 1e9)
        except OSError:
            continue
    return min(free) if free else 0.0


def _ram_gb() -> float:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


image_builder = ImageBuilder()
