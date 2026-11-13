"""Container specs and runner for the two-phase quant decoder sandbox.

The sandbox gets the decoder (and, in the run phase, the artifact) read-only and nothing else:
no network, no GPU, no credentials, a pinned image, and CPU/memory/PID/wall-clock caps.
"""

import asyncio
import math
import os
import re
import time
import uuid
from enum import Enum
from pathlib import Path

import docker
from docker.errors import DockerException
from docker.models.containers import Container
from pydantic import BaseModel
from pydantic import field_validator

import trainer.constants as cst
from core.constants.quant import DECODE_CPUS
from core.constants.quant import DECODE_MEMORY_MULTIPLIER
from core.constants.quant import DECODE_PIDS_LIMIT
from core.constants.quant import DECODE_TIMEOUT_SECONDS
from core.constants.quant import DECODER_ENTRYPOINT
from trainer.telemetry import logger


FORBIDDEN_ENV_KEY_PARTS = ("TOKEN", "KEY", "SECRET", "HF_", "WANDB")
FAILURE_DETAIL_MAX_CHARS = 350
DECODE_TMPFS = {"/tmp": "size=1g"}
# The sandbox runs as the trainer's uid, which has no passwd entry in the image, so docker would set HOME=/ on the
# read-only root; point it at the tmpfs instead. Not a credential, and the only variable the sandbox gets.
DECODE_ENVIRONMENT = {"HOME": "/tmp"}
# A build log line that reads as a failure ("build failed", gcc's "error:"). Only ever a warning, so a
# false positive costs a log line, never a score.
BUILD_FAILURE_LINE = re.compile(r"\b(fail(ed|ure|s)?|error)\b", re.IGNORECASE)


class DecodePhase(str, Enum):
    BUILD = "build"
    RUN = "run"


class DecodeContainerSpec(BaseModel):
    """Everything `client.containers.run` needs for one sandboxed decoder phase."""

    phase: DecodePhase
    image: str
    command: list[str]
    volumes: dict[str, dict[str, str]]  # docker-py bind form: {host_path: {"bind": path, "mode": "ro"|"rw"}}
    mem_limit: str
    nano_cpus: int
    pids_limit: int
    timeout_seconds: float
    network_disabled: bool = True
    read_only: bool = True
    # uid:gid of the trainer process, never root: everything the decoder writes to /build and /output is then owned by
    # the trainer, which can always clean it up, and the rw mounts need no "other" write bits.
    user: str
    cap_drop: list[str] = ["ALL"]
    security_opt: list[str] = ["no-new-privileges"]
    environment: dict[str, str] = DECODE_ENVIRONMENT
    tmpfs: dict[str, str] = DECODE_TMPFS

    @field_validator("environment")
    @classmethod
    def validate_no_credentials(cls, v: dict[str, str]) -> dict[str, str]:
        for key in v:
            if any(part in key.upper() for part in FORBIDDEN_ENV_KEY_PARTS):
                raise ValueError(f"decoder sandbox environment must not carry credentials: {key}")
        return v

    def docker_run_kwargs(self) -> dict:
        """Kwargs for `client.containers.run`. No device_requests (no GPU) and no bridge network:
        even the internal bridge allows container-to-container traffic, so networking is disabled outright."""
        return self.model_dump(exclude={"phase", "timeout_seconds"})


class DecodeOutcome(BaseModel):
    phase: DecodePhase
    ok: bool
    failure: str | None = None
    exit_code: int | None = None
    wall_seconds: float
    # Build phase only, on a clean exit: the log tail (stdout and stderr), read for build_phase_warning.
    log_lines: list[str] = []


def sandbox_user() -> str:
    """The trainer process's own uid:gid, which owns the work dirs the sandbox writes."""
    return f"{os.geteuid()}:{os.getegid()}"


def _bind(host_path: Path, container_path: str, mode: str) -> dict[str, dict[str, str]]:
    return {str(host_path): {"bind": container_path, "mode": mode}}


def build_phase_spec(decoder_src: Path, build_out: Path) -> DecodeContainerSpec:
    """Build phase: only the decoder source (read-only) and an empty build dir; the compiler is available."""
    if (decoder_src / cst.DECODE_BUILD_SCRIPT).exists():
        command = ["/bin/sh", f"{cst.DECODE_DECODER_PATH}/{cst.DECODE_BUILD_SCRIPT}"]
    else:
        command = ["true"]
    return DecodeContainerSpec(
        phase=DecodePhase.BUILD,
        image=cst.QUANT_DECODER_BUILD_IMAGE,
        command=command,
        volumes={
            **_bind(decoder_src, cst.DECODE_DECODER_PATH, "ro"),
            **_bind(build_out, cst.DECODE_BUILD_OUT_PATH, "rw"),
        },
        mem_limit=f"{cst.QUANT_DECODE_MIN_MEMORY_GB}g",
        nano_cpus=DECODE_CPUS * 1_000_000_000,
        pids_limit=DECODE_PIDS_LIMIT,
        timeout_seconds=DECODE_TIMEOUT_SECONDS,
        user=sandbox_user(),
    )


def run_phase_mem_limit_bytes(bf16_model_bytes: int) -> int:
    """Memory cap for the run phase: DECODE_MEMORY_MULTIPLIER x the decoded bf16 size, never below the floor."""
    if bf16_model_bytes <= 0:
        raise ValueError(f"bf16_model_bytes must be positive, got {bf16_model_bytes}")
    floor_bytes = cst.QUANT_DECODE_MIN_MEMORY_GB * 1024**3
    return max(floor_bytes, math.ceil(DECODE_MEMORY_MULTIPLIER * bf16_model_bytes))


def run_phase_spec(
    artifact_dir: Path, decoder_dir: Path, output_dir: Path, bf16_model_bytes: int, build_dir: Path | None = None
) -> DecodeContainerSpec:
    """Run phase: artifact, decoder source and build output read-only, an empty output dir; no compiler in the image.

    The build output is mounted read-only at /build rather than copied, so the run phase cannot modify it.
    """
    build_mount = _bind(build_dir, cst.DECODE_BUILD_OUT_PATH, "ro") if build_dir is not None else {}
    return DecodeContainerSpec(
        phase=DecodePhase.RUN,
        image=cst.QUANT_DECODER_RUN_IMAGE,
        command=[
            "python",
            f"{cst.DECODE_DECODER_PATH}/{DECODER_ENTRYPOINT}",
            cst.DECODE_ARTIFACT_PATH,
            cst.DECODE_OUTPUT_PATH,
        ],
        volumes={
            **_bind(artifact_dir, cst.DECODE_ARTIFACT_PATH, "ro"),
            **_bind(decoder_dir, cst.DECODE_DECODER_PATH, "ro"),
            **_bind(output_dir, cst.DECODE_OUTPUT_PATH, "rw"),
            **build_mount,
        },
        mem_limit=str(run_phase_mem_limit_bytes(bf16_model_bytes)),
        nano_cpus=DECODE_CPUS * 1_000_000_000,
        pids_limit=DECODE_PIDS_LIMIT,
        timeout_seconds=DECODE_TIMEOUT_SECONDS,
        user=sandbox_user(),
    )


def sanitize_failure(text: str) -> str:
    printable = "".join(c if c.isprintable() else " " for c in text)
    return " ".join(printable.split())[:FAILURE_DETAIL_MAX_CHARS]


def build_phase_warning(outcome: DecodeOutcome, build_out: Path, ran_build_script: bool) -> str | None:
    """Why a build that exited 0 probably did not build, or None.

    A build.sh can print an error, swallow it and exit 0. That is never a scored fail, but it must be
    visible: an empty /build or a failure line in the build log becomes a warning carried to the eval record.
    """
    if not outcome.ok or not ran_build_script:
        return None
    reasons = []
    if not any(build_out.iterdir()):
        reasons.append(f"left {cst.DECODE_BUILD_OUT_PATH} empty")
    failure_lines = [line for line in outcome.log_lines if BUILD_FAILURE_LINE.search(line)]
    if failure_lines:
        reasons.append(f"logged {failure_lines[-1]!r}")
    if not reasons:
        return None
    return sanitize_failure(f"{cst.DECODE_BUILD_SCRIPT} exited 0 but {' and '.join(reasons)}")


async def _exit_failure(container: Container, exit_code: int) -> str:
    logs = await asyncio.to_thread(container.logs, stdout=False, stderr=True, tail=cst.QUANT_DECODE_LOG_TAIL_LINES)
    stderr = logs.decode("utf-8", errors="ignore")
    lines = [line for line in stderr.splitlines() if line.strip()]
    if not lines:
        return f"exit {exit_code}"
    return sanitize_failure(f"exit {exit_code}: {lines[-1]}")


async def _log_lines(container: Container) -> list[str]:
    logs = await asyncio.to_thread(container.logs, stdout=True, stderr=True, tail=cst.QUANT_DECODE_LOG_TAIL_LINES)
    return [sanitize_failure(line) for line in logs.decode("utf-8", errors="ignore").splitlines() if line.strip()]


async def _remove(container: Container) -> None:
    try:
        await asyncio.to_thread(container.remove, force=True, v=True)
    except DockerException as e:
        logger.warning(f"Failed to remove decoder container {container.name}: {e}")


async def run_phase(client: docker.DockerClient, spec: DecodeContainerSpec) -> DecodeOutcome:
    """Run one sandboxed decoder phase to completion, killing it at the wall-clock cap.

    A non-ok outcome is a scored fail for the submission, never a training failure,
    so it must not count towards the majority-failure hold. Mapping the outcome onto
    task_nodes.quant_failure is the caller's job.
    """
    started = time.monotonic()
    container: Container | None = None

    def outcome(failure: str | None, exit_code: int | None = None, log_lines: list[str] | None = None) -> DecodeOutcome:
        return DecodeOutcome(
            phase=spec.phase,
            ok=failure is None,
            failure=failure,
            exit_code=exit_code,
            wall_seconds=time.monotonic() - started,
            log_lines=log_lines or [],
        )

    try:
        container = await asyncio.to_thread(
            client.containers.run,
            **spec.docker_run_kwargs(),
            detach=True,
            name=f"quant-decode-{spec.phase.value}-{uuid.uuid4().hex}",
        )
        result = await asyncio.wait_for(asyncio.to_thread(container.wait), timeout=spec.timeout_seconds)
        exit_code = result.get("StatusCode", -1)
        if exit_code != 0:
            return outcome(await _exit_failure(container, exit_code), exit_code)
        # Only the build log is read on success: the run phase sees the artifact, and nothing reads its output.
        log_lines = await _log_lines(container) if spec.phase == DecodePhase.BUILD else None
        return outcome(None, exit_code, log_lines)
    except asyncio.TimeoutError:
        try:
            await asyncio.to_thread(container.kill)
        except DockerException as e:
            logger.warning(f"Failed to kill timed-out decoder container {container.name}: {e}")
        return outcome("timeout")
    except DockerException as e:
        return outcome(f"container error: {sanitize_failure(str(e))}")
    finally:
        if container is not None:
            await _remove(container)
