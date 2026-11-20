"""Hand a training container's output dir back to the trainer process before the trainer prunes it.

Training containers run as root with every capability dropped, so everything a miner writes under the checkpoints
volume is root-owned: 0o755 dirs by default, or any mode the miner chose (0o000 included). A non-root trainer can read
most of it but can delete nothing in it, and the quant pre-upload check must delete every unlisted file.

`reclaim_output_dir` runs one short container with a fixed command: `chown -R` the output dir to the trainer's
uid:gid, then `chmod -R u+rwX`, so the trainer owns and can traverse, check and prune all of it. Only the
checkpoints volume is mounted (by name, so no host path is ever resolved through a miner's symlink) and the root
filesystem is read-only. Symlinks are never followed: the output dir and its task dir must not be symlinks, and the
recursive chown/chmod change links themselves or skip them. chown also clears any setuid/setgid bit a miner set.
"""

import asyncio
import uuid
from pathlib import Path

import docker
from docker.errors import DockerException
from docker.models.containers import Container
from pydantic import BaseModel

import trainer.constants as cst
from trainer.quant_decode.sandbox import sandbox_user
from trainer.quant_decode.sandbox import sanitize_failure
from trainer.telemetry import logger
from trainer.training_paths import get_checkpoints_output_path


SYMLINK_EXIT_CODE = 3
SYMLINKED_OUTPUT_FAILURE = "output dir or its task dir is a symlink"
# $1 = output dir (container path), $2 = uid:gid. Exit 0 without changes when the dir does not exist: the manifest
# check then rejects it as unfinished.
RECLAIM_SCRIPT = (
    'set -eu; out="$1"; owner="$2"; '
    'for p in "$(dirname "$out")" "$out"; do if [ -L "$p" ]; then exit ' + str(SYMLINK_EXIT_CODE) + "; fi; done; "
    '[ -d "$out" ] || exit 0; '
    'chown -R -P "$owner" "$out"; '
    'chmod -R u+rwX "$out"'
)


class OutputReclaimError(Exception):
    """Ours (image missing, daemon down, chown failed): the output was not reclaimed and must not be uploaded."""


class ReclaimContainerSpec(BaseModel):
    """Everything `client.containers.run` needs for the reclaim container."""

    image: str = cst.OUTPUT_RECLAIM_DOCKER_IMAGE
    entrypoint: list[str]
    command: list[str]
    volumes: dict[str, dict[str, str]]
    user: str = "0"
    cap_drop: list[str] = ["ALL"]
    cap_add: list[str] = cst.OUTPUT_RECLAIM_CAPS
    security_opt: list[str] = ["no-new-privileges"]
    network_disabled: bool = True
    read_only: bool = True
    mem_limit: str = cst.OUTPUT_RECLAIM_MEM_LIMIT
    pids_limit: int = cst.OUTPUT_RECLAIM_PIDS_LIMIT

    def docker_run_kwargs(self) -> dict:
        return self.model_dump()


def reclaim_spec(
    task_id: str, repo_name: str, owner: str, checkpoints_source: str = cst.CHECKPOINTS_VOLUME_NAME
) -> ReclaimContainerSpec:
    """checkpoints_source is the checkpoints volume name on a trainer; tests pass a host dir instead."""
    output_path = str(Path(get_checkpoints_output_path(task_id, repo_name)))
    return ReclaimContainerSpec(
        entrypoint=["/bin/sh", "-c", RECLAIM_SCRIPT, "reclaim"],
        command=[output_path, owner],
        volumes={checkpoints_source: {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"}},
    )


async def _remove(container: Container) -> None:
    try:
        await asyncio.to_thread(container.remove, force=True, v=True)
    except DockerException as e:
        logger.warning(f"Failed to remove output reclaim container {container.name}: {e}")


async def _run(client: docker.DockerClient, spec: ReclaimContainerSpec) -> tuple[int, Container]:
    container = await asyncio.to_thread(
        client.containers.run, **spec.docker_run_kwargs(), detach=True, name=f"output-reclaim-{uuid.uuid4().hex}"
    )
    try:
        result = await asyncio.wait_for(asyncio.to_thread(container.wait), timeout=cst.OUTPUT_RECLAIM_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        try:
            await asyncio.to_thread(container.kill)
        except DockerException as e:
            logger.warning(f"Failed to kill timed-out output reclaim container {container.name}: {e}")
        await _remove(container)
        raise OutputReclaimError(f"output reclaim timed out after {cst.OUTPUT_RECLAIM_TIMEOUT_SECONDS}s") from None
    return result.get("StatusCode", -1), container


async def reclaim_output_dir(
    client: docker.DockerClient,
    *,
    task_id: str,
    repo_name: str,
    owner: str | None = None,
    checkpoints_source: str = cst.CHECKPOINTS_VOLUME_NAME,
) -> str | None:
    """chown a finished training job's output dir to the trainer (owner defaults to its uid:gid).

    Returns a rejection reason when the miner made the output dir or its task dir a symlink, else None.
    Raises OutputReclaimError when the container could not run or did not finish cleanly.
    """
    spec = reclaim_spec(task_id, repo_name, owner or sandbox_user(), checkpoints_source)
    try:
        exit_code, container = await _run(client, spec)
    except DockerException as e:
        raise OutputReclaimError(f"output reclaim container error: {sanitize_failure(str(e))}") from None
    try:
        if exit_code == SYMLINK_EXIT_CODE:
            return SYMLINKED_OUTPUT_FAILURE
        if exit_code != 0:
            logs = await asyncio.to_thread(container.logs, stdout=True, stderr=True, tail=cst.OUTPUT_RECLAIM_LOG_TAIL_LINES)
            detail = sanitize_failure(logs.decode("utf-8", errors="ignore"))
            raise OutputReclaimError(f"output reclaim exited {exit_code}: {detail}")
        return None
    finally:
        await _remove(container)
