"""Each training job's own directory on the checkpoints volume.

A job's containers (training, output reclaim, upload) mount only `<checkpoints volume>/<task_id>/<hotkey>` at
OUTPUT_CHECKPOINTS_PATH, never the volume itself, so a job cannot read or change what another job wrote. Paths inside
the container are the same for every job: the output is `/app/checkpoints/<task_id>/<expected_repo_name>` and the wandb
files are under `/app/checkpoints/wandb_logs`.

The directory is bind-mounted by host path. The Docker daemon creates it on first use, owned by root like everything a
training container writes. The trainer chooses both path components and no training container can write to either of
their parents; a component that is a symlink is still refused, so the bind never resolves outside the volume.
"""

import os
import stat
from pathlib import Path

import docker
from docker.models.containers import Container

import trainer.constants as cst
from trainer.path_names import is_plain_path_component


class JobCheckpointsError(Exception):
    """The job's checkpoints directory cannot be named, or is not a plain directory under the volume."""


def job_checkpoints_subdir(task_id: str, hotkey: str) -> Path:
    """The job's directory relative to the checkpoints volume root."""
    for name, value in (("task_id", task_id), ("hotkey", hotkey)):
        if not is_plain_path_component(value):
            raise JobCheckpointsError(f"{name} is not usable as a directory name")
    return Path(task_id) / hotkey


def _refuse_unless_plain_dir_or_absent(path: Path, what: str) -> None:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    except OSError as e:
        raise JobCheckpointsError(f"{what} on the checkpoints volume cannot be inspected ({type(e).__name__})") from None
    if stat.S_ISLNK(mode):
        raise JobCheckpointsError(f"{what} on the checkpoints volume is a symlink")
    if not stat.S_ISDIR(mode):
        raise JobCheckpointsError(f"{what} on the checkpoints volume is not a directory")


def job_checkpoints_host_dir(client: docker.DockerClient, task_id: str, hotkey: str) -> Path:
    """Host path of the job's directory on the checkpoints volume. It may not exist yet."""
    mountpoint = Path(client.volumes.get(cst.CHECKPOINTS_VOLUME_NAME).attrs["Mountpoint"])
    if not mountpoint.is_absolute():
        raise JobCheckpointsError("the checkpoints volume has no absolute host path")
    job_dir = mountpoint / job_checkpoints_subdir(task_id, hotkey)
    _refuse_unless_plain_dir_or_absent(job_dir.parent, "the task directory")
    _refuse_unless_plain_dir_or_absent(job_dir, "the job directory")
    return job_dir


def job_checkpoints_volume(client: docker.DockerClient, task_id: str, hotkey: str) -> dict[str, dict[str, str]]:
    """docker-py `volumes` entry that mounts only this job's directory, read-write, at OUTPUT_CHECKPOINTS_PATH."""
    job_dir = job_checkpoints_host_dir(client, task_id, hotkey)
    return {str(job_dir): {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"}}


def mounts_whole_checkpoints_volume(container: Container) -> bool:
    """True for a container that mounts the checkpoints volume itself, and so sees every job's directory."""
    return any(
        mount.get("Type") == "volume" and mount.get("Name") == cst.CHECKPOINTS_VOLUME_NAME
        for mount in container.attrs.get("Mounts") or []
    )
