"""Each training job's own directory on the checkpoints volume.

A job's containers (training, output reclaim, upload) mount only `<checkpoints volume>/<task_id>/<hotkey>` at
OUTPUT_CHECKPOINTS_PATH, never the volume itself, so a job cannot read or change what another job wrote. Paths inside
the container are the same for every job: the output is `/app/checkpoints/<task_id>/<expected_repo_name>` and the wandb
files are under `/app/checkpoints/wandb_logs`.

The directory is bind-mounted by host path. The Docker daemon creates it on first use, owned by root like everything a
training container writes. The trainer chooses both path components and no training container can write to either of
their parents; a component that is a symlink is still refused, so the bind never resolves outside the volume.
"""

import re
from pathlib import Path

import docker

import trainer.constants as cst


# One path component, and safe in docker's "source:target:mode" bind form: no separator, no colon, not "." or "..".
_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")


class JobCheckpointsError(Exception):
    """The job's checkpoints directory cannot be named, or is not a plain directory under the volume."""


def job_checkpoints_subdir(task_id: str, hotkey: str) -> Path:
    """The job's directory relative to the checkpoints volume root."""
    for name, value in (("task_id", task_id), ("hotkey", hotkey)):
        if not _PATH_COMPONENT.fullmatch(value):
            raise JobCheckpointsError(f"{name} is not usable as a directory name")
    return Path(task_id) / hotkey


def job_checkpoints_host_dir(client: docker.DockerClient, task_id: str, hotkey: str) -> Path:
    """Host path of the job's directory on the checkpoints volume. It may not exist yet."""
    mountpoint = Path(client.volumes.get(cst.CHECKPOINTS_VOLUME_NAME).attrs["Mountpoint"])
    job_dir = mountpoint / job_checkpoints_subdir(task_id, hotkey)
    for path in (job_dir.parent, job_dir):
        if path.is_symlink():
            raise JobCheckpointsError("the job's checkpoints directory or its task directory is a symlink")
    return job_dir


def job_checkpoints_volume(client: docker.DockerClient, task_id: str, hotkey: str) -> dict[str, dict[str, str]]:
    """docker-py `volumes` entry that mounts only this job's directory, read-write, at OUTPUT_CHECKPOINTS_PATH."""
    job_dir = job_checkpoints_host_dir(client, task_id, hotkey)
    return {str(job_dir): {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"}}
