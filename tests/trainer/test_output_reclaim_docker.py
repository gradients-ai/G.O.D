"""Output reclaim against real containers: a miner's root `--cap-drop ALL` training container writes the output, the
reclaim container (the real uploader image) hands it to the trainer, and the quant pre-upload check can prune it.

Skipped without a docker daemon, alpine (the stand-in miner image) or the uploader image.
"""

import json
import os
import stat
from pathlib import Path

import docker
import pytest
from docker.errors import DockerException

import trainer.constants as cst
from core.constants.quant import QUANT_MANIFEST_NAME
from core.models.task_models import TaskType
from trainer.output_reclaim import SYMLINKED_OUTPUT_FAILURE
from trainer.output_reclaim import reclaim_output_dir
from trainer.quant_upload_check import check_quant_output


ALPINE = "alpine:3.20"
TASK_ID = "task-1"
REPO = "repo-1"
OUT = f"/app/checkpoints/{TASK_ID}/{REPO}"
MANIFEST = {"version": 1, "task_id": TASK_ID, "track": "freeform", "files": [{"name": "model.safetensors", "bytes": 3}]}
# What a miner can leave behind as capability-less root: nested dirs, a 0o000 dir, a setuid file, a symlink out.
MINER_SCRIPT = (
    f"mkdir -p {OUT}/checkpoint-100/deep {OUT}/locked && printf abc > {OUT}/model.safetensors"
    f" && printf '%s' \"$MANIFEST\" > {OUT}/{QUANT_MANIFEST_NAME}"
    f" && touch {OUT}/checkpoint-100/deep/optimizer.pt {OUT}/locked/x {OUT}/suid"
    f" && chmod 4755 {OUT}/suid && chmod 0 {OUT}/locked && ln -s /etc {OUT}/etc"
)


@pytest.fixture(scope="module")
def client() -> docker.DockerClient:
    try:
        client = docker.from_env()
        client.ping()
    except DockerException as e:
        pytest.skip(f"no docker daemon: {e}")
    for image in (ALPINE, cst.OUTPUT_RECLAIM_DOCKER_IMAGE):
        try:
            client.images.get(image)
        except DockerException:
            pytest.skip(f"{image} not present")
    return client


def _alpine(client: docker.DockerClient, checkpoints: Path, script: str, **kwargs) -> None:
    client.containers.run(
        ALPINE,
        ["sh", "-c", script],
        user="0",
        security_opt=["no-new-privileges"],
        network_disabled=True,
        volumes={str(checkpoints): {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"}},
        remove=True,
        **kwargs,
    )


@pytest.fixture
def checkpoints(client, tmp_path: Path) -> Path:
    """Stands in for the volume's _data (root:docker 0o775 on a trainer): writable by the capless root miner."""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    checkpoints.chmod(0o777)
    yield checkpoints
    # The task dir stays root-owned; clear it with a root container, as the trainer's cleanup container does.
    _alpine(client, checkpoints, "rm -rf /app/checkpoints/*", cap_drop=["ALL"], cap_add=["DAC_OVERRIDE"])


def _miner_writes_output(client: docker.DockerClient, checkpoints: Path) -> Path:
    _alpine(client, checkpoints, MINER_SCRIPT, cap_drop=["ALL"], environment={"MANIFEST": json.dumps(MANIFEST)})
    return checkpoints / TASK_ID / REPO


def _check(output_dir: Path):
    return check_quant_output(output_dir, task_id=TASK_ID, task_type=TaskType.QUANTFREEFORMTASK, max_bytes=None)


@pytest.mark.skipif(os.geteuid() == 0, reason="a root trainer can prune root-owned files without the reclaim")
def test_without_the_reclaim_a_non_root_trainer_cannot_prune(client, checkpoints: Path):
    output_dir = _miner_writes_output(client, checkpoints)

    verdict = _check(output_dir)

    assert not verdict.ok
    assert "output not accessible to the trainer" in verdict.failure


async def test_reclaimed_output_is_owned_by_the_trainer_and_prunes_cleanly(client, checkpoints: Path):
    output_dir = _miner_writes_output(client, checkpoints)

    rejection = await reclaim_output_dir(client, task_id=TASK_ID, repo_name=REPO, checkpoints_source=str(checkpoints))
    me = (os.geteuid(), os.getegid())
    owners = {(p.lstat().st_uid, p.lstat().st_gid) for p in [output_dir, *output_dir.rglob("*")]}
    suid_mode = (output_dir / "suid").stat().st_mode
    verdict = _check(output_dir)

    assert rejection is None
    assert owners == {me}
    assert not suid_mode & stat.S_ISUID
    assert verdict.ok, verdict
    assert sorted(p.name for p in output_dir.iterdir()) == ["model.safetensors", QUANT_MANIFEST_NAME]


async def test_symlinked_output_dir_is_rejected_and_never_followed(client, checkpoints: Path):
    _alpine(
        client,
        checkpoints,
        f"mkdir -p /app/checkpoints/{TASK_ID} /app/checkpoints/victim && touch /app/checkpoints/victim/f"
        f" && ln -s /app/checkpoints/victim {OUT}",
        cap_drop=["ALL"],
    )

    rejection = await reclaim_output_dir(client, task_id=TASK_ID, repo_name=REPO, checkpoints_source=str(checkpoints))

    assert rejection == SYMLINKED_OUTPUT_FAILURE
    assert (checkpoints / "victim" / "f").stat().st_uid == 0


async def test_missing_output_dir_is_left_to_the_manifest_check(client, checkpoints: Path):
    rejection = await reclaim_output_dir(client, task_id=TASK_ID, repo_name=REPO, checkpoints_source=str(checkpoints))

    assert rejection is None
    assert "quant_manifest.json missing" in _check(checkpoints / TASK_ID / REPO).failure
