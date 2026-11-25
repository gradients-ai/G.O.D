"""A job's containers mount only that job's directory of the checkpoints volume, and the shared cache read-only."""

import asyncio
import os
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import trainer.constants as cst
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.task_models import TaskType
from trainer import cleanup
from trainer import endpoints
from trainer import runtime
from trainer.containers import cache_cleanup
from trainer.job_checkpoints import JobCheckpointsError
from trainer.job_checkpoints import job_checkpoints_host_dir
from trainer.job_checkpoints import job_checkpoints_subdir
from trainer.job_checkpoints import job_checkpoints_volume
from trainer.job_checkpoints import mounts_whole_checkpoints_volume
from trainer.training_paths import get_checkpoints_output_path


TASK_ID = "0b1c2d3e-0000-4000-8000-000000000001"
HOTKEY_A = "5Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
HOTKEY_B = "5Bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
REPO = "repo-1"
CACHE_READ_ONLY = {"bind": cst.CACHE_ROOT_PATH, "mode": "ro"}


class FakeVolume:
    def __init__(self, mountpoint: Path):
        self.attrs = {"Mountpoint": str(mountpoint)}


class FakeContainer:
    name = "fake"
    status = "exited"

    def wait(self) -> dict:
        return {"StatusCode": 0}

    def logs(self) -> bytes:
        return b""

    def reload(self) -> None:
        return None

    def remove(self, force: bool = False, v: bool = False) -> None:
        return None


class FakeDockerClient:
    """Volumes live under root/<volume name>; every `containers.run` call is recorded."""

    def __init__(self, root: Path):
        self.root = root
        self.volumes = self
        self.containers = self
        self.runs: list[dict] = []

    def get(self, name: str) -> FakeVolume:
        return FakeVolume(self.root / name)

    def run(self, **kwargs) -> FakeContainer:
        self.runs.append(kwargs)
        return FakeContainer()


class FinishedJob:
    status = "success"
    started_at = None


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeDockerClient:
    fake = FakeDockerClient(tmp_path)
    (tmp_path / cst.CHECKPOINTS_VOLUME_NAME).mkdir()
    (tmp_path / cst.CACHE_VOLUME_NAME).mkdir()
    monkeypatch.setattr(runtime.docker, "from_env", lambda: fake)
    monkeypatch.setattr(runtime, "ensure_internal_network", lambda: None)
    monkeypatch.setattr(runtime, "stream_container_logs", lambda *args: None)
    return fake


def _job_dir(client: FakeDockerClient, hotkey: str) -> Path:
    return client.root / cst.CHECKPOINTS_VOLUME_NAME / TASK_ID / hotkey


def _expected_volumes(client: FakeDockerClient, hotkey: str) -> dict:
    return {
        str(_job_dir(client, hotkey)): {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"},
        cst.CACHE_VOLUME_NAME: CACHE_READ_ONLY,
    }


async def _run_text(hotkey: str) -> None:
    await runtime.run_trainer_container_text(
        task_id=TASK_ID,
        hotkey=hotkey,
        tag="tag",
        model="org/model",
        dataset="s3://bucket/train.json",
        dataset_type=InstructTextDatasetType(),
        task_type=TaskType.INSTRUCTTEXTTASK,
        file_format=FileFormat.S3,
        expected_repo_name=REPO,
        hours_to_complete=1.0,
    )


async def test_text_container_mounts_only_its_job_dir_and_a_read_only_cache(client: FakeDockerClient):
    await _run_text(HOTKEY_A)

    [run] = client.runs
    assert run["volumes"] == _expected_volumes(client, HOTKEY_A)
    assert cst.CHECKPOINTS_VOLUME_NAME not in run["volumes"]


async def test_image_container_mounts_only_its_job_dir_and_a_read_only_cache(client: FakeDockerClient):
    await runtime.run_trainer_container_image(
        task_id=TASK_ID,
        tag="tag",
        model="org/model",
        dataset_zip="s3://bucket/images.zip",
        model_type="flux",
        expected_repo_name=REPO,
        hours_to_complete=1.0,
        hotkey=HOTKEY_A,
    )

    [run] = client.runs
    assert run["volumes"] == _expected_volumes(client, HOTKEY_A)
    assert cst.CHECKPOINTS_VOLUME_NAME not in run["volumes"]


async def test_image_container_gets_no_hugging_face_token(client: FakeDockerClient, monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf-write-token")
    await runtime.run_trainer_container_image(
        task_id=TASK_ID,
        tag="tag",
        model="org/model",
        dataset_zip="s3://bucket/images.zip",
        model_type="flux",
        expected_repo_name=REPO,
        hours_to_complete=1.0,
        hotkey=HOTKEY_A,
    )

    [run] = client.runs
    assert not {"HUGGINGFACE_TOKEN", "HF_TOKEN"} & set(run["environment"])
    assert "hf-write-token" not in run["environment"].values()


async def test_two_jobs_of_one_task_share_no_writable_directory(client: FakeDockerClient):
    await _run_text(HOTKEY_A)
    await _run_text(HOTKEY_B)

    volume_root = client.root / cst.CHECKPOINTS_VOLUME_NAME
    sources = [
        Path(source)
        for run in client.runs
        for source, mount in run["volumes"].items()
        if mount["bind"] == cst.OUTPUT_CHECKPOINTS_PATH
    ]
    assert sources == [_job_dir(client, HOTKEY_A), _job_dir(client, HOTKEY_B)]
    first, second = sources
    assert first != second
    assert first not in second.parents and second not in first.parents
    for source in sources:
        assert source != volume_root and volume_root in source.parents
    # Every other mount of a training container is read-only.
    for run in client.runs:
        assert [mount["mode"] for mount in run["volumes"].values() if mount["bind"] != cst.OUTPUT_CHECKPOINTS_PATH] == ["ro"]


async def test_upload_container_reads_the_same_job_dir_at_the_same_container_path(client: FakeDockerClient, monkeypatch):
    async def _noop(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(runtime, "log_task", _noop)

    await runtime.upload_repo_to_hf(
        task_id=TASK_ID,
        hotkey=HOTKEY_A,
        expected_repo_name=REPO,
        huggingface_token="token",
        huggingface_username="user",
        model="org/model",
    )
    await asyncio.sleep(0)

    [run] = client.runs
    assert run["image"] == cst.HF_UPLOAD_DOCKER_IMAGE
    assert run["volumes"] == _expected_volumes(client, HOTKEY_A)
    assert run["environment"]["LOCAL_FOLDER"] == get_checkpoints_output_path(TASK_ID, REPO)
    assert run["environment"]["WANDB_LOGS_PATH"] == f"{cst.WANDB_LOGS_DIR}/{TASK_ID}_{HOTKEY_A}"


def test_job_dir_is_per_task_and_hotkey():
    assert job_checkpoints_subdir(TASK_ID, HOTKEY_A) == Path(TASK_ID) / HOTKEY_A
    assert job_checkpoints_subdir(TASK_ID, HOTKEY_A) != job_checkpoints_subdir(TASK_ID, HOTKEY_B)


@pytest.mark.parametrize("bad", ["", ".", "..", "../other", "a/b", "/abs", "a:b", ".hidden", "a b", "a\x00b"])
def test_names_that_are_not_one_plain_path_component_are_refused(bad: str):
    with pytest.raises(JobCheckpointsError):
        job_checkpoints_subdir(bad, HOTKEY_A)
    with pytest.raises(JobCheckpointsError):
        job_checkpoints_subdir(TASK_ID, bad)


@pytest.mark.parametrize("linked", ["task", "job"])
def test_symlinked_job_or_task_dir_is_refused(client: FakeDockerClient, tmp_path: Path, linked: str):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    job_dir = _job_dir(client, HOTKEY_A)
    if linked == "task":
        job_dir.parent.symlink_to(elsewhere, target_is_directory=True)
    else:
        job_dir.parent.mkdir()
        job_dir.symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(JobCheckpointsError, match="symlink"):
        job_checkpoints_volume(client, TASK_ID, HOTKEY_A)


def test_task_dir_that_is_a_file_is_refused(client: FakeDockerClient):
    _job_dir(client, HOTKEY_A).parent.write_text("not a directory")

    with pytest.raises(JobCheckpointsError, match="the task directory .* is not a directory"):
        job_checkpoints_volume(client, TASK_ID, HOTKEY_A)


def test_task_dir_that_cannot_be_inspected_is_refused_with_a_clear_error(client: FakeDockerClient, monkeypatch):
    def denied(path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "lstat", denied)

    with pytest.raises(JobCheckpointsError, match="the task directory .* cannot be inspected \\(PermissionError\\)"):
        job_checkpoints_volume(client, TASK_ID, HOTKEY_A)


def test_volume_without_an_absolute_host_path_is_refused():
    relative = FakeDockerClient(Path("volumes"))

    with pytest.raises(JobCheckpointsError, match="absolute"):
        job_checkpoints_volume(relative, TASK_ID, HOTKEY_A)


def _inspected(mounts: list[dict]) -> MagicMock:
    return MagicMock(attrs={"Mounts": mounts})


def test_only_a_mount_of_the_checkpoints_volume_itself_counts_as_the_whole_volume(client: FakeDockerClient):
    cache = {"Type": "volume", "Name": cst.CACHE_VOLUME_NAME, "Destination": cst.CACHE_ROOT_PATH}
    whole = {"Type": "volume", "Name": cst.CHECKPOINTS_VOLUME_NAME, "Destination": "/app/checkpoints"}
    own = {"Type": "bind", "Source": str(_job_dir(client, HOTKEY_A)), "Destination": "/app/checkpoints"}

    assert mounts_whole_checkpoints_volume(_inspected([whole, cache]))
    assert not mounts_whole_checkpoints_volume(_inspected([own, cache]))
    assert not mounts_whole_checkpoints_volume(_inspected([cache]))
    assert not mounts_whole_checkpoints_volume(MagicMock(attrs={}))


@pytest.mark.parametrize(("task_id", "hotkey"), [("../cache", HOTKEY_A), (TASK_ID, "a/b"), ("", HOTKEY_A), (TASK_ID, "..")])
async def test_training_request_with_a_path_like_name_is_refused(monkeypatch, task_id: str, hotkey: str):
    async def unexpected(req) -> None:
        raise AssertionError("job accepted")

    monkeypatch.setattr(endpoints, "_start_task_unlocked", unexpected)
    request = MagicMock(hotkey=hotkey)
    request.training_data.task_id = task_id

    with pytest.raises(HTTPException) as refused:
        await endpoints.start_training(request)

    assert refused.value.status_code == 400


def test_job_dir_need_not_exist_yet(client: FakeDockerClient):
    job_dir = job_checkpoints_host_dir(client, TASK_ID, HOTKEY_A)

    assert job_dir == _job_dir(client, HOTKEY_A)
    assert not job_dir.exists()


def _age_tree(root: Path, hours: float) -> None:
    stamp = time.time() - hours * 3600
    for path in [root, *root.rglob("*")]:
        os.utime(path, (stamp, stamp))


def test_checkpoints_cleanup_removes_every_job_dir_of_a_finished_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(cache_cleanup, "CHECKPOINTS_DIR", tmp_path)
    old = (datetime.utcnow() - timedelta(hours=cache_cleanup.CUTOFF_HOURS + 1)).isoformat()
    for task_id in ("old-task", "live-task", "rewritten-task"):
        for hotkey in (HOTKEY_A, HOTKEY_B):
            (tmp_path / job_checkpoints_subdir(task_id, hotkey) / task_id / REPO).mkdir(parents=True)
        _age_tree(tmp_path / task_id, cache_cleanup.CUTOFF_HOURS + 1)
    # Written after the history was read: a job accepted since.
    (tmp_path / job_checkpoints_subdir("rewritten-task", HOTKEY_A) / "new-file").write_text("x")
    history = [
        {"training_data": {"task_id": "rewritten-task"}, "hotkey": HOTKEY_A, "finished_at": old},
        {"training_data": {"task_id": "old-task"}, "hotkey": HOTKEY_A, "finished_at": old},
        {"training_data": {"task_id": "old-task"}, "hotkey": HOTKEY_B, "finished_at": old},
        {"training_data": {"task_id": "live-task"}, "hotkey": HOTKEY_A, "finished_at": old},
        {"training_data": {"task_id": "live-task"}, "hotkey": HOTKEY_B, "finished_at": None},
    ]

    cache_cleanup.clean_checkpoints(history)

    assert not (tmp_path / "old-task").exists()
    assert (tmp_path / "rewritten-task" / HOTKEY_A / "new-file").exists()
    assert (tmp_path / job_checkpoints_subdir("live-task", HOTKEY_A)).is_dir()
    assert (tmp_path / job_checkpoints_subdir("live-task", HOTKEY_B)).is_dir()


@pytest.mark.parametrize("task_id", ["../outside", "..", ".", "", "/abs", "a/../../outside", None, 7])
def test_checkpoints_cleanup_never_deletes_for_a_task_id_that_is_not_a_plain_name(monkeypatch, tmp_path: Path, task_id):
    checkpoints = tmp_path / "checkpoints"
    outside = tmp_path / "outside"
    for directory in (checkpoints, outside):
        directory.mkdir()
        (directory / "keep").write_text("x")
        _age_tree(directory, cache_cleanup.CUTOFF_HOURS + 1)
    monkeypatch.setattr(cache_cleanup, "CHECKPOINTS_DIR", checkpoints)
    old = (datetime.utcnow() - timedelta(hours=cache_cleanup.CUTOFF_HOURS + 1)).isoformat()

    cache_cleanup.clean_checkpoints([{"training_data": {"task_id": task_id}, "hotkey": HOTKEY_A, "finished_at": old}])

    assert (checkpoints / "keep").exists()
    assert (outside / "keep").exists()


async def test_cleanup_container_mounts_the_volumes_where_the_cleaner_deletes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class StopLoop(Exception):
        pass

    async def stop(seconds: float) -> None:
        raise StopLoop

    history_file = tmp_path / "task_history.json"
    history_file.write_text("[]")
    fake = FakeDockerClient(tmp_path)
    monkeypatch.setattr(cst, "TASKS_FILE_PATH", str(history_file))
    monkeypatch.setattr(cleanup.docker, "from_env", lambda: fake)
    monkeypatch.setattr(cleanup, "stream_container_logs", lambda *args: None)
    monkeypatch.setattr(cleanup, "remove_stale_trainer_containers", lambda client: None)
    monkeypatch.setattr(cleanup, "prune_build_leftovers", lambda: None)
    monkeypatch.setattr(cleanup.asyncio, "sleep", stop)
    monkeypatch.setattr(cleanup, "task_history", [FinishedJob()])

    with pytest.raises(StopLoop):
        await cleanup.periodically_cleanup_tasks_and_cache()

    [run] = fake.runs
    assert run["image"] == cst.CACHE_CLEANER_DOCKER_IMAGE
    assert run["volumes"][cst.CHECKPOINTS_VOLUME_NAME] == {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"}
    assert run["volumes"][cst.CACHE_VOLUME_NAME] == {"bind": cst.CACHE_ROOT_PATH, "mode": "rw"}
    assert Path(cst.OUTPUT_CHECKPOINTS_PATH) == cache_cleanup.CHECKPOINTS_DIR
    assert Path(cst.CACHE_ROOT_PATH) in cache_cleanup.CACHE_MODELS_DIR.parents

