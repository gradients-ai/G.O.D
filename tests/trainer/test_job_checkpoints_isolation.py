"""A job's containers mount only that job's directory of the checkpoints volume, and the shared cache read-only."""

import asyncio
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pytest

import trainer.constants as cst
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.task_models import TaskType
from trainer import cleanup
from trainer import runtime
from trainer.containers import cache_cleanup
from trainer.job_checkpoints import JobCheckpointsError
from trainer.job_checkpoints import job_checkpoints_host_dir
from trainer.job_checkpoints import job_checkpoints_subdir
from trainer.job_checkpoints import job_checkpoints_volume
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


def test_job_dir_need_not_exist_yet(client: FakeDockerClient):
    job_dir = job_checkpoints_host_dir(client, TASK_ID, HOTKEY_A)

    assert job_dir == _job_dir(client, HOTKEY_A)
    assert not job_dir.exists()


def test_checkpoints_cleanup_removes_every_job_dir_of_a_finished_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(cache_cleanup, "CHECKPOINTS_DIR", tmp_path)
    old = (datetime.utcnow() - timedelta(hours=cache_cleanup.CUTOFF_HOURS + 1)).isoformat()
    for task_id in ("old-task", "live-task"):
        for hotkey in (HOTKEY_A, HOTKEY_B):
            (tmp_path / job_checkpoints_subdir(task_id, hotkey) / task_id / REPO).mkdir(parents=True)
    history = [
        {"training_data": {"task_id": "old-task"}, "hotkey": HOTKEY_A, "finished_at": old},
        {"training_data": {"task_id": "old-task"}, "hotkey": HOTKEY_B, "finished_at": old},
        {"training_data": {"task_id": "live-task"}, "hotkey": HOTKEY_A, "finished_at": old},
        {"training_data": {"task_id": "live-task"}, "hotkey": HOTKEY_B, "finished_at": None},
    ]

    cache_cleanup.clean_checkpoints(history)

    assert not (tmp_path / "old-task").exists()
    assert (tmp_path / job_checkpoints_subdir("live-task", HOTKEY_A)).is_dir()
    assert (tmp_path / job_checkpoints_subdir("live-task", HOTKEY_B)).is_dir()


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

