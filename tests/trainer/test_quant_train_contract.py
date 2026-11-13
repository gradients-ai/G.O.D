"""Trainer contract for quant jobs: --quant-spec reaches the miner container, the output is uploaded verbatim only with a
valid quant_manifest.json, and a freeform decode never fails training."""

import asyncio
import json
import threading
from pathlib import Path

import pytest

import core.constants as core_cst
import trainer.constants as trainer_cst
from core.constants.quant import QUANT_LARGE_FILE_REVIEW_BYTES
from core.constants.quant import QUANT_MANIFEST_NAME
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestQuant
from core.models.payload_models import TrainRequestText
from core.models.quant_artifact_models import QuantDecodeRequest
from core.models.quant_artifact_models import QuantDecodeResult
from core.models.quant_models import QuantCalibrationReference
from core.models.quant_models import QuantCalibrationRequest
from core.models.quant_models import QuantFormat
from core.models.quant_models import QuantTarget
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from trainer import cleanup
from trainer import runtime
from trainer.model_artifacts import get_anonymous_model_dir
from trainer.output_reclaim import OutputReclaimError


SPEC = QuantFormat.W4A16_SYM_G128.spec()
REAL_UPLOAD_REPO_TO_HF = runtime.upload_repo_to_hf
DECODE = QuantDecodeRequest(
    decoder_snapshot_url="https://minio.example/decoder.tar.xz?sig=x",
    decoder_sha256="a" * 64,
    bucket="decoded",
    object_prefix="quant-decoded/task-1/hk",
)


FREEFORM_TARGET = QuantTarget(
    reference_model="org/own-bf16",
    reference_loss=2.0,
    reference_sku="h100-sxm",
    bar_cap=0.04,
    bar=2.08,
    knee=1.04,
    knee_loss=2.08,
    penalty_curvature=2500.0,
    cap=1.08,
    cap_loss=2.16,
)


FIXED_TARGET = QuantTarget(reference_model="org/own-bf16", reference_loss=2.0, reference_sku="h100-sxm")
CALIBRATION = QuantCalibrationRequest(
    examples_url="https://minio.example/cal.jsonl?sig=secret",
    eval_set_fingerprint="fp",
    n_examples=3,
    reference=QuantCalibrationReference(model="org/own-bf16", loss=2.0, per_example_losses=[1.5, None, 2.5], sku="h100-sxm"),
)
CALIBRATION_ENV = {core_cst.QUANT_CALIBRATION_PATH_ENV: "/cache/quant_calibration_task-1_hk.json"}


def _quant_request(
    task_type: TaskType, quant_spec=None, decode=None, hours_to_complete: float = 1.0, quant_target=None, calibration=CALIBRATION
) -> TrainerProxyRequest:
    if quant_target is None:
        quant_target = FREEFORM_TARGET if task_type == TaskType.QUANTFREEFORMTASK else FIXED_TARGET
    training_data = TrainRequestQuant(
        model="org/model",
        task_id="task-1",
        hours_to_complete=hours_to_complete,
        expected_repo_name="repo-1",
        dataset="s3://bucket/train.json",
        file_format=FileFormat.S3,
        dataset_type=InstructTextDatasetType(),
        task_type=task_type,
        quant_spec=quant_spec,
        decode=decode,
        quant_target=quant_target,
        calibration=calibration,
    )
    return TrainerProxyRequest(training_data=training_data, github_repo="https://github.com/org/repo", gpu_ids=[0], hotkey="hk")


def _instruct_request() -> TrainerProxyRequest:
    training_data = TrainRequestText(
        model="org/model",
        task_id="task-1",
        hours_to_complete=1.0,
        expected_repo_name="repo-1",
        dataset="s3://bucket/train.json",
        file_format=FileFormat.S3,
        dataset_type=InstructTextDatasetType(),
    )
    return TrainerProxyRequest(training_data=training_data, github_repo="https://github.com/org/repo", gpu_ids=[0], hotkey="hk")


def _write_quant_output(mountpoint: Path, track: str = "fixed") -> None:
    output_dir = mountpoint / "task-1" / "repo-1"
    output_dir.mkdir(parents=True)
    (output_dir / "model.safetensors").write_bytes(b"x" * 8)
    manifest = {"version": 1, "task_id": "task-1", "track": track, "files": [{"name": "model.safetensors", "bytes": 8}]}
    (output_dir / QUANT_MANIFEST_NAME).write_text(json.dumps(manifest))


class FakeTrainContainer:
    name = "text-trainer-fake"

    def wait(self) -> dict:
        return {"StatusCode": 0}

    def logs(self) -> bytes:
        return b""


class HangingTrainContainer(FakeTrainContainer):
    def __init__(self):
        self.released = threading.Event()

    def wait(self) -> dict:
        self.released.wait(timeout=10)
        return {"StatusCode": 0}


async def _release_orphaned_wait(container: HangingTrainContainer) -> None:
    """The runtime leaves container.wait running on timeout; release it and await its task before the loop closes."""
    container.released.set()
    orphans = asyncio.all_tasks() - {asyncio.current_task()}
    await asyncio.gather(*orphans)


class FakeVolume:
    def __init__(self, mountpoint: Path):
        self.attrs = {"Mountpoint": str(mountpoint)}


class FakeVolumes:
    def __init__(self, mountpoint: Path):
        self.mountpoint = mountpoint

    def get(self, name: str) -> FakeVolume:
        return FakeVolume(self.mountpoint)


class FakeDockerClient:
    def __init__(self, mountpoint: Path):
        self.volumes = FakeVolumes(mountpoint)
        self.run_kwargs: dict | None = None
        self.containers = self

    def run(self, **kwargs) -> FakeTrainContainer:
        self.run_kwargs = kwargs
        return FakeTrainContainer()


class Recorder:
    def __init__(self):
        self.text_container_kwargs: dict | None = None
        self.completed: list[bool] = []
        self.decode_calls: list[dict] = []
        self.stored: list[QuantDecodeResult] = []
        self.logs: list[str] = []
        self.decode_result = QuantDecodeResult(quant_failure="decoder: exit 1: boom")
        self.uploads: list[dict] = []
        self.train_container: FakeTrainContainer = FakeTrainContainer()
        self.docker_client: FakeDockerClient | None = None
        self.reclaims: list[dict] = []
        self.reclaim_outcome: str | Exception | None = None
        self.calibration_calls: list[tuple] = []
        self.calibration_failure: Exception | None = None
        self.image_builds: list[dict] = []


async def _noop(*args, **kwargs) -> None:
    return None


@pytest.fixture
def recorder(monkeypatch, tmp_path: Path) -> Recorder:
    rec = Recorder()

    async def run_text(**kwargs):
        rec.text_container_kwargs = kwargs
        return rec.train_container

    async def complete(task_id, hotkey, success=True):
        rec.completed.append(success)

    async def log(task_id, hotkey, message):
        rec.logs.append(message)

    async def decode(request, **kwargs):
        rec.decode_calls.append({"request": request, **kwargs})
        return rec.decode_result

    async def store(task_id, hotkey, result):
        rec.stored.append(result)

    async def upload(**kwargs):
        rec.uploads.append(kwargs)

    async def reclaim(client, **kwargs):
        rec.reclaims.append(kwargs)
        if isinstance(rec.reclaim_outcome, Exception):
            raise rec.reclaim_outcome
        return rec.reclaim_outcome

    async def write_calibration(task_id, hotkey, request, volume_dir):
        rec.calibration_calls.append((task_id, hotkey, request, volume_dir))
        if rec.calibration_failure:
            raise rec.calibration_failure
        return CALIBRATION_ENV

    rec.docker_client = FakeDockerClient(tmp_path)

    monkeypatch.setattr(runtime, "write_quant_calibration_file", write_calibration)
    monkeypatch.setattr(runtime, "create_volumes_if_dont_exist", _noop)
    monkeypatch.setattr(runtime, "get_dockerfile_path", lambda *args: "Dockerfile")
    monkeypatch.setattr(runtime, "run_downloader_container", lambda **kwargs: (0, None))
    def build_image(**kwargs):
        rec.image_builds.append(kwargs)
        return "tag", None

    monkeypatch.setattr(runtime, "build_docker_image", build_image)
    monkeypatch.setattr(runtime, "delete_image_and_cleanup", lambda tag: None)
    monkeypatch.setattr(runtime, "run_trainer_container_text", run_text)
    monkeypatch.setattr(runtime, "update_container_name", _noop)
    monkeypatch.setattr(runtime, "update_training_started_at", _noop)
    monkeypatch.setattr(runtime, "upload_repo_to_hf", upload)
    monkeypatch.setattr(runtime, "reclaim_output_dir", reclaim)
    monkeypatch.setattr(runtime, "log_task", log)
    monkeypatch.setattr(runtime, "complete_task", complete)
    monkeypatch.setattr(runtime, "run_freeform_decode", decode)
    monkeypatch.setattr(runtime, "update_quant_decode_result", store)
    monkeypatch.setattr(runtime.docker, "from_env", lambda: rec.docker_client)
    monkeypatch.setattr(runtime, "stream_container_logs", lambda *args: None)
    monkeypatch.setenv("HUGGINGFACE_USERNAME", "trainer-user")
    (tmp_path / "models" / "org--model").mkdir(parents=True)
    return rec


@pytest.fixture
def quant_output(tmp_path: Path) -> None:
    _write_quant_output(tmp_path)


@pytest.fixture
def freeform_output(tmp_path: Path) -> None:
    _write_quant_output(tmp_path, track="freeform")


async def test_fixed_task_passes_quant_spec_to_text_container(recorder: Recorder, quant_output):
    await runtime.start_training_task(_quant_request(TaskType.QUANTFIXEDTASK, quant_spec=SPEC), "/repo")

    assert recorder.text_container_kwargs["quant_spec"] == SPEC
    assert recorder.text_container_kwargs["task_type"] == TaskType.QUANTFIXEDTASK
    assert recorder.completed == [True]
    assert recorder.decode_calls == []


async def test_text_container_gets_quant_spec_argument_and_env(monkeypatch, tmp_path: Path):
    client = FakeDockerClient(tmp_path)
    monkeypatch.setattr(runtime.docker, "from_env", lambda: client)
    monkeypatch.setattr(runtime, "ensure_internal_network", lambda: None)
    monkeypatch.setattr(runtime, "stream_container_logs", lambda *args: None)
    training_data = _quant_request(TaskType.QUANTFIXEDTASK, quant_spec=SPEC).training_data

    await runtime.run_trainer_container_text(
        task_id="task-1",
        hotkey="hk",
        tag="tag",
        model="org/model",
        dataset=training_data.dataset,
        dataset_type=training_data.dataset_type,
        task_type=TaskType.QUANTFIXEDTASK,
        file_format=training_data.file_format,
        expected_repo_name="repo-1",
        hours_to_complete=1.0,
        quant_spec=SPEC,
    )

    command = client.run_kwargs["command"]
    assert command[command.index("--task-type") + 1] == TaskType.QUANTFIXEDTASK
    assert command[command.index("--quant-spec") + 1] == SPEC.model_dump_json()
    assert client.run_kwargs["environment"][core_cst.QUANT_SPEC_ENV] == SPEC.model_dump_json()


async def test_text_container_without_quant_spec_has_no_quant_contract(monkeypatch, tmp_path: Path):
    client = FakeDockerClient(tmp_path)
    monkeypatch.setattr(runtime.docker, "from_env", lambda: client)
    monkeypatch.setattr(runtime, "ensure_internal_network", lambda: None)
    monkeypatch.setattr(runtime, "stream_container_logs", lambda *args: None)
    training_data = _quant_request(TaskType.QUANTFREEFORMTASK).training_data

    await runtime.run_trainer_container_text(
        task_id="task-1",
        hotkey="hk",
        tag="tag",
        model="org/model",
        dataset=training_data.dataset,
        dataset_type=training_data.dataset_type,
        task_type=TaskType.QUANTFREEFORMTASK,
        file_format=training_data.file_format,
        expected_repo_name="repo-1",
        hours_to_complete=1.0,
    )

    assert "--quant-spec" not in client.run_kwargs["command"]
    assert core_cst.QUANT_SPEC_ENV not in client.run_kwargs["environment"]
    assert "--quant-target" not in client.run_kwargs["command"]
    assert core_cst.QUANT_TARGET_ENV not in client.run_kwargs["environment"]


async def test_request_quant_target_reaches_the_text_container(recorder: Recorder, freeform_output):
    await runtime.start_training_task(_quant_request(TaskType.QUANTFREEFORMTASK, quant_target=FREEFORM_TARGET), "/repo")

    assert recorder.text_container_kwargs["quant_target"] == FREEFORM_TARGET


@pytest.mark.parametrize(
    "task_type,quant_spec,target,output",
    [
        (TaskType.QUANTFIXEDTASK, SPEC, FIXED_TARGET, "quant_output"),
        (TaskType.QUANTFREEFORMTASK, None, FREEFORM_TARGET, "freeform_output"),
    ],
)
async def test_every_quant_job_starts_with_its_target_and_the_calibration_reference(
    recorder: Recorder, request, task_type, quant_spec, target, output
):
    request.getfixturevalue(output)

    await runtime.start_training_task(_quant_request(task_type, quant_spec=quant_spec), "/repo")

    assert recorder.text_container_kwargs["quant_target"] == target
    assert recorder.text_container_kwargs["extra_environment"] == CALIBRATION_ENV
    assert [call[2] for call in recorder.calibration_calls] == [CALIBRATION]
    assert recorder.calibration_calls[0][2].reference is not None


async def test_text_container_gets_quant_target_env_and_no_flag(monkeypatch, tmp_path: Path):
    client = FakeDockerClient(tmp_path)
    monkeypatch.setattr(runtime.docker, "from_env", lambda: client)
    monkeypatch.setattr(runtime, "ensure_internal_network", lambda: None)
    monkeypatch.setattr(runtime, "stream_container_logs", lambda *args: None)
    training_data = _quant_request(TaskType.QUANTFREEFORMTASK, quant_target=FREEFORM_TARGET).training_data

    await runtime.run_trainer_container_text(
        task_id="task-1",
        hotkey="hk",
        tag="tag",
        model="org/own-bf16",
        dataset=training_data.dataset,
        dataset_type=training_data.dataset_type,
        task_type=TaskType.QUANTFREEFORMTASK,
        file_format=training_data.file_format,
        expected_repo_name="repo-1",
        hours_to_complete=1.0,
        quant_target=FREEFORM_TARGET,
    )

    assert "--quant-target" not in client.run_kwargs["command"]
    environment = client.run_kwargs["environment"]
    assert QuantTarget.model_validate_json(environment[core_cst.QUANT_TARGET_ENV]) == FREEFORM_TARGET


async def test_freeform_without_decode_request_skips_decode(recorder: Recorder, freeform_output):
    await runtime.start_training_task(_quant_request(TaskType.QUANTFREEFORMTASK), "/repo")

    assert recorder.completed == [True]
    assert recorder.decode_calls == []
    assert recorder.stored == []
    assert "freeform decode skipped: no decode request" in recorder.logs


async def test_freeform_decode_failure_keeps_training_success(recorder: Recorder, freeform_output, tmp_path: Path):
    await runtime.start_training_task(_quant_request(TaskType.QUANTFREEFORMTASK, decode=DECODE), "/repo")

    assert recorder.completed == [True]
    assert recorder.stored == [recorder.decode_result]
    call = recorder.decode_calls[0]
    assert call["request"] == DECODE
    assert call["artifact_repo"] == "trainer-user/repo-1"
    assert call["base_model_dir"] == tmp_path / "models" / "org--model"
    assert call["work_root"] == trainer_cst.QUANT_DECODE_WORK_ROOT
    assert not call["work_root"].is_relative_to(tmp_path)  # tmp_path is the fake cache volume mountpoint
    assert "freeform decode scored fail: decoder: exit 1: boom" in recorder.logs


async def test_freeform_decode_crash_is_stored_as_infra_error(recorder: Recorder, freeform_output, monkeypatch):
    async def crash(request, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(runtime, "run_freeform_decode", crash)

    await runtime.start_training_task(_quant_request(TaskType.QUANTFREEFORMTASK, decode=DECODE), "/repo")

    assert recorder.completed == [True]
    assert recorder.stored == [QuantDecodeResult(infra_error="disk full")]


async def test_failed_upload_skips_decode(recorder: Recorder, freeform_output, monkeypatch):
    async def failing_upload(**kwargs):
        raise RuntimeError("hf down")

    monkeypatch.setattr(runtime, "upload_repo_to_hf", failing_upload)

    await runtime.start_training_task(_quant_request(TaskType.QUANTFREEFORMTASK, decode=DECODE), "/repo")

    assert recorder.completed == [False]
    assert recorder.decode_calls == []


def test_decode_base_model_dir_uses_anonymised_dir_after_model_prep(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MODEL_HASH_SALT", "salt")
    (tmp_path / "models" / "org--model").mkdir(parents=True)
    training_data = _quant_request(TaskType.QUANTFREEFORMTASK).training_data
    prepped = training_data.model_copy(update={"baseline_stats": object()})

    assert runtime._decode_base_model_dir(training_data, tmp_path) == tmp_path / "models" / "org--model"
    assert runtime._decode_base_model_dir(prepped, tmp_path) == tmp_path / "models" / get_anonymous_model_dir("org/model")


def test_decode_base_model_dir_falls_back_to_anonymised_dir_without_baseline_stats(monkeypatch, tmp_path: Path):
    """task_history.json drops baseline_stats; a reloaded job must still find the prepped model."""
    monkeypatch.setenv("MODEL_HASH_SALT", "salt")
    training_data = _quant_request(TaskType.QUANTFREEFORMTASK).training_data

    assert runtime._decode_base_model_dir(training_data, tmp_path) == tmp_path / "models" / get_anonymous_model_dir("org/model")


async def test_quant_job_without_manifest_fails_without_upload_or_decode(recorder: Recorder, tmp_path: Path):
    output_dir = tmp_path / "task-1" / "repo-1"
    output_dir.mkdir(parents=True)
    (output_dir / "model.safetensors").write_bytes(b"x" * 8)

    await runtime.start_training_task(_quant_request(TaskType.QUANTFREEFORMTASK, decode=DECODE), "/repo")

    assert recorder.completed == [False]
    assert recorder.uploads == []
    assert recorder.decode_calls == []
    assert "[ERROR] quant output rejected: quant_manifest.json missing: the pipeline did not finish" in recorder.logs


async def test_quant_output_is_reclaimed_before_the_check(recorder: Recorder, quant_output):
    await runtime.start_training_task(_quant_request(TaskType.QUANTFIXEDTASK, quant_spec=SPEC), "/repo")

    assert recorder.reclaims == [{"task_id": "task-1", "repo_name": "repo-1"}]
    assert recorder.completed == [True]


async def test_instruct_job_is_never_reclaimed(recorder: Recorder):
    await runtime.start_training_task(_instruct_request(), "/repo")

    assert recorder.reclaims == []
    assert recorder.completed == [True]


async def test_symlinked_quant_output_is_rejected_without_upload(recorder: Recorder, freeform_output):
    recorder.reclaim_outcome = "output dir or its task dir is a symlink"

    await runtime.start_training_task(_quant_request(TaskType.QUANTFREEFORMTASK, decode=DECODE), "/repo")

    assert recorder.completed == [False]
    assert recorder.uploads == []
    assert recorder.decode_calls == []
    assert "[ERROR] quant output rejected: output dir or its task dir is a symlink" in recorder.logs


async def test_failed_reclaim_fails_the_job_without_upload(recorder: Recorder, freeform_output):
    recorder.reclaim_outcome = OutputReclaimError("output reclaim container error: image missing")

    await runtime.start_training_task(_quant_request(TaskType.QUANTFREEFORMTASK, decode=DECODE), "/repo")

    assert recorder.completed == [False]
    assert recorder.uploads == []
    assert recorder.decode_calls == []
    assert any("Quant output check crashed: output reclaim container error" in message for message in recorder.logs)


async def test_quant_job_with_manifest_uploads_verbatim_then_decodes(recorder: Recorder, freeform_output, monkeypatch):
    monkeypatch.setattr(runtime, "upload_repo_to_hf", REAL_UPLOAD_REPO_TO_HF)

    await runtime.start_training_task(_quant_request(TaskType.QUANTFREEFORMTASK, decode=DECODE), "/repo")

    upload_run = recorder.docker_client.run_kwargs
    assert upload_run["image"] == runtime.cst.HF_UPLOAD_DOCKER_IMAGE
    assert upload_run["environment"]["UPLOAD_VERBATIM"] == "1"
    assert len(recorder.decode_calls) == 1
    assert recorder.completed == [True]


async def test_instruct_job_skips_the_quant_check_and_verbatim_upload(recorder: Recorder, monkeypatch):
    def unexpected_check(*args, **kwargs):
        raise AssertionError("check_quant_output called for an instruct job")

    monkeypatch.setattr(runtime, "check_quant_output", unexpected_check)
    monkeypatch.setattr(runtime, "upload_repo_to_hf", REAL_UPLOAD_REPO_TO_HF)

    await runtime.start_training_task(_instruct_request(), "/repo")

    assert "UPLOAD_VERBATIM" not in recorder.docker_client.run_kwargs["environment"]
    assert recorder.completed == [True]


@pytest.mark.parametrize(
    ("request_factory", "verbatim"),
    [
        (_instruct_request, False),
        (lambda: _quant_request(TaskType.QUANTFREEFORMTASK), True),
    ],
)
async def test_only_quant_jobs_upload_verbatim(recorder: Recorder, freeform_output, request_factory, verbatim):
    await runtime.start_training_task(request_factory(), "/repo")

    [upload] = recorder.uploads
    assert upload["verbatim"] is verbatim


async def test_timed_out_quant_job_without_manifest_is_a_failure(recorder: Recorder):
    container = HangingTrainContainer()
    recorder.train_container = container
    try:
        await runtime.start_training_task(
            _quant_request(TaskType.QUANTFIXEDTASK, quant_spec=SPEC, hours_to_complete=0.0001), "/repo"
        )
    finally:
        await _release_orphaned_wait(container)

    assert any(message.startswith("Timeout reached") for message in recorder.logs)
    assert recorder.completed == [False]
    assert recorder.uploads == []


@pytest.mark.parametrize(("decode", "expected"), [(DECODE, True), (None, False)])
def test_cleanup_extends_deadline_only_for_jobs_that_decode(decode, expected):
    request = _quant_request(TaskType.QUANTFREEFORMTASK, decode=decode)
    job = TrainerTaskLog(**request.model_dump(), status=TaskStatus.TRAINING)

    assert cleanup._decodes_after_upload(job) is expected
    assert cleanup.QUANT_DECODE_GRACE_MINUTES == 2 * 900 // 60 + 30


async def test_large_build_context_file_is_logged_to_the_task_before_build(recorder: Recorder, quant_output, tmp_path: Path):
    repo = tmp_path / "miner-repo"
    repo.mkdir()
    (repo / "blob.bin").write_bytes(b"\0" * (QUANT_LARGE_FILE_REVIEW_BYTES + 1))

    await runtime.start_training_task(_quant_request(TaskType.QUANTFIXEDTASK, quant_spec=SPEC), str(repo))

    review = [message for message in recorder.logs if message.startswith("[REVIEW]")]
    assert len(review) == 1 and "blob.bin" in review[0]
    assert recorder.completed == [True]  # flagged for review, never failed



async def test_calibration_file_is_written_before_the_container_and_its_env_reaches_it(recorder: Recorder, freeform_output, tmp_path):
    await runtime.start_training_task(_quant_request(TaskType.QUANTFREEFORMTASK), "/repo")

    assert recorder.calibration_calls == [("task-1", "hk", CALIBRATION, str(tmp_path))]
    assert recorder.text_container_kwargs["extra_environment"] == CALIBRATION_ENV


async def test_fixed_task_also_gets_the_calibration_env(recorder: Recorder, quant_output):
    await runtime.start_training_task(_quant_request(TaskType.QUANTFIXEDTASK, quant_spec=SPEC), "/repo")

    assert recorder.text_container_kwargs["extra_environment"] == CALIBRATION_ENV


async def test_job_fails_before_the_image_build_when_the_calibration_file_cannot_be_produced(
    recorder: Recorder, freeform_output
):
    recorder.calibration_failure = RuntimeError("calibration set unavailable after 3 attempts (OSError)")

    await runtime.start_training_task(_quant_request(TaskType.QUANTFREEFORMTASK), "/repo")

    assert recorder.image_builds == [] and recorder.text_container_kwargs is None
    assert recorder.completed and not any(recorder.completed)  # reported failed, never successful
    assert recorder.uploads == [] and recorder.decode_calls == []


async def test_a_non_quant_job_writes_no_calibration_file(recorder: Recorder):
    await runtime.start_training_task(_instruct_request(), "/repo")

    assert recorder.calibration_calls == []
    assert recorder.text_container_kwargs["extra_environment"] is None


async def test_calibration_fetch_is_not_bounded_by_the_container_start_timeout(recorder: Recorder, freeform_output, monkeypatch):
    """The fetch runs before the 60 s container-start wait_for, so a slow fetch cannot be killed by it."""
    timeouts = []
    real_wait_for = asyncio.wait_for

    async def spy(awaitable, timeout):
        timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(runtime.asyncio, "wait_for", spy)

    await runtime.start_training_task(_quant_request(TaskType.QUANTFREEFORMTASK), "/repo")

    attempts = trainer_cst.QUANT_CALIBRATION_FETCH_ATTEMPTS
    assert len(recorder.calibration_calls) == 1
    assert timeouts[0] >= attempts * trainer_cst.QUANT_CALIBRATION_FETCH_TIMEOUT_SECONDS and timeouts[-1] == 60


def _text_container_kwargs(training_data, extra_environment) -> dict:
    return dict(
        task_id="task-1",
        hotkey="hk",
        tag="tag",
        model="org/own-bf16",
        dataset=training_data.dataset,
        dataset_type=training_data.dataset_type,
        task_type=TaskType.QUANTFREEFORMTASK,
        file_format=training_data.file_format,
        expected_repo_name="repo-1",
        hours_to_complete=1.0,
        extra_environment=extra_environment,
    )


async def test_text_container_gets_the_extra_environment(monkeypatch, tmp_path: Path):
    client = FakeDockerClient(tmp_path)
    monkeypatch.setattr(runtime.docker, "from_env", lambda: client)
    monkeypatch.setattr(runtime, "ensure_internal_network", lambda: None)
    monkeypatch.setattr(runtime, "stream_container_logs", lambda *args: None)
    training_data = _quant_request(TaskType.QUANTFREEFORMTASK).training_data

    await runtime.run_trainer_container_text(**_text_container_kwargs(training_data, CALIBRATION_ENV))

    environment = client.run_kwargs["environment"]
    assert environment[core_cst.QUANT_CALIBRATION_PATH_ENV] == CALIBRATION_ENV[core_cst.QUANT_CALIBRATION_PATH_ENV]
    assert "--quant-calibration" not in client.run_kwargs["command"]


async def test_text_container_without_extra_environment_has_no_calibration_env(monkeypatch, tmp_path: Path):
    client = FakeDockerClient(tmp_path)
    monkeypatch.setattr(runtime.docker, "from_env", lambda: client)
    monkeypatch.setattr(runtime, "ensure_internal_network", lambda: None)
    monkeypatch.setattr(runtime, "stream_container_logs", lambda *args: None)
    training_data = _quant_request(TaskType.QUANTFREEFORMTASK).training_data

    await runtime.run_trainer_container_text(**_text_container_kwargs(training_data, None))

    assert core_cst.QUANT_CALIBRATION_PATH_ENV not in client.run_kwargs["environment"]
