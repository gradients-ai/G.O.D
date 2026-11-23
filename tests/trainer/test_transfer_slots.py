"""Image builds, uploads and freeform decodes take turns on a trainer, and queueing never eats a step's time limit."""

import asyncio
import threading
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestText
from core.models.quant_artifact_models import QuantDecodeRequest
from core.models.task_models import TaskStatus
from trainer import cleanup
from trainer import constants as cst
from trainer import runtime
from trainer import slots
from trainer.quant_decode import job
from trainer.slots import SlotLimiter


HOLD_SECONDS = 0.3
JOB = ("task-1", "hk")


class JobLogs:
    """Stands in for the job log in task_history.json."""

    def __init__(self):
        self.lines: dict[tuple[str, str], list[str]] = {}

    async def log(self, task_id: str, hotkey: str, message: str) -> None:
        self.lines.setdefault((task_id, hotkey), []).append(message)

    def waited(self, task_id: str, hotkey: str, slot_name: str) -> bool:
        return any(f"Waiting for {slot_name} slot" in line for line in self.lines.get((task_id, hotkey), []))


class Gauge:
    """Counts how many holders are inside at once."""

    def __init__(self):
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        with self._lock:
            self.current -= 1


@pytest.fixture
def job_logs(monkeypatch) -> JobLogs:
    logs = JobLogs()
    monkeypatch.setattr(runtime, "log_task", logs.log)
    return logs


@pytest.fixture(autouse=True)
def fresh_wait_records():
    yield
    slots.forget_waits(JOB)


async def _until(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


# --- the limiter itself


async def test_holders_beyond_the_limit_wait_in_arrival_order():
    limiter = SlotLimiter("test", 2)
    order: list[int] = []
    gauge = Gauge()

    async def hold(n: int) -> float:
        async with limiter.slot() as waited:
            order.append(n)
            gauge.enter()
            await asyncio.sleep(0.05)
            gauge.leave()
            return waited

    waits = await asyncio.gather(*(hold(n) for n in range(5)))

    assert gauge.peak == 2
    assert order == [0, 1, 2, 3, 4]
    assert waits[0] == waits[1] == 0.0
    assert all(wait > 0 for wait in waits[2:])


async def test_a_wait_is_reported_and_recorded_against_the_job():
    limiter = SlotLimiter("test", 1)
    reports: list[str] = []

    async def report(message: str) -> None:
        reports.append(message)

    async with limiter.slot():
        queued = asyncio.create_task(_take(limiter, report))
        await _until(lambda: slots.is_waiting(JOB))
        assert reports == ["Waiting for test slot (1 at a time on this trainer)"]
        await asyncio.sleep(0.05)
    await queued

    assert not slots.is_waiting(JOB)
    assert slots.waited_seconds(JOB) >= 0.05
    assert reports[1].startswith("Got test slot after waiting")

    slots.forget_waits(JOB)
    assert slots.waited_seconds(JOB) == 0.0


async def _take(limiter: SlotLimiter, report) -> None:
    async with limiter.slot(report, JOB):
        pass


async def test_a_free_slot_is_taken_without_any_report():
    reports: list[str] = []

    async def report(message: str) -> None:
        reports.append(message)

    async with SlotLimiter("test", 1).slot(report, JOB) as waited:
        assert waited == 0.0
    assert reports == []
    assert slots.waited_seconds(JOB) == 0.0


async def test_a_cancelled_waiter_leaves_the_queue_and_the_slot_stays_usable():
    limiter = SlotLimiter("test", 1)
    async with limiter.slot():
        queued = asyncio.create_task(_take(limiter, slots._no_report))
        await _until(lambda: slots.is_waiting(JOB))
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
    assert not slots.is_waiting(JOB)
    async with limiter.slot() as waited:
        assert waited == 0.0


def test_a_limit_below_one_is_rejected():
    with pytest.raises(ValueError):
        SlotLimiter("test", 0)


def test_default_limits():
    assert slots.image_builds.limit == cst.MAX_CONCURRENT_IMAGE_BUILDS == 2
    assert slots.uploads.limit == cst.MAX_CONCURRENT_UPLOADS == 2
    assert slots.quant_decodes.limit == cst.MAX_CONCURRENT_QUANT_DECODES == 2


# --- image builds


@pytest.fixture
def slow_builds(monkeypatch) -> Gauge:
    """A Docker client whose every build takes HOLD_SECONDS inside its stream, the part the build timeout covers."""
    gauge = Gauge()

    def stream():
        gauge.enter()
        try:
            time.sleep(HOLD_SECONDS)
        finally:
            gauge.leave()
        yield {"stream": "Successfully built\n"}

    def build(**kwargs):
        return stream()

    client = MagicMock()
    client.api.build.side_effect = build
    monkeypatch.setattr(runtime.docker, "from_env", MagicMock(return_value=client))
    return gauge


async def test_builds_beyond_the_cap_wait_and_the_wait_is_not_counted_in_the_build_timeout(monkeypatch, slow_builds, job_logs):
    monkeypatch.setattr(slots, "image_builds", SlotLimiter("image build", 1))
    # Enough for one build, not for a build plus the queue ahead of the third job.
    monkeypatch.setattr(cst, "DOCKER_BUILD_TIMEOUT_MINUTES", HOLD_SECONDS * 1.8 / 60)

    results = await asyncio.gather(
        *(runtime.build_job_image("task-1", f"hk{n}", dockerfile_path="Dockerfile", tag=f"t:{n}") for n in range(3))
    )

    assert results == [("t:0", None), ("t:1", None), ("t:2", None)]
    assert slow_builds.peak == 1
    assert not job_logs.waited("task-1", "hk0", "image build")
    assert job_logs.waited("task-1", "hk1", "image build")
    assert job_logs.waited("task-1", "hk2", "image build")


async def test_builds_within_the_cap_run_together(monkeypatch, slow_builds, job_logs):
    monkeypatch.setattr(slots, "image_builds", SlotLimiter("image build", 2))

    await asyncio.gather(
        *(runtime.build_job_image("task-1", f"hk{n}", dockerfile_path="Dockerfile", tag=f"t:{n}") for n in range(2))
    )

    assert slow_builds.peak == 2
    assert job_logs.lines == {}


# --- uploads


class FakeUploadContainer:
    name = "hf-upload-fake"

    def __init__(self, gauge: Gauge, released: threading.Event):
        self._gauge = gauge
        self._released = released

    def wait(self) -> dict:
        self._gauge.enter()
        try:
            self._released.wait(timeout=10)
        finally:
            self._gauge.leave()
        return {"StatusCode": 0}

    def logs(self) -> bytes:
        return b""


class HfUploads:
    """Fake uploader containers that run until `released` is set."""

    def __init__(self):
        self.gauge = Gauge()
        self.released = threading.Event()
        self.containers = self

    def run(self, **kwargs) -> FakeUploadContainer:
        return FakeUploadContainer(self.gauge, self.released)

    async def upload(self, hotkey: str = "hk") -> None:
        await runtime.upload_repo_to_hf(
            task_id="task-1",
            hotkey=hotkey,
            expected_repo_name="repo-1",
            huggingface_token="token",
            huggingface_username="user",
            model="org/model",
            gpu_ids=[0],
        )


@pytest.fixture
def hf_uploads(monkeypatch, job_logs) -> HfUploads:
    uploads = HfUploads()
    monkeypatch.setattr(runtime.docker, "from_env", lambda: uploads)
    monkeypatch.setattr(runtime, "stream_container_logs", lambda *args: None)
    return uploads


@pytest.fixture
def decoded_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"decoded")
    return path


async def test_hf_and_decode_uploads_share_one_pool_and_the_wait_is_not_counted_in_the_deadline(
    monkeypatch, hf_uploads, decoded_file
):
    monkeypatch.setattr(slots, "uploads", SlotLimiter("upload", 1))
    # A deadline shorter than the time the decoded file spends queued behind the model upload.
    monkeypatch.setattr(job, "upload_timeout_seconds", lambda size: HOLD_SECONDS / 2)
    sent: list[str] = []
    reports: list[str] = []

    async def put(bucket, object_name, path, report, policy, sha256):
        assert hf_uploads.gauge.current == 0, "decoded file sent while a model upload held the only slot"
        sent.append(object_name)

    async def report(message: str) -> None:
        reports.append(message)

    monkeypatch.setattr(job, "upload_with_retry", put)

    model_upload = asyncio.create_task(hf_uploads.upload())
    await _until(lambda: hf_uploads.gauge.current == 1)
    decode_upload = asyncio.create_task(job._upload("bucket", "prefix/model.safetensors", decoded_file, report, JOB))
    await asyncio.sleep(HOLD_SECONDS)
    assert sent == []
    assert slots.is_waiting(JOB)

    hf_uploads.released.set()
    await asyncio.gather(model_upload, decode_upload)

    assert sent == ["prefix/model.safetensors"]
    assert reports[0].startswith("Waiting for upload slot")
    assert slots.waited_seconds(JOB) >= HOLD_SECONDS / 2


async def test_model_uploads_beyond_the_cap_wait(monkeypatch, hf_uploads, job_logs):
    monkeypatch.setattr(slots, "uploads", SlotLimiter("upload", 2))

    uploads = [asyncio.create_task(hf_uploads.upload(f"hk{n}")) for n in range(3)]
    await _until(lambda: job_logs.waited("task-1", "hk2", "upload"))
    assert hf_uploads.gauge.peak == 2

    hf_uploads.released.set()
    await asyncio.gather(*uploads)

    assert hf_uploads.gauge.peak == 2
    assert not job_logs.waited("task-1", "hk0", "upload")


async def test_a_stuck_model_upload_times_out_and_frees_its_slot(monkeypatch, hf_uploads):
    monkeypatch.setattr(slots, "uploads", SlotLimiter("upload", 1))
    monkeypatch.setattr(cst, "HF_UPLOAD_TIMEOUT_MINUTES", 0.1 / 60)

    with pytest.raises(RuntimeError, match="HF upload exceeded"):
        await hf_uploads.upload()

    async with slots.uploads.slot() as waited:
        assert waited == 0.0
    hf_uploads.released.set()
    await _until(lambda: hf_uploads.gauge.current == 0)  # let the abandoned wait thread end before the loop closes


# --- freeform decodes


async def test_decodes_beyond_the_cap_wait(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(slots, "quant_decodes", SlotLimiter("freeform decode", 1))
    gauge = Gauge()
    reports: dict[str, list[str]] = {}

    async def decode(request, **kwargs):
        gauge.enter()
        await asyncio.sleep(0.05)
        gauge.leave()

    monkeypatch.setattr(job, "_decode", decode)
    request = QuantDecodeRequest(
        decoder_snapshot_url="http://127.0.0.1/decoder.tar.xz", decoder_sha256="0" * 64, bucket="decoded", object_prefix="p"
    )

    async def run(hotkey: str):
        async def report(message: str) -> None:
            reports.setdefault(hotkey, []).append(message)

        return await job.run_freeform_decode(
            request,
            task_id="task-1",
            hotkey=hotkey,
            artifact_repo="user/artifact",
            base_model_dir=tmp_path,
            hf_token=None,
            work_root=tmp_path / "work",
            docker_client=MagicMock(),
            report=report,
        )

    results = await asyncio.gather(*(run(f"hotkey-{n}") for n in range(3)))

    assert all(result.ok for result in results)
    assert gauge.peak == 1
    assert "hotkey-0" not in reports
    assert reports["hotkey-1"][0].startswith("Waiting for freeform decode slot")
    assert reports["hotkey-2"][0].startswith("Waiting for freeform decode slot")


# --- a queued job is not overdue


def _running_job(started_at: datetime) -> TrainerTaskLog:
    training_data = TrainRequestText(
        model="org/model",
        task_id=JOB[0],
        hours_to_complete=1.0,
        expected_repo_name="repo-1",
        dataset="s3://bucket/train.json",
        file_format=FileFormat.S3,
        dataset_type=InstructTextDatasetType(),
    )
    return TrainerTaskLog(
        training_data=training_data,
        github_repo="https://github.com/org/repo",
        gpu_ids=[0],
        hotkey=JOB[1],
        status=TaskStatus.TRAINING,
        started_at=started_at,
        finished_at=None,
    )


async def test_a_queued_job_has_no_stale_deadline_and_its_wait_extends_the_deadline():
    started_at = datetime.utcnow()
    running = _running_job(started_at)
    base_deadline = started_at + timedelta(hours=1, minutes=cst.STALE_TASK_GRACE_MINUTES)
    assert cleanup.stale_deadline(running) == base_deadline

    limiter = SlotLimiter("upload", 1)
    async with limiter.slot():
        queued = asyncio.create_task(_take(limiter, slots._no_report))
        await _until(lambda: slots.is_waiting(JOB))
        assert cleanup.stale_deadline(running) is None
        await asyncio.sleep(0.05)
    await queued

    waited = slots.waited_seconds(JOB)
    assert waited >= 0.05
    assert cleanup.stale_deadline(running) == base_deadline + timedelta(seconds=waited)


async def test_a_job_queued_for_too_long_is_overdue_again(monkeypatch):
    started_at = datetime.utcnow()
    running = _running_job(started_at)
    base_deadline = started_at + timedelta(hours=1, minutes=cst.STALE_TASK_GRACE_MINUTES)
    max_wait_seconds = 0.1
    monkeypatch.setattr(cst, "SLOT_WAIT_STALE_EXEMPT_MAX_HOURS", max_wait_seconds / 3600)

    limiter = SlotLimiter("upload", 1)
    async with limiter.slot():
        queued = asyncio.create_task(_take(limiter, slots._no_report))
        await _until(lambda: slots.is_waiting(JOB))
        assert cleanup.stale_deadline(running) is None
        await _until(lambda: slots.waiting_seconds(JOB) > max_wait_seconds)
        assert cleanup.stale_deadline(running) == base_deadline  # one unbroken wait past the cap
    await queued

    assert slots.waiting_seconds(JOB) == 0.0
    assert slots.waited_seconds(JOB) > max_wait_seconds
    assert cleanup.stale_deadline(running) == base_deadline + timedelta(seconds=max_wait_seconds)  # credit is capped too


async def test_a_stalled_artifact_download_times_out_as_an_infra_error(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cst, "QUANT_ARTIFACT_FETCH_TIMEOUT_SECONDS", 0.1)
    released = threading.Event()
    monkeypatch.setattr(job, "snapshot_download", lambda **kwargs: released.wait(timeout=10))

    try:
        with pytest.raises(job._InfraFault, match="artifact download timed out after 0s"):
            await job._fetch_artifact("user/artifact", tmp_path, None)
    finally:
        released.set()
