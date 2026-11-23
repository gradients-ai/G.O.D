"""Ops re-run of a freeform decode against a task_history.json in a tmp dir and a fake decode."""

import asyncio
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import pytest

from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestQuant
from core.models.quant_artifact_models import QuantDecodeRequest
from core.models.quant_artifact_models import QuantDecodeResult
from core.models.quant_models import QuantCalibrationReference
from core.models.quant_models import QuantCalibrationRequest
from core.models.quant_models import QuantTarget
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from trainer import constants as cst
from trainer import endpoints
from trainer import job_state
from trainer.quant_decode import job as decode_job
from trainer.quant_decode import rerun
from trainer.quant_decode.job import PendingUpload
from trainer.quant_decode.job import _DecodeProgress
from trainer.quant_decode.job_lock import JobLock


DECODE = QuantDecodeRequest(
    decoder_snapshot_url="https://minio.example/decoder.tar.xz?sig=x",
    decoder_sha256="a" * 64,
    bucket="decoded",
    object_prefix="quant-decoded/task-1/hk",
)
OK = QuantDecodeResult(ok=True, n_tensors=3, uploaded_files=["model.safetensors", "index.json"])
INFRA = QuantDecodeResult(infra_error="artifact download failed: hf down")
PENDING = QuantDecodeResult(upload_pending=True, n_tensors=3, uploaded_files=["model.safetensors"])


def _job(decode: QuantDecodeRequest | None) -> TrainerTaskLog:
    training_data = TrainRequestQuant(
        model="org/model",
        task_id="task-1",
        hours_to_complete=1.0,
        expected_repo_name="repo-1",
        dataset="s3://bucket/train.json",
        file_format=FileFormat.S3,
        dataset_type=InstructTextDatasetType(),
        task_type=TaskType.QUANTFREEFORMTASK,
        decode=decode,
        quant_target=QuantTarget(
        reference_model="m", reference_loss=2.0, reference_sku="s", bar_cap=0.04, bar=2.08, knee=1.04, knee_loss=2.08,
        penalty_curvature=2500.0, cap=1.08, cap_loss=2.16,
    ),
        calibration=QuantCalibrationRequest(
            examples_url="https://minio.example/cal.jsonl",
            eval_set_fingerprint="fp",
            n_examples=1,
            reference=QuantCalibrationReference(model="m", loss=2.0, per_example_losses=[1.0], sku="s"),
        ),
    )
    return TrainerTaskLog(
        training_data=training_data,
        github_repo="https://github.com/org/repo",
        gpu_ids=[0],
        hotkey="hk",
        status=TaskStatus.SUCCESS,
        quant_decode=QuantDecodeResult(infra_error="minio down"),
    )


@pytest.fixture
def history(monkeypatch, tmp_path: Path):
    path = tmp_path / "task_history.json"
    monkeypatch.setattr(job_state, "TASK_HISTORY_FILE", path)
    monkeypatch.setattr(job_state, "task_history", [])
    monkeypatch.setattr(cst, "QUANT_DECODE_WORK_ROOT", tmp_path / "work_root")

    def write(job: TrainerTaskLog) -> None:
        path.write_text(json.dumps([job.model_dump(mode="json")]))

    return write


class FakeDecode:
    def __init__(self):
        self.calls: list[TrainerProxyRequest] = []
        self.result = OK

    async def __call__(self, task: TrainerProxyRequest) -> QuantDecodeResult:
        self.calls.append(task)
        return self.result


@pytest.fixture
def decode(monkeypatch) -> FakeDecode:
    fake = FakeDecode()
    monkeypatch.setattr(rerun, "freeform_decode_result", fake)
    return fake


def _stored_decode() -> QuantDecodeResult | None:
    job_state.load_task_history()
    return job_state.get_task("task-1", "hk").quant_decode


async def test_ok_result_is_stored_and_exits_zero(history, decode, capsys):
    history(_job(DECODE))

    assert await rerun.rerun_decode("task-1", "hk") == 0

    assert _stored_decode() == OK
    assert decode.calls[0].training_data.decode == DECODE
    assert decode.calls[0].hotkey == "hk"
    assert QuantDecodeResult.model_validate_json(capsys.readouterr().out) == OK


async def test_task_details_serves_a_result_written_by_another_process(history, decode):
    history(_job(DECODE))
    job_state.load_task_history()
    history(_job(DECODE).model_copy(update={"quant_decode": OK}))

    assert (await endpoints.get_task_details("task-1", "hk")).quant_decode == OK


async def test_task_details_hides_a_pending_upload_from_a_caller_that_does_not_ask_for_it(history, decode):
    """A validator that predates the upload_pending state cannot parse it: it gets "no decode result yet"."""
    history(_job(DECODE).model_copy(update={"quant_decode": PENDING}))

    plain = await endpoints.get_task_details("task-1", "hk")
    asked = await endpoints.get_task_details("task-1", "hk", accepts_upload_pending=True)

    assert plain.quant_decode is None
    assert plain.status == TaskStatus.SUCCESS and plain.hotkey == "hk"
    assert asked.quant_decode == PENDING
    assert _stored_decode() == PENDING  # only the response is changed


async def test_task_details_serves_a_final_result_to_every_caller(history, decode):
    history(_job(DECODE).model_copy(update={"quant_decode": INFRA}))

    assert (await endpoints.get_task_details("task-1", "hk")).quant_decode == INFRA
    assert (await endpoints.get_task_details("task-1", "hk", accepts_upload_pending=True)).quant_decode == INFRA


async def test_recent_tasks_hides_a_pending_upload(history, decode, monkeypatch):
    pending_job = _job(DECODE).model_copy(update={"quant_decode": PENDING})
    monkeypatch.setattr(endpoints, "get_recent_tasks", lambda hours: [pending_job])

    listed = await endpoints.get_recent_tasks_list(1)

    assert [job.quant_decode for job in listed] == [None]


async def test_infra_error_is_stored_and_exits_one(history, decode):
    history(_job(DECODE))
    decode.result = INFRA

    assert await rerun.rerun_decode("task-1", "hk") == 1

    assert _stored_decode() == INFRA


async def test_pending_upload_is_retried_in_the_foreground_and_its_result_stored(history, decode, monkeypatch, capsys):
    history(_job(DECODE))
    decode.result = PENDING
    stored_while_retrying: list[QuantDecodeResult | None] = []

    async def retry(work_root, task_id, hotkey, report, store):
        stored_while_retrying.append(_stored_decode())
        await report("decode: upload not complete")
        await store(OK)
        return OK

    monkeypatch.setattr(rerun.upload_retry, "retry_pending_upload", retry)

    assert await rerun.rerun_decode("task-1", "hk") == 0

    assert stored_while_retrying == [PENDING]
    assert _stored_decode() == OK
    captured = capsys.readouterr()
    assert QuantDecodeResult.model_validate_json(captured.out) == OK
    assert "decode: upload not complete" in captured.err


async def test_pending_upload_that_runs_out_of_window_exits_one(history, decode, monkeypatch):
    history(_job(DECODE))
    decode.result = PENDING

    async def retry(work_root, task_id, hotkey, report, store):
        await store(INFRA)
        return INFRA

    monkeypatch.setattr(rerun.upload_retry, "retry_pending_upload", retry)

    assert await rerun.rerun_decode("task-1", "hk") == 1
    assert _stored_decode() == INFRA


async def test_upload_retry_taken_over_by_the_trainer_exits_three(history, decode, monkeypatch, capsys):
    history(_job(DECODE))
    decode.result = PENDING

    async def retry(work_root, task_id, hotkey, report, store):
        return None

    monkeypatch.setattr(rerun.upload_retry, "retry_pending_upload", retry)

    assert await rerun.rerun_decode("task-1", "hk") == 3
    assert _stored_decode() == PENDING
    assert "took over" in capsys.readouterr().err


async def test_a_rerun_that_never_gets_the_job_lock_stores_nothing_and_exits_four(history, decode, capsys):
    """The trainer is still uploading for the job: its pending result must not be replaced by an error."""
    history(_job(DECODE).model_copy(update={"quant_decode": PENDING}))
    decode.result = QuantDecodeResult(infra_error=decode_job.JOB_BUSY_FAILURE)

    assert await rerun.rerun_decode("task-1", "hk") == 4

    assert _stored_decode() == PENDING
    captured = capsys.readouterr()
    assert captured.out == "" and "nothing was stored" in captured.err


def _pending_work_dir(token: str) -> decode_job._WorkDirs:
    """The job's work dir with a pending upload in it, as the trainer's retry or an earlier decode left it."""
    decode_job.prepare_work_root(cst.QUANT_DECODE_WORK_ROOT)
    work = decode_job.work_dirs(cst.QUANT_DECODE_WORK_ROOT, "task-1", "hk")
    work.root.mkdir(exist_ok=True)
    work.output.mkdir(exist_ok=True)
    (work.output / "model.safetensors").write_bytes(b"x" * 16)
    until = datetime(2030, 1, 1, tzinfo=timezone.utc)
    pending = PendingUpload(
        token=token, task_id="task-1", hotkey="hk", request=DECODE, progress=_DecodeProgress(), retry_until=until, last_error="e"
    )
    decode_job.save_pending(work, pending)
    return work


async def test_a_rerun_interrupted_after_it_replaced_a_pending_upload_stores_an_infra_error(history, monkeypatch):
    history(_job(DECODE).model_copy(update={"quant_decode": PENDING}))
    work = _pending_work_dir("trainer-token")

    async def decode_then_interrupted(task) -> QuantDecodeResult:
        decode_job.remove_work_tree(work.root)  # the decode emptied the work dir: the trainer's retry is over
        work.root.mkdir()
        raise asyncio.CancelledError()

    monkeypatch.setattr(rerun, "freeform_decode_result", decode_then_interrupted)

    with pytest.raises(asyncio.CancelledError):
        await rerun.rerun_decode("task-1", "hk")

    assert _stored_decode() == QuantDecodeResult(infra_error=rerun.INTERRUPTED_FAILURE)
    assert not work.root.exists()


async def test_a_rerun_interrupted_during_its_own_upload_retry_stores_an_infra_error(history, decode, monkeypatch):
    history(_job(DECODE))
    decode.result = PENDING
    work = decode_job.work_dirs(cst.QUANT_DECODE_WORK_ROOT, "task-1", "hk")

    async def retry(work_root, task_id, hotkey, report, store):
        _pending_work_dir("rerun-token")  # what the command's own decode left for the retry
        raise KeyboardInterrupt()

    monkeypatch.setattr(rerun.upload_retry, "retry_pending_upload", retry)

    with pytest.raises(KeyboardInterrupt):
        await rerun.rerun_decode("task-1", "hk")

    assert _stored_decode() == QuantDecodeResult(infra_error=rerun.INTERRUPTED_FAILURE)
    assert not work.root.exists()


async def test_a_rerun_interrupted_before_its_decode_started_leaves_the_trainers_retry_alone(history, monkeypatch):
    history(_job(DECODE).model_copy(update={"quant_decode": PENDING}))
    work = _pending_work_dir("trainer-token")

    async def interrupted_while_waiting(task) -> QuantDecodeResult:
        raise asyncio.CancelledError()

    monkeypatch.setattr(rerun, "freeform_decode_result", interrupted_while_waiting)

    with pytest.raises(asyncio.CancelledError):
        await rerun.rerun_decode("task-1", "hk")

    assert _stored_decode() == PENDING
    assert decode_job.load_pending(work).token == "trainer-token"
    assert (work.output / "model.safetensors").exists()


async def test_a_rerun_interrupted_while_the_trainer_holds_the_job_lock_changes_nothing(history, monkeypatch):
    history(_job(DECODE).model_copy(update={"quant_decode": PENDING}))
    work = _pending_work_dir("trainer-token")
    trainer_round = JobLock(cst.QUANT_DECODE_WORK_ROOT, work.root.name)
    assert trainer_round.try_acquire()

    async def interrupted_while_waiting(task) -> QuantDecodeResult:
        decode_job.save_pending(work, decode_job.load_pending(work).model_copy(update={"token": "trainer-token-2"}))
        raise asyncio.CancelledError()

    monkeypatch.setattr(rerun, "freeform_decode_result", interrupted_while_waiting)
    try:
        with pytest.raises(asyncio.CancelledError):
            await rerun.rerun_decode("task-1", "hk")
    finally:
        trainer_round.release()

    assert _stored_decode() == PENDING
    assert work.pending.exists()


async def test_sigterm_interrupts_the_rerun_like_ctrl_c(history, monkeypatch):
    history(_job(DECODE).model_copy(update={"quant_decode": PENDING}))
    work = _pending_work_dir("trainer-token")

    async def decode_until_killed(task) -> QuantDecodeResult:
        decode_job.remove_work_tree(work.root)
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.sleep(30)
        return OK

    monkeypatch.setattr(rerun, "freeform_decode_result", decode_until_killed)
    try:
        with pytest.raises(asyncio.CancelledError):
            await rerun._run_until_signalled("task-1", "hk")
    finally:
        asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM)

    assert _stored_decode() == QuantDecodeResult(infra_error=rerun.INTERRUPTED_FAILURE)


async def test_job_without_decode_request_exits_two(history, decode, capsys):
    history(_job(None))

    assert await rerun.rerun_decode("task-1", "hk") == 2

    assert decode.calls == []
    assert "has no freeform decode request" in capsys.readouterr().err


async def test_unknown_job_exits_two(history, decode, capsys):
    history(_job(DECODE))

    assert await rerun.rerun_decode("task-1", "other-hotkey") == 2

    assert decode.calls == []
    assert "no training job" in capsys.readouterr().err


def test_help_exits_zero():
    result = subprocess.run([sys.executable, "-m", "trainer.quant_decode.rerun", "--help"], capture_output=True, text=True)

    assert result.returncode == 0
    assert "task_id" in result.stdout
