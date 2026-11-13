"""Ops re-run of a freeform decode against a task_history.json in a tmp dir and a fake decode."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestQuant
from core.models.quant_artifact_models import QuantDecodeRequest
from core.models.quant_artifact_models import QuantDecodeResult
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from trainer import endpoints
from trainer import job_state
from trainer.quant_decode import rerun


DECODE = QuantDecodeRequest(
    decoder_snapshot_url="https://minio.example/decoder.tar.xz?sig=x",
    decoder_sha256="a" * 64,
    bucket="decoded",
    object_prefix="quant-decoded/task-1/hk",
)
OK = QuantDecodeResult(ok=True, n_tensors=3, uploaded_files=["model.safetensors", "index.json"])
INFRA = QuantDecodeResult(infra_error="artifact download failed: hf down")


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


async def test_infra_error_is_stored_and_exits_one(history, decode):
    history(_job(DECODE))
    decode.result = INFRA

    assert await rerun.rerun_decode("task-1", "hk") == 1

    assert _stored_decode() == INFRA


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
