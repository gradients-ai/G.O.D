from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainRequestQuant
from core.models.quant_artifact_models import QuantDecodeRequest
from core.models.quant_models import QuantCalibrationReference
from core.models.quant_models import QuantCalibrationRequest
from core.models.quant_models import QuantTarget
from core.models.task_models import TaskType
from validator.tournament import orchestrator


GITHUB_TOKEN = "ghp_secretsecretsecret"
SIGNATURE = "X-Amz-Signature=deadbeef"
DECODER_URL = f"https://bucket.s3.amazonaws.com/quant-decoders/c/s.tar.xz?{SIGNATURE}"
DATASET_URL = "https://bucket.s3.amazonaws.com/train.json?X-Amz-Signature=cafe"
CALIBRATION_URL = "https://bucket.s3.amazonaws.com/cal.jsonl?X-Amz-Signature=beef"


def _request() -> TrainerProxyRequest:
    training_data = TrainRequestQuant(
        model="org/model",
        task_id="task-1",
        hours_to_complete=1.0,
        expected_repo_name="repo-1",
        dataset=DATASET_URL,
        file_format=FileFormat.S3,
        dataset_type=InstructTextDatasetType(),
        task_type=TaskType.QUANTFREEFORMTASK,
        quant_target=QuantTarget(
            reference_model="m", reference_loss=2.0, reference_sku="s", bar_cap=0.04, bar=2.08, knee=1.04, knee_loss=2.08,
            penalty_curvature=2500.0, cap=1.08, cap_loss=2.16,
        ),
        calibration=QuantCalibrationRequest(
            examples_url=CALIBRATION_URL,
            eval_set_fingerprint="fp",
            n_examples=1,
            reference=QuantCalibrationReference(model="m", loss=2.0, per_example_losses=[1.0], sku="s"),
        ),
        decode=QuantDecodeRequest(
            decoder_snapshot_url=DECODER_URL, decoder_sha256="a" * 64, bucket="b", object_prefix="quant-decoded/t/h"
        ),
    )
    return TrainerProxyRequest(
        training_data=training_data,
        github_repo="https://github.com/org/repo",
        gpu_ids=[0],
        hotkey="hk",
        github_commit_hash="c" * 40,
        github_token=GITHUB_TOKEN,
    )


def _assert_clean(text: str) -> None:
    assert GITHUB_TOKEN not in text
    assert "deadbeef" not in text and "cafe" not in text and "beef" not in text
    assert "quant-decoders" not in text


def test_loggable_request_redacts_token_decoder_url_and_queries():
    text = orchestrator.loggable_training_request(_request())
    _assert_clean(text)
    # Everything else stays readable.
    assert "task-1" in text and "https://github.com/org/repo" in text and "train.json?***" in text


def test_loggable_request_without_secrets():
    request = _request().model_copy(update={"github_token": None})
    request.training_data.decode = None
    text = orchestrator.loggable_training_request(request)
    assert "'github_token': None" in text and "'decode': None" in text


@pytest.mark.parametrize("fail_validation", [False, True])
async def test_start_training_task_never_logs_secrets(monkeypatch, fail_validation):
    logger = MagicMock()
    monkeypatch.setattr(orchestrator, "logger", logger)
    response = MagicMock()
    response.json.return_value = {"message": "ok"}
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(orchestrator.httpx, "AsyncClient", MagicMock(return_value=client))
    if fail_validation:
        monkeypatch.setattr(orchestrator.TrainerProxyRequest, "model_validate", MagicMock(side_effect=ValueError("bad")))

    await orchestrator.start_training_task.__wrapped__("10.0.0.1", _request())

    logged = " ".join(
        str(call) for method in ("info", "error", "warning", "debug") for call in getattr(logger, method).call_args_list
    )
    assert ("Request payload" in logged) is fail_validation
    assert ("with payload" in logged) is not fail_validation
    _assert_clean(logged)
