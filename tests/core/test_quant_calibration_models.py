"""The calibration contract models (shared with the miner) and the TrainerProxyRequest quant hardening."""

import pytest
from pydantic import ValidationError

import core.constants as core_cst
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainRequestQuant
from core.models.payload_models import TrainRequestText
from core.models.quant_models import QuantCalibrationExample
from core.models.quant_models import QuantCalibrationReference
from core.models.quant_models import QuantCalibrationRequest
from core.models.quant_models import QuantCalibrationSet
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskType


def _quant_payload(**overrides) -> dict:
    payload = {
        "model": "org/model",
        "task_id": "task-1",
        "hours_to_complete": 1.0,
        "dataset": "s3://bucket/train.json",
        "dataset_type": InstructTextDatasetType().model_dump(),
        "file_format": FileFormat.S3,
        "task_type": TaskType.QUANTFREEFORMTASK,
    }
    return {**payload, **overrides}


def _proxy(training_data: dict) -> dict:
    return {"training_data": training_data, "github_repo": "https://github.com/o/r", "gpu_ids": [0], "hotkey": "hk"}


def test_the_calibration_env_constant_is_the_contract_name():
    assert core_cst.QUANT_CALIBRATION_PATH_ENV == "QUANT_CALIBRATION_PATH"


@pytest.mark.parametrize(
    "model,kwargs",
    [
        (QuantCalibrationReference, dict(model="m", loss=1.0, per_example_losses=[1.0, None], sku="s", extra=1)),
        (QuantCalibrationRequest, dict(examples_url="u", eval_set_fingerprint="f", n_examples=1, reference=None, extra=1)),
        (QuantCalibrationExample, dict(input_ids=[1], labels=[-100], reference_loss=None, extra=1)),
        (
            QuantCalibrationSet,
            dict(
                task_id="t", eval_set_fingerprint="f", reference_model=None, reference_loss=None, reference_sku=None,
                examples=[], extra=1,
            ),
        ),
    ],
)  # fmt: skip
def test_contract_models_forbid_extra_fields(model, kwargs):
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_contract_models_are_frozen():
    example = QuantCalibrationExample(input_ids=[1], labels=[-100], reference_loss=None)
    with pytest.raises(ValidationError):
        example.reference_loss = 1.0


def test_calibration_set_round_trips_through_json_with_none_losses():
    calibration_set = QuantCalibrationSet(
        task_id="t",
        eval_set_fingerprint="f",
        reference_model="m",
        reference_loss=2.0,
        reference_sku="s",
        examples=[
            QuantCalibrationExample(input_ids=[1, 2], labels=[-100, 2], reference_loss=1.5),
            QuantCalibrationExample(input_ids=[1], labels=[-100], reference_loss=None),
        ],
    )

    assert QuantCalibrationSet.model_validate_json(calibration_set.model_dump_json()) == calibration_set


def test_a_quant_request_without_calibration_defaults_to_none():
    assert TrainRequestQuant(**_quant_payload()).calibration is None


def test_a_fixed_request_may_carry_a_calibration():
    calibration = {"examples_url": "u", "eval_set_fingerprint": "f", "n_examples": 1, "reference": None}

    request = TrainRequestQuant(
        **_quant_payload(task_type=TaskType.QUANTFIXEDTASK, quant_spec=QuantFormat.W4A16_SYM_G128.spec(), calibration=calibration)
    )

    assert request.calibration.n_examples == 1


def test_a_quant_payload_round_trips_as_quant_with_calibration():
    calibration = {"examples_url": "u", "eval_set_fingerprint": "f", "n_examples": 1, "reference": None}

    request = TrainerProxyRequest.model_validate(_proxy(_quant_payload(calibration=calibration)))

    assert isinstance(request.training_data, TrainRequestQuant)
    assert request.training_data.calibration.n_examples == 1


def test_a_quant_payload_with_an_unknown_field_raises_instead_of_becoming_text():
    with pytest.raises(ValidationError):
        TrainerProxyRequest.model_validate(_proxy(_quant_payload(some_future_field=1)))


def test_a_quant_payload_with_a_bad_calibration_raises_instead_of_becoming_text():
    with pytest.raises(ValidationError):
        TrainerProxyRequest.model_validate(_proxy(_quant_payload(calibration={"examples_url": "u"})))


def test_a_text_payload_is_untouched():
    payload = {
        "model": "org/model",
        "task_id": "task-1",
        "hours_to_complete": 1.0,
        "dataset": "s3://bucket/train.json",
        "dataset_type": InstructTextDatasetType().model_dump(),
        "file_format": FileFormat.S3,
    }

    request = TrainerProxyRequest.model_validate(_proxy(payload))

    assert isinstance(request.training_data, TrainRequestText)
