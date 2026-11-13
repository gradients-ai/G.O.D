"""Quant contracts: fixed-format specs, the fixed-format rotation, and the TrainRequestQuant payload."""

import pytest
from pydantic import ValidationError

from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestQuant
from core.models.payload_models import TrainRequestText
from core.models.quant_models import LAUNCH_QUANT_FORMATS
from core.models.quant_models import QuantActivationFormat
from core.models.quant_models import QuantFixedSpec
from core.models.quant_models import QuantFormat
from core.models.quant_models import QuantTarget
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType


SPEC = {"weight_bits": 4, "activation": "bf16", "group_size": 128, "symmetric": True}
TEXT_TRAINING_DATA = {
    "model": "org/model",
    "task_id": "task-1",
    "hours_to_complete": 2.0,
    "dataset": "s3://bucket/train.json",
    "dataset_type": {"field_instruction": "instruction", "field_output": "output"},
    "file_format": "s3",
}
FIXED_TRAINING_DATA = {**TEXT_TRAINING_DATA, "task_type": TaskType.QUANTFIXEDTASK.value, "quant_spec": SPEC}
PROXY_FIELDS = {"github_repo": "https://github.com/org/repo", "gpu_ids": [0], "hotkey": "hk"}


@pytest.mark.parametrize(
    "override",
    [{"group_size": 96}, {"weight_bits": 1}, {"weight_bits": 9}, {"act_order": True}, {"unknown": 1}],
)
def test_fixed_spec_rejects_invalid_fields(override):
    with pytest.raises(ValidationError):
        QuantFixedSpec(**{**SPEC, **override})


def test_w3_launch_format_is_asymmetric_g128():
    spec = QuantFormat.W3A16_ASYM_G128.spec()
    assert (spec.weight_bits, spec.symmetric, spec.group_size) == (3, False, 128)
    assert spec.activation == QuantActivationFormat.BF16


def test_w2_launch_format_is_asymmetric_g64():
    spec = QuantFormat.W2A16_ASYM_G64.spec()
    assert (spec.weight_bits, spec.symmetric, spec.group_size) == (2, False, 64)
    assert spec.activation == QuantActivationFormat.BF16
    assert spec.name == QuantFormat.W2A16_ASYM_G64.value == "W2A16-asym-g64"


def test_launch_formats():
    assert LAUNCH_QUANT_FORMATS == (
        QuantFormat.W3A16_ASYM_G128,
        QuantFormat.W2A16_ASYM_G64,
    )


def test_w4a16_stays_scorable_but_is_not_drawn():
    assert QuantFormat.W4A16_SYM_G128 not in LAUNCH_QUANT_FORMATS
    assert QuantFormat.W4A16_SYM_G128.spec().weight_bits == 4


def test_w4a8fp8_stays_scorable_but_is_not_drawn():
    assert QuantFormat.W4A8_FP8_G128 not in LAUNCH_QUANT_FORMATS
    spec = QuantFormat.W4A8_FP8_G128.spec()
    assert (spec.weight_bits, spec.activation, spec.group_size) == (4, QuantActivationFormat.FP8, 128)


@pytest.mark.parametrize("quant_format", list(QuantFormat))
def test_format_value_is_its_spec_name(quant_format):
    assert quant_format.spec().name == quant_format.value


def test_spec_names():
    assert [f.spec().name for f in LAUNCH_QUANT_FORMATS] == ["W3A16-asym-g128", "W2A16-asym-g64"]


def test_fixed_request_requires_spec():
    with pytest.raises(ValidationError, match="requires quant_spec"):
        TrainRequestQuant(**{**FIXED_TRAINING_DATA, "quant_spec": None})


def test_freeform_request_rejects_spec():
    with pytest.raises(ValidationError, match="must not set quant_spec"):
        TrainRequestQuant(**{**FIXED_TRAINING_DATA, "task_type": TaskType.QUANTFREEFORMTASK.value})


def test_quant_request_rejects_non_quant_task_type():
    with pytest.raises(ValidationError):
        TrainRequestQuant(**{**FIXED_TRAINING_DATA, "task_type": TaskType.INSTRUCTTEXTTASK.value})


def test_proxy_request_resolves_quant_payload():
    request = TrainerProxyRequest.model_validate({**PROXY_FIELDS, "training_data": FIXED_TRAINING_DATA})
    assert isinstance(request.training_data, TrainRequestQuant)
    assert request.training_data.quant_spec == QuantFormat.W4A16_SYM_G128.spec()


def test_proxy_request_still_resolves_text_payload():
    request = TrainerProxyRequest.model_validate({**PROXY_FIELDS, "training_data": TEXT_TRAINING_DATA})
    assert type(request.training_data) is TrainRequestText


def test_task_log_json_round_trip_keeps_quant_class():
    log = TrainerTaskLog(
        **PROXY_FIELDS,
        status=TaskStatus.TRAINING,
        training_data=TrainRequestQuant(**FIXED_TRAINING_DATA),
    )
    restored = TrainerTaskLog.model_validate_json(log.model_dump_json())
    assert isinstance(restored.training_data, TrainRequestQuant)
    assert restored.training_data == log.training_data


FIXED_TARGET = {"reference_model": "org/own-bf16", "reference_loss": 2.0, "reference_sku": "h100-sxm"}
FREEFORM_FIELDS = {
    "bar_cap": 0.04,
    "bar": 2.08,
    "knee": 1.04,
    "knee_loss": 2.08,
    "penalty_curvature": 2500.0,
    "cap": 1.08,
    "cap_loss": 2.16,
}
FREEFORM_TARGET = {**FIXED_TARGET, **FREEFORM_FIELDS}
FREEFORM_TRAINING_DATA = {**FIXED_TRAINING_DATA, "task_type": TaskType.QUANTFREEFORMTASK.value, "quant_spec": None}


def test_quant_target_defaults_to_none_and_fixed_fields_are_optional():
    assert TrainRequestQuant(**FIXED_TRAINING_DATA).quant_target is None
    target = QuantTarget(**FIXED_TARGET)
    assert all(getattr(target, name) is None for name in FREEFORM_FIELDS)


@pytest.mark.parametrize("override", [{"reference_loss": 0.0}, {"reference_loss": -1.0}, {"extra": 1}])
def test_quant_target_rejects_bad_fields(override):
    with pytest.raises(ValidationError):
        QuantTarget(**{**FIXED_TARGET, **override})


def test_fixed_request_carries_a_target_through_json():
    request = TrainRequestQuant(**{**FIXED_TRAINING_DATA, "quant_target": FIXED_TARGET})
    assert request.quant_target == QuantTarget(**FIXED_TARGET)
    assert TrainRequestQuant.model_validate_json(request.model_dump_json()) == request


def test_freeform_request_carries_the_bar_through_json():
    request = TrainRequestQuant(**{**FREEFORM_TRAINING_DATA, "quant_target": FREEFORM_TARGET})
    restored = TrainRequestQuant.model_validate_json(request.model_dump_json())
    target = restored.quant_target
    assert target.bar == target.knee_loss == 2.08 and target.bar_cap == 0.04
    assert (target.knee, target.penalty_curvature, target.cap, target.cap_loss) == (1.04, 2500.0, 1.08, 2.16)


def test_task_log_round_trip_keeps_the_target():
    log = TrainerTaskLog(
        **PROXY_FIELDS,
        status=TaskStatus.TRAINING,
        training_data=TrainRequestQuant(**{**FIXED_TRAINING_DATA, "quant_target": FIXED_TARGET}),
    )
    restored = TrainerTaskLog.model_validate_json(log.model_dump_json())
    assert restored.training_data.quant_target == QuantTarget(**FIXED_TARGET)


@pytest.mark.parametrize("field", list(FREEFORM_FIELDS))
def test_fixed_request_rejects_each_freeform_field(field):
    with pytest.raises(ValidationError, match="must not set"):
        TrainRequestQuant(**{**FIXED_TRAINING_DATA, "quant_target": {**FIXED_TARGET, field: FREEFORM_FIELDS[field]}})


@pytest.mark.parametrize(
    "target",
    [FIXED_TARGET, {**FIXED_TARGET, "bar_cap": 0.04, "bar": 2.08}]
    + [{k: v for k, v in FREEFORM_TARGET.items() if k != field} for field in FREEFORM_FIELDS],
)
def test_freeform_request_requires_the_penalty_fields(target):
    with pytest.raises(ValidationError, match="requires bar_cap, bar, knee, knee_loss, penalty_curvature, cap, cap_loss"):
        TrainRequestQuant(**{**FREEFORM_TRAINING_DATA, "quant_target": target})


def test_quant_request_still_forbids_extra_fields():
    with pytest.raises(ValidationError):
        TrainRequestQuant(**{**FIXED_TRAINING_DATA, "bar": 2.1})
