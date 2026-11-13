"""Quant prep's device probes on the baseline stats: readers and writers of different versions stay compatible."""

import json

import pytest
from pydantic import TypeAdapter
from pydantic import ValidationError

from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.model_prep_models import BaselineStats
from core.models.model_prep_models import DeviceProbeStats
from core.models.model_prep_models import InstructBaselineStats
from core.models.payload_models import ModelPrepJob
from core.models.payload_models import ModelPrepResponse
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainRequestQuant
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType


ADAPTER = TypeAdapter(BaselineStats)

SEQ = {"mean": 300.0, "p50": 280, "p95": 600, "p99": 800, "max": 1000}
# What a trainer before the device probes writes.
OLD_STATS = {
    "task_type": "instruct",
    "dataset": {
        "total_tokens": 5_000_000,
        "num_records": 20_000,
        "seq_length_distribution": SEQ,
        "near_duplicate_rate": 0.01,
        "bits_per_byte": 1.1,
        "vocab_size": 32_000,
        "prompt_tokens": 3_000_000,
        "completion_tokens": 2_000_000,
        "completion_length_distribution": SEQ,
    },
    "weights": {"by_group": {}},
    "training": {"init_loss": 2.0, "activation_rms": {}, "output_entropy": 1.0, "masked_completion_loss": 1.5},
    "throughput": {
        "tokens_per_sec": 9_000.0,
        "seq_len": 600,
        "micro_batch_size": 13,
        "n_gpus": 1,
        "gpu_name": "NVIDIA H100 PCIe",
    },
}
DEVICE = {
    "train_tokens_per_sec": 9_100.0,
    "train_window_tokens_per_sec": [9_050.0, 9_100.0, 9_150.0],
    "train_timed_seconds": 20.5,
    "cpu_single_thread_seconds": 1.2,
    "cpu_quota": 8.0,
    "gpu_name": "NVIDIA H100 80GB HBM3",
    "gpu_sm_count": 132,
}


def test_stats_from_a_trainer_without_probes_parse_with_no_device():
    stats = ADAPTER.validate_python(OLD_STATS)
    assert isinstance(stats, InstructBaselineStats)
    assert stats.device is None
    assert stats.throughput.tokens_per_sec == 9_000.0


def test_stats_with_probes_round_trip_through_json():
    stats = ADAPTER.validate_python({**OLD_STATS, "device": DEVICE})
    again = ADAPTER.validate_json(json.dumps(stats.model_dump()))
    assert again == stats
    assert again.device.gpu_sm_count == 132
    assert again.device.cpu_quota == 8.0


def test_quant_prep_stats_may_omit_the_training_dynamics():
    data = {key: value for key, value in OLD_STATS.items() if key != "training"}
    stats = ADAPTER.validate_python({**data, "device": DEVICE})
    assert stats.training is None
    assert ADAPTER.validate_json(json.dumps(stats.model_dump())) == stats


def test_every_probe_field_is_optional():
    assert DeviceProbeStats().model_dump(exclude_none=True) == {}
    assert all(not field.is_required() for field in DeviceProbeStats.model_fields.values())


def test_a_reader_that_predates_a_field_ignores_it():
    """The stats models ignore unknown keys, which is how a validator or trainer process on older code reads newer stats."""
    stats = ADAPTER.validate_python({**OLD_STATS, "device": {**DEVICE, "some_later_probe": 1.0}, "some_later_block": {"x": 1}})
    assert stats.device.train_tokens_per_sec == 9_100.0
    assert "some_later_probe" not in stats.device.model_dump()
    assert "some_later_block" not in stats.model_dump()


def test_stats_recorded_with_dropped_probe_fields_still_parse():
    removed = {"some_removed_probe": 1.0}
    stats = ADAPTER.validate_python({**OLD_STATS, "device": {**DEVICE, **removed}})
    dumped = stats.device.model_dump()
    assert stats.device.train_tokens_per_sec == 9_100.0
    assert not set(removed) & set(dumped)


def test_the_trainer_job_status_carries_the_probes_to_the_validator():
    response = ModelPrepResponse.model_validate_json(
        json.dumps({"augmented_model_id": None, "baseline_stats": {**OLD_STATS, "device": DEVICE}})
    )
    job = ModelPrepJob(task_id="t", model_id="m", gpu_ids=[0], status=TaskStatus.SUCCESS, result=response)
    recovered = ModelPrepJob.model_validate(json.loads(job.model_dump_json()))
    assert recovered.result.baseline_stats.device.train_tokens_per_sec == 9_100.0


def _quant_train_payload(baseline_stats: dict) -> dict:
    training_data = {
        "model": "org/model",
        "task_id": "task-1",
        "hours_to_complete": 1.0,
        "dataset": "s3://bucket/train.json",
        "dataset_type": InstructTextDatasetType().model_dump(),
        "file_format": FileFormat.S3,
        "task_type": TaskType.QUANTFREEFORMTASK,
        "baseline_stats": baseline_stats,
        "quant_target": {
            "reference_model": "m",
            "reference_loss": 2.0,
            "reference_sku": "s",
            "bar_cap": 0.04,
            "bar": 2.08,
            "knee": 1.04,
            "knee_loss": 2.08,
            "penalty_curvature": 2500.0,
            "cap": 1.08,
            "cap_loss": 2.16,
        },
        "calibration": {
            "examples_url": "u",
            "eval_set_fingerprint": "f",
            "n_examples": 1,
            "reference": {"model": "m", "loss": 2.0, "per_example_losses": [1.0], "sku": "s"},
        },
    }
    return {"training_data": training_data, "github_repo": "https://github.com/o/r", "gpu_ids": [0], "hotkey": "hk"}


def test_the_quant_train_request_carries_the_probes_inside_baseline_stats():
    request = TrainerProxyRequest.model_validate(_quant_train_payload({**OLD_STATS, "device": DEVICE}))
    assert isinstance(request.training_data, TrainRequestQuant)
    assert request.training_data.baseline_stats.device.gpu_name == "NVIDIA H100 80GB HBM3"


def test_a_trainer_that_predates_a_probe_field_still_accepts_the_quant_train_request():
    """TrainRequestQuant forbids unknown fields at its own level only: the stats nested in it ignore them, so a trainer on
    older code takes a request with newer stats as a quant request (never the plain-SFT fallback) and drops the extras."""
    stats = {**OLD_STATS, "device": {**DEVICE, "some_later_probe": 1.0}, "some_later_block": {"x": 1}}
    request = TrainerProxyRequest.model_validate(_quant_train_payload(stats))
    assert isinstance(request.training_data, TrainRequestQuant)
    assert request.training_data.baseline_stats.throughput.tokens_per_sec == 9_000.0


def test_probe_fields_never_go_on_the_train_request_itself():
    payload = _quant_train_payload(OLD_STATS)
    payload["training_data"]["device"] = DEVICE
    with pytest.raises(ValidationError):
        TrainerProxyRequest.model_validate(payload)
