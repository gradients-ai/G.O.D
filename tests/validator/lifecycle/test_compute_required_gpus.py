"""Eval GPU sizing must only request Runpod-sellable counts (1/2/4/8), never 3."""

from datetime import datetime
from uuid import uuid4

import pytest

from core.models.task_models import TaskType
from validator.lifecycle import constants as lifecycle_cst
from validator.lifecycle.tasks import compute_required_gpus
from validator.tasks import requests as task_requests
from validator.tasks.models import InstructTextRawTask


def _instruct_task(*, model_id: str = "org/model", model_params_count: int = 0) -> InstructTextRawTask:
    now = datetime.utcnow()
    return InstructTextRawTask(
        task_id=uuid4(),
        is_organic=False,
        model_id=model_id,
        ds="dataset",
        hours_to_complete=1.0,
        created_at=now,
        account_id=uuid4(),
        status="pending",
        task_type=TaskType.INSTRUCTTEXTTASK,
        model_params_count=model_params_count,
        field_instruction="instruct",
        field_output="output",
    )


@pytest.mark.parametrize(
    ("num_params", "expected_gpus"),
    [
        (1 * 10**9, 1),
        (lifecycle_cst.MODEL_SIZE_REQUIRING_2_GPUS - 1, 1),
        (lifecycle_cst.MODEL_SIZE_REQUIRING_2_GPUS, 2),
        (70_553_706_496, 2),  # cogito-70B / 70B-class boss round
        (lifecycle_cst.MODEL_SIZE_REQUIRING_4_GPUS - 1, 2),
        (lifecycle_cst.MODEL_SIZE_REQUIRING_4_GPUS, 4),
        (100 * 10**9, 4),
        (120 * 10**9, 4),
    ],
)
def test_compute_required_gpus_sizing(num_params: int, expected_gpus: int):
    task = _instruct_task(model_params_count=num_params)
    assert compute_required_gpus(task) == expected_gpus


def test_compute_required_gpus_never_returns_3():
    # Sweep the historical 70-100B band that previously mapped to 3.
    for params in range(70 * 10**9, 101 * 10**9, 5 * 10**9):
        gpus = compute_required_gpus(_instruct_task(model_params_count=params))
        assert gpus != 3
        assert gpus in {1, 2, 4, 8}


def test_compute_required_gpus_uses_hub_count_when_task_count_missing(monkeypatch):
    monkeypatch.setattr(task_requests, "get_model_num_params", lambda _model_id: 70_553_706_496)
    # Import after? compute_required_gpus imports get_model_num_params at module level.
    from validator.lifecycle import tasks as lifecycle_tasks

    monkeypatch.setattr(lifecycle_tasks, "get_model_num_params", lambda _model_id: 70_553_706_496)
    task = _instruct_task(model_id="deepcogito/cogito-v1-preview-llama-70B", model_params_count=0)
    assert compute_required_gpus(task) == 2


def test_parse_model_size_ignores_uuid_hex_fragments():
    # Continuous-SFT training repo names embed UUIDs; the old (\d+)(?=[bB]) regex
    # mis-read "...c8068e356b7c..." as 356B.
    assert task_requests._parse_model_size_from_name(
        "gradients-opensource/god-text-tourn-c8068e356b7c22c8-20260928-position-1"
    ) == 0
    assert task_requests._parse_model_size_from_name("org/llama-70B-chat") == 70_000_000_000
    assert task_requests._parse_model_size_from_name("org/Qwen2.5-7B-Instruct") == 7_000_000_000
