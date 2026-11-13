"""Chained quant training requests: once released, every player compresses the task's shared input.

A quant request starts from the miner's own starting model when one is set (own-model rounds), else from
the task's (augmented) model: for a released chained task that is the better upstream output, shared by both players.
"""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainRequestQuant
from core.models.quant_models import QuantFormat
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tournament import orchestrator


BOSS = "5Boss"
CHALLENGER = "5Challenger"
SHARED_INPUT = "validator-org/better-upstream-output"
AUGMENTED_MODEL = "validator-org/augmented-model"
CONFIG = SimpleNamespace(psql_db=None, discord_url=None)


def _base_fields() -> dict:
    return {
        "is_organic": False,
        "task_id": uuid.uuid4(),
        "status": "looking_for_nodes",
        "model_id": SHARED_INPUT,
        "ds": "org/dataset",
        "account_id": uuid.uuid4(),
        "hours_to_complete": 2.0,
        "training_data": "s3://bucket/train.json",
        "created_at": datetime(2026, 10, 19),
        "dataset_type": InstructTextDatasetType(field_instruction="instruction", field_output="output"),
        "upstream_task_id": uuid.uuid4(),
    }


def _fixed_task(**overrides) -> QuantFixedRawTask:
    return QuantFixedRawTask(**{**_base_fields(), "quant_spec": QuantFormat.W4A8_FP8_G128.spec(), **overrides})


def _freeform_task(**overrides) -> QuantFreeformRawTask:
    return QuantFreeformRawTask(**{**_base_fields(), **overrides})


@pytest.fixture
def starting_model(monkeypatch):
    lookup = AsyncMock(return_value="validator-org/some-player-model")
    monkeypatch.setattr(orchestrator.task_sql, "get_expected_repo_name", AsyncMock(return_value="expected-repo"))
    monkeypatch.setattr(orchestrator.task_sql, "get_starting_model_repo", lookup)
    monkeypatch.setattr(orchestrator.task_sql, "get_miner_baseline_stats", AsyncMock(return_value=None))
    monkeypatch.setattr(orchestrator, "get_quant_sku_references", AsyncMock(return_value={}))
    monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=None))
    return lookup


async def _request(task, hotkey: str):
    return await orchestrator._create_training_request(
        task,
        hotkey=hotkey,
        available_gpu_ids=[0],
        training_repo="https://github.com/miner/repo",
        training_commit_hash="abc123",
        github_token=None,
        requested_datasets=None,
        config=CONFIG,
    )


class TestChainedTrainingRequest:
    @pytest.mark.parametrize("make_task", [_fixed_task, _freeform_task])
    async def test_both_players_compress_the_shared_input(self, starting_model, make_task):
        starting_model.return_value = None
        task = make_task()

        models = [(await _request(task, hotkey)).training_data for hotkey in (BOSS, CHALLENGER)]

        assert all(isinstance(request, TrainRequestQuant) for request in models)
        assert [request.model for request in models] == [SHARED_INPUT, SHARED_INPUT]
        assert starting_model.await_count == 2

    async def test_augmented_input_is_preferred(self, starting_model):
        starting_model.return_value = None
        request = await _request(_fixed_task(augmented_model_id=AUGMENTED_MODEL), BOSS)

        assert request.training_data.model == AUGMENTED_MODEL

    @pytest.mark.parametrize("make_task", [_fixed_task, _freeform_task])
    async def test_starting_model_wins_when_set(self, starting_model, make_task):
        request = await _request(make_task(augmented_model_id=AUGMENTED_MODEL), BOSS)

        assert request.training_data.model == "validator-org/some-player-model"
