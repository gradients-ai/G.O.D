"""The boss round always forces one of its instruct-text tasks onto a large (35B+) model.

The other two instruct-text tasks (and the DPO/GRPO tasks) draw from the normal standard
model pool - there's no more probabilistic "big model" draw now that one slot is guaranteed.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from core.models.task_models import TaskType
from validator.tournament import constants as t_cst
from validator.tournament import task_creator


def _patch_boss_seams(monkeypatch, get_text_models) -> AsyncMock:
    monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator, "_get_text_models", get_text_models)
    monkeypatch.setattr(task_creator, "_get_instruct_text_datasets", lambda *a, **k: MagicMock())
    monkeypatch.setattr(task_creator, "_get_dpo_datasets", lambda *a, **k: MagicMock())
    monkeypatch.setattr(task_creator, "warn_orphaned_continuous_sft_state", AsyncMock())
    monkeypatch.setattr(task_creator, "_create_continuous_sft_boss_task", AsyncMock(return_value=MagicMock()))

    created = []

    async def fake_create_single(task_type, tournament_id, round_id, pair_id, config, models, *args, **kwargs):
        created.append((task_type, models))
        return SimpleNamespace(task_id="t", task_type=task_type)

    monkeypatch.setattr(task_creator, "_create_single_new_text_task", fake_create_single)
    return created


def _model_pool_tag(keypair, smallest_size_b=0.1, largest_size_b=12.0):
    """A unique, comparable stand-in for the (min, max) size band a model pool was built with."""
    return (smallest_size_b, largest_size_b)


async def test_exactly_one_instruct_task_uses_the_large_model_pool(monkeypatch):
    created = _patch_boss_seams(monkeypatch, _model_pool_tag)

    await task_creator._create_new_text_boss_round_tasks("tourn", "tourn_round_009", MagicMock())

    instruct_pools = [models for task_type, models in created if task_type == TaskType.INSTRUCTTEXTTASK]
    assert len(instruct_pools) == 3

    large_pool = (t_cst.BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B, t_cst.BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B)
    standard_pool = (0.1, 12.0)
    assert instruct_pools.count(large_pool) == 1
    assert instruct_pools.count(standard_pool) == 2


async def test_dpo_and_grpo_tasks_use_the_standard_pool(monkeypatch):
    created = _patch_boss_seams(monkeypatch, _model_pool_tag)

    await task_creator._create_new_text_boss_round_tasks("tourn", "tourn_round_009", MagicMock())

    standard_pool = (0.1, 12.0)
    non_instruct_pools = [models for task_type, models in created if task_type != TaskType.INSTRUCTTEXTTASK]
    assert non_instruct_pools == [standard_pool] * len(non_instruct_pools)


async def test_large_instruct_slot_survives_a_resumed_round(monkeypatch):
    """If 2 of 3 instruct tasks already exist, the resumed call still forces the 3rd onto the large pool."""
    monkeypatch.setattr(
        task_creator,
        "_get_existing_tasks_by_identifier",
        AsyncMock(
            return_value=[
                SimpleNamespace(task_id="existing-1"),
                SimpleNamespace(task_id="existing-2"),
            ]
        ),
    )
    monkeypatch.setattr(
        task_creator.task_sql,
        "get_task",
        AsyncMock(
            side_effect=lambda task_id, psql_db: SimpleNamespace(
                task_type=TaskType.INSTRUCTTEXTTASK, ds="some-dataset", model_id="already-existing-model"
            )
        ),
    )
    monkeypatch.setattr(task_creator, "_get_text_models", _model_pool_tag)
    monkeypatch.setattr(task_creator, "_get_instruct_text_datasets", lambda *a, **k: MagicMock())
    monkeypatch.setattr(task_creator, "_get_dpo_datasets", lambda *a, **k: MagicMock())
    monkeypatch.setattr(task_creator, "warn_orphaned_continuous_sft_state", AsyncMock())
    monkeypatch.setattr(task_creator, "_create_continuous_sft_boss_task", AsyncMock(return_value=MagicMock()))

    created = []

    async def fake_create_single(task_type, tournament_id, round_id, pair_id, config, models, *args, **kwargs):
        created.append((task_type, models))
        return SimpleNamespace(task_id="t", task_type=task_type)

    monkeypatch.setattr(task_creator, "_create_single_new_text_task", fake_create_single)

    await task_creator._create_new_text_boss_round_tasks("tourn", "tourn_round_009", MagicMock())

    instruct_pools = [models for task_type, models in created if task_type == TaskType.INSTRUCTTEXTTASK]
    large_pool = (t_cst.BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B, t_cst.BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B)
    assert instruct_pools == [large_pool]
