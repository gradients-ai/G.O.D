"""The boss round always forces one of its instruct-text tasks onto a large (35B+) model.

The other two instruct-text tasks (and the DPO/GRPO tasks) draw from the normal standard
model pool - there's no more probabilistic "big model" draw now that one slot is guaranteed.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from core.models.task_models import TaskStatus
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


async def test_large_instruct_slot_disables_augmentation(monkeypatch):
    _patch_boss_seams(monkeypatch, _model_pool_tag)
    calls = []

    async def fake_create_single(task_type, tournament_id, round_id, pair_id, config, models, *args, **kwargs):
        calls.append((task_type, models, kwargs.get("allow_augmentation", True)))
        return SimpleNamespace(task_id="t", task_type=task_type)

    monkeypatch.setattr(task_creator, "_create_single_new_text_task", fake_create_single)
    await task_creator._create_new_text_boss_round_tasks("tourn", "tourn_round_009", MagicMock())

    large_pool = (t_cst.BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B, t_cst.BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B)
    instruct_calls = [c for c in calls if c[0] == TaskType.INSTRUCTTEXTTASK]
    assert len(instruct_calls) == 3
    large_calls = [c for c in instruct_calls if c[1] == large_pool]
    other_calls = [c for c in instruct_calls if c[1] != large_pool]
    assert large_calls == [(TaskType.INSTRUCTTEXTTASK, large_pool, False)]
    assert all(allow for _, _, allow in other_calls)


def _patch_replace_seams(monkeypatch, original):
    monkeypatch.setattr(task_creator.task_sql, "get_task", AsyncMock(return_value=original))
    monkeypatch.setattr(task_creator.task_sql, "get_nodes_assigned_to_task", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator.task_sql, "delete_task", AsyncMock())
    monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", AsyncMock())
    monkeypatch.setattr(task_creator, "_get_instruct_text_datasets", lambda *a, **k: MagicMock())
    same_type_mock = AsyncMock()
    monkeypatch.setattr(task_creator, "create_new_task_of_same_type", same_type_mock)
    return same_type_mock


def _large_instruct_original(**overrides):
    original = SimpleNamespace(
        task_id="orig-task",
        task_type=TaskType.INSTRUCTTEXTTASK,
        status=TaskStatus.PREP_TASK_FAILURE.value,
        model_id="NousResearch/Llama-2-70b-hf",
        model_params_count=0,
    )
    for key, value in overrides.items():
        setattr(original, key, value)
    return original


async def test_large_instruct_replacement_redraws_from_large_pool_without_pinning(monkeypatch):
    """Prep failure on the 70B slot must redraw from 35-71B, not retry the failed model or the 1-10B default."""
    original = _large_instruct_original()
    same_type_mock = _patch_replace_seams(monkeypatch, original)
    monkeypatch.setattr(task_creator, "get_model_num_params", lambda model_id: 68_976_648_192)

    captured = {}

    def fake_get_text_models(keypair, smallest_size_b=0.1, largest_size_b=12.0):
        captured["pool"] = (smallest_size_b, largest_size_b)
        return MagicMock()

    monkeypatch.setattr(task_creator, "_get_text_models", fake_get_text_models)
    instruct_mock = AsyncMock(return_value=SimpleNamespace(task_id="new-task", task_type=TaskType.INSTRUCTTEXTTASK))
    monkeypatch.setattr(task_creator, "create_synthetic_instruct_text_task", instruct_mock)

    new_task_id = await task_creator.replace_tournament_task(
        "orig-task", "tourn", "tourn_round_003", None, "pair-1", MagicMock(), is_final_round=True
    )

    same_type_mock.assert_not_awaited()
    assert new_task_id == "new-task"
    assert captured["pool"] == (
        t_cst.BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B,
        t_cst.BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B,
    )
    assert instruct_mock.call_args.kwargs["allow_augmentation"] is False
    assert instruct_mock.call_args.kwargs["enable_kl"] is True
    assert instruct_mock.call_args.kwargs.get("model_id_override") is None


async def test_large_instruct_replacement_uses_persisted_params_without_hub_lookup(monkeypatch):
    original = _large_instruct_original(model_params_count=70_553_706_496)
    same_type_mock = _patch_replace_seams(monkeypatch, original)
    hub = MagicMock(side_effect=AssertionError("should not hit Hugging Face when params are persisted"))
    monkeypatch.setattr(task_creator, "get_model_num_params", hub)
    replacement = AsyncMock(return_value=SimpleNamespace(task_id="new-task", task_type=TaskType.INSTRUCTTEXTTASK))
    monkeypatch.setattr(task_creator, "_create_boss_round_large_instruct_replacement_task", replacement)

    new_task_id = await task_creator.replace_tournament_task(
        "orig-task", "tourn", "tourn_round_003", None, "pair-1", MagicMock(), is_final_round=True
    )

    assert new_task_id == "new-task"
    replacement.assert_awaited_once()
    same_type_mock.assert_not_awaited()
    hub.assert_not_called()


async def test_small_boss_instruct_replacement_does_not_use_the_large_pool(monkeypatch):
    original = _large_instruct_original(model_id="defog/sqlcoder-7b-2", model_params_count=7_000_000_000)
    same_type_mock = _patch_replace_seams(monkeypatch, original)
    same_type_mock.return_value = SimpleNamespace(task_id="redrawn-task", task_type=TaskType.INSTRUCTTEXTTASK)
    large_replace = AsyncMock()
    monkeypatch.setattr(task_creator, "_create_boss_round_large_instruct_replacement_task", large_replace)

    new_task_id = await task_creator.replace_tournament_task(
        "orig-task", "tourn", "tourn_round_003", None, "pair-1", MagicMock(), is_final_round=True
    )

    large_replace.assert_not_awaited()
    same_type_mock.assert_awaited_once()
    assert new_task_id == "redrawn-task"


async def test_non_final_70b_instruct_is_not_treated_as_boss_large_slot(monkeypatch):
    original = _large_instruct_original(model_params_count=68_976_648_192)
    same_type_mock = _patch_replace_seams(monkeypatch, original)
    same_type_mock.return_value = SimpleNamespace(task_id="redrawn-task", task_type=TaskType.INSTRUCTTEXTTASK)
    large_replace = AsyncMock()
    monkeypatch.setattr(task_creator, "_create_boss_round_large_instruct_replacement_task", large_replace)

    new_task_id = await task_creator.replace_tournament_task(
        "orig-task", "tourn", "tourn_round_002", None, "pair-1", MagicMock(), is_final_round=False
    )

    large_replace.assert_not_awaited()
    same_type_mock.assert_awaited_once()
    assert new_task_id == "redrawn-task"
