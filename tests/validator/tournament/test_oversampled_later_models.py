"""Text tournaments oversample the 2026+ model pool into two guaranteed slots.

Round 1: exactly one task across all groups plays on a model from OVERSAMPLED_LATER_MODELS,
whatever the group count is. Boss round: the first of the two instruct tasks does.

Both slots are drawn from a round_id-seeded rng, so a partially created round that gets retried
resumes onto the same slot and the same model rather than minting a second oversampled task.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from core.models.task_models import TaskType
from core.oversampled_later_models import OVERSAMPLED_LATER_MODELS
from core.oversampled_later_models import sample_oversampled_later_model
from validator.tournament import constants as t_cst
from validator.tournament import task_creator
from validator.tournament.models import Group
from validator.tournament.models import GroupRound


POOL_IDS = {m.model_id for m in OVERSAMPLED_LATER_MODELS}


def _group_round(num_groups: int, members_per_group: int = 12, round_number: int = 1) -> GroupRound:
    return GroupRound(
        round_id="tourn_round_001",
        round_number=round_number,
        groups=[
            Group(member_ids=[f"miner-{g}-{m}" for m in range(members_per_group)])
            for g in range(num_groups)
        ],
    )


def _patch_seams(monkeypatch, existing: list | None = None) -> AsyncMock:
    """Stub pool generators, DB lookups and registration; return the instruct-creation mock."""
    for name in ("_get_text_models", "_get_instruct_text_datasets", "_get_dpo_datasets"):
        monkeypatch.setattr(task_creator, name, lambda *a, **k: MagicMock())
    monkeypatch.setattr(
        task_creator, "_get_existing_tasks_by_identifier", AsyncMock(return_value=existing or [])
    )
    monkeypatch.setattr(task_creator, "_get_existing_tasks", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", AsyncMock())

    instruct_mock = AsyncMock(
        return_value=SimpleNamespace(task_id="task", task_type=TaskType.INSTRUCTTEXTTASK)
    )
    monkeypatch.setattr(task_creator, "create_synthetic_instruct_text_task", instruct_mock)
    return instruct_mock


def _overrides(instruct_mock: AsyncMock) -> list[str | None]:
    return [call.kwargs.get("model_id_override") for call in instruct_mock.call_args_list]


@pytest.mark.parametrize("num_groups", [2, 3, 8])
async def test_round_one_gives_exactly_one_task_an_oversampled_model(monkeypatch, num_groups):
    instruct_mock = _patch_seams(monkeypatch)

    await task_creator._create_group_text_tasks(
        _group_round(num_groups), "tourn", MagicMock(), is_final_round=False
    )

    overrides = _overrides(instruct_mock)
    assert len(overrides) == num_groups * t_cst.TEXT_TASKS_PER_GROUP
    picked = [o for o in overrides if o is not None]
    assert len(picked) == 1
    assert picked[0] in POOL_IDS


async def test_small_tournament_round_one_still_gets_exactly_one(monkeypatch):
    """A single group in the small band plays SMALL_TOURNAMENT_GROUP_TASKS matches — still one
    oversampled task in total, not one per match."""
    instruct_mock = _patch_seams(monkeypatch)

    await task_creator._create_group_text_tasks(
        _group_round(1, members_per_group=8), "tourn", MagicMock(), is_final_round=False
    )

    overrides = _overrides(instruct_mock)
    assert len(overrides) == t_cst.SMALL_TOURNAMENT_GROUP_TASKS
    picked = [o for o in overrides if o is not None]
    assert len(picked) == 1
    assert picked[0] in POOL_IDS


async def test_round_one_skips_when_the_round_already_has_a_pool_model_task(monkeypatch):
    """Whatever mints an extra task mid-round, R1 never hands out a second oversampled slot."""
    instruct_mock = _patch_seams(monkeypatch)
    monkeypatch.setattr(
        task_creator, "_round_already_has_oversampled_task", AsyncMock(return_value=True)
    )

    await task_creator._create_group_text_tasks(_group_round(3), "tourn", MagicMock(), is_final_round=False)

    assert _overrides(instruct_mock) == [None, None, None]


async def test_later_group_rounds_do_not_oversample(monkeypatch):
    instruct_mock = _patch_seams(monkeypatch)

    await task_creator._create_group_text_tasks(
        _group_round(3, round_number=2), "tourn", MagicMock(), is_final_round=False
    )

    assert _overrides(instruct_mock) == [None, None, None]


async def test_round_one_pick_is_stable_across_retries(monkeypatch):
    """A round re-created from scratch lands on the same slot and the same model."""
    first = _patch_seams(monkeypatch)
    await task_creator._create_group_text_tasks(_group_round(5), "tourn", MagicMock(), is_final_round=False)
    second = _patch_seams(monkeypatch)
    await task_creator._create_group_text_tasks(_group_round(5), "tourn", MagicMock(), is_final_round=False)

    assert _overrides(first) == _overrides(second)


def _patch_boss_seams(monkeypatch) -> AsyncMock:
    monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", AsyncMock(return_value=[]))
    for name in ("_get_text_models", "_get_instruct_text_datasets", "_get_dpo_datasets"):
        monkeypatch.setattr(task_creator, name, lambda *a, **k: MagicMock())
    monkeypatch.setattr(task_creator, "warn_orphaned_continuous_sft_state", AsyncMock())
    monkeypatch.setattr(task_creator, "_create_continuous_sft_boss_task", AsyncMock(return_value=MagicMock()))
    created = AsyncMock(side_effect=lambda task_type, *a, **k: SimpleNamespace(task_id="t", task_type=task_type))
    monkeypatch.setattr(task_creator, "_create_single_new_text_task", created)
    return created


async def test_boss_round_gives_exactly_one_task_an_oversampled_model(monkeypatch):
    created = _patch_boss_seams(monkeypatch)

    await task_creator._create_new_text_boss_round_tasks("tourn", "tourn_round_009", MagicMock())

    overrides = [c.kwargs.get("model_id_override") for c in created.call_args_list]
    assert len(overrides) == sum(t_cst.FINAL_ROUND_TEXT_TASK_DISTRIBUTION.values())
    picked = [o for o in overrides if o is not None]
    assert len(picked) == 1
    assert picked[0] in POOL_IDS


async def test_boss_round_oversampled_slot_is_not_always_instruct(monkeypatch):
    """The slot is drawn over instruct/dpo/grpo, not pinned to instruct."""
    seen: set[TaskType] = set()
    for round_number in range(40):
        created = _patch_boss_seams(monkeypatch)
        await task_creator._create_new_text_boss_round_tasks(
            "tourn", f"tourn_round_{round_number:03d}", MagicMock()
        )
        seen.update(
            call.args[0] for call in created.call_args_list if call.kwargs.get("model_id_override")
        )

    assert seen == set(t_cst.FINAL_ROUND_TEXT_TASK_DISTRIBUTION)


@pytest.mark.parametrize(
    "existing_model_id, expected_picks",
    [
        (next(iter(POOL_IDS)), 0),  # the oversampled task already exists -> don't mint a second
        ("Qwen/Qwen2.5-7B-Instruct", 1),  # a normal draw exists -> the round still owes one
    ],
)
async def test_boss_round_resume_keeps_exactly_one_oversampled_task(
    monkeypatch, existing_model_id, expected_picks
):
    """A partially created round is completed by looking at what it already plays, not at how many
    tasks have been created."""
    existing = [SimpleNamespace(task_id="existing")]
    monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", AsyncMock(return_value=existing))
    monkeypatch.setattr(
        task_creator.task_sql,
        "get_task",
        AsyncMock(
            return_value=SimpleNamespace(
                task_id="existing", task_type=TaskType.INSTRUCTTEXTTASK, ds="ds", model_id=existing_model_id
            )
        ),
    )
    monkeypatch.setattr(t_cst, "is_continuous_sft_task", lambda task: False)
    for name in ("_get_text_models", "_get_instruct_text_datasets", "_get_dpo_datasets"):
        monkeypatch.setattr(task_creator, name, lambda *a, **k: MagicMock())
    monkeypatch.setattr(task_creator, "warn_orphaned_continuous_sft_state", AsyncMock())
    monkeypatch.setattr(task_creator, "_create_continuous_sft_boss_task", AsyncMock(return_value=MagicMock()))
    created = AsyncMock(side_effect=lambda task_type, *a, **k: SimpleNamespace(task_id="t", task_type=task_type))
    monkeypatch.setattr(task_creator, "_create_single_new_text_task", created)

    await task_creator._create_new_text_boss_round_tasks("tourn", "tourn_round_009", MagicMock())

    instruct_calls = [c for c in created.call_args_list if c.args[0] == TaskType.INSTRUCTTEXTTASK]
    assert len(instruct_calls) == 1  # the round had one already
    picks = [c for c in created.call_args_list if c.kwargs.get("model_id_override") in POOL_IDS]
    assert len(picks) == expected_picks


async def test_draw_deciders_never_get_an_oversampled_model(monkeypatch):
    """A decider restores a decided result on the drawn task's type — it is not a second chance to
    put the boss round on a pool model, however many tasks draw."""
    for name in ("_get_text_models", "_get_instruct_text_datasets", "_get_dpo_datasets"):
        monkeypatch.setattr(task_creator, name, lambda *a, **k: MagicMock())
    created = AsyncMock(side_effect=lambda task_type, *a, **k: SimpleNamespace(task_id="t", task_type=task_type))
    monkeypatch.setattr(task_creator, "_create_single_new_text_task", created)

    await task_creator.create_boss_round_decider_tasks(
        "tourn",
        "tourn_round_009",
        MagicMock(),
        [TaskType.INSTRUCTTEXTTASK, TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK],
    )

    assert [c.kwargs.get("model_id_override") for c in created.call_args_list] == [None, None, None]


def test_pool_is_sampled_uniformly_and_is_seedable():
    import random

    seeded = random.Random("tourn_round_001:oversampled_model")
    assert sample_oversampled_later_model(seeded) == sample_oversampled_later_model(
        random.Random("tourn_round_001:oversampled_model")
    )
    drawn = {sample_oversampled_later_model(random.Random(i)) for i in range(500)}
    assert drawn == POOL_IDS
