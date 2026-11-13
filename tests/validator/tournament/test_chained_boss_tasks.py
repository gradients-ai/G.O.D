"""Chained quant tasks held in AWAITING_UPSTREAM on an instruct task, and replacement keeping the chain intact."""

import random
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from core.constants.environments import TrainingStartPoint
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tasks.models import InstructTextRawTask
from validator.tasks.synthetics import scheduler
from validator.tasks.synthetics.scheduler import draw_quant_format
from validator.tournament import chaining
from validator.tournament import orchestrator
from validator.tournament import task_creator
from validator.tournament.models import TournamentTask


QUANT_TYPES = (TaskType.QUANTFREEFORMTASK, TaskType.QUANTFIXEDTASK)
ROUND_ID = "tourn_round_004"
PAIR_ID = f"{ROUND_ID}_pair_001"
FRESH_COLUMNS = SimpleNamespace(field_instruction="q", field_input=None, field_output="a")


# Flat boss budgets: freeform 4 h; fixed 1.25 h up to 4B, rising linearly to 2 h at 8B.
BOSS_HOURS_3B = {TaskType.QUANTFREEFORMTASK: 2.0, TaskType.QUANTFIXEDTASK: 1.25}

def _seeded_format(slot: int) -> QuantFormat:
    return draw_quant_format(random.Random(f"{ROUND_ID}:quant_format:{TaskType.QUANTFIXEDTASK.value}:{slot}"))


def _instruct_task(**overrides) -> InstructTextRawTask:
    fields = dict(
        task_id=uuid4(),
        model_id="org/boss-instruct-base",
        ds="org/boss-instruct-ds",
        field_system="sys",
        field_instruction="instruction",
        field_input="input",
        field_output="output",
        format="{instruction} {input}",
        no_input_format="{instruction}",
        status=TaskStatus.SUCCESS.value,
        is_organic=False,
        account_id=uuid4(),
        hours_to_complete=6.0,
        created_at=datetime(2026, 1, 1),
        model_params_count=3_000_000_000,
    )
    fields.update(overrides)
    return InstructTextRawTask(**fields)


def _tournament_tasks(task_ids) -> list[TournamentTask]:
    return [TournamentTask(tournament_id="tourn", round_id=ROUND_ID, task_id=task_id, pair_id=PAIR_ID) for task_id in task_ids]


def _with_id(task, psql_db):
    task.task_id = uuid4()
    return task


@pytest.fixture
def scheduler_seams(monkeypatch):
    """Real task construction; persistence, the dataset pool and the hub are faked."""
    monkeypatch.setattr(scheduler, "add_task", AsyncMock(side_effect=_with_id))
    monkeypatch.setattr(scheduler, "get_model_num_params", lambda model_id: 1_500_000_000)
    monkeypatch.setattr(scheduler, "maybe_get_augmentation_config", lambda task_type: None)
    monkeypatch.setattr(scheduler, "get_dataset", AsyncMock(return_value=SimpleNamespace(dataset_id="ds-fresh", num_rows=40_000)))
    monkeypatch.setattr(scheduler, "_get_columns_for_instruct_dataset", AsyncMock(return_value=FRESH_COLUMNS))


async def test_recreate_keeps_upstream_data_and_spec(scheduler_seams):
    original = await scheduler.create_chained_quant_task(
        MagicMock(), TaskType.QUANTFIXEDTASK, _instruct_task(), quant_format=QuantFormat.W3A16_ASYM_G128
    )
    original.status = TaskStatus.PREP_TASK_FAILURE

    fresh = await scheduler.recreate_chained_quant_task(MagicMock(), original)

    assert fresh.task_id != original.task_id
    assert fresh.status == TaskStatus.AWAITING_UPSTREAM
    kept = ("upstream_task_id", "model_id", "ds", "dataset_type", "quant_spec", "hours_to_complete", "model_params_count")
    assert {name: getattr(fresh, name) for name in kept} == {name: getattr(original, name) for name in kept}


async def test_recreate_keeps_the_own_upstream_start_point_and_is_released_per_player(scheduler_seams, monkeypatch):
    original = await scheduler.create_own_upstream_quant_task(
        MagicMock(), TaskType.QUANTFIXEDTASK, _instruct_task(), quant_format=QuantFormat.W3A16_ASYM_G128
    )
    original.status = TaskStatus.PREP_TASK_FAILURE

    fresh = await scheduler.recreate_chained_quant_task(MagicMock(), original)

    assert fresh.training_start_point == TrainingStartPoint.OWN_UPSTREAM_MODEL
    shared, own = AsyncMock(), AsyncMock()
    monkeypatch.setattr(orchestrator, "_release_chained_task", shared)
    monkeypatch.setattr(orchestrator, "_release_own_upstream_task", own)
    monkeypatch.setattr(orchestrator.task_sql, "get_tasks_with_status", AsyncMock(return_value=[fresh]))
    monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value="tourn"))
    monkeypatch.setattr(orchestrator.asyncio, "sleep", AsyncMock(side_effect=StopAsyncIteration))
    with pytest.raises(StopAsyncIteration):
        await orchestrator.process_awaiting_upstream_tasks(SimpleNamespace(psql_db=None))
    own.assert_awaited_once()
    shared.assert_not_awaited()


@pytest.fixture
def replace_seams(monkeypatch):
    monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", AsyncMock())
    monkeypatch.setattr(task_creator.task_sql, "get_nodes_assigned_to_task", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator.task_sql, "detach_task_from_tournament", AsyncMock())
    # The replaced task is not a task of a bound quant tournament here: it has no source row to move.
    monkeypatch.setattr(task_creator.quant_sources_sql, "get_quant_task_source", AsyncMock(return_value=None))
    update = AsyncMock()
    monkeypatch.setattr(task_creator.task_sql, "update_task", update)
    return update


def _round_rows(monkeypatch, tasks) -> None:
    rows = {str(task.task_id): task for task in tasks}
    monkeypatch.setattr(
        task_creator,
        "get_tournament_tasks",
        AsyncMock(return_value=_tournament_tasks(rows)),
    )
    monkeypatch.setattr(task_creator.task_sql, "get_task", AsyncMock(side_effect=lambda task_id, psql_db: rows.get(str(task_id))))


async def test_replacing_a_chained_task_recreates_a_held_copy(monkeypatch, scheduler_seams, replace_seams):
    upstream = _instruct_task()
    chained = await scheduler.create_chained_quant_task(
        MagicMock(), TaskType.QUANTFIXEDTASK, upstream, quant_format=QuantFormat.W4A16_SYM_G128
    )
    _round_rows(monkeypatch, [upstream, chained])

    new_task_id = await task_creator.replace_tournament_task(
        str(chained.task_id), "tourn", ROUND_ID, None, PAIR_ID, MagicMock(), is_final_round=True
    )

    new_task = scheduler.add_task.await_args.args[0]
    assert new_task.task_id == new_task_id != chained.task_id
    assert (new_task.status, new_task.upstream_task_id, new_task.quant_spec) == (
        TaskStatus.AWAITING_UPSTREAM,
        upstream.task_id,
        chained.quant_spec,
    )
    replace_seams.assert_not_awaited()


async def test_same_type_routes_a_chained_task_to_the_chained_copy(monkeypatch, scheduler_seams):
    chained = await scheduler.create_chained_quant_task(
        MagicMock(), TaskType.QUANTFREEFORMTASK, _instruct_task(), quant_format=None
    )

    # Also the route of an R1-R3 task: identical held copy on the same upstream.
    new_task = await task_creator.create_new_task_of_same_type(chained, MagicMock())

    assert new_task.status == TaskStatus.AWAITING_UPSTREAM and new_task.upstream_task_id == chained.upstream_task_id


class TestSharedInputCandidates:
    @pytest.fixture
    def reads(self, monkeypatch):
        statuses = {"5A": "success", "5B": "success", "5C": "failure", "5D": "success", "5E": "success"}
        losses = [
            {"hotkey": "5A", "test_loss": 1.1},
            {"hotkey": "5B", "test_loss": 1.0},
            {"hotkey": "5C", "test_loss": 0.9},
            {"hotkey": "5D", "test_loss": None},  # non-finite losses are read back as None
            {"hotkey": "5E", "test_loss": 0.95},
        ]
        monkeypatch.setattr(
            chaining.tournament_sql, "get_training_status_for_task_and_hotkeys", AsyncMock(return_value=statuses)
        )
        # the eligible set leaves out 5B, whose output was not a fine-tune.
        monkeypatch.setattr(
            chaining.submissions_and_scoring_sql,
            "get_eligible_hotkeys_for_task",
            AsyncMock(return_value={"5A", "5C", "5D", "5E"}),
        )
        monkeypatch.setattr(chaining.submissions_and_scoring_sql, "get_task_node_losses", AsyncMock(return_value=losses))

    async def test_only_usable_runs_best_loss_first(self, reads):
        hotkeys = ["5A", "5B", "5C", "5D", "5E", "5Missing"]

        assert await chaining.shared_input_candidates(uuid4(), hotkeys, MagicMock()) == ["5E", "5A"]

    def test_lower_loss_comes_first_even_over_the_boss(self):
        assert chaining.order_by_upstream_loss({EMISSION_BURN_HOTKEY: 1.2, "5Challenger": 1.1}) == [
            "5Challenger",
            EMISSION_BURN_HOTKEY,
        ]

    @pytest.mark.parametrize("challenger", ["0Challenger", "zChallenger"])
    def test_exact_tie_goes_to_the_challenger(self, challenger):
        assert chaining.order_by_upstream_loss({challenger: 1.0, EMISSION_BURN_HOTKEY: 1.0}) == [
            challenger,
            EMISSION_BURN_HOTKEY,
        ]


async def test_chained_input_repo_is_the_expected_upstream_repo(monkeypatch):
    names = {"5A": "abc123"}
    monkeypatch.setattr(
        chaining.task_sql, "get_expected_repo_name", AsyncMock(side_effect=lambda task_id, hotkey, psql_db: names.get(hotkey))
    )

    assert await chaining.chained_input_repo(uuid4(), "5A", MagicMock()) == f"{chaining.RAYONLABS_HF_USERNAME}/abc123"
    assert await chaining.chained_input_repo(uuid4(), "5B", MagicMock()) is None
