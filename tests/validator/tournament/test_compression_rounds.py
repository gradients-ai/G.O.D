"""The three miner rounds of a quant tournament on the text boss's models (validator/tournament/README.md).

R1 freeform on the text tournament's text_r1 seed, R2 fixed on text_r2, R3 freeform on text_r3: one chained quant task per
round, a new task held until the text boss's model has landed, released on that model with the seed's split and calibration
and non-boss hours. The round's task is normally its quant seed (attached; see test_quant_text_linkage.py); the creation tests
here take a round whose quant seed role is `failed`, which creates the same task without the boss. Cuts 8 / 4 / 1 by average
rank (ties by the most recent round) among the challengers, the boss ranked alongside them in each round, the seed hiding
rule and the endpoints that apply it.
"""

import math
import random
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import UUID
from uuid import uuid4

import asyncpg
import pytest

import validator.tournament.round_results as rr
from core.constants.environments import TrainingStartPoint
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from tests.validator.tournament.quant_source_fakes import FakeQuantSourceStore
from validator.db.sql import tournament_seeds as seeds_sql
from validator.endpoints import tasks as tasks_endpoint
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.tasks.details import hide_sensitive_data_till_finished
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import InstructTextTask
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tasks.synthetics import scheduler
from validator.tasks.synthetics.scheduler import draw_quant_format
from validator.tasks.synthetics.scheduler import quant_training_hours
from validator.tournament import constants as t_cst
from validator.tournament import orchestrator
from validator.tournament import task_creator
from validator.tournament.boss_rank_shares import compare_with_boss
from validator.tournament.models import INPUT_SEED_ROLES
from validator.tournament.models import Group
from validator.tournament.models import GroupRound
from validator.tournament.models import QuantRoundSpec
from validator.tournament.models import QuantSourceSlot
from validator.tournament.models import RoundType
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType
from validator.tournament.tournament_manager import _round_field_without_boss
from validator.tournament.tournament_manager import organise_tournament_round


TID = "tourn"
TEXT_TID = "text_tourn"
PARAMS = 1_500_000_000
CALIBRATION = "https://minio/seed_calibration.json"


# --------------------------------------------------------------------------------------------- the round plan


def test_the_plan_is_freeform_on_text_r1_then_fixed_on_text_r2_then_freeform_on_text_r3_with_cuts_8_4_1():
    assert t_cst.QUANT_ROUND_PLAN == (
        QuantRoundSpec(
            round_number=1,
            slot=QuantSourceSlot.R1,
            task_type=TaskType.QUANTFREEFORMTASK,
            input_role=SeedRole.TEXT_R1,
            advance=8,
        ),
        QuantRoundSpec(
            round_number=2,
            slot=QuantSourceSlot.R2,
            task_type=TaskType.QUANTFIXEDTASK,
            input_role=SeedRole.TEXT_R2,
            advance=4,
            quant_format=QuantFormat.W2A16_ASYM_G64,
        ),
        QuantRoundSpec(
            round_number=3,
            slot=QuantSourceSlot.R3,
            task_type=TaskType.QUANTFREEFORMTASK,
            input_role=SeedRole.TEXT_R3,
            advance=1,
        ),
    )
    assert (t_cst.QUANT_ROUND_ONE_ADVANCE, t_cst.QUANT_ROUND_TWO_ADVANCE, t_cst.QUANT_ROUND_THREE_ADVANCE) == (8, 4, 1)
    assert [spec.advance for spec in t_cst.QUANT_ROUND_PLAN] == [
        t_cst.QUANT_ROUND_ONE_ADVANCE,
        t_cst.QUANT_ROUND_TWO_ADVANCE,
        t_cst.QUANT_ROUND_THREE_ADVANCE,
    ]


def test_a_round_outside_the_plan_has_no_spec():
    with pytest.raises(ValueError, match="not a miner round"):
        task_creator.quant_round_spec(4)


@pytest.mark.parametrize("round_number", [1, 2, 3])
def test_each_miner_round_is_one_group_without_the_boss(round_number):
    nodes = [SimpleNamespace(hotkey=f"hk_{i:02d}") for i in range(40)]

    round_data = organise_tournament_round(
        nodes, MagicMock(), TournamentType.QUANT, round_id=f"{TID}_round_{round_number:03d}", round_number=round_number
    )

    assert isinstance(round_data, GroupRound) and len(round_data.groups) == 1
    assert _round_field_without_boss([*(n.hotkey for n in nodes), EMISSION_BURN_HOTKEY]) == [n.hotkey for n in nodes]


# --------------------------------------------------------------------------------------------- creation


def _seed_task(**overrides) -> InstructTextRawTask:
    fields = dict(
        task_id=uuid4(),
        model_id="org/base",
        ds="org/ds",
        field_system="sys",
        field_instruction="instruction",
        field_input="input",
        field_output="output",
        format="{instruction} {input}",
        no_input_format="{instruction}",
        status=TaskStatus.TRAINING,
        is_organic=False,
        account_id=uuid4(),
        hours_to_complete=6.0,
        created_at=datetime(2026, 1, 1),
        model_params_count=PARAMS,
        training_data="s3://bucket/seed_train.json",
        test_data="s3://bucket/seed_test.json",
        calibration_data=CALIBRATION,
    )
    fields.update(overrides)
    return InstructTextRawTask(**fields)


def _seed_row(role: SeedRole, task: InstructTextRawTask) -> TournamentSeedTask:
    return TournamentSeedTask(tournament_id=TID, role=role, attempt=1, task_id=task.task_id, status=SeedStatus.TRAINING)


def _round(round_number: int, members: tuple[str, ...] = ("hk_a", "hk_b")) -> GroupRound:
    return GroupRound(
        round_id=f"{TID}_round_{round_number:03d}", round_number=round_number, groups=[Group(member_ids=list(members))]
    )


def _text_row(role: SeedRole, task: InstructTextRawTask) -> TournamentSeedTask:
    """A live seed row of the bound TEXT tournament."""
    return TournamentSeedTask(tournament_id=TEXT_TID, role=role, attempt=1, task_id=task.task_id, status=SeedStatus.TRAINING)


@pytest.fixture
def creation(monkeypatch):
    """The bound text tournament has its round seeds; persistence is faked, non-boss hours are spied on."""
    seeds = {
        SeedRole.TEXT_R1: _seed_task(),
        SeedRole.TEXT_R2: _seed_task(),
        SeedRole.TEXT_R3: _seed_task(model_params_count=6_000_000_000),
    }
    rows = [_text_row(role, task) for role, task in seeds.items()]
    # The quant tournament's round seed roles are out of replacements: each round creates its own task.
    failed_quant_rows = [
        TournamentSeedTask(tournament_id=TID, role=role, attempt=3, task_id=uuid4(), status=SeedStatus.FAILED)
        for role in (SeedRole.QUANT_R1, SeedRole.QUANT_R2, SeedRole.QUANT_R3)
    ]
    by_id = {task.task_id: task for task in seeds.values()}
    store = FakeQuantSourceStore().install(monkeypatch)
    store.bind(TID, TEXT_TID)
    register = AsyncMock()
    hours_calls: list[bool] = []
    real_hours = scheduler.quant_training_hours

    def spy_hours(task_type, model_params_count, boss, baseline_stats=None):
        hours_calls.append(boss)
        return real_hours(task_type, model_params_count, boss, baseline_stats)

    monkeypatch.setattr(scheduler, "quant_training_hours", spy_hours)

    async def add_with_id(task, psql_db):
        task = task.model_copy(update={"task_id": uuid4()})
        by_id[task.task_id] = task
        return task

    monkeypatch.setattr(scheduler, "add_task", add_with_id)
    monkeypatch.setattr(task_creator.seeds_sql, "get_active_seed_tasks", AsyncMock(return_value=rows))
    monkeypatch.setattr(task_creator.seeds_sql, "get_latest_seed_tasks", AsyncMock(return_value=rows + failed_quant_rows))
    monkeypatch.setattr(task_creator.seeds_sql, "get_seed_task_by_task_id", AsyncMock(return_value=None))
    monkeypatch.setattr(task_creator.tournament_sql, "get_tournament_task_by_task_id", AsyncMock(return_value=None))
    monkeypatch.setattr(task_creator.task_sql, "get_task", AsyncMock(side_effect=lambda task_id, psql_db: by_id.get(task_id)))
    monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator, "_get_existing_tasks", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", register)
    return SimpleNamespace(
        seeds=seeds,
        rows=rows,
        failed_quant_rows=failed_quant_rows,
        register=register,
        hours_calls=hours_calls,
        store=store,
        by_id=by_id,
    )


async def _create(round_number: int) -> None:
    await task_creator.create_quant_tournament_tasks(_round(round_number), TID, MagicMock())


@pytest.mark.parametrize(
    ("round_number", "task_type", "input_role"),
    [
        (1, TaskType.QUANTFREEFORMTASK, SeedRole.TEXT_R1),
        (2, TaskType.QUANTFIXEDTASK, SeedRole.TEXT_R2),
        (3, TaskType.QUANTFREEFORMTASK, SeedRole.TEXT_R3),
    ],
)
async def test_each_round_creates_one_held_chained_task_on_its_seed(creation, round_number, task_type, input_role):
    await _create(round_number)

    (call,) = creation.register.call_args_list
    task = call.args[0]
    seed = creation.seeds[input_role]
    assert task.task_type == task_type
    assert task.status == TaskStatus.AWAITING_UPSTREAM
    assert task.upstream_task_id == seed.task_id
    assert task.model_params_count == seed.model_params_count
    assert task.training_start_point == TrainingStartPoint.DEFAULT
    assert call.kwargs["group_id"] == f"{TID}_round_{round_number:03d}_group_001"
    assert creation.hours_calls == [False]  # the non-boss budget, never the boss one
    # The task is a new task of its own, linked to the text tournament's seed task by a persisted source row.
    assert task.task_id is not None and task.task_id != seed.task_id
    source = creation.store.live(TID, QuantSourceSlot(f"r{round_number}"))
    assert (source.quant_task_id, source.source_tournament_id, source.source_role, source.source_task_id) == (
        task.task_id,
        TEXT_TID,
        input_role,
        seed.task_id,
    )
    assert source.source_model_repo is None and not source.source_is_fallback


async def test_round_two_is_always_w2_whatever_the_round_seeded_rng_would_draw(creation, monkeypatch):
    await _create(2)

    task = creation.register.call_args.args[0]
    assert task.quant_spec == QuantFormat.W2A16_ASYM_G64.spec()
    expected = quant_training_hours(TaskType.QUANTFIXEDTASK, PARAMS, boss=False)
    assert task.hours_to_complete == expected

    # The pin wins over any rng: force the draw to the other launch format and over many round ids.
    monkeypatch.setattr(task_creator, "draw_quant_format", lambda rng: QuantFormat.W3A16_ASYM_G128)
    for tournament_id in ("tourn_a", "tourn_b", "tourn_c", "tourn_d"):
        creation.register.reset_mock()
        creation.store.bind(tournament_id, f"text_{tournament_id}")
        await task_creator.create_quant_tournament_tasks(_round(2), tournament_id, MagicMock())
        assert creation.register.call_args.args[0].quant_spec == QuantFormat.W2A16_ASYM_G64.spec()


async def test_a_fixed_round_without_a_pinned_format_keeps_the_round_seeded_draw(creation, monkeypatch):
    monkeypatch.setattr(
        t_cst, "QUANT_ROUND_PLAN", tuple(spec.model_copy(update={"quant_format": None}) for spec in t_cst.QUANT_ROUND_PLAN)
    )
    await _create(2)

    task = creation.register.call_args.args[0]
    assert task.quant_spec == draw_quant_format(random.Random(f"{TID}_round_002:quant_format:0")).spec()


@pytest.mark.parametrize("round_number", [1, 3])
async def test_freeform_rounds_have_no_spec(creation, round_number):
    await _create(round_number)

    task = creation.register.call_args.args[0]
    assert task.task_type == TaskType.QUANTFREEFORMTASK
    assert not hasattr(task, "quant_spec")


async def test_rerun_with_the_round_task_present_creates_nothing(creation, monkeypatch):
    existing = [SimpleNamespace(task_id="t", task_type=TaskType.QUANTFREEFORMTASK)]
    monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", AsyncMock(return_value=[MagicMock()]))
    monkeypatch.setattr(task_creator, "_get_existing_tasks", AsyncMock(return_value=existing))

    created = await task_creator._create_seed_input_round_task(_round(1), TID, MagicMock())

    assert created == existing
    creation.register.assert_not_awaited()


async def test_a_missing_text_seed_keeps_the_round_pending(creation, monkeypatch):
    monkeypatch.setattr(task_creator.seeds_sql, "get_active_seed_tasks", AsyncMock(return_value=[creation.rows[1]]))
    monkeypatch.setattr(
        task_creator.seeds_sql, "get_latest_seed_tasks", AsyncMock(return_value=[creation.rows[1], *creation.failed_quant_rows])
    )
    # the missing role's row is gone from the store: nothing final, and the text tournament is still running

    with pytest.raises(task_creator.MissingBossRoundSeeds) as error:
        await _create(1)

    assert error.value.roles == ["text_r1"]
    assert creation.store.sources == [] and creation.register.await_count == 0


async def test_no_instruct_task_is_created_for_any_miner_round(creation, monkeypatch):
    instruct = AsyncMock()
    monkeypatch.setattr(task_creator, "create_synthetic_instruct_text_task", instruct)

    for round_number in (1, 2, 3):
        await _create(round_number)

    instruct.assert_not_awaited()


# --------------------------------------------------------------------------------------------- release


@pytest.fixture
def release(monkeypatch):
    """R1 held on the small input seed; the boss (and nobody else) has an output on the seed."""
    seed = _seed_task(status=TaskStatus.TRAINING)
    mocks = SimpleNamespace(
        seed=seed,
        candidates=AsyncMock(return_value=[EMISSION_BURN_HOTKEY]),
        full_model=AsyncMock(return_value=True),
        update_task=AsyncMock(),
        nodes=AsyncMock(return_value=[SimpleNamespace(hotkey="hk_a"), SimpleNamespace(hotkey="hk_b")]),
    )
    monkeypatch.setattr(orchestrator.task_sql, "get_task", AsyncMock(return_value=seed))
    monkeypatch.setattr(orchestrator.task_sql, "get_nodes_assigned_to_task", mocks.nodes)
    monkeypatch.setattr(orchestrator.task_sql, "update_task", mocks.update_task)
    monkeypatch.setattr(
        orchestrator.seeds_sql,
        "get_seed_task_by_task_id",
        AsyncMock(return_value=_seed_row(SeedRole.INPUT_SMALL, seed)),
    )
    monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=TID))
    monkeypatch.setattr(orchestrator, "is_final_round_task", AsyncMock(return_value=False))
    monkeypatch.setattr(orchestrator.chaining, "shared_input_candidates", mocks.candidates)
    monkeypatch.setattr(orchestrator.chaining.task_sql, "get_expected_repo_name", AsyncMock(return_value="boss-i1-repo"))
    monkeypatch.setattr(orchestrator, "_is_full_model_repo", mocks.full_model)
    return mocks


def _held_round_task(seed: InstructTextRawTask, task_type: TaskType = TaskType.QUANTFREEFORMTASK):
    fields = dict(
        task_id=uuid4(),
        model_id=seed.model_id,
        ds=seed.ds,
        status=TaskStatus.AWAITING_UPSTREAM,
        upstream_task_id=seed.task_id,
        hours_to_complete=99.0,
        is_organic=False,
        account_id=uuid4(),
        created_at=datetime(2026, 1, 1),
        model_params_count=seed.model_params_count,
        dataset_type=orchestrator.InstructTextDatasetType(field_instruction="raw"),
    )
    if task_type == TaskType.QUANTFIXEDTASK:
        return QuantFixedRawTask(**fields, quant_spec=draw_quant_format(random.Random("x")).spec())
    return QuantFreeformRawTask(**fields)


@pytest.mark.parametrize("status", [TaskStatus.PENDING, TaskStatus.TRAINING, TaskStatus.PREEVALUATION, TaskStatus.EVALUATING])
async def test_r1_waits_for_i1_to_be_success(release, status):
    release.seed.status = status
    task = _held_round_task(release.seed)

    await orchestrator._release_chained_task(task, MagicMock())

    assert task.status == TaskStatus.AWAITING_UPSTREAM
    release.candidates.assert_not_awaited()
    release.update_task.assert_not_awaited()


async def test_r1_is_released_onto_the_boss_output_with_the_seeds_data_and_non_boss_hours(release):
    release.seed.status = TaskStatus.SUCCESS
    task = _held_round_task(release.seed)
    before = datetime.utcnow()

    await orchestrator._release_chained_task(task, MagicMock())

    # The miners of the round have no run on the seed: the boss's output is the only candidate, and a full model.
    assert release.candidates.await_args.args[:2] == (release.seed.task_id, [EMISSION_BURN_HOTKEY])
    release.full_model.assert_awaited_once_with("gradients-io-tournaments/boss-i1-repo")
    assert task.model_id == "gradients-io-tournaments/boss-i1-repo"
    assert task.status == TaskStatus.PENDING
    assert (task.training_data, task.test_data, task.calibration_data) == (
        release.seed.training_data,
        release.seed.test_data,
        CALIBRATION,
    )
    expected = quant_training_hours(TaskType.QUANTFREEFORMTASK, PARAMS, boss=False, baseline_stats=None)
    assert task.hours_to_complete == expected != 99.0
    assert task.termination_at >= before + orchestrator.timedelta(hours=expected)
    release.update_task.assert_awaited_once()


async def test_a_seed_without_a_usable_boss_output_holds_the_round_task_instead_of_failing_it(release):
    release.seed.status = TaskStatus.SUCCESS
    release.candidates.return_value = []
    task = _held_round_task(release.seed, TaskType.QUANTFIXEDTASK)

    await orchestrator._release_chained_task(task, MagicMock())

    assert task.status == TaskStatus.AWAITING_UPSTREAM
    release.update_task.assert_not_awaited()


async def test_the_held_task_has_its_nodes_before_the_release(release):
    """Nodes are assigned while the task is held (a held task does not stall the round); no nodes, no release."""
    release.seed.status = TaskStatus.SUCCESS
    release.nodes.return_value = []
    task = _held_round_task(release.seed)

    await orchestrator._release_chained_task(task, MagicMock())

    assert task.status == TaskStatus.AWAITING_UPSTREAM
    release.candidates.assert_not_awaited()


async def test_a_held_round_task_does_not_block_node_assignment_checks(monkeypatch):
    """check_if_all_tasks_have_nodes_assigned reads only the assigned nodes, never the task status."""
    from validator.tournament import tournament_manager

    monkeypatch.setattr(
        tournament_manager,
        "get_tournament_tasks",
        AsyncMock(return_value=[TournamentTask(tournament_id=TID, round_id="r", task_id="held")]),
    )
    monkeypatch.setattr(
        tournament_manager.task_sql, "get_nodes_assigned_to_task", AsyncMock(return_value=[SimpleNamespace(hotkey="hk_a")])
    )

    assert await tournament_manager.check_if_all_tasks_have_nodes_assigned("r", MagicMock()) is True


# --------------------------------------------------------------------------------------------- replacement


async def test_replacing_a_round_task_is_an_identical_held_copy_on_the_same_seed(creation, monkeypatch):
    await _create(2)
    original = creation.register.call_args.args[0]
    monkeypatch.setattr(task_creator.task_sql, "get_task", AsyncMock(return_value=original))
    monkeypatch.setattr(task_creator.task_sql, "get_nodes_assigned_to_task", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator.task_sql, "detach_task_from_tournament", AsyncMock())
    monkeypatch.setattr(task_creator, "get_tournament_tasks", AsyncMock(return_value=[]))
    register = AsyncMock()
    monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", register)

    new_id = await task_creator.replace_tournament_task(
        str(original.task_id), TID, f"{TID}_round_002", f"{TID}_round_002_group_001", None, MagicMock()
    )

    new_task = register.call_args.args[0]
    assert new_id == new_task.task_id != original.task_id
    assert new_task.status == TaskStatus.AWAITING_UPSTREAM
    assert new_task.upstream_task_id == original.upstream_task_id
    assert (new_task.task_type, new_task.quant_spec, new_task.ds) == (original.task_type, original.quant_spec, original.ds)
    assert new_task.quant_spec == QuantFormat.W2A16_ASYM_G64.spec()  # a replaced R2 stays on its fixed format
    assert new_task.hours_to_complete == original.hours_to_complete  # still the non-boss budget
    assert register.call_args.kwargs["group_id"] == f"{TID}_round_002_group_001"
    # The slot's link moves to the replacement and the replaced task's row is superseded.
    live = creation.store.live(TID, QuantSourceSlot.R2)
    old = next(row for row in creation.store.sources if row.quant_task_id == original.task_id)
    assert live.quant_task_id == new_id and old.superseded and live.source_task_id == old.source_task_id


# --------------------------------------------------------------------------------------------- cuts and standings

T1_TASK = "11111111-2222-3333-4444-555555555555"
T2_TASK = "21111111-2222-3333-4444-555555555555"
T3_TASK = "31111111-2222-3333-4444-555555555555"


def _freeform(hotkey: str, total_bytes: int) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=1.0,
        synth_loss=1.0,
        is_finetune=True,
        task_type=TaskType.QUANTFREEFORMTASK,
        artifact_bytes=total_bytes,
        decoder_bytes=0,
        passed_bar=True,
        reference_loss=1.0,
    )


def _freeform_fail(hotkey: str) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=math.nan,
        synth_loss=math.nan,
        is_finetune=True,
        task_type=TaskType.QUANTFREEFORMTASK,
        quant_failure="did not load",
    )


def _fixed(hotkey: str, loss: float) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey, test_loss=loss, synth_loss=loss, is_finetune=True, task_type=TaskType.QUANTFIXEDTASK, reference_loss=1.0
    )


def _round_data(number: int, is_final: bool = False) -> TournamentRoundData:
    return TournamentRoundData(
        round_id=f"r{number}",
        tournament_id=TID,
        round_number=number,
        round_type=RoundType.GROUP,
        is_final_round=is_final,
    )


def _round_task(number: int, task_id: str) -> TournamentTask:
    return TournamentTask(tournament_id=TID, round_id=f"r{number}", task_id=task_id, group_id=f"g{number}")


@pytest.fixture
def tournament(monkeypatch):
    """Fake a tournament DB: results per task, cohort per round (round number -> hotkeys)."""

    def _wire(results: dict[str, list[MinerResultsText]], cohorts: dict[int, list[str]]):
        task_ids = {1: T1_TASK, 2: T2_TASK, 3: T3_TASK}

        async def fake_results(task_id, psql_db):
            return [result.model_copy() for result in results[task_id]]

        async def fake_tasks(round_id, psql_db):
            number = int(round_id.removeprefix("r"))
            return [_round_task(number, task_ids[number])]

        async def fake_members(group_id, psql_db):
            return [SimpleNamespace(hotkey=hotkey) for hotkey in cohorts[int(group_id.removeprefix("g"))]]

        monkeypatch.setattr(rr, "get_task_results_for_ranking", fake_results)
        monkeypatch.setattr(rr, "get_task", AsyncMock(return_value=SimpleNamespace(task_type=TaskType.QUANTFREEFORMTASK)))
        rounds = AsyncMock(side_effect=lambda *_: [_round_data(n) for n in sorted(cohorts)])
        monkeypatch.setattr(rr, "get_tournament_rounds", rounds)
        monkeypatch.setattr(rr, "get_tournament_tasks", fake_tasks)
        monkeypatch.setattr(rr, "get_tournament_group_members", fake_members)
        monkeypatch.setattr(
            rr, "get_tournament", AsyncMock(return_value=TournamentData(tournament_id=TID, tournament_type=TournamentType.QUANT))
        )

    return _wire


def _by_bytes(order: list[str]) -> list[MinerResultsText]:
    """Freeform results in the given best-first order (fewer bytes is better)."""
    return [_freeform(hotkey, 100 * (i + 1)) for i, hotkey in enumerate(order)]


async def test_round_one_advances_the_best_eight_and_drops_absent_and_failed_entrants(tournament):
    entrants = [f"h{i:02d}" for i in range(14)]
    # h00..h08 scored best-first, h09 scored a fail, h10-h13 never reported.
    results = _by_bytes(entrants[:9]) + [_freeform_fail("h09")]
    tournament({T1_TASK: results}, {1: entrants})

    winners = await rr.get_group_winners(_round_data(1), [_round_task(1, T1_TASK)], psql_db=None)

    assert winners == entrants[:8]
    assert "h09" not in winners


@pytest.mark.parametrize(("scored", "advance"), [(5, 5), (8, 8), (9, 8)])
async def test_a_small_field_advances_min_of_n_and_eight(tournament, scored, advance):
    entrants = [f"h{i}" for i in range(scored)]
    tournament({T1_TASK: _by_bytes(entrants)}, {1: entrants})

    winners = await rr.get_group_winners(_round_data(1), [_round_task(1, T1_TASK)], psql_db=None)

    assert len(winners) == advance


async def test_round_two_takes_the_best_four_by_average_rank_and_the_latest_round_breaks_ties(tournament):
    cohort = [f"h{i}" for i in range(8)]
    # R1 order h0..h7. R2 (fixed): h1 beats h0, so h0 and h1 tie on average rank 1.5 and h1 wins on R2; h4 is absent from R2.
    r2 = [_fixed(hk, 1.0 + i / 10) for i, hk in enumerate(["h1", "h0", "h3", "h2", "h5", "h6", "h7"])]
    tournament({T1_TASK: _by_bytes(cohort), T2_TASK: r2}, {1: cohort, 2: cohort})

    top_four = await rr.quant_cumulative_winners(TID, _round_data(2), t_cst.QUANT_ROUND_TWO_ADVANCE, None)

    assert top_four == ["h1", "h0", "h3", "h2"]  # h2 and h3 tie on 3.5 too; h3 was better in R2
    # h4 (absent from R2) ranks last there and misses the cut
    standings = await rr._cumulative_standings(TID, _round_data(2), None)
    by_hotkey = {s.hotkey: s for s in standings}
    assert (by_hotkey["h0"].recent_round_rank, by_hotkey["h1"].recent_round_rank) == (2.0, 1.0)
    assert by_hotkey["h4"].has_error


async def test_a_scored_fail_in_round_two_ranks_last(tournament):
    cohort = ["A", "B", "C", "D"]
    r2 = [_fixed("B", 1.0), _fixed("C", 1.1), _fixed("D", 1.2), _freeform_fail("A")]
    tournament({T1_TASK: _by_bytes(cohort), T2_TASK: r2}, {1: cohort, 2: cohort})

    standings = await rr._cumulative_standings(TID, _round_data(2), None)

    a = next(s for s in standings if s.hotkey == "A")
    assert a.recent_round_rank == 4.0 and a.total_rank == 1 + 4


async def test_the_viability_cull_after_round_two_leaves_a_lone_survivor(tournament):
    cohort = [f"h{i}" for i in range(4)]
    # h0 leads both rounds by a margin the others cannot close in a three-round cut.
    r2 = [_fixed(hk, 1.0 + i / 10) for i, hk in enumerate(cohort)]
    tournament({T1_TASK: _by_bytes(cohort), T2_TASK: r2}, {1: cohort, 2: cohort})

    winners = await rr.get_group_winners(_round_data(2), [_round_task(2, T2_TASK)], psql_db=None)

    assert winners == ["h0"]


async def test_round_three_crowns_the_best_average_rank_over_all_three_rounds(tournament):
    cohort = ["A", "B", "C"]
    r1 = _by_bytes(["A", "C", "B"])  # A1 C2 B3
    r2 = [_fixed("C", 1.0), _fixed("A", 1.1), _fixed("B", 1.2)]  # C1 A2 B3
    r3 = _by_bytes(["B", "A", "C"])  # B1 A2 C3 -> A 1+2+2=5, B 3+3+1=7, C 2+1+3=6
    tournament({T1_TASK: r1, T2_TASK: r2, T3_TASK: r3}, {1: cohort, 2: cohort, 3: cohort})

    winners = await rr.get_group_winners(_round_data(3), [_round_task(3, T3_TASK)], psql_db=None)

    assert winners == ["A"]


async def test_a_three_round_tie_goes_to_the_better_round_three_rank(tournament):
    cohort = ["A", "B", "C"]
    r1 = _by_bytes(["A", "B", "C"])  # A1 B2 C3
    r2 = [_fixed("C", 1.0), _fixed("A", 1.1), _fixed("B", 1.2)]  # C1 A2 B3
    r3 = _by_bytes(["B", "A", "C"])  # B1 A2 C3 -> A 1+2+2=5, B 2+3+1=6, C 3+1+3=7: no tie yet
    tournament({T1_TASK: r1, T2_TASK: r2, T3_TASK: r3}, {1: cohort, 2: cohort, 3: cohort})
    assert [s.hotkey for s in await rr._cumulative_standings(TID, _round_data(3), None)] == ["A", "B", "C"]

    r3 = _by_bytes(["B", "C", "A"])  # B1 C2 A3 -> A 1+2+3=6, B 2+3+1=6, C 3+1+2=6: a three-way tie on 2.0
    tournament({T1_TASK: r1, T2_TASK: r2, T3_TASK: r3}, {1: cohort, 2: cohort, 3: cohort})
    standings = await rr._cumulative_standings(TID, _round_data(3), None)
    assert {s.average_rank for s in standings} == {2.0}
    assert [s.hotkey for s in standings] == ["B", "C", "A"]  # ordered by the round-three rank


async def test_third_place_is_the_pre_boss_round_three_standing(tournament):
    cohort = ["A", "B", "C", "D"]
    r1 = _by_bytes(["A", "B", "C", "D"])
    r2 = [_fixed("A", 1.0), _fixed("B", 1.1), _fixed("C", 1.2), _fixed("D", 1.3)]
    r3 = _by_bytes(["A", "B", "C", "D"])
    tournament({T1_TASK: r1, T2_TASK: r2, T3_TASK: r3}, {1: cohort, 2: cohort, 3: cohort})

    third = await rr.get_quant_pre_boss_runner_up(TID, _round_data(4, is_final=True), "A", None)

    assert third == "B"


async def test_third_place_after_a_lone_round_three_survivor_is_the_first_two_round_standing_of_the_top_four(tournament):
    """8 miners in the same order in R1 and R2: the viability cut keeps h0 alone, who plays round 3 and the boss in round 4.
    Round 3's group names nobody else, so third place is the best of the round-2 advance set after h0."""
    cohort = [f"h{i}" for i in range(8)]
    r2 = [_fixed(hk, 1.0 + i / 10) for i, hk in enumerate(cohort)]
    results = {T1_TASK: _by_bytes(cohort), T2_TASK: r2, T3_TASK: _by_bytes(["h0"])}
    tournament(results, {1: cohort, 2: cohort})
    assert await rr.get_group_winners(_round_data(2), [_round_task(2, T2_TASK)], psql_db=None) == ["h0"]

    tournament(results, {1: cohort, 2: cohort, 3: ["h0"]})

    assert await rr.get_quant_pre_boss_runner_up(TID, _round_data(4, is_final=True), "h0", None) == "h1"
    assert await rr.get_quant_round_runner_up(TID, _round_data(3), "h0", None) == "h1"  # the early stop after round 3


async def test_the_third_place_from_rounds_one_and_two_is_compared_with_the_boss_over_those_rounds(tournament):
    cohort = [f"h{i}" for i in range(8)]
    r1 = _by_bytes(["h0", BOSS, "h1", *cohort[2:]])  # the boss second in R1
    r2 = [_fixed(hk, 1.0 + i / 10) for i, hk in enumerate(["h0", "h1", BOSS, *cohort[2:]])]  # third in R2
    r3 = _by_bytes([BOSS, "h0"])
    tournament({T1_TASK: r1, T2_TASK: r2, T3_TASK: r3}, {1: cohort, 2: cohort, 3: ["h0"]})

    third = await rr.get_quant_round_runner_up(TID, _round_data(3), "h0", None)
    comparison = compare_with_boss(await rr.round_ranks_with_boss(TID, psql_db=None), third, 3)

    assert third == "h1"
    assert comparison.compared_round_numbers == [1, 2]
    assert (comparison.contender_average_rank, comparison.boss_average_rank) == (2.5, 2.5)
    assert comparison.share_to_boss is False


# --------------------------------------------------------------------------------------------- hiding


def _hide_input(status: TaskStatus) -> InstructTextTask:
    raw = _seed_task(status=status, format=None, no_input_format=None)
    return InstructTextTask(**{k: v for k, v in raw.model_dump().items() if k in InstructTextTask.model_fields})


def test_a_forced_hidden_task_is_blanked_even_when_it_is_success():
    shown = hide_sensitive_data_till_finished(_hide_input(TaskStatus.SUCCESS))
    hidden = hide_sensitive_data_till_finished(_hide_input(TaskStatus.SUCCESS), force_hidden=True)

    assert shown.test_data and shown.ds == "org/ds"
    assert hidden.test_data is None and hidden.training_data is None and hidden.calibration_data is None
    assert hidden.ds == "Hidden"


class _Cluster:
    """A throwaway local Postgres (the real SQL of get_hidden_seed_task_ids runs against the real seed schema columns)."""

    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="pg_hidden_seeds_"))
        self.port = 55000 + random.randint(0, 900)
        subprocess.run(["initdb", "-D", str(self.dir / "data"), "-A", "trust", "-U", "postgres"], check=True, capture_output=True)
        subprocess.run(
            [
                "pg_ctl", "-D", str(self.dir / "data"), "-w", "start", "-l", str(self.dir / "log"),
                "-o", f"-c listen_addresses='' -k {self.dir} -p {self.port}",
            ],  # fmt: skip
            check=True,
            capture_output=True,
        )

    async def connect(self) -> asyncpg.Connection:
        return await asyncpg.connect(host=str(self.dir), port=self.port, user="postgres", database="postgres")

    def stop(self) -> None:
        subprocess.run(["pg_ctl", "-D", str(self.dir / "data"), "-m", "immediate", "stop"], capture_output=True)
        shutil.rmtree(self.dir, ignore_errors=True)


@pytest.fixture(scope="module")
def cluster():
    if not all(shutil.which(binary) for binary in ("initdb", "pg_ctl")):
        pytest.skip("no local postgres binaries")
    cluster = _Cluster()
    yield cluster
    cluster.stop()


class _Db:
    def __init__(self, connection: asyncpg.Connection) -> None:
        self.connection_obj = connection

    async def connection(self):
        outer = self

        class _Ctx:
            async def __aenter__(self):
                return outer.connection_obj

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


_SCHEMA = """
DROP TABLE IF EXISTS tournaments, tournament_rounds, tournament_seed_tasks, quant_tasks, tournament_tasks, quant_task_sources;
CREATE TABLE tournaments (tournament_id text PRIMARY KEY, status text);
CREATE TABLE tournament_rounds (round_id text PRIMARY KEY, tournament_id text, round_number int, status text);
CREATE TABLE tournament_seed_tasks (
    tournament_id text, role text, attempt int, task_id uuid, status text, attached_round_id text
);
CREATE TABLE quant_tasks (task_id uuid PRIMARY KEY, upstream_task_id uuid);
CREATE TABLE tournament_tasks (tournament_id text, task_id uuid);
CREATE TABLE quant_task_sources (
    quant_task_id uuid PRIMARY KEY, quant_tournament_id text, source_task_id uuid, superseded boolean NOT NULL DEFAULT FALSE
);
"""


async def _hidden(connection, task_ids: list[UUID]) -> set[str]:
    return await seeds_sql.get_hidden_seed_task_ids([str(t) for t in task_ids], _Db(connection))


async def test_seed_hiding_rule_per_role_before_and_after_the_deciding_round(cluster):
    connection = await cluster.connect()
    try:
        await connection.execute(_SCHEMA)
        ids = {role: uuid4() for role in SeedRole}
        round_tasks = {"r1": uuid4(), "r2": uuid4(), "r3": uuid4()}  # chained on the small, small and large input seeds
        plain = uuid4()
        await connection.execute("INSERT INTO tournaments VALUES ('t', 'active')")
        for role, task_id in ids.items():
            await connection.execute(
                "INSERT INTO tournament_seed_tasks VALUES ('t', $1, 1, $2, 'ready')", role.value, task_id
            )
        chained_on = {"r1": SeedRole.INPUT_SMALL, "r2": SeedRole.INPUT_SMALL, "r3": SeedRole.INPUT_LARGE}
        for name, role in chained_on.items():
            upstream = ids[role]
            await connection.execute("INSERT INTO quant_tasks VALUES ($1, $2)", round_tasks[name], upstream)
        every = [*ids.values(), *round_tasks.values(), plain]

        async def set_round(number: int, status: str) -> None:
            await connection.execute("DELETE FROM tournament_rounds WHERE round_number = $1", number)
            await connection.execute(
                "INSERT INTO tournament_rounds VALUES ($1, 't', $2, $3)", f"t_round_{number:03d}", number, status
            )

        everything = {str(t) for t in every} - {str(plain)}
        i1 = str(ids[SeedRole.INPUT_SMALL])
        on_i1 = {i1, str(round_tasks["r1"]), str(round_tasks["r2"])}

        # Nothing played yet: every seed (and every round task chained on one) is hidden, other tasks never.
        assert await _hidden(connection, every) == everything
        # R1 done, R2 running: the small input seed still hidden (R2 plays on it).
        await set_round(1, "completed")
        await set_round(2, "active")
        assert await _hidden(connection, every) == everything
        # R2 completed: the small input seed and the tasks on it are released; the large input seed and the boss-round seeds
        # stay hidden.
        await set_round(2, "completed")
        assert await _hidden(connection, every) == everything - on_i1
        # R3 completed: the large input seed released; every other seed (the boss round's input and compressions) stays hidden
        # until the end.
        await set_round(3, "completed")
        boss_round_seeds = {str(task_id) for role, task_id in ids.items() if role not in INPUT_SEED_ROLES}
        assert await _hidden(connection, every) == boss_round_seeds
        # Tournament completed: everything is released.
        await connection.execute("UPDATE tournaments SET status = 'completed'")
        assert await _hidden(connection, every) == set()
        # Cancelled releases too.
        await connection.execute("UPDATE tournaments SET status = 'cancelled'")
        assert await _hidden(connection, every) == set()
        assert await _hidden(connection, []) == set()
    finally:
        await connection.close()


async def test_a_tournament_without_round_three_releases_input_large_only_with_the_tournament(cluster):
    """A lone survivor skips R3, so the boss round is round 3: the tournament, not the round, releases the large input seed."""
    connection = await cluster.connect()
    try:
        await connection.execute(_SCHEMA)
        i4 = uuid4()
        await connection.execute("INSERT INTO tournaments VALUES ('t', 'active')")
        await connection.execute("INSERT INTO tournament_seed_tasks VALUES ('t', 'input_large', 1, $1, 'ready')", i4)
        await connection.execute("INSERT INTO tournament_rounds VALUES ('t_round_002', 't', 2, 'completed')")
        assert await _hidden(connection, [i4]) == {str(i4)}
        await connection.execute("UPDATE tournaments SET status = 'completed'")
        assert await _hidden(connection, [i4]) == set()
    finally:
        await connection.close()


async def test_the_task_endpoints_apply_the_hiding_rule(monkeypatch):
    task = _hide_input(TaskStatus.SUCCESS)
    helper = AsyncMock(return_value={str(task.task_id)})
    monkeypatch.setattr(tasks_endpoint.seeds_sql, "get_hidden_seed_task_ids", helper)
    monkeypatch.setattr(tasks_endpoint.task_sql, "get_task_by_id", AsyncMock(return_value=task))
    monkeypatch.setattr(tasks_endpoint.task_sql, "get_tasks_by_account_id", AsyncMock(return_value=[task.model_copy()]))
    config = SimpleNamespace(psql_db=MagicMock())

    details = await tasks_endpoint.get_task_details(task.task_id, config)
    listed = await tasks_endpoint.get_task_details_by_account(uuid4(), 100, 1, config)

    assert details.ds_repo == "Hidden" and listed[0].ds_repo == "Hidden"
    assert helper.await_args_list[0].args[0] == [str(task.task_id)]  # one batched call per request

    helper.return_value = set()
    task = _hide_input(TaskStatus.SUCCESS)
    monkeypatch.setattr(tasks_endpoint.task_sql, "get_task_by_id", AsyncMock(return_value=task))
    shown = await tasks_endpoint.get_task_details(task.task_id, config)
    assert shown.ds_repo == "org/ds"


# --------------------------------------------------------------------------------------------- the boss in the rounds

BOSS = EMISSION_BURN_HOTKEY


async def test_the_boss_is_ranked_in_each_round_but_takes_no_advancement_slot(tournament):
    entrants = [f"h{i:02d}" for i in range(10)]
    # The boss's entry on the round seed is second best of the round.
    results = _by_bytes([entrants[0], BOSS, *entrants[1:]])
    tournament({T1_TASK: results}, {1: entrants})

    winners = await rr.get_group_winners(_round_data(1), [_round_task(1, T1_TASK)], psql_db=None)
    ranks = await rr.round_ranks_with_boss(TID, psql_db=None)

    assert winners == entrants[:8] and BOSS not in winners
    (round_one,) = ranks.rounds
    assert round_one.round_number == 1 and round_one.task_ids == [T1_TASK]
    assert round_one.ranks[BOSS] == 2
    assert round_one.ranks[entrants[0]] == 1 and round_one.ranks[entrants[1]] == 3 and round_one.ranks[entrants[9]] == 11


async def test_a_boss_entry_with_no_valid_result_ranks_behind_every_valid_entry(tournament):
    cohort = ["A", "B", "C"]
    r1 = [*_by_bytes(["A", "B", "C"]), _freeform_fail(BOSS)]
    r2 = [_fixed("B", 1.0), _fixed("C", 1.1)]  # the boss has no row at all, A scored nothing
    tournament({T1_TASK: r1, T2_TASK: r2}, {1: cohort, 2: cohort})

    ranks = await rr.round_ranks_with_boss(TID, psql_db=None)

    first, second = ranks.rounds
    assert first.ranks == {"A": 1, "B": 2, "C": 3, BOSS: 4}
    assert (second.ranks["B"], second.ranks["C"]) == (1, 2)
    assert second.ranks["A"] == second.ranks[BOSS] == 3  # both behind every valid entry, level with each other


async def test_an_exact_tie_with_the_boss_shares_the_rank(tournament):
    cohort = ["A", "B", "C"]
    r1 = [_freeform("B", 100), _freeform(BOSS, 200), _freeform("A", 200), _freeform("C", 300)]
    tournament({T1_TASK: r1}, {1: cohort})

    (first,) = (await rr.round_ranks_with_boss(TID, psql_db=None)).rounds

    assert first.ranks == {"B": 1, BOSS: 2, "A": 2, "C": 4}
    # Advancement among the challengers is untouched: A still advances ahead of C, behind B.
    assert await rr.get_group_winners(_round_data(1), [_round_task(1, T1_TASK)], psql_db=None) == ["B", "A", "C"]


async def test_the_average_rank_over_rounds_one_to_three(tournament):
    cohort = ["A", "B"]
    r1 = _by_bytes([BOSS, "A", "B"])
    r2 = [_fixed("A", 1.0), _fixed(BOSS, 1.1), _fixed("B", 1.2)]
    r3 = _by_bytes(["A", "B", BOSS])
    tournament({T1_TASK: r1, T2_TASK: r2, T3_TASK: r3}, {1: cohort, 2: cohort, 3: ["A"]})

    ranks = await rr.round_ranks_with_boss(TID, psql_db=None)

    assert [round_ranks.round_number for round_ranks in ranks.rounds] == [1, 2, 3]
    assert ranks.average_rank(BOSS) == pytest.approx((1 + 2 + 2) / 3)  # round 3's field is A and the boss: B is not in it
    assert ranks.average_rank("A") == pytest.approx((2 + 1 + 1) / 3)
    assert ranks.average_rank("B") == pytest.approx((3 + 3) / 2)
    assert ranks.average_rank("nobody") is None


async def test_a_text_tournament_ranks_its_boss_the_same_way(tournament, monkeypatch):
    cohort = ["A", "B"]
    loss = {"A": 1.2, BOSS: 1.1, "B": 1.3}
    text = [
        MinerResultsText(hotkey=hk, test_loss=value, synth_loss=value, is_finetune=True, task_type=TaskType.INSTRUCTTEXTTASK)
        for hk, value in loss.items()
    ]
    tournament({T1_TASK: text}, {1: cohort})
    monkeypatch.setattr(
        rr, "get_tournament", AsyncMock(return_value=TournamentData(tournament_id=TID, tournament_type=TournamentType.TEXT))
    )

    ranks = await rr.round_ranks_with_boss(TID, psql_db=None)

    assert ranks.rounds[0].ranks == {BOSS: 1, "A": 2, "B": 3}
