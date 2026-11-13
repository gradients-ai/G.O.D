"""Quant tournament task creation: each miner's own R1 repo (own-model tasks), the boss mix and its budgets,
replacements. The miner rounds R1-R3 on the seeds are in test_compression_rounds.py."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from core.constants.environments import TrainingStartPoint
from core.models.dataset_models import InstructTextDatasetType
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.tasks import quant_input_repo
from validator.tasks.datasets import constants as data_cst
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tasks.models import freeform_penalty_policy
from validator.tasks.synthetics import scheduler
from validator.tournament import constants as t_cst
from validator.tournament import task_creator
from validator.tournament.models import TournamentTask


NAMESPACE = "gradients-io-tournaments"
R2_ROUND_ID = "tourn_round_002"
R3_ROUND_ID = "tourn_round_003"


def _patch_scheduler_persistence(monkeypatch):
    monkeypatch.setattr(scheduler, "add_task", AsyncMock(side_effect=lambda task, psql_db: task))
    monkeypatch.setattr(scheduler, "get_model_num_params", lambda model_id: 1_543_714_304)
    monkeypatch.setattr(scheduler, "maybe_get_augmentation_config", lambda task_type: None)


def _patch_own_r1_repo(monkeypatch, repo_files: dict[str, list[str]], ranks: dict[str, dict[str, int]], repo_name: str | None):
    r1_tasks = [TournamentTask(tournament_id="tourn", round_id="tourn_round_001", task_id=t) for t in ranks]
    monkeypatch.setattr(task_creator, "_get_round_one_tasks", AsyncMock(return_value=r1_tasks))
    monkeypatch.setattr(task_creator, "_task_ranks", AsyncMock(side_effect=lambda task_id, psql_db: ranks[task_id]))
    monkeypatch.setattr(task_creator.task_sql, "get_expected_repo_name", AsyncMock(return_value=repo_name))

    def list_files(repo_id, token=None):
        if repo_id not in repo_files:
            raise FileNotFoundError(repo_id)
        return repo_files[repo_id]

    monkeypatch.setattr(quant_input_repo, "list_repo_files", list_files)
    monkeypatch.setattr(task_creator.t_cst, "QUANT_INPUT_LISTING_RETRY_DELAY_SECONDS", 0)


async def test_own_r1_model_repo_is_the_full_model_our_trainer_uploaded(monkeypatch):
    _patch_own_r1_repo(
        monkeypatch, {f"{NAMESPACE}/r1-repo": ["config.json", "model.safetensors"]}, {"r1-task-a": {"hk_a": 1}}, "r1-repo"
    )

    assert await task_creator.own_r1_model_repo("tourn", "hk_a", MagicMock()) == f"{NAMESPACE}/r1-repo"


async def test_own_r1_model_repo_takes_the_best_ranked_task(monkeypatch):
    _patch_own_r1_repo(
        monkeypatch,
        {f"{NAMESPACE}/r1-repo": ["config.json", "model.safetensors"]},
        {"r1-task-a": {"hk_a": 3}, "r1-task-b": {"hk_a": 1}},
        "r1-repo",
    )

    await task_creator.own_r1_model_repo("tourn", "hk_a", MagicMock())

    assert task_creator.task_sql.get_expected_repo_name.await_args.args[:2] == ("r1-task-b", "hk_a")


async def test_own_r1_model_repo_is_none_for_an_adapter(monkeypatch):
    _patch_own_r1_repo(
        monkeypatch, {f"{NAMESPACE}/r1-repo": ["config.json", "adapter_config.json"]}, {"r1-task-a": {"hk_a": 1}}, "r1-repo"
    )

    assert await task_creator.own_r1_model_repo("tourn", "hk_a", MagicMock()) is None


async def test_own_r1_model_repo_is_none_without_an_expected_repo(monkeypatch):
    _patch_own_r1_repo(monkeypatch, {}, {"r1-task-a": {"hk_a": 1}}, None)

    assert await task_creator.own_r1_model_repo("tourn", "hk_a", MagicMock()) is None


async def test_own_r1_model_repo_is_none_without_a_ranked_result(monkeypatch):
    _patch_own_r1_repo(monkeypatch, {}, {"r1-task-a": {"hk_other": 1}}, "r1-repo")

    assert await task_creator.own_r1_model_repo("tourn", "hk_a", MagicMock()) is None

def _freeform_task(**overrides) -> QuantFreeformRawTask:
    fields = dict(
        task_id=uuid4(),
        model_id="org/old-dense",
        ds="org/instruct-ds",
        dataset_type=InstructTextDatasetType(field_instruction="instruction", field_output="output"),
        status="prep_task_failure",
        is_organic=False,
        account_id=uuid4(),
        hours_to_complete=4.0,
        created_at=datetime(2026, 1, 1),
    )
    fields.update(overrides)
    return QuantFreeformRawTask(**fields)


def _fixed_task(**overrides) -> QuantFixedRawTask:
    fields = dict(
        task_id=uuid4(),
        model_id=f"{NAMESPACE}/r1-winner",
        ds="ds-r1",
        dataset_type=InstructTextDatasetType(field_instruction="instruction", field_output="output"),
        quant_spec=QuantFormat.W3A16_ASYM_G128.spec(),
        status="training",
        is_organic=False,
        account_id=uuid4(),
        hours_to_complete=1.75,
        created_at=datetime(2026, 1, 1),
        model_params_count=1_543_714_304,
    )
    fields.update(overrides)
    return QuantFixedRawTask(**fields)


async def test_synthetic_quant_task_hours_columns_and_track_fields(monkeypatch):
    _patch_scheduler_persistence(monkeypatch)
    monkeypatch.setattr(scheduler, "get_dataset", AsyncMock(return_value=SimpleNamespace(dataset_id="ds", num_rows=150_000)))
    columns = SimpleNamespace(field_instruction="q", field_input="ctx", field_output="a")
    monkeypatch.setattr(scheduler, "_get_columns_for_instruct_dataset", AsyncMock(return_value=columns))
    monkeypatch.setattr(scheduler, "get_model_num_params", lambda model_id: int(3e9))
    fixed = await scheduler.create_synthetic_quant_task(
        MagicMock(), TaskType.QUANTFIXEDTASK, None, MagicMock(), quant_format=QuantFormat.W4A16_SYM_G128, model_id_override="m"
    )
    freeform = await scheduler.create_synthetic_quant_task(
        MagicMock(), TaskType.QUANTFREEFORMTASK, None, MagicMock(), model_id_override="m"
    )

    assert freeform.hours_to_complete == data_cst.QUANT_FREEFORM_EARLY_HOURS
    assert fixed.hours_to_complete == scheduler.quant_fixed_hours(3.0)
    for task in (fixed, freeform):
        assert task.yarn_factor is None
        assert task.dataset_type == InstructTextDatasetType(
            field_instruction="q", field_input="ctx", field_output="a", field_system=None
        )
    assert fixed.quant_spec == QuantFormat.W4A16_SYM_G128.spec()
    assert freeform.bar_cap == scheduler.QUANT_BAR_CAP == 0.08
    assert (freeform.penalty_knee, freeform.penalty_curvature) == (0.04, 2500.0)
    assert not hasattr(fixed, "penalty_knee")
    assert not hasattr(freeform, "quant_spec")


def _freeform_source(**penalty) -> QuantFreeformRawTask:
    return scheduler._new_quant_task(
        TaskType.QUANTFREEFORMTASK,
        model_id="m",
        ds="ds",
        dataset_type=InstructTextDatasetType(field_instruction="q", field_output="a"),
        hours_to_complete=2.0,
        **penalty,
    )


# A retuned policy and an unset (NULL) one: both must survive a recreate untouched.
PINNED_PENALTIES = [
    dict(penalty_knee=0.03, penalty_curvature=900.0, bar_cap=0.06),
    dict(penalty_knee=None, penalty_curvature=None, bar_cap=0.05),
]


@pytest.mark.parametrize("penalty", PINNED_PENALTIES)
async def test_recreate_quant_task_copies_the_pinned_penalty(monkeypatch, penalty):
    """A source with NULL knee and curvature keeps them NULL; it is never re-pinned to the current constants."""
    _patch_scheduler_persistence(monkeypatch)
    source = _freeform_source(**penalty)

    fresh = await scheduler.recreate_quant_task(MagicMock(), source)

    assert (fresh.penalty_knee, fresh.penalty_curvature, fresh.bar_cap) == (
        penalty["penalty_knee"],
        penalty["penalty_curvature"],
        penalty["bar_cap"],
    )


@pytest.mark.parametrize("penalty", PINNED_PENALTIES)
async def test_recreate_chained_quant_task_copies_the_pinned_penalty(monkeypatch, penalty):
    _patch_scheduler_persistence(monkeypatch)
    source = _freeform_source(**penalty, upstream_task_id=uuid4(), status=TaskStatus.AWAITING_UPSTREAM)

    fresh = await scheduler.recreate_chained_quant_task(MagicMock(), source)

    assert (fresh.penalty_knee, fresh.penalty_curvature, fresh.bar_cap) == (
        penalty["penalty_knee"],
        penalty["penalty_curvature"],
        penalty["bar_cap"],
    )


def test_changing_the_constants_does_not_move_an_existing_tasks_policy(monkeypatch):
    task = _freeform_source()
    before = freeform_penalty_policy(task)

    monkeypatch.setattr(scheduler, "QUANT_PENALTY_KNEE", 0.01)
    monkeypatch.setattr(scheduler, "QUANT_PENALTY_CURVATURE", 1.0)
    monkeypatch.setattr(scheduler, "QUANT_BAR_CAP", 0.5)

    assert freeform_penalty_policy(task) == before
    assert (before.knee, before.curvature, before.cap) == (1.04, 2500.0, 1.08)


async def test_boss_quant_hours(monkeypatch):
    """A boss-flagged quant task is priced like any other: freeform flat 2 h, fixed by size, no instruct term."""
    _patch_scheduler_persistence(monkeypatch)
    monkeypatch.setattr(scheduler, "get_dataset", AsyncMock(return_value=SimpleNamespace(dataset_id="ds", num_rows=20_000)))
    columns = SimpleNamespace(field_instruction="q", field_input="ctx", field_output="a")
    monkeypatch.setattr(scheduler, "_get_columns_for_instruct_dataset", AsyncMock(return_value=columns))
    monkeypatch.setattr(scheduler, "get_model_num_params", lambda model_id: int(6e9))
    fixed = await scheduler.create_synthetic_quant_task(
        MagicMock(),
        TaskType.QUANTFIXEDTASK,
        None,
        MagicMock(),
        quant_format=QuantFormat.W4A16_SYM_G128,
        model_id_override="m",
        boss=True,
    )
    freeform = await scheduler.create_synthetic_quant_task(
        MagicMock(), TaskType.QUANTFREEFORMTASK, None, MagicMock(), model_id_override="m", boss=True
    )

    assert freeform.hours_to_complete == data_cst.QUANT_FREEFORM_EARLY_HOURS
    assert fixed.hours_to_complete == pytest.approx(1.625)


async def test_synthetic_quant_task_carries_the_training_start_point(monkeypatch):
    _patch_scheduler_persistence(monkeypatch)
    monkeypatch.setattr(scheduler, "get_dataset", AsyncMock(return_value=SimpleNamespace(dataset_id="ds", num_rows=50_000)))
    columns = SimpleNamespace(field_instruction="q", field_input=None, field_output="a")
    monkeypatch.setattr(scheduler, "_get_columns_for_instruct_dataset", AsyncMock(return_value=columns))

    task = await scheduler.create_synthetic_quant_task(
        MagicMock(), TaskType.QUANTFREEFORMTASK, None, MagicMock(), model_id_override="m",
        training_start_point=TrainingStartPoint.OWN_R1_MODEL,
    )
    default = await scheduler.create_synthetic_quant_task(
        MagicMock(), TaskType.QUANTFREEFORMTASK, None, MagicMock(), model_id_override="m"
    )

    assert task.training_start_point == TrainingStartPoint.OWN_R1_MODEL
    assert default.training_start_point == TrainingStartPoint.DEFAULT


async def test_replacing_a_task_copies_each_nodes_starting_model_and_quant_failure(monkeypatch):
    original = _fixed_task(training_start_point=TrainingStartPoint.OWN_R1_MODEL)
    replacement = _fixed_task(task_id=uuid4(), training_start_point=TrainingStartPoint.OWN_R1_MODEL)
    nodes = [SimpleNamespace(hotkey="hk_ok"), SimpleNamespace(hotkey="hk_failed")]
    monkeypatch.setattr(task_creator.task_sql, "get_task", AsyncMock(return_value=original))
    monkeypatch.setattr(task_creator, "create_new_task_of_same_type", AsyncMock(return_value=replacement))
    monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", AsyncMock())
    monkeypatch.setattr(task_creator, "_repoint_chained_dependents", AsyncMock())
    monkeypatch.setattr(task_creator.task_sql, "get_nodes_assigned_to_task", AsyncMock(return_value=nodes))
    monkeypatch.setattr(task_creator.task_sql, "assign_node_to_task", AsyncMock())
    monkeypatch.setattr(task_creator.task_sql, "get_expected_repo_name", AsyncMock(side_effect=lambda t, hk, db: f"repo-{hk}"))
    monkeypatch.setattr(task_creator.task_sql, "set_expected_repo_name", AsyncMock())
    starting = {"hk_ok": f"{NAMESPACE}/r1-ok"}
    monkeypatch.setattr(
        task_creator.task_sql, "get_starting_model_repo", AsyncMock(side_effect=lambda t, hk, db: starting.get(hk))
    )
    set_starting = AsyncMock()
    monkeypatch.setattr(task_creator.task_sql, "set_starting_model_repo", set_starting)
    monkeypatch.setattr(task_creator.task_sql, "detach_task_from_tournament", AsyncMock())
    # An own-model task is not a task of a bound quant tournament: it has no source row to move.
    monkeypatch.setattr(task_creator.quant_sources_sql, "get_quant_task_source", AsyncMock(return_value=None))
    decode_rows = {
        "hk_failed": SimpleNamespace(quant_failure=t_cst.NO_USABLE_R1_MODEL_REASON),
        "hk_ok": SimpleNamespace(quant_failure=None),
    }
    monkeypatch.setattr(
        task_creator.submissions_and_scoring_sql, "get_quant_decode_rows", AsyncMock(return_value=decode_rows)
    )
    set_failure = AsyncMock()
    monkeypatch.setattr(task_creator.submissions_and_scoring_sql, "set_task_node_quant_failure", set_failure)
    config = MagicMock()

    await task_creator.replace_tournament_task(
        str(original.task_id), "tourn", R3_ROUND_ID, f"{R3_ROUND_ID}_group_001", None, config
    )

    set_starting.assert_awaited_once_with(replacement.task_id, "hk_ok", f"{NAMESPACE}/r1-ok", config.psql_db)
    set_failure.assert_awaited_once_with(replacement.task_id, "hk_failed", t_cst.NO_USABLE_R1_MODEL_REASON, config.psql_db)
