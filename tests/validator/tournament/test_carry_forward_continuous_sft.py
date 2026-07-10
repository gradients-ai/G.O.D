"""_carry_forward_continuous_sft routes each lineage's lowest-loss winner to the right state row.

Mocks the DB seams (get_task, get_lowest_loss_repo_for_task, advance) and asserts the routing:
lineage recovered from the task's ds, winner carried as-is, None-winner still advances (so the
cursor never stalls), a malformed ds is skipped, and one lineage's failure doesn't abort the rest.
The lineage<->winner pairing is the thing that must never cross wires (a quasar winner landing in
the qwen row silently poisons a lineage).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.constants.environments import TrainingStartPoint
from core.models.task_models import TaskType
from validator.tournament import constants as t_cst
from validator.tournament import tournament_manager


SRID = "round-42"
PSQL = object()


def _round_task(task_id):
    return SimpleNamespace(task_id=task_id)


def _cont_task(lineage_label, *, ds=None):
    """A continuous-SFT task object (CHATTASK + CONTINUOUS_SFT) with an encoded ds."""
    ds = ds if ds is not None else t_cst.continuous_sft_ds(lineage_label, "chunk")
    return SimpleNamespace(task_type=TaskType.CHATTASK, training_start_point=TrainingStartPoint.CONTINUOUS_SFT, ds=ds)


def _wire(monkeypatch, *, tasks_by_id, winners_by_id, advance=None):
    monkeypatch.setattr(
        tournament_manager.task_sql, "get_task", AsyncMock(side_effect=lambda tid, db: tasks_by_id.get(tid))
    )
    monkeypatch.setattr(
        tournament_manager.task_sql,
        "get_lowest_loss_repo_for_task",
        AsyncMock(side_effect=lambda tid, db: winners_by_id.get(tid)),
    )
    advance = advance or AsyncMock()
    monkeypatch.setattr(tournament_manager, "advance_continuous_sft_state", advance)
    return advance


async def test_advances_each_lineage_with_its_lowest_loss_winner(monkeypatch):
    tasks = {"t1": _cont_task("quasar"), "t2": _cont_task("qwen")}
    winners = {"t1": "org/quasar-win", "t2": "org/qwen-win"}
    advance = _wire(monkeypatch, tasks_by_id=tasks, winners_by_id=winners)

    await tournament_manager._carry_forward_continuous_sft([_round_task("t1"), _round_task("t2")], SRID, PSQL)

    calls = [c.args for c in advance.call_args_list]
    assert calls == [("quasar", "org/quasar-win", SRID, PSQL), ("qwen", "org/qwen-win", SRID, PSQL)]


async def test_none_winner_still_advances(monkeypatch):
    # Empty week: no eligible submission -> advance(None) so the +1 / COALESCE-preserve path runs.
    tasks = {"t1": _cont_task("qwen")}
    advance = _wire(monkeypatch, tasks_by_id=tasks, winners_by_id={"t1": None})
    await tournament_manager._carry_forward_continuous_sft([_round_task("t1")], SRID, PSQL)
    assert advance.call_args_list[0].args == ("qwen", None, SRID, PSQL)


async def test_unrecognized_ds_is_skipped_but_loop_continues(monkeypatch):
    tasks = {"bad": _cont_task("qwen", ds="not-a-continuous-ds"), "good": _cont_task("quasar")}
    advance = _wire(monkeypatch, tasks_by_id=tasks, winners_by_id={"bad": "x", "good": "org/quasar-win"})
    await tournament_manager._carry_forward_continuous_sft([_round_task("bad"), _round_task("good")], SRID, PSQL)
    calls = [c.args for c in advance.call_args_list]
    assert calls == [("quasar", "org/quasar-win", SRID, PSQL)]  # bad ds skipped, good still advanced


async def test_one_lineage_failure_does_not_block_the_other(monkeypatch):
    tasks = {"t1": _cont_task("quasar"), "t2": _cont_task("qwen")}
    winners = {"t1": "org/q", "t2": "org/w"}
    advance = _wire(
        monkeypatch, tasks_by_id=tasks, winners_by_id=winners, advance=AsyncMock(side_effect=[Exception("boom"), None])
    )
    # Must not raise; both lineages attempted.
    await tournament_manager._carry_forward_continuous_sft([_round_task("t1"), _round_task("t2")], SRID, PSQL)
    assert advance.call_count == 2


async def test_non_continuous_and_missing_tasks_are_skipped(monkeypatch):
    plain_chat = SimpleNamespace(
        task_type=TaskType.CHATTASK, training_start_point=TrainingStartPoint.DEFAULT, ds="whatever"
    )
    tasks = {"plain": plain_chat, "missing": None, "cont": _cont_task("qwen")}
    advance = _wire(monkeypatch, tasks_by_id=tasks, winners_by_id={"cont": "org/w"})
    await tournament_manager._carry_forward_continuous_sft(
        [_round_task("plain"), _round_task("missing"), _round_task("cont")], SRID, PSQL
    )
    calls = [c.args for c in advance.call_args_list]
    assert calls == [("qwen", "org/w", SRID, PSQL)]  # only the real continuous task advanced
