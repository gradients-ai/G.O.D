"""_carry_forward_continuous_sft routes each lineage's lowest-loss winner to the right state row.

Mocks the DB seams (get_task, get_lowest_loss_repo_for_task, advance) and asserts the routing:
lineage recovered from the task's ds, winner carried as-is, None-winner still advances (so the
cursor never stalls), a malformed ds is skipped, a failed advance propagates (and a retry is idempotent),
and with only_hotkey that hotkey's evaluated repo is carried instead of the lowest-loss one.
The lineage<->winner pairing is the thing that must never cross wires (one lineage's winner
landing in another's row silently poisons a lineage).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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
    tasks = {"t1": _cont_task("alt"), "t2": _cont_task("qwen")}
    winners = {"t1": "org/alt-win", "t2": "org/qwen-win"}
    advance = _wire(monkeypatch, tasks_by_id=tasks, winners_by_id=winners)

    await tournament_manager._carry_forward_continuous_sft([_round_task("t1"), _round_task("t2")], SRID, PSQL)

    calls = [(*c.args, c.kwargs["task_id"]) for c in advance.call_args_list]
    assert calls == [("alt", "org/alt-win", SRID, PSQL, "t1"), ("qwen", "org/qwen-win", SRID, PSQL, "t2")]


async def test_none_winner_still_advances(monkeypatch):
    # Empty week: no eligible submission -> advance(None) so the +1 / COALESCE-preserve path runs.
    tasks = {"t1": _cont_task("qwen")}
    advance = _wire(monkeypatch, tasks_by_id=tasks, winners_by_id={"t1": None})
    await tournament_manager._carry_forward_continuous_sft([_round_task("t1")], SRID, PSQL)
    assert advance.call_args_list[0].args == ("qwen", None, SRID, PSQL)
    assert advance.call_args_list[0].kwargs == {"task_id": "t1"}


async def test_unrecognized_ds_is_skipped_but_loop_continues(monkeypatch):
    tasks = {"bad": _cont_task("qwen", ds="not-a-continuous-ds"), "good": _cont_task("alt")}
    advance = _wire(monkeypatch, tasks_by_id=tasks, winners_by_id={"bad": "x", "good": "org/alt-win"})
    await tournament_manager._carry_forward_continuous_sft([_round_task("bad"), _round_task("good")], SRID, PSQL)
    calls = [(*c.args, c.kwargs["task_id"]) for c in advance.call_args_list]
    assert calls == [("alt", "org/alt-win", SRID, PSQL, "good")]  # bad ds skipped, good still advanced


async def test_a_failed_advance_propagates(monkeypatch):
    # The caller must not complete the tournament if the cursor did not move.
    failing = AsyncMock(side_effect=RuntimeError("db down"))
    advance = _wire(monkeypatch, tasks_by_id={"t1": _cont_task("qwen")}, winners_by_id={"t1": "org/w"}, advance=failing)

    with pytest.raises(RuntimeError, match="db down"):
        await tournament_manager._carry_forward_continuous_sft([_round_task("t1")], SRID, PSQL)

    assert advance.call_count == 1


class _FakeCursor:
    """advance_continuous_sft_state's contract: idempotent on the source round id; a failure advances nothing."""

    def __init__(self):
        self.train_index = 0
        self.last_round = None
        self.fail_next = False
        self.repos = []

    async def advance(self, lineage, winner_repo, source_round_id, psql_db, task_id=None):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("transient")
        if self.last_round == source_round_id:
            return
        self.train_index += 1
        self.last_round = source_round_id
        self.repos.append(winner_repo)


async def test_after_a_propagated_failure_the_retry_advances_exactly_once(monkeypatch):
    cursor = _FakeCursor()
    cursor.fail_next = True
    _wire(monkeypatch, tasks_by_id={"t1": _cont_task("qwen")}, winners_by_id={"t1": "org/w"}, advance=cursor.advance)
    round_tasks = [_round_task("t1")]

    with pytest.raises(RuntimeError):
        await tournament_manager._carry_forward_continuous_sft(round_tasks, SRID, PSQL)
    assert cursor.train_index == 0
    await tournament_manager._carry_forward_continuous_sft(round_tasks, SRID, PSQL)
    await tournament_manager._carry_forward_continuous_sft(round_tasks, SRID, PSQL)

    assert cursor.train_index == 1 and cursor.repos == ["org/w"]


async def test_only_hotkey_carries_that_hotkeys_repo_never_the_lowest_loss_one(monkeypatch):
    advance = _wire(monkeypatch, tasks_by_id={"t1": _cont_task("qwen")}, winners_by_id={"t1": "org/challenger-lowest"})
    repo_of = AsyncMock(return_value="org/boss-repo")
    monkeypatch.setattr(tournament_manager.task_sql, "get_evaluated_repo_for_hotkey", repo_of)
    lowest = tournament_manager.task_sql.get_lowest_loss_repo_for_task

    await tournament_manager._carry_forward_continuous_sft(
        [_round_task("t1")], SRID, PSQL, only_hotkey=tournament_manager.EMISSION_BURN_HOTKEY
    )

    repo_of.assert_awaited_once_with("t1", tournament_manager.EMISSION_BURN_HOTKEY, PSQL)
    lowest.assert_not_awaited()
    assert advance.call_args_list[0].args == ("qwen", "org/boss-repo", SRID, PSQL)
    assert advance.call_args_list[0].kwargs == {"task_id": "t1"}


async def test_only_hotkey_with_no_evaluated_repo_still_advances_with_none(monkeypatch):
    advance = _wire(monkeypatch, tasks_by_id={"t1": _cont_task("qwen")}, winners_by_id={"t1": "org/challenger-lowest"})
    monkeypatch.setattr(tournament_manager.task_sql, "get_evaluated_repo_for_hotkey", AsyncMock(return_value=None))

    await tournament_manager._carry_forward_continuous_sft(
        [_round_task("t1")], SRID, PSQL, only_hotkey=tournament_manager.EMISSION_BURN_HOTKEY
    )

    assert advance.call_args_list[0].args == ("qwen", None, SRID, PSQL)
    assert advance.call_args_list[0].kwargs == {"task_id": "t1"}


async def test_non_continuous_and_missing_tasks_are_skipped(monkeypatch):
    plain_chat = SimpleNamespace(
        task_type=TaskType.CHATTASK, training_start_point=TrainingStartPoint.DEFAULT, ds="whatever"
    )
    tasks = {"plain": plain_chat, "missing": None, "cont": _cont_task("qwen")}
    advance = _wire(monkeypatch, tasks_by_id=tasks, winners_by_id={"cont": "org/w"})
    await tournament_manager._carry_forward_continuous_sft(
        [_round_task("plain"), _round_task("missing"), _round_task("cont")], SRID, PSQL
    )
    calls = [(*c.args, c.kwargs["task_id"]) for c in advance.call_args_list]
    assert calls == [("qwen", "org/w", SRID, PSQL, "cont")]  # only the real continuous task advanced
