"""The boss-round seed hold holds on every finalising path (validator/tournament/README.md, "Design notes").

A boss-round seed (the Free and Fixed compression tasks) trains as a TRAINING task whose only row is the boss's until the boss
round attaches it and adds the challenger. The boss's row is evaluated at seed time, as soon as it trained (its evaluation
row is seeded and the lifecycle picks the task up); until the hold is satisfied (attached, a non-boss player assigned,
every assigned row terminal) the task is not finalised and not moved to PREEVALUATION; then it is ranked once, from the
stored rows of both players. The compressions are released on the BOSS_INSTRUCT seed's SUCCESS, which is never held.
The seed-time evaluation itself is covered end to end by test_boss_seed_time_eval.py.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from core.models.task_models import TaskStatus
from tests.validator.tournament.test_boss_round_seeds import CHALLENGER
from tests.validator.tournament.test_boss_round_seeds import CONFIG
from tests.validator.tournament.test_boss_round_seeds import ROUND_ID
from tests.validator.tournament.test_boss_round_seeds import SUCCESS
from tests.validator.tournament.test_boss_round_seeds import TRAINING
from tests.validator.tournament.test_boss_round_seeds import World
from tests.validator.tournament.test_boss_round_seeds import _chained
from tests.validator.tournament.test_boss_round_seeds import _instruct
from tests.validator.tournament.test_boss_round_seeds import _output
from validator.lifecycle import tasks as lifecycle_tasks
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament import orchestrator
from validator.tournament import task_creator
from validator.tournament.models import SeedRole
from validator.tournament.models import TrainingStatus


BOSS = EMISSION_BURN_HOTKEY


class _StopPass(Exception):
    pass


async def _seeding_pass() -> None:
    """One pass of orchestrator.seed_tournament_evaluations_from_training: the sleep that ends the pass stops the loop."""
    with patch.object(orchestrator.asyncio, "sleep", AsyncMock(side_effect=_StopPass)):
        with pytest.raises(_StopPass):
            await orchestrator.seed_tournament_evaluations_from_training(CONFIG)


class Pipeline(World):
    """World plus the lifecycle's task-listing queries and evaluation rows, all in memory."""

    def __init__(self, monkeypatch) -> None:
        super().__init__(monkeypatch)
        self.eval_rows: dict[str, set[str]] = {}
        self.pair_calls: list[str] = []
        self.registered: list[str] = []
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value="tid"))
        monkeypatch.setattr(orchestrator.task_sql, "get_tasks_with_status", AsyncMock(side_effect=self._with_status))
        monkeypatch.setattr(
            orchestrator.task_sql, "get_task_ids_with_evaluation_statuses", AsyncMock(side_effect=self._ids_with_eval_rows)
        )
        monkeypatch.setattr(orchestrator.task_sql, "add_task_evaluation_pairs", AsyncMock(side_effect=self._add_pairs))
        monkeypatch.setattr(
            orchestrator.tournament_sql, "get_tasks_with_all_training_completed", AsyncMock(side_effect=self._all_completed)
        )
        monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", self._register)
        monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", self._existing)
        monkeypatch.setattr(task_creator.seeds_sql, "mark_seed_attached", AsyncMock())
        monkeypatch.setattr(
            task_creator.seeds_sql, "get_latest_seed_tasks", AsyncMock(side_effect=lambda tid, db: list(self.seeds.values()))
        )

    async def _with_status(self, status, psql_db=None, tournament_filter="all", benchmark_filter="include", **_):
        if tournament_filter == "exclude":
            return []
        return [task for task in self.tasks.values() if task.status == status]

    async def _ids_with_eval_rows(self, evaluation_statuses, psql_db, task_statuses=None, tournament_only=False):
        return [
            task_id
            for task_id in self.eval_rows
            if TaskStatus(self.tasks[task_id].status).value in (task_statuses or [TaskStatus(self.tasks[task_id].status).value])
        ]

    async def _add_pairs(self, task_id, psql_db, include_failed_training=False):
        self.pair_calls.append(str(task_id))
        done = {hk for hk, st in self.training.get(str(task_id), {}).items() if st == SUCCESS}
        self.eval_rows.setdefault(str(task_id), set()).update(done)

    async def _all_completed(self, psql_db):
        terminal = {SUCCESS, TrainingStatus.FAILURE.value}
        return [
            task_id
            for task_id, rows in self.training.items()
            if rows and self.tasks[task_id].status == TaskStatus.TRAINING and all(st in terminal for st in rows.values())
        ]

    async def _register(self, task, tournament_id, round_id, config, group_id=None, pair_id=None):
        self.registered.append(str(task.task_id))
        self.attached.add(str(task.task_id))
        self.assigned[str(task.task_id)] = [BOSS, CHALLENGER]

    async def _existing(self, round_id, config, group_id=None, pair_id=None):
        return [SimpleNamespace(task_id=task_id) for task_id in self.registered]

    def seed_set(self):
        """The BOSS_INSTRUCT seed (SUCCESS, an input seed) and the two compression seeds chained on it, unattached, only the
        boss assigned. Returns (instruct, [free, fixed])."""
        instruct = self.add(_instruct(), [BOSS], SeedRole.BOSS_INSTRUCT)
        instruct.status = TaskStatus.SUCCESS
        self.training[str(instruct.task_id)] = {BOSS: SUCCESS}
        chained = [
            self.add(_chained(instruct), [BOSS], SeedRole.BOSS_CHAINED_FREEFORM),
            self.add(_chained(instruct, freeform=False), [BOSS], SeedRole.BOSS_CHAINED_FIXED),
        ]
        for task in chained:
            self.training[str(task.task_id)] = {BOSS: TRAINING}
        return instruct, chained

    async def ready_ids(self) -> set[str]:
        return {str(task.task_id) for task in await lifecycle_tasks._get_tasks_ready_for_evaluation(CONFIG)}


@pytest.fixture
def pipe(monkeypatch):
    return Pipeline(monkeypatch)


class TestBossRoundSeedLifecycle:
    async def test_boss_is_evaluated_at_seed_time_and_the_task_finalised_after_the_challenger(self, pipe):
        instruct, (_, fixed) = pipe.seed_set()
        key = str(fixed.task_id)

        # The Fixed compression is released on the boss's output of the instruct seed, alone, and the boss trains it.
        await orchestrator._release_chained_task(fixed, CONFIG)
        assert fixed.status == TaskStatus.PENDING and fixed.model_id == _output(instruct, BOSS)
        fixed.status = TaskStatus.TRAINING

        # The boss's run finishes: the seed stays TRAINING, but the boss's row gets its evaluation row and the lifecycle
        # picks the task up to evaluate it (finalisation is held, see TestSkipSites).
        pipe.training[key][BOSS] = SUCCESS
        await _seeding_pass()
        assert pipe.eval_rows[key] == {BOSS}
        assert key in await pipe.ready_ids()
        await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)
        assert fixed.status == TaskStatus.TRAINING

        # The boss round attaches the two compression seeds; still no challenger row, still held.
        await task_creator._attach_boss_round_seeds(pipe.seeds, "tid", ROUND_ID, f"{ROUND_ID}_pair_001", CONFIG)
        assert key in pipe.attached and len(pipe.registered) == 2
        assert str(instruct.task_id) not in pipe.registered
        await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)
        assert fixed.status == TaskStatus.TRAINING

        # The challenger's row is added; until it is terminal the seed is still held.
        await orchestrator._boss_round_seed_cycle(CONFIG)
        assert pipe.training[key][CHALLENGER] == TrainingStatus.PENDING.value
        for state in (TrainingStatus.PENDING.value, TRAINING):
            pipe.training[key][CHALLENGER] = state
            await _seeding_pass()
            await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)
            assert fixed.status == TaskStatus.TRAINING and pipe.eval_rows[key] == {BOSS}

        # The challenger finishes: its own evaluation row is added beside the boss's stored one, and the task moves on.
        pipe.training[key][CHALLENGER] = SUCCESS
        await _seeding_pass()
        assert pipe.eval_rows[key] == {BOSS, CHALLENGER}
        await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)
        assert fixed.status == TaskStatus.PREEVALUATION

    async def test_the_boss_instruct_seed_finalises_like_any_task_without_a_challenger(self, pipe):
        instruct, _ = pipe.seed_set()
        instruct.status = TaskStatus.TRAINING
        key = str(instruct.task_id)

        await _seeding_pass()
        await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)

        assert pipe.eval_rows[key] == {BOSS} and instruct.status == TaskStatus.PREEVALUATION


class TestSkipSites:
    """Evaluation of a seed's rows is never held; only finalisation and the PREEVALUATION move are."""

    @pytest.fixture
    def tasks(self, pipe):
        _, (held, _) = pipe.seed_set()
        held.status = TaskStatus.TRAINING
        plain = pipe.add(_instruct(), [BOSS, CHALLENGER])
        for task in (held, plain):
            pipe.training[str(task.task_id)] = {BOSS: SUCCESS}
        return SimpleNamespace(held=held, plain=plain)

    async def test_completed_training_query_skips_a_held_seed_with_nothing_to_evaluate(self, pipe, tasks):
        ready = await pipe.ready_ids()

        assert str(tasks.plain.task_id) in ready and str(tasks.held.task_id) not in ready

    async def test_pending_evaluation_rows_pick_up_a_held_seed_to_evaluate_the_boss(self, pipe, tasks, monkeypatch):
        monkeypatch.setattr(orchestrator.tournament_sql, "get_tasks_with_all_training_completed", AsyncMock(return_value=[]))
        pipe.eval_rows = {str(tasks.held.task_id): {BOSS}}

        assert await pipe.ready_ids() == {str(tasks.held.task_id)}

    async def test_a_parked_seed_time_evaluation_is_skipped_until_attached(self, pipe, tasks, monkeypatch):
        monkeypatch.setattr(orchestrator.tournament_sql, "get_tasks_with_all_training_completed", AsyncMock(return_value=[]))
        monkeypatch.setattr(lifecycle_tasks.boss_seed_eval, "seed_time_retries_spent", lambda task_id: True)
        pipe.eval_rows = {str(tasks.held.task_id): {BOSS}}
        assert await pipe.ready_ids() == set()

        pipe.attached.add(str(tasks.held.task_id))
        assert await pipe.ready_ids() == {str(tasks.held.task_id)}

    async def test_seeding_pass_seeds_the_boss_row_of_a_held_seed(self, pipe, tasks):
        await _seeding_pass()

        assert str(tasks.held.task_id) in pipe.pair_calls and str(tasks.plain.task_id) in pipe.pair_calls
        assert pipe.eval_rows[str(tasks.held.task_id)] == {BOSS}

    async def test_finalisation_of_a_held_seed_is_deferred(self, pipe, tasks, monkeypatch):
        finalise = AsyncMock()
        monkeypatch.setattr(lifecycle_tasks, "finalize_task_scores_from_raw_losses", finalise)

        assert not await lifecycle_tasks._finalize_task_status_from_evaluations(tasks.held, CONFIG)
        finalise.assert_not_awaited()
        assert tasks.held.status == TaskStatus.TRAINING

    async def test_preevaluation_move_skips_a_held_seed(self, pipe, tasks):
        await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)

        assert tasks.held.status == TaskStatus.TRAINING and tasks.plain.status == TaskStatus.PREEVALUATION

    async def test_a_text_task_that_is_not_a_seed_is_never_held(self, pipe):
        plain = pipe.add(_instruct(), [BOSS, CHALLENGER])
        pipe.training[str(plain.task_id)] = {BOSS: SUCCESS}

        await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)

        assert plain.status == TaskStatus.PREEVALUATION


class TestRoundOpenCheck:
    async def test_the_cycle_checks_each_attached_seed_whose_boss_row_trained(self, pipe, monkeypatch):
        alert = AsyncMock(return_value=True)
        monkeypatch.setattr(orchestrator.boss_seed_eval, "alert_if_boss_evaluation_missing", alert)
        _, (free, fixed) = pipe.seed_set()
        pipe.training[str(free.task_id)][BOSS] = SUCCESS
        free.status = fixed.status = TaskStatus.TRAINING
        await task_creator._attach_boss_round_seeds(pipe.seeds, "tid", ROUND_ID, f"{ROUND_ID}_pair_001", CONFIG)

        await orchestrator._boss_round_seed_cycle(CONFIG)

        # Only the seed whose boss row is SUCCESS is checked, after the challenger's row.
        assert [call.args[0].task_id for call in alert.await_args_list] == [free.task_id]
        assert pipe.training[str(free.task_id)][CHALLENGER] == TrainingStatus.PENDING.value

    async def test_a_failing_check_never_blocks_the_challengers_rows(self, pipe, monkeypatch):
        monkeypatch.setattr(
            orchestrator.boss_seed_eval.tasks_sql, "get_task_evaluation_rows", AsyncMock(side_effect=RuntimeError("db"))
        )
        _, (free, fixed) = pipe.seed_set()
        pipe.training[str(free.task_id)][BOSS] = SUCCESS
        free.status = fixed.status = TaskStatus.TRAINING
        await task_creator._attach_boss_round_seeds(pipe.seeds, "tid", ROUND_ID, f"{ROUND_ID}_pair_001", CONFIG)

        await orchestrator._boss_round_seed_cycle(CONFIG)

        assert pipe.training[str(free.task_id)][CHALLENGER] == TrainingStatus.PENDING.value
