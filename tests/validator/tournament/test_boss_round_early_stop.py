"""Stopping a planned image or compression boss round once the crown is out of reach: the work still to come is stopped
without touching a running training, and a compression tournament ends the same way as an image one."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.tournament.orchestrator as orchestrator
import validator.tournament.seed_tasks as seeds
import validator.tournament.tournament_manager as tm
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournaments as tournament_sql
from validator.tournament.crown import BossRoundProgress
from validator.tournament.crown import CrownVerdict
from validator.tournament.models import RoundStatus
from validator.tournament.models import RoundType
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus


TID = "tourn_stop"


def _round(number: int, final: bool = False) -> TournamentRoundData:
    return TournamentRoundData(
        round_id=f"r{number}",
        tournament_id=TID,
        round_number=number,
        round_type=RoundType.KNOCKOUT if final else RoundType.GROUP,
        is_final_round=final,
        status=RoundStatus.ACTIVE,
    )


class TestStopUnfinishedBossRoundTasks:
    @pytest.fixture
    def world(self, monkeypatch):
        tasks = {
            "unstarted": SimpleNamespace(task_id="unstarted", status=TaskStatus.PENDING),
            "training": SimpleNamespace(task_id="training", status=TaskStatus.TRAINING),
            "evaluating": SimpleNamespace(task_id="evaluating", status=TaskStatus.EVALUATING),
        }
        pending = {"unstarted": ["boss", "miner"], "training": ["miner"], "evaluating": []}
        updated_rows: list[tuple[str, str, TrainingStatus]] = []
        updated_tasks: list[str] = []

        async def rows(task_id, status, _db):
            assert status == TrainingStatus.PENDING
            return [SimpleNamespace(hotkey=hotkey) for hotkey in pending[task_id]]

        async def fail_pending(task_id, hotkey, _db):
            updated_rows.append((task_id, hotkey, TrainingStatus.FAILURE))
            return True

        seeds_of = {"unstarted": SimpleNamespace(tournament_id=TID, role="role", attempt=1, status=SeedStatus.TRAINING)}
        failed_seeds: list[str] = []

        async def update_seed(_tid, role, attempt, status, _db):
            failed_seeds.append(f"{role}/{attempt}/{status.value}")

        async def update_task(task, _db):
            updated_tasks.append(task.task_id)

        monkeypatch.setattr(seeds.tournament_sql, "get_training_rows_for_task", rows)
        monkeypatch.setattr(seeds.tournament_sql, "fail_pending_training_row", fail_pending)
        monkeypatch.setattr(
            seeds.seeds_sql, "get_seed_task_by_task_id", AsyncMock(side_effect=lambda task_id, _db: seeds_of.get(task_id))
        )
        monkeypatch.setattr(seeds.seeds_sql, "update_seed_status", update_seed)
        monkeypatch.setattr(seeds.task_sql, "get_task", AsyncMock(side_effect=lambda task_id, _db: tasks[task_id]))
        monkeypatch.setattr(seeds.task_sql, "update_task", update_task)
        return SimpleNamespace(tasks=tasks, rows=updated_rows, updated_tasks=updated_tasks, failed_seeds=failed_seeds)

    async def test_pending_rows_never_dispatch_and_only_an_unstarted_task_is_failed(self, world):
        config = SimpleNamespace(psql_db=None)
        stopped = await seeds.stop_unfinished_boss_round_tasks(["unstarted", "training", "evaluating"], config)

        assert stopped == ["unstarted"]
        assert world.rows == [
            ("unstarted", "boss", TrainingStatus.FAILURE),
            ("unstarted", "miner", TrainingStatus.FAILURE),
            ("training", "miner", TrainingStatus.FAILURE),
        ]
        assert world.updated_tasks == ["unstarted"]
        assert world.failed_seeds == ["role/1/failed"]
        assert world.tasks["unstarted"].status == TaskStatus.FAILURE
        assert world.tasks["training"].status == TaskStatus.TRAINING
        assert world.tasks["evaluating"].status == TaskStatus.EVALUATING


class TestQuantBossRoundStop:
    @pytest.fixture
    def stop(self, monkeypatch):
        tournament = TournamentData(tournament_id=TID, tournament_type=TournamentType.QUANT)
        boss_round = _round(4, final=True)
        third = _round(3)
        verdict = CrownVerdict(crowned_if_complete=False, non_wins=3, decided=3, reason="3 non-wins of 3 decided")
        progress = BossRoundProgress(challenger="C", verdict=verdict, decided=[], unfinished_task_ids=["t2", "t3"])
        holder = SimpleNamespace(progress=progress)
        complete = AsyncMock()
        stopped = AsyncMock(return_value=[])
        status = AsyncMock()
        monkeypatch.setattr(tm, "boss_round_progress", AsyncMock(side_effect=lambda *_: holder.progress))
        monkeypatch.setattr(tm, "plays_round_plan", AsyncMock(return_value=True))
        monkeypatch.setattr(tm, "get_tournament_rounds", AsyncMock(return_value=[third, boss_round]))
        monkeypatch.setattr(tm, "stop_unfinished_boss_round_tasks", stopped)
        monkeypatch.setattr(tm, "persist_decided_boss_round_winners", AsyncMock())
        monkeypatch.setattr(tm, "update_round_status", status)
        monkeypatch.setattr(tm, "_complete_without_boss_round", complete)
        config = SimpleNamespace(psql_db=None)
        return SimpleNamespace(
            run=lambda: tm.stop_unreachable_boss_round(tournament, boss_round, config, None),
            holder=holder,
            complete=complete,
            stopped=stopped,
            status=status,
            third=third,
            boss_round=boss_round,
        )

    async def test_an_unreachable_crown_stops_the_rest_and_ends_the_tournament_through_the_early_stop(self, stop):
        assert await stop.run() is True

        stop.stopped.assert_awaited_once_with(["t2", "t3"], stop.stopped.await_args.args[1])
        stop.status.assert_awaited_once_with(stop.boss_round.round_id, RoundStatus.COMPLETED, None)
        assert stop.complete.await_args.args[1] is stop.third
        assert stop.complete.await_args.args[2] == "C"
        assert stop.complete.await_args.kwargs == {"boss_round": stop.boss_round}

    async def test_a_reachable_crown_does_nothing(self, stop):
        stop.holder.progress = stop.holder.progress.model_copy(
            update={"verdict": CrownVerdict(crowned_if_complete=True, non_wins=1, decided=3, reason="ok")}
        )

        assert await stop.run() is False

        stop.stopped.assert_not_awaited()
        stop.complete.assert_not_awaited()

    async def test_a_round_with_nothing_left_to_stop_does_nothing(self, stop):
        stop.holder.progress = stop.holder.progress.model_copy(update={"unfinished_task_ids": []})

        assert await stop.run() is False

        stop.complete.assert_not_awaited()


class _Connection:
    def __init__(self, sink: list[str]) -> None:
        self.sink = sink

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def fetch(self, query, *_args):
        self.sink.append(query)
        return []

    async def execute(self, query, *_args):
        self.sink.append(query)
        return "UPDATE 1"

    async def fetchval(self, query, *_args):
        self.sink.append(query)
        return 1


class _Db:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def connection(self):
        return _Connection(self.queries)


class TestSchedulerNeverStartsAStoppedRound:
    async def test_pending_rows_of_a_decided_final_round_are_left_out_of_the_fetch(self):
        db = _Db()

        await tournament_sql.get_tournament_training_tasks(db, TrainingStatus.PENDING)
        await tournament_sql.get_tournament_training_tasks(db, TrainingStatus.TRAINING)

        pending, training = db.queries
        assert "AND NOT EXISTS" in pending and "winner_hotkey IS NOT NULL" in pending
        assert "EXISTS" not in training

    async def test_a_failure_is_written_only_while_the_row_is_pending(self):
        db = _Db()

        assert await tournament_sql.fail_pending_training_row("task", "hotkey", db) is True

        assert "training_status = $4" in db.queries[0]

    async def test_a_task_of_a_decided_final_round_gets_no_nodes_and_no_training_rows(self, monkeypatch):
        decided = AsyncMock(return_value=True)
        nodes = AsyncMock(return_value=[])
        monkeypatch.setattr(orchestrator.tournament_sql, "is_task_of_decided_final_round", decided)
        monkeypatch.setattr(orchestrator.task_sql, "get_nodes_assigned_to_task", nodes)
        failed_task = AsyncMock(return_value=True)
        monkeypatch.setattr(orchestrator.task_sql, "fail_task_if_status", failed_task)
        task = SimpleNamespace(task_id="t", task_type=TaskType.IMAGETASK)

        await orchestrator._process_tasks_for_training([task], SimpleNamespace(psql_db=None), priority=2)

        decided.assert_awaited_once()
        nodes.assert_not_awaited()
        failed_task.assert_awaited_once_with("t", [TaskStatus.LOOKING_FOR_NODES], None)

    async def test_pending_rows_of_a_decided_final_round_are_failed_on_the_cycle(self):
        db = _Db()

        assert await tournament_sql.fail_pending_rows_of_decided_final_rounds(db) == 1

        assert "training_status = $2" in db.queries[0] and "winner_hotkey IS NOT NULL" in db.queries[0]

    async def test_a_task_is_failed_only_from_the_given_statuses(self):
        db = _Db()

        assert await task_sql.fail_task_if_status("t", [TaskStatus.LOOKING_FOR_NODES], db) is True

        assert "status = ANY($3)" in db.queries[0]
