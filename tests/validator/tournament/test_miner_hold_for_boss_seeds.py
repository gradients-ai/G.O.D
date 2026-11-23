"""Miner rows of a tournament are not dispatched until the boss is training every text seed of that tournament.

The boss trains its text seeds upfront (the compression tournament compresses their models), so they take the GPUs before
the round's miners. The boss's own rows, benchmark rows, other tournaments' rows and organic rows are never held. A seed
stops holding once the boss's row on it is training or finished (or failed with no replacement to come), it left
`training`, it is past its time bound, or its tournament is decided; it holds again while a run a trainer gave back waits
for its next attempt. Held rows do not count towards the pending-queue thresholds, so the seed tasks they wait for still
reach training.
"""

import uuid
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.tournament.constants as t_cst
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from tests.validator.tournament.test_quant_dispatch_bucket import Dispatch
from tests.validator.tournament.test_quant_scheduling_isolation import CONFIG
from tests.validator.tournament.test_quant_scheduling_isolation import Scheduling
from tests.validator.tournament.test_quant_scheduling_isolation import _row
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament import orchestrator
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus


TID = "tourn_text"
OTHER_TID = "tourn_other"
SEED_HOURS = 3.0
ENDED_WITHOUT_A_RESULT = [
    pytest.param(TaskStatus.FAILURE, TrainingStatus.FAILURE.value, id="task-failed"),
    pytest.param(TaskStatus.TRAINING, TrainingStatus.FAILURE.value, id="boss-row-failed"),
    pytest.param(None, None, id="task-gone"),
]


class Seeds:
    """The seed rows, seed task statuses, boss row statuses and tournaments the database would hold."""

    def __init__(self, monkeypatch) -> None:
        self.rows: dict[str, list[TournamentSeedTask]] = {}
        self.history: dict[str, list[TournamentSeedTask]] = {}
        self.task_status: dict[str, TaskStatus | None] = {}
        self.boss_status: dict[str, str] = {}
        self.in_a_round: set[str] = set()
        self.tournament = SimpleNamespace(winner_hotkey=None, status=TournamentStatus.ACTIVE)
        self.lookups: list[str] = []
        self.alert = AsyncMock()

        async def latest(tournament_id, _db):
            self.lookups.append(tournament_id)
            return self.rows.get(tournament_id, [])

        async def history(tournament_id, _db):
            return self.history.get(tournament_id, []) + self.rows.get(tournament_id, [])

        async def get_task(task_id, _db):
            status = self.task_status.get(str(task_id), TaskStatus.TRAINING)
            if status is None:
                return None
            return SimpleNamespace(task_id=task_id, status=status, hours_to_complete=SEED_HOURS, ds="org/dataset")

        async def boss_rows(task_id, hotkeys, _db):
            assert hotkeys == [EMISSION_BURN_HOTKEY]
            status = self.boss_status.get(str(task_id))
            return {EMISSION_BURN_HOTKEY: status} if status else {}

        async def attached_to_round(task_id, _db):
            return str(task_id) in self.in_a_round

        async def get_tournament(tournament_id, _db):
            return self.tournament

        monkeypatch.setattr(orchestrator.seeds_sql, "get_latest_seed_tasks", latest)
        monkeypatch.setattr(orchestrator.seeds_sql, "get_seed_task_history", history)
        monkeypatch.setattr(orchestrator.seeds_sql, "is_task_attached_to_round", attached_to_round)
        monkeypatch.setattr(orchestrator.task_sql, "get_task", get_task)
        monkeypatch.setattr(orchestrator.tournament_sql, "get_training_status_for_task_and_hotkeys", boss_rows)
        monkeypatch.setattr(orchestrator.tournament_sql, "get_tournament", get_tournament)
        monkeypatch.setattr(orchestrator, "alert_seed_issue", self.alert)

    def add(
        self,
        role: SeedRole = SeedRole.TEXT_R2,
        status: SeedStatus = SeedStatus.TRAINING,
        boss: str | None = TrainingStatus.PENDING.value,
        tournament_id: str = TID,
        attempt: int = 1,
        age_hours: float = 0.0,
    ) -> TournamentSeedTask:
        seed = TournamentSeedTask(
            tournament_id=tournament_id,
            role=role,
            attempt=attempt,
            task_id=uuid.uuid4(),
            status=status,
            created_at=datetime.now(timezone.utc) - timedelta(hours=age_hours),
        )
        self.rows.setdefault(tournament_id, []).append(seed)
        if boss is not None:
            self.boss_status[str(seed.task_id)] = boss
        return seed

    def add_charged_replacements(self, seed: TournamentSeedTask, count: int) -> None:
        """Give the seed's role `count` earlier attempts that were replaced for a charged failure."""
        seed.attempt = count + 1
        self.history.setdefault(seed.tournament_id, []).extend(
            TournamentSeedTask(
                tournament_id=seed.tournament_id,
                role=seed.role,
                attempt=attempt,
                task_id=uuid.uuid4(),
                status=SeedStatus.REPLACED,
            )
            for attempt in range(1, count + 1)
        )


@pytest.fixture
def scheduling(monkeypatch) -> Scheduling:
    return Scheduling(monkeypatch)


@pytest.fixture
def seeds(monkeypatch, scheduling) -> Seeds:
    """Built after `scheduling`, whose own seed lookup (no seeds at all) this one replaces."""
    return Seeds(monkeypatch)


class TestASeedInTraining:
    @pytest.mark.parametrize("boss", [None, TrainingStatus.PENDING.value])
    async def test_holds_while_the_boss_row_is_absent_or_pending(self, seeds, boss):
        seed = seeds.add(boss=boss)

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == [seed]

    @pytest.mark.parametrize("boss", [TrainingStatus.TRAINING.value, TrainingStatus.SUCCESS.value])
    async def test_releases_once_the_boss_row_is_training_or_succeeded(self, seeds, boss):
        seeds.add(boss=boss)

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == []

    async def test_only_the_seeds_the_boss_is_not_training_yet_hold(self, seeds):
        seeds.add(role=SeedRole.TEXT_R1, boss=TrainingStatus.TRAINING.value)
        seeds.add(role=SeedRole.TEXT_R2, boss=TrainingStatus.SUCCESS.value)
        waiting = seeds.add(role=SeedRole.TEXT_R3, boss=TrainingStatus.PENDING.value)
        unprepped = seeds.add(role=SeedRole.TEXT_CONTINUOUS, boss=None)

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == [waiting, unprepped]

    async def test_holds_again_while_a_run_given_back_waits_for_its_next_attempt(self, seeds):
        seed = seeds.add(boss=TrainingStatus.TRAINING.value)
        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == []

        seeds.boss_status[str(seed.task_id)] = TrainingStatus.PENDING.value
        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == [seed]

        seeds.boss_status[str(seed.task_id)] = TrainingStatus.TRAINING.value
        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == []

    @pytest.mark.parametrize("task_status", [TaskStatus.PENDING, TaskStatus.AWAITING_MODEL_PREP, TaskStatus.LOOKING_FOR_NODES])
    async def test_holds_before_its_task_reached_training(self, seeds, task_status):
        seed = seeds.add(boss=None)
        seeds.task_status[str(seed.task_id)] = task_status

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == [seed]

    @pytest.mark.parametrize("role", sorted(orchestrator.TEXT_SEED_ROLES, key=lambda role: role.value))
    async def test_every_text_role_holds(self, seeds, role):
        seed = seeds.add(role=role)

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == [seed]

    async def test_releases_once_its_task_succeeded(self, seeds):
        seed = seeds.add()
        seeds.task_status[str(seed.task_id)] = TaskStatus.SUCCESS

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == []

    @pytest.mark.parametrize("status", [SeedStatus.TRAINED, SeedStatus.READY, SeedStatus.REPLACED, SeedStatus.FAILED])
    async def test_releases_once_the_seed_left_training(self, seeds, status):
        seeds.add(status=status)

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == []

    @pytest.mark.parametrize("role", [SeedRole.INPUT_SMALL, SeedRole.QUANT_R1, SeedRole.BOSS_R2_FREEFORM])
    async def test_a_seed_of_another_kind_never_holds(self, seeds, role):
        seeds.add(role=role)

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == []


class TestASeedWhoseRunEndedWithoutAResult:
    def _end(self, seeds, seed, task_status, boss):
        seeds.task_status[str(seed.task_id)] = task_status
        seeds.boss_status.pop(str(seed.task_id), None)
        if boss is not None:
            seeds.boss_status[str(seed.task_id)] = boss

    @pytest.mark.parametrize("task_status, boss", ENDED_WITHOUT_A_RESULT)
    async def test_holds_while_a_replacement_is_to_come(self, seeds, task_status, boss):
        seed = seeds.add()
        self._end(seeds, seed, task_status, boss)

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == [seed]

    @pytest.mark.parametrize("task_status, boss", ENDED_WITHOUT_A_RESULT)
    async def test_releases_when_no_charged_replacement_is_left(self, seeds, task_status, boss):
        seed = seeds.add()
        seeds.add_charged_replacements(seed, t_cst.SEED_MAX_REPLACEMENTS)
        self._end(seeds, seed, task_status, boss)

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == []

    @pytest.mark.parametrize("task_status, boss", ENDED_WITHOUT_A_RESULT)
    @pytest.mark.parametrize("marker", ["attached_round_id", "tournament_tasks row"])
    async def test_releases_when_a_round_attached_it(self, seeds, task_status, boss, marker):
        seed = seeds.add()
        if marker == "attached_round_id":
            seed.attached_round_id = "round_1"
        else:
            seeds.in_a_round.add(str(seed.task_id))
        self._end(seeds, seed, task_status, boss)

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == []

    @pytest.mark.parametrize("attached", [False, True])
    async def test_a_prep_failure_holds_attached_or_not_and_whatever_was_charged(self, seeds, attached):
        seed = seeds.add(boss=None)
        seeds.add_charged_replacements(seed, t_cst.SEED_MAX_REPLACEMENTS)
        seed.attached_round_id = "round_1" if attached else None
        seeds.task_status[str(seed.task_id)] = TaskStatus.PREP_TASK_FAILURE

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == [seed]

    async def test_a_prep_failure_of_the_continuous_seed_needs_a_charged_replacement(self, seeds):
        seed = seeds.add(role=SeedRole.TEXT_CONTINUOUS, boss=None)
        seeds.add_charged_replacements(seed, t_cst.SEED_MAX_REPLACEMENTS)
        seeds.task_status[str(seed.task_id)] = TaskStatus.PREP_TASK_FAILURE

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == []


class TestTheTimeBound:
    # Two text seeds of SEED_HOURS each, every attempt of both, plus the margin.
    BOUND_HOURS = t_cst.MINER_HOLD_MARGIN_HOURS + t_cst.MAX_TRAINING_ATTEMPTS * 2 * SEED_HOURS

    async def test_a_seed_within_its_bound_holds_and_is_not_reported(self, seeds):
        seeds.add(role=SeedRole.TEXT_R1, status=SeedStatus.TRAINED)
        seed = seeds.add(age_hours=self.BOUND_HOURS - 0.5)

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == [seed]
        seeds.alert.assert_not_awaited()

    async def test_a_seed_past_its_bound_is_reported_and_holds_no_more(self, seeds):
        seeds.add(role=SeedRole.TEXT_R1, status=SeedStatus.TRAINED)
        seed = seeds.add(age_hours=self.BOUND_HOURS + 0.5)

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == []
        key, message, _config = seeds.alert.await_args.args
        assert key == f"{TID}:{seed.role.value}:{seed.attempt}:miner-hold-expired"
        assert str(seed.task_id) in message and f"{self.BOUND_HOURS:.1f} hours" in message

    async def test_a_seed_past_its_bound_does_not_release_a_younger_one(self, seeds):
        seeds.add(role=SeedRole.TEXT_R1, age_hours=self.BOUND_HOURS + 0.5)
        young = seeds.add(role=SeedRole.TEXT_R2)

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == [young]


class TestADecidedTournament:
    @pytest.mark.parametrize("status", [TournamentStatus.COMPLETED, TournamentStatus.CANCELLED])
    async def test_a_tournament_that_is_over_holds_nothing(self, seeds, status):
        seeds.add()
        seeds.tournament.status = status

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == []

    async def test_a_tournament_with_a_winner_holds_nothing(self, seeds):
        seeds.add()
        seeds.tournament.winner_hotkey = "5Winner"

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == []

    @pytest.mark.parametrize("status", [TournamentStatus.PENDING, TournamentStatus.ACTIVE])
    async def test_a_running_tournament_holds(self, seeds, status):
        seed = seeds.add()
        seeds.tournament.status = status

        assert await orchestrator._text_seeds_awaiting_training(TID, CONFIG) == [seed]


class TestScheduling:
    @pytest.fixture(autouse=True)
    def tournaments(self, scheduling, monkeypatch):
        self.tournament_of: dict[str, str | None] = {}

        async def tournament_id(task_id, _db):
            return self.tournament_of.get(str(task_id), TID)

        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", tournament_id)

    def _row_of(self, tournament_id: str | None, hotkey: str = "5Miner") -> SimpleNamespace:
        row = _row(TaskType.INSTRUCTTEXTTASK, hotkey)
        self.tournament_of[str(row.task.task_id)] = tournament_id
        return row

    async def test_miner_rows_wait_and_the_boss_rows_are_dispatched(self, scheduling, seeds):
        seeds.add(role=SeedRole.TEXT_R2)
        r1_miner = self._row_of(TID)
        r1_boss, r2_boss = self._row_of(TID, EMISSION_BURN_HOTKEY), self._row_of(TID, EMISSION_BURN_HOTKEY)
        pending = [r1_miner, r2_boss, r1_boss]

        all_held = await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        assert scheduling.built == [r1_boss.task.task_id, r2_boss.task.task_id]
        assert pending == []
        assert not all_held

    async def test_a_cycle_holding_every_row_does_no_gpu_lookup_and_waits(self, scheduling, seeds, caplog):
        seed = seeds.add(role=SeedRole.TEXT_CONTINUOUS)
        pending = [self._row_of(TID, f"5Miner{i}") for i in range(3)]

        with caplog.at_level("INFO"):
            all_held = await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        assert all_held
        assert scheduling.built == []
        orchestrator._check_suitable_gpus.assert_not_awaited()
        orchestrator.tournament_sql.update_tournament_task_training_status.assert_not_awaited()
        assert seeds.lookups == [TID]  # once per tournament per cycle
        held_lines = [record for record in caplog.records if "Held 3 miner training row(s)" in record.getMessage()]
        assert len(held_lines) == 1
        assert f"{TID}/{SeedRole.TEXT_CONTINUOUS.value} (task {seed.task_id})" in held_lines[0].getMessage()

    async def test_other_tournaments_and_organic_rows_are_not_held(self, scheduling, seeds):
        seeds.add()
        held, other, organic = self._row_of(TID), self._row_of(OTHER_TID), self._row_of(None)
        pending = [held, organic, other]

        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        assert scheduling.built == [other.task.task_id, organic.task.task_id]

    async def test_a_benchmark_row_of_the_tournament_is_not_held(self, scheduling, seeds):
        seeds.add()
        miner, benchmark = self._row_of(TID), self._row_of(TID, "5Winner")
        benchmark.priority = 3

        await orchestrator.schedule_tasks_for_training([miner, benchmark], CONFIG)

        assert scheduling.built == [benchmark.task.task_id]

    def _boss_rows_follow_dispatch(self, seeds):
        """A dispatched boss row becomes TRAINING in the seeds' view, as the database row does."""

        async def set_status(task_id, hotkey, status, *_args, **_kwargs):
            if hotkey == EMISSION_BURN_HOTKEY:
                seeds.boss_status[str(task_id)] = status.value

        orchestrator.tournament_sql.update_tournament_task_training_status.side_effect = set_status

    def _boss_row_on(self, seed: TournamentSeedTask) -> SimpleNamespace:
        row = self._row_of(TID, EMISSION_BURN_HOTKEY)
        row.task.task_id = seed.task_id
        self.tournament_of[str(seed.task_id)] = TID
        return row

    async def test_the_miners_follow_the_last_boss_row_dispatched_in_the_same_cycle(self, scheduling, seeds):
        self._boss_rows_follow_dispatch(seeds)
        seeds.add(role=SeedRole.TEXT_R1, boss=TrainingStatus.TRAINING.value)
        boss = self._boss_row_on(seeds.add(role=SeedRole.TEXT_R2))
        miners = [self._row_of(TID, f"5Miner{i}") for i in range(2)]

        all_held = await orchestrator.schedule_tasks_for_training([*miners, boss], CONFIG)  # the last row is taken first

        assert not all_held
        assert scheduling.built[0] == boss.task.task_id
        assert sorted(scheduling.built[1:], key=str) == sorted((row.task.task_id for row in miners), key=str)

    async def test_the_miners_wait_while_a_boss_row_finds_no_gpu(self, scheduling, seeds):
        self._boss_rows_follow_dispatch(seeds)
        boss = self._boss_row_on(seeds.add())
        miner = self._row_of(TID)
        orchestrator._check_suitable_gpus.return_value = None

        await orchestrator.schedule_tasks_for_training([miner, boss], CONFIG)

        assert scheduling.built == []

    async def test_a_boss_row_given_back_is_dispatched_again_before_the_pending_miners(self, scheduling, seeds):
        self._boss_rows_follow_dispatch(seeds)
        seed = seeds.add(boss=TrainingStatus.TRAINING.value)
        early, late = self._row_of(TID, "5Early"), self._row_of(TID, "5Late")
        await orchestrator.schedule_tasks_for_training([early], CONFIG)
        assert scheduling.built == [early.task.task_id]

        seeds.boss_status[str(seed.task_id)] = TrainingStatus.PENDING.value  # the trainer gave the boss's run back
        orchestrator._check_suitable_gpus.return_value = None
        assert await orchestrator.schedule_tasks_for_training([late], CONFIG)  # held again, waits a cycle
        assert scheduling.built == [early.task.task_id]

        orchestrator._check_suitable_gpus.return_value = ("1.2.3.4", [0], "NVLINK")
        await orchestrator.schedule_tasks_for_training([late, self._boss_row_on(seed)], CONFIG)
        assert scheduling.built == [early.task.task_id, seed.task_id, late.task.task_id]

    async def test_miner_rows_are_dispatched_once_the_boss_trains_or_trained_every_seed(self, scheduling, seeds):
        seeds.add(role=SeedRole.TEXT_R1, boss=TrainingStatus.TRAINING.value)
        seeds.add(role=SeedRole.TEXT_R2, boss=TrainingStatus.SUCCESS.value)
        seeds.add(role=SeedRole.TEXT_R3, status=SeedStatus.FAILED)
        miner = self._row_of(TID)

        all_held = await orchestrator.schedule_tasks_for_training([miner], CONFIG)

        assert scheduling.built == [miner.task.task_id]
        assert not all_held

    async def test_a_held_row_is_not_dispatched_marked_or_charged_an_attempt(self, scheduling, seeds):
        seeds.add()
        miner = self._row_of(TID)

        await orchestrator.schedule_tasks_for_training([miner], CONFIG)

        orchestrator.start_training_task.assert_not_awaited()
        orchestrator.tournament_sql.update_tournament_task_training_status.assert_not_awaited()
        orchestrator.tournament_sql.update_gpu_availability.assert_not_awaited()


class TestTheFetchCycle:
    """The real fetch cycle and the real scheduling cycle, over the same pending rows, one after the other."""

    @pytest.fixture
    def dispatch(self, monkeypatch, seeds) -> Dispatch:
        dispatch = Dispatch(monkeypatch)
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=TID))
        self.benchmark_passes = 0

        async def process(tasks, config, priority):
            if priority == 2:
                dispatch.processed.append((list(tasks), priority))
            elif priority == 3:
                self.benchmark_passes += 1

        monkeypatch.setattr(orchestrator, "_process_tasks_for_training", process)
        return dispatch

    def _miner_rows(self, dispatch: Dispatch, count: int) -> list[SimpleNamespace]:
        rows = [_row(TaskType.INSTRUCTTEXTTASK, f"5Miner{i}") for i in range(count)]
        for row in rows:
            dispatch.types[str(row.task.task_id)] = TournamentType.TEXT
        dispatch.pending = list(rows)
        return rows

    def _seed_task_looking_for_nodes(self, dispatch: Dispatch, seeds: Seeds) -> SimpleNamespace:
        seed = seeds.add(boss=None)
        seeds.task_status[str(seed.task_id)] = TaskStatus.LOOKING_FOR_NODES
        task = SimpleNamespace(task_id=seed.task_id, task_type=TaskType.INSTRUCTTEXTTASK)
        dispatch.types[str(task.task_id)] = TournamentType.TEXT
        dispatch.waiting = [task]
        return task

    @pytest.mark.parametrize("n_miners", [t_cst.PENDING_QUEUE_THRESHOLD_PER_TYPE, 20])
    async def test_a_seed_task_reaches_training_while_a_full_queue_of_miners_waits_for_it(
        self, scheduling, seeds, dispatch, n_miners
    ):
        self._miner_rows(dispatch, n_miners)
        seed_task = self._seed_task_looking_for_nodes(dispatch, seeds)

        buckets = await dispatch.run()
        all_held = await orchestrator.schedule_tasks_for_training(list(dispatch.pending), CONFIG)

        assert buckets["text"] == [seed_task]
        assert all_held and scheduling.built == []

    async def test_the_miners_are_dispatched_once_the_boss_trains_that_seed(self, scheduling, seeds, dispatch):
        rows = self._miner_rows(dispatch, 20)
        seed_task = self._seed_task_looking_for_nodes(dispatch, seeds)
        assert (await dispatch.run())["text"] == [seed_task]

        seeds.task_status[str(seed_task.task_id)] = TaskStatus.TRAINING
        seeds.boss_status[str(seed_task.task_id)] = TrainingStatus.TRAINING.value
        all_held = await orchestrator.schedule_tasks_for_training(list(dispatch.pending), CONFIG)

        assert not all_held
        assert sorted(scheduling.built, key=str) == sorted((row.task.task_id for row in rows), key=str)

    async def test_rows_that_are_not_held_still_fill_the_queue(self, scheduling, seeds, dispatch):
        self._miner_rows(dispatch, t_cst.PENDING_QUEUE_THRESHOLD_PER_TYPE)
        dispatch.waiting = [dispatch.task(TaskType.INSTRUCTTEXTTASK, TournamentType.TEXT)]

        assert (await dispatch.run())["text"] == []

    async def test_held_rows_do_not_keep_benchmark_tasks_out(self, scheduling, seeds, dispatch):
        self._miner_rows(dispatch, 20)
        seeds.add()

        await dispatch.run()

        assert self.benchmark_passes == 1

    async def test_rows_that_are_not_held_keep_benchmark_tasks_out(self, scheduling, seeds, dispatch):
        self._miner_rows(dispatch, t_cst.PENDING_QUEUE_THRESHOLD_FOR_BENCHMARK)

        await dispatch.run()

        assert self.benchmark_passes == 0
