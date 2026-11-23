"""Miner rows of a tournament are not dispatched while the boss's text seeds of that tournament are in training.

The boss trains its text seeds upfront (the compression tournament compresses their models), so they take the GPUs before
the round's miners. The boss's own rows, other tournaments' rows and organic rows are never held; a seed stops holding once
the boss's row on it is terminal, its task ended, or it left `training` (trained, ready, replaced or failed).
"""

import uuid
from types import SimpleNamespace

import pytest

from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from tests.validator.tournament.test_quant_scheduling_isolation import CONFIG
from tests.validator.tournament.test_quant_scheduling_isolation import Scheduling
from tests.validator.tournament.test_quant_scheduling_isolation import _row
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament import orchestrator
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TrainingStatus


TID = "tourn_text"
OTHER_TID = "tourn_other"


class Seeds:
    """The seed rows, seed task statuses and boss row statuses the database would hold."""

    def __init__(self, monkeypatch) -> None:
        self.rows: dict[str, list[TournamentSeedTask]] = {}
        self.task_status: dict[str, TaskStatus | None] = {}
        self.boss_status: dict[str, str] = {}
        self.lookups: list[str] = []

        async def latest(tournament_id, _db):
            self.lookups.append(tournament_id)
            return self.rows.get(tournament_id, [])

        async def get_task(task_id, _db):
            status = self.task_status.get(str(task_id), TaskStatus.TRAINING)
            return None if status is None else SimpleNamespace(task_id=task_id, status=status)

        async def boss_rows(task_id, hotkeys, _db):
            assert hotkeys == [EMISSION_BURN_HOTKEY]
            status = self.boss_status.get(str(task_id))
            return {EMISSION_BURN_HOTKEY: status} if status else {}

        monkeypatch.setattr(orchestrator.seeds_sql, "get_latest_seed_tasks", latest)
        monkeypatch.setattr(orchestrator.task_sql, "get_task", get_task)
        monkeypatch.setattr(orchestrator.tournament_sql, "get_training_status_for_task_and_hotkeys", boss_rows)

    def add(
        self,
        role: SeedRole = SeedRole.TEXT_R2,
        status: SeedStatus = SeedStatus.TRAINING,
        boss: str | None = TrainingStatus.PENDING.value,
        tournament_id: str = TID,
    ) -> TournamentSeedTask:
        seed = TournamentSeedTask(tournament_id=tournament_id, role=role, task_id=uuid.uuid4(), status=status)
        self.rows.setdefault(tournament_id, []).append(seed)
        if boss is not None:
            self.boss_status[str(seed.task_id)] = boss
        return seed


@pytest.fixture
def seeds(monkeypatch) -> Seeds:
    return Seeds(monkeypatch)


@pytest.fixture
def scheduling(monkeypatch) -> Scheduling:
    return Scheduling(monkeypatch)


class TestASeedInTraining:
    @pytest.mark.parametrize("boss", [None, TrainingStatus.PENDING.value, TrainingStatus.TRAINING.value])
    async def test_holds_while_the_boss_row_is_not_terminal(self, seeds, boss):
        seed = seeds.add(boss=boss)

        assert await orchestrator._text_seeds_in_training(TID, CONFIG) == [seed]

    @pytest.mark.parametrize("role", sorted(orchestrator.TEXT_SEED_ROLES, key=lambda role: role.value))
    async def test_every_text_role_holds(self, seeds, role):
        seed = seeds.add(role=role)

        assert await orchestrator._text_seeds_in_training(TID, CONFIG) == [seed]

    @pytest.mark.parametrize("boss", [TrainingStatus.SUCCESS.value, TrainingStatus.FAILURE.value])
    async def test_releases_once_the_boss_row_is_terminal(self, seeds, boss):
        seeds.add(boss=boss)

        assert await orchestrator._text_seeds_in_training(TID, CONFIG) == []

    @pytest.mark.parametrize("status", [SeedStatus.TRAINED, SeedStatus.READY, SeedStatus.REPLACED, SeedStatus.FAILED])
    async def test_releases_once_the_seed_left_training(self, seeds, status):
        seeds.add(status=status)

        assert await orchestrator._text_seeds_in_training(TID, CONFIG) == []

    @pytest.mark.parametrize("task_status", [TaskStatus.FAILURE, TaskStatus.PREP_TASK_FAILURE, TaskStatus.SUCCESS, None])
    async def test_releases_once_its_task_ended(self, seeds, task_status):
        seed = seeds.add()
        seeds.task_status[str(seed.task_id)] = task_status

        assert await orchestrator._text_seeds_in_training(TID, CONFIG) == []

    @pytest.mark.parametrize("role", [SeedRole.INPUT_SMALL, SeedRole.QUANT_R1, SeedRole.BOSS_R2_FREEFORM])
    async def test_a_seed_of_another_kind_never_holds(self, seeds, role):
        seeds.add(role=role)

        assert await orchestrator._text_seeds_in_training(TID, CONFIG) == []


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

    async def test_miner_rows_are_dispatched_once_the_seeds_are_trained(self, scheduling, seeds):
        seed = seeds.add()
        seeds.boss_status[str(seed.task_id)] = TrainingStatus.SUCCESS.value
        seeds.add(role=SeedRole.TEXT_R3, status=SeedStatus.FAILED)
        miner = self._row_of(TID)

        all_held = await orchestrator.schedule_tasks_for_training([miner], CONFIG)

        assert scheduling.built == [miner.task.task_id]
        assert not all_held

    async def test_a_held_row_is_not_marked_or_charged_an_attempt(self, scheduling, seeds):
        seeds.add()
        miner = self._row_of(TID)
        miner.n_training_attempts = 1

        await orchestrator.schedule_tasks_for_training([miner], CONFIG)

        orchestrator.start_training_task.assert_not_awaited()
        orchestrator.tournament_sql.update_tournament_task_training_status.assert_not_awaited()
        assert miner.n_training_attempts == 1
