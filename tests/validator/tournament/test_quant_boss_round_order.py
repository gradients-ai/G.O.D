"""The compression boss trains in round order, with the field.

A compression seed is created and prepped ahead, but no training row exists for it until its round attaches it: the boss's
row is then created together with the miners'. Rounds 1-3 attach their seed when the round is created, the boss round its
four when it is created, and a boss round the early stop skips leaves its four seeds untrained and retired. A text seed is
not held: the text boss trains it upfront.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.models.task_models import TaskStatus
from tests.validator.tournament.test_quant_text_linkage import CHALLENGER
from tests.validator.tournament.test_quant_text_linkage import QUANT_TID
from tests.validator.tournament.test_quant_text_linkage import QuantWorld
from tests.validator.tournament.test_seed_tasks import env  # noqa: F401  (fixture)
from validator.db.sql import tournament_seeds as seeds_sql
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament import orchestrator
from validator.tournament import seed_tasks
from validator.tournament import task_creator
from validator.tournament.models import BOSS_ROUND_SEED_ROLES
from validator.tournament.models import QUANT_SLOT_SPECS
from validator.tournament.models import QuantSourceSlot
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TaskTrainingAssignment


BOSS = EMISSION_BURN_HOTKEY
ROUND_SEED_ROLES = {n: QUANT_SLOT_SPECS[QuantSourceSlot(f"r{n}")].quant_seed_role for n in (1, 2, 3)}
SEVEN_ROLES = tuple(seed_tasks.SEED_CREATION_ORDER)


class RoundOrderWorld(QuantWorld):
    """QuantWorld plus the orchestrator's move to TRAINING, with the training rows it writes recorded."""

    def __init__(self, env, monkeypatch) -> None:  # noqa: F811
        super().__init__(env, monkeypatch)
        self.rows: list[list[str]] = []  # the hotkeys of each add_tournament_task_hotkey_pairs_for_training call, per task
        self.rows_by_task: dict[str, list[str]] = {}

        async def add_rows(assignments: list[TaskTrainingAssignment], _db):
            for assignment in assignments:
                self.rows_by_task.setdefault(assignment.task_id, []).append(assignment.hotkey)
            self.rows.append([assignment.hotkey for assignment in assignments])

        monkeypatch.setattr(orchestrator.tournament_sql, "add_tournament_task_hotkey_pairs_for_training", add_rows)
        monkeypatch.setattr(orchestrator.tournament_sql, "get_tournament_id_by_task_id", AsyncMock(return_value=QUANT_TID))
        monkeypatch.setattr(orchestrator.tournament_sql, "is_task_of_decided_final_round", AsyncMock(return_value=False))

    async def _existing_in_round(self, round_id, config, group_id=None, pair_id=None):
        registered = [task_id for task_id, _, registered_in, _ in self.registrations if registered_in == round_id]
        return [SimpleNamespace(task_id=task_id) for task_id in registered]

    def prepped(self, role: SeedRole):
        """The role's seed task as the release and prep leave it: LOOKING_FOR_NODES."""
        task = self.quant_task(role)
        task.status = TaskStatus.LOOKING_FOR_NODES
        task.created_at = datetime(2026, 1, 1)
        return task

    async def dispatch(self, roles) -> None:
        """One orchestrator pass over the given roles' tasks (the compression bucket of tournament tasks)."""
        await orchestrator._process_tasks_for_training([self.prepped(role) for role in roles], self.config, priority=2)

    def trained_rows(self, role: SeedRole) -> list[str]:
        return self.rows_by_task.get(str(self.quant_seed(role).task_id), [])

    def join(self, role: SeedRole, hotkey: str = CHALLENGER) -> None:
        self.env.assigned.setdefault(str(self.quant_seed(role).task_id), []).append(hotkey)


@pytest.fixture
def world(env, monkeypatch):  # noqa: F811
    return RoundOrderWorld(env, monkeypatch)


class TestNoRowBeforeTheRound:
    async def test_every_compression_seed_is_prepped_but_held_without_a_training_row(self, world):
        await world.create_seeds()

        await world.dispatch(SEVEN_ROLES)

        assert world.rows == []
        for role in SEVEN_ROLES:
            assert world.quant_task(role).status == TaskStatus.LOOKING_FOR_NODES
            assert world.env.assigned[str(world.quant_seed(role).task_id)] == [BOSS]  # assigned, not trained

    async def test_the_hold_is_the_attachment(self, world):
        await world.create_seeds()
        held = world.quant_seed(SeedRole.QUANT_R1)

        assert await seeds_sql.is_unattached_quant_seed(held.task_id, None)
        await world.create_round_task(1)
        assert not await seeds_sql.is_unattached_quant_seed(held.task_id, None)

    async def test_a_text_seed_is_not_held(self, world):
        text_task = world.text_task(SeedRole.TEXT_R1)
        text_task.status = TaskStatus.LOOKING_FOR_NODES
        world.env.assigned[str(text_task.task_id)] = [BOSS]

        await orchestrator._process_tasks_for_training([text_task], world.config, priority=2)

        assert world.rows_by_task == {str(text_task.task_id): [BOSS]}
        assert text_task.status == TaskStatus.TRAINING


class TestRoundsOneToThree:
    @pytest.mark.parametrize("round_number", [1, 2, 3])
    async def test_the_boss_row_is_created_with_the_miners_when_its_round_is_created(self, world, round_number):
        role = ROUND_SEED_ROLES[round_number]
        await world.create_seeds()
        await world.dispatch([role])
        assert world.trained_rows(role) == []

        await world.create_round_task(round_number)
        world.join(role)
        await world.dispatch([role])

        assert world.rows == [[BOSS, CHALLENGER]]  # one move to TRAINING, both rows together
        assert world.quant_task(role).status == TaskStatus.TRAINING

    async def test_dispatch_follows_the_rounds(self, world):
        await world.create_seeds()

        for round_number in (1, 2, 3):
            await world.create_round_task(round_number)
            world.join(ROUND_SEED_ROLES[round_number])
            await world.dispatch(SEVEN_ROLES)  # every seed the orchestrator sees: only the attached ones train
            trained = [role for role in SEVEN_ROLES if world.trained_rows(role)]
            assert trained == [ROUND_SEED_ROLES[n] for n in range(1, round_number + 1)]
        assert not any(world.trained_rows(role) for role in BOSS_ROUND_SEED_ROLES)


class TestBossRound:
    async def test_the_four_entries_train_only_when_the_boss_round_is_created(self, world):
        await world.create_seeds()
        boss_roles = [role for role in SEVEN_ROLES if role in BOSS_ROUND_SEED_ROLES]
        await world.dispatch(boss_roles)
        assert world.rows == []

        tasks = await task_creator._create_quant_boss_round_tasks(QUANT_TID, f"{QUANT_TID}_round_004", world.config)
        for role in boss_roles:
            world.join(role)
        await world.dispatch(boss_roles)

        assert sorted(str(task.task_id) for task in tasks) == sorted(str(world.quant_seed(r).task_id) for r in boss_roles)
        for role in boss_roles:
            assert world.trained_rows(role) == [BOSS, CHALLENGER]

    async def test_an_early_stop_leaves_them_untrained_and_retires_them(self, world, monkeypatch):
        for round_number in (1, 2, 3):
            await world.create_round_task(round_number)
        boss_roles = [role for role in SEVEN_ROLES if role in BOSS_ROUND_SEED_ROLES]
        await world.dispatch(boss_roles)
        assert world.rows == []
        failed_tasks: list = []

        async def update_task(task, _db):
            failed_tasks.append((task.task_id, task.status))

        monkeypatch.setattr(seed_tasks.task_sql, "update_task", update_task)

        retired = await seed_tasks.retire_unattached_quant_seeds(QUANT_TID, world.config)

        assert sorted(retired) == sorted(boss_roles)
        assert world.rows == []
        assert sorted(failed_tasks) == sorted((world.quant_task(r).task_id, TaskStatus.FAILURE) for r in boss_roles)
        latest = {row.role: row for row in await seeds_sql.get_latest_seed_tasks(QUANT_TID, None)}
        assert all(latest[role].status == SeedStatus.FAILED for role in boss_roles)
