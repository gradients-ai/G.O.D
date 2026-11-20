"""A seed whose prep cannot fetch its data is retried in place: no replacement attempt, no `failed` role. A deterministic
prep rejection still replaces it as before. The real _prep_task drives the seed loop's fakes (test_text_seed_rounds' world).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.lifecycle.tasks as lifecycle_tasks
from core.models.task_models import TaskStatus
from tests.validator.tournament.test_seed_tasks import TID
from tests.validator.tournament.test_seed_tasks import env  # noqa: F401  (fixture)
from tests.validator.tournament.test_text_seed_rounds import TextWorld
from validator.db.sql import tasks as task_sql
from validator.lifecycle import constants as lifecycle_cst
from validator.tasks.datasets.fetch import DataFetchError
from validator.tasks.datasets.preparation import NearDuplicateDatasetError
from validator.tournament import seed_tasks
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus


@pytest.fixture
def world(env, monkeypatch) -> TextWorld:  # noqa: F811
    world = TextWorld(env, monkeypatch)
    world.prep = AsyncMock()
    world.notify = AsyncMock()

    async def update_task(task, _db):
        world.env.tasks[task.task_id] = task

    monkeypatch.setattr(task_sql, "update_task", update_task)
    monkeypatch.setattr(lifecycle_tasks, "get_task_config", lambda task: SimpleNamespace(task_prep_function=world.prep))
    monkeypatch.setattr(lifecycle_tasks, "notify_evaluation_exception", world.notify)
    monkeypatch.setattr(lifecycle_tasks, "_PREP_FETCH_BACKOFF", {})
    return world


async def _prep_then_cycle(world, role: SeedRole) -> None:
    seed = next(row for row in world.env.store.rows if row.role == role and row.status != SeedStatus.REPLACED)
    await lifecycle_tasks._prep_task(world.env.tasks[seed.task_id], world.env.config)
    await seed_tasks.process_tournament_seed_tasks(TID, world.env.config)


@pytest.mark.parametrize("role", [SeedRole.TEXT_CONTINUOUS, SeedRole.TEXT_R1])
async def test_a_fetch_failure_keeps_the_same_seed_and_consumes_no_attempt(world, role):
    seeds = await world.create_seeds()
    world.prep.side_effect = DataFetchError("could not download the chunk: 404 NoSuchKey")

    for _ in range(lifecycle_cst.PREP_FETCH_FAILURES_BEFORE_ALERT + 3):
        await _prep_then_cycle(world, role)

    rows = [row for row in world.env.store.rows if row.role == role]
    assert [(row.attempt, row.task_id, row.status) for row in rows] == [(1, seeds[role].task_id, SeedStatus.TRAINING)]
    assert world.env.tasks[seeds[role].task_id].status == TaskStatus.PENDING
    world.env.alerts.assert_not_awaited()  # no seed alert: nothing was replaced or failed
    world.notify.assert_awaited_once()  # the one prep-fetch alert


async def test_a_deterministic_rejection_of_the_continuous_seed_is_still_a_charged_replacement(world):
    seeds = await world.create_seeds()
    world.prep.side_effect = NearDuplicateDatasetError("near-duplicate rate 0.40 >= 0.15")

    await _prep_then_cycle(world, SeedRole.TEXT_CONTINUOUS)

    rows = sorted((row for row in world.env.store.rows if row.role == SeedRole.TEXT_CONTINUOUS), key=lambda r: r.attempt)
    assert [(row.attempt, row.status) for row in rows] == [(1, SeedStatus.REPLACED), (2, SeedStatus.TRAINING)]
    assert rows[0].task_id == seeds[SeedRole.TEXT_CONTINUOUS].task_id and rows[0].prep_rejected is False
    world.notify.assert_not_awaited()


async def test_a_fetch_outage_that_ends_lets_the_same_seed_prep(world):
    seeds = await world.create_seeds()
    world.prep.side_effect = DataFetchError("timed out")
    await _prep_then_cycle(world, SeedRole.TEXT_CONTINUOUS)
    world.prep.side_effect = lambda task, *args: task

    await _prep_then_cycle(world, SeedRole.TEXT_CONTINUOUS)

    task = world.env.tasks[seeds[SeedRole.TEXT_CONTINUOUS].task_id]
    assert task.status in (TaskStatus.LOOKING_FOR_NODES, TaskStatus.AWAITING_MODEL_PREP)
    assert len([row for row in world.env.store.rows if row.role == SeedRole.TEXT_CONTINUOUS]) == 1
