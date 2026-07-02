"""The pre-boss knockout (2 competitors left) always plays on quasar.

Guards the routing in _create_probability_based_text_tasks: the pre-boss round is detected by
COMPETITOR count (a small-tournament round 1 also creates a single task, so task count is not a
valid key), and its task is a standard instruct task with only the model forced to the quasar
seed — no KL, no augmentation, normal dataset pull.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from core.models.task_models import TaskType
from validator.tournament import constants as t_cst
from validator.tournament import task_creator
from validator.tournament.models import KnockoutRound


def _knockout(pairs: list[tuple[str, str]]) -> KnockoutRound:
    return KnockoutRound(round_id="tourn_round_003", round_number=3, pairs=pairs)


def _patch_seams(monkeypatch):
    """Stub the pool generators, DB lookups and registration; return the two creation mocks."""
    for name in ("_get_text_models", "_get_instruct_text_datasets", "_get_dpo_datasets"):
        monkeypatch.setattr(task_creator, name, lambda *a, **k: MagicMock())
    monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", AsyncMock())

    quasar_task = SimpleNamespace(task_id="quasar-task", task_type=TaskType.INSTRUCTTEXTTASK)
    instruct_mock = AsyncMock(return_value=quasar_task)
    monkeypatch.setattr(task_creator, "create_synthetic_instruct_text_task", instruct_mock)

    probability_task = SimpleNamespace(task_id="prob-task", task_type=TaskType.DPOTASK)
    probability_mock = AsyncMock(return_value=probability_task)
    monkeypatch.setattr(task_creator, "_create_single_probability_task", probability_mock)
    return instruct_mock, probability_mock


async def test_two_competitors_forces_the_quasar_task(monkeypatch):
    instruct_mock, probability_mock = _patch_seams(monkeypatch)

    tasks = await task_creator._create_probability_based_text_tasks(
        _knockout([("miner-a", "miner-b")]), "tourn", MagicMock()
    )

    probability_mock.assert_not_awaited()
    assert [t.task_id for t in tasks] == ["quasar-task"]
    args, kwargs = instruct_mock.call_args
    assert args[1] is None  # no model pool: the model is forced
    assert kwargs["model_id_override"] == t_cst.PRE_BOSS_QUASAR_MODEL
    assert kwargs["enable_kl"] is False
    assert kwargs["allow_augmentation"] is False
    assert kwargs["allow_yarn"] is False


async def test_more_than_two_competitors_keeps_probability_routing(monkeypatch):
    instruct_mock, probability_mock = _patch_seams(monkeypatch)

    tasks = await task_creator._create_probability_based_text_tasks(
        _knockout([("miner-a", "miner-b"), ("miner-c", "miner-d")]), "tourn", MagicMock()
    )

    instruct_mock.assert_not_awaited()
    assert probability_mock.await_count == 2
    assert len(tasks) == 2
