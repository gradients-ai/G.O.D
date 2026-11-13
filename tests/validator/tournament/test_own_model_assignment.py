"""Node assignment for an own-model quant task (R2/R3): each player starts from its own full-model R1 repo, else
it is persisted as a scored fail (no GPU spent). The step is idempotent."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from core.constants.environments import TrainingStartPoint
from core.models.task_models import TaskType
from validator.tournament import constants as t_cst
from validator.tournament import tournament_manager as manager
from validator.tournament.models import Group
from validator.tournament.models import GroupRound
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType


TOURNAMENT_ID = "tourn"
ROUND_ID = "tourn_round_002"
TASK_ID = "t2-task"
GOOD = "hk_good"
ADAPTER = "hk_adapter"


class AssignmentWorld:
    """In-memory task_nodes for one task: assigned hotkeys, starting models and quant failures."""

    def __init__(self, repos: dict[str, str | None]) -> None:
        self.repos = repos  # hotkey -> own R1 repo, None for no usable model
        self.assigned: set[str] = set()
        self.starting: dict[str, str] = {}
        self.failures: dict[str, str] = {}
        self.psql_db = MagicMock()

    async def get_nodes_assigned_to_task(self, task_id, psql_db):
        return [SimpleNamespace(hotkey=hotkey) for hotkey in self.assigned]

    async def assign_node_to_task(self, task_id, node, psql_db):
        self.assigned.add(node.hotkey)

    async def get_starting_model_repo(self, task_id, hotkey, psql_db):
        return self.starting.get(hotkey)

    async def set_starting_model_repo(self, task_id, hotkey, repo, psql_db):
        self.starting[hotkey] = repo

    async def get_quant_decode_rows(self, task_id, psql_db):
        return {hotkey: SimpleNamespace(quant_failure=reason) for hotkey, reason in self.failures.items()}

    async def set_task_node_quant_failure(self, task_id, hotkey, reason, psql_db):
        self.failures[hotkey] = reason

    async def own_r1_model_repo(self, tournament_id, hotkey, psql_db):
        return self.repos[hotkey]


@pytest.fixture
def world(monkeypatch) -> AssignmentWorld:
    world = AssignmentWorld({GOOD: "gradients-io-tournaments/r1-good", ADAPTER: None})
    tasks = [TournamentTask(tournament_id=TOURNAMENT_ID, round_id=ROUND_ID, task_id=TASK_ID, group_id=f"{ROUND_ID}_group_001")]
    task = SimpleNamespace(
        task_type=TaskType.QUANTFREEFORMTASK, training_start_point=TrainingStartPoint.OWN_R1_MODEL, model_id="org/r1-base"
    )
    tournament = TournamentData(tournament_id=TOURNAMENT_ID, tournament_type=TournamentType.QUANT)
    monkeypatch.setattr(manager, "get_tournament", AsyncMock(return_value=tournament))
    monkeypatch.setattr(manager, "get_tournament_tasks", AsyncMock(return_value=tasks))
    monkeypatch.setattr(manager, "get_node_by_hotkey", AsyncMock(side_effect=lambda hotkey, db: SimpleNamespace(hotkey=hotkey)))
    monkeypatch.setattr(manager.task_sql, "get_task", AsyncMock(return_value=task))
    monkeypatch.setattr(manager.task_sql, "set_expected_repo_name", AsyncMock())
    for name in ("get_nodes_assigned_to_task", "assign_node_to_task", "get_starting_model_repo", "set_starting_model_repo"):
        monkeypatch.setattr(manager.task_sql, name, getattr(world, name))
    for name in ("get_quant_decode_rows", "set_task_node_quant_failure"):
        monkeypatch.setattr(manager.submissions_and_scoring_sql, name, getattr(world, name))
    monkeypatch.setattr(manager, "own_r1_model_repo", AsyncMock(side_effect=world.own_r1_model_repo))
    return world


def _round() -> GroupRound:
    return GroupRound(round_id=ROUND_ID, round_number=2, groups=[Group(member_ids=[GOOD, ADAPTER])])


async def test_full_model_sets_the_starting_repo_and_an_adapter_is_a_scored_fail(world):
    await manager.assign_nodes_to_tournament_tasks(TOURNAMENT_ID, _round(), world.psql_db)

    assert world.assigned == {GOOD, ADAPTER}  # a failed miner still has its task_nodes row
    assert world.starting == {GOOD: "gradients-io-tournaments/r1-good"}
    assert world.failures == {ADAPTER: t_cst.NO_USABLE_R1_MODEL_REASON}
    assert world.starting.get(ADAPTER) is None


async def test_rerun_after_completion_changes_nothing(world):
    await manager.assign_nodes_to_tournament_tasks(TOURNAMENT_ID, _round(), world.psql_db)
    manager.own_r1_model_repo.reset_mock()

    await manager.assign_nodes_to_tournament_tasks(TOURNAMENT_ID, _round(), world.psql_db)

    manager.own_r1_model_repo.assert_not_awaited()
    assert world.starting == {GOOD: "gradients-io-tournaments/r1-good"}
    assert world.failures == {ADAPTER: t_cst.NO_USABLE_R1_MODEL_REASON}


async def test_rerun_completes_a_hotkey_assigned_without_a_model_or_failure(world):
    # A crash between assign_node_to_task and the starting-model write.
    world.assigned = {GOOD, ADAPTER}

    await manager.assign_nodes_to_tournament_tasks(TOURNAMENT_ID, _round(), world.psql_db)

    assert world.starting == {GOOD: "gradients-io-tournaments/r1-good"}
    assert world.failures == {ADAPTER: t_cst.NO_USABLE_R1_MODEL_REASON}


async def test_other_task_kinds_get_no_starting_model(world, monkeypatch):
    plain = SimpleNamespace(
        task_type=TaskType.INSTRUCTTEXTTASK, training_start_point=TrainingStartPoint.DEFAULT, model_id="org/base"
    )
    monkeypatch.setattr(manager.task_sql, "get_task", AsyncMock(return_value=plain))

    await manager.assign_nodes_to_tournament_tasks(TOURNAMENT_ID, _round(), world.psql_db)

    assert world.assigned == {GOOD, ADAPTER}
    assert world.starting == {} and world.failures == {}
    manager.own_r1_model_repo.assert_not_awaited()
