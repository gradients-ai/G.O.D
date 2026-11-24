"""Boss-round analytics skip the tasks an early-stopped boss round never ran."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.tournament.performance_calculator as pc
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsImage
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus


CHALLENGER = "challenger"


@pytest.fixture
def world(monkeypatch):
    state = SimpleNamespace(status=TaskStatus.FAILURE.value, training={CHALLENGER: TrainingStatus.FAILURE})
    tournament = TournamentData(tournament_id="t", tournament_type=TournamentType.IMAGE, winner_hotkey=EMISSION_BURN_HOTKEY)
    task = SimpleNamespace(task_id="task")
    monkeypatch.setattr(pc, "get_tournament", AsyncMock(return_value=tournament))
    monkeypatch.setattr(pc, "get_final_round_id", AsyncMock(return_value="r4"))
    monkeypatch.setattr(pc, "get_tournament_tasks", AsyncMock(return_value=[task]))
    monkeypatch.setattr(
        pc, "get_task", AsyncMock(side_effect=lambda *_: SimpleNamespace(task_type=TaskType.IMAGETASK, status=state.status))
    )
    monkeypatch.setattr(
        pc,
        "get_task_results_for_ranking",
        AsyncMock(return_value=[MinerResultsImage(hotkey=CHALLENGER, test_loss=1.0, synth_loss=1.0, is_finetune=True)]),
    )
    monkeypatch.setattr(pc, "get_training_status_for_task_and_hotkeys", AsyncMock(side_effect=lambda *_: state.training))
    return state


async def test_an_unfinished_task_with_no_finished_challenger_training_is_not_reported_as_a_challenger_win(world):
    assert await pc.calculate_boss_round_performance_differences("t", None) == []


async def test_a_task_the_boss_failed_and_the_challenger_trained_is_the_challengers(world):
    world.training = {CHALLENGER: TrainingStatus.SUCCESS}

    differences = await pc.calculate_boss_round_performance_differences("t", None)

    assert [difference.challenger_won for difference in differences] == [True]
