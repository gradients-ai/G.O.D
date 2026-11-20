"""A failed-but-scored quant entry has an infinite loss. The boss-round performance difference reports no number for it."""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.tournament.performance_calculator as performance_calculator
from core.models.task_models import TaskType
from core.models.tournament_models import TournamentType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.tournament.models import TournamentData


BOSS = EMISSION_BURN_HOTKEY
CHALLENGER = "challenger_hotkey"
FIXED = TaskType.QUANTFIXEDTASK


def _result(hotkey: str, loss: float | None) -> MinerResultsText:
    return MinerResultsText(hotkey=hotkey, test_loss=1.0, synth_loss=1.0, is_finetune=True, task_type=FIXED, adjusted_loss=loss)


async def _boss_round_diffs(monkeypatch, boss_loss: float, challenger_loss: float, challenger_won: bool = True):
    for name, value in (
        ("get_tournament", TournamentData(tournament_id="t", tournament_type=TournamentType.QUANT)),
        ("get_final_round_id", "round"),
        ("get_tournament_tasks", [SimpleNamespace(task_id="task-1")]),
        ("get_task", SimpleNamespace(task_type=FIXED, training_start_point=None)),
        ("get_task_results_for_ranking", [SimpleNamespace()]),
    ):
        monkeypatch.setattr(performance_calculator, name, AsyncMock(return_value=value))
    monkeypatch.setattr(
        performance_calculator,
        "calculate_miner_ranking_and_scores",
        lambda _r: [_result(BOSS, boss_loss), _result(CHALLENGER, challenger_loss)],
    )
    analytics = SimpleNamespace(challenger_won=challenger_won, paired_comparison=None, reason="x", within_allowed_loss=None)
    monkeypatch.setattr(performance_calculator, "_quant_challenger_won_task", AsyncMock(return_value=analytics))
    return await performance_calculator.calculate_boss_round_performance_differences("t", None)


async def test_boss_inf_challenger_finite_has_no_performance_difference(monkeypatch):
    (diff,) = await _boss_round_diffs(monkeypatch, math.inf, 1.0, challenger_won=True)

    assert diff.performance_difference is None
    assert diff.challenger_won is True  # who won is decided independently of the number


async def test_boss_finite_challenger_inf_has_no_performance_difference(monkeypatch):
    (diff,) = await _boss_round_diffs(monkeypatch, 1.0, math.inf, challenger_won=False)

    assert diff.performance_difference is None
    assert diff.challenger_won is False


async def test_finite_scores_keep_their_performance_difference(monkeypatch):
    (diff,) = await _boss_round_diffs(monkeypatch, 1.2, 1.0)

    assert diff.performance_difference == pytest.approx(0.2)
