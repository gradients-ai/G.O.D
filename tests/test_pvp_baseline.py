"""Tests for the in-harness MCTS baseline (LLMBot vs pyspiel MCTSBot).

A scripted ChatFn plays the model side (first legal action each turn) against a
real low-simulation MCTSBot, so a full baseline run is exercised without SGLang.
"""

import re

import pytest

from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatResult
from core.models.pvp_models import EnvironmentName
from core.models.pvp_models import ToolCall


try:
    import pyspiel  # noqa: F401

    from core.pvp.baseline import MctsBaselineResult
    from core.pvp.baseline import run_mcts_baseline

    HAS_PYSPIEL = True
except ImportError:
    HAS_PYSPIEL = False

needs_pyspiel = pytest.mark.skipif(not HAS_PYSPIEL, reason="pyspiel not installed")


def _first_legal_chat(config, messages, tools=None) -> ChatResult:
    action_id = 0
    for tool in tools or []:
        if tool.function.name == "game_action":
            ids = re.findall(r"\d+", tool.function.description)
            action_id = int(ids[0]) if ids else 0
    return ChatResult(tool_calls=[ToolCall(id="c", name="game_action", arguments={"action_id": action_id})])


def _config() -> ChatCompletionConfig:
    return ChatCompletionConfig(inference_model="test", base_url="http://localhost/v1")


class TestMctsBaselineResult:
    def test_mean_score_weights_draws_half(self):
        r = MctsBaselineResult(wins=2, draws=2, losses=0, num_games=4)
        assert r.mean_score == 0.75  # (2 + 0.5*2)/4

    def test_mean_score_zero_games(self):
        assert MctsBaselineResult().mean_score == 0.0


@needs_pyspiel
class TestRunMctsBaseline:
    def test_baseline_completes_and_tallies(self):
        result = run_mcts_baseline(
            EnvironmentName.LEDUC_POKER,
            _first_legal_chat,
            _config(),
            num_games=2,
            mcts_simulations=8,
            base_seed=1,
        )
        assert result.num_games == 2
        assert result.wins + result.draws + result.losses == 2
        assert 0.0 <= result.mean_score <= 1.0

    def test_runs_for_liars_dice_too(self):
        result = run_mcts_baseline(
            EnvironmentName.LIARS_DICE,
            _first_legal_chat,
            _config(),
            num_games=1,
            mcts_simulations=8,
            base_seed=3,
        )
        assert result.num_games == 1

    def test_exhausted_time_budget_returns_partial_tally(self):
        result = run_mcts_baseline(
            EnvironmentName.LEDUC_POKER,
            _first_legal_chat,
            _config(),
            num_games=5,
            mcts_simulations=8,
            base_seed=1,
            time_budget_seconds=0.0,
        )
        assert result.num_games == 0
        assert result.mean_score == 0.0

    def test_generous_time_budget_plays_all_games(self):
        result = run_mcts_baseline(
            EnvironmentName.LEDUC_POKER,
            _first_legal_chat,
            _config(),
            num_games=2,
            mcts_simulations=8,
            base_seed=1,
            time_budget_seconds=600.0,
        )
        assert result.num_games == 2

    def test_forfeiting_model_skips_reflection(self):
        """A bot that commits no move forfeits; reflection (a memory-tools-only
        call) must then be skipped, mirroring game_runner."""
        calls = {"turn": 0, "reflect": 0}

        def no_move_chat(config, messages, tools=None) -> ChatResult:
            tool_names = {t.function.name for t in tools or []}
            calls["turn" if "game_action" in tool_names else "reflect"] += 1
            return ChatResult()  # never calls game_action -> forfeit

        result = run_mcts_baseline(
            EnvironmentName.LEDUC_POKER,
            no_move_chat,
            _config(),
            num_games=2,
            mcts_simulations=8,
            base_seed=1,
        )
        assert result.losses == 2  # forfeits score as losses
        assert calls["turn"] > 0
        assert calls["reflect"] == 0
