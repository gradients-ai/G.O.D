"""Tests for the single-player OpenSpiel 2048 pathway."""

import re

import pytest

from core.constants import ENVIRONMENT_CONFIGS
from core.constants import VALIDATOR_DOCKER_IMAGE_ENV
from core.constants import EnvironmentName
from core.constants import EvalType
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatResult
from core.models.pvp_models import ToolCall
from core.models.pvp_models import TwentyFortyEightParams


try:
    import pyspiel

    from core.pvp.agents import TwentyFortyEightAgent
    from core.pvp.individual import run_individual_open_spiel_eval

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


def test_2048_environment_uses_individual_env_image():
    cfg = ENVIRONMENT_CONFIGS[EnvironmentName.TWENTY_FORTY_EIGHT]

    assert cfg.eval_type == EvalType.INDIVIDUAL
    assert cfg.env_image == VALIDATOR_DOCKER_IMAGE_ENV
    assert cfg.tournament_eval_image == VALIDATOR_DOCKER_IMAGE_ENV
    assert cfg.tournament_eval_command == ["python", "-m", "validator.evaluation.individual"]


@needs_pyspiel
class TestTwentyFortyEightAgent:
    def test_matches_openspiel_registration(self):
        game = pyspiel.load_game("2048")
        game_type = game.get_type()

        assert game.num_players() == 1
        assert game_type.short_name == "2048"
        assert game_type.chance_mode == pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC
        assert game_type.utility == pyspiel.GameType.Utility.GENERAL_SUM

    def test_params_render_to_pyspiel(self):
        params = TwentyFortyEightAgent().generate_params(0)

        assert params == TwentyFortyEightParams(max_tile=2048)
        assert params.to_pyspiel() == {"max_tile": 2048}

    def test_action_strings_are_slide_directions(self):
        state = pyspiel.load_game("2048").new_initial_state()
        for _ in range(2):
            state.apply_action(state.chance_outcomes()[0][0])

        actions = {action: state.action_to_string(0, action) for action in state.legal_actions()}
        assert set(actions.values()) <= {"Up", "Right", "Down", "Left"}

    def test_individual_runner_returns_average_raw_score(self, monkeypatch):
        monkeypatch.setattr(
            TwentyFortyEightAgent,
            "generate_params",
            lambda self, config_id: TwentyFortyEightParams(max_tile=8),
        )

        result = run_individual_open_spiel_eval(
            env_name=EnvironmentName.TWENTY_FORTY_EIGHT,
            chat_fn=_first_legal_chat,
            config=_config(),
            num_games=1,
            base_seed=1,
        )

        assert result.num_games == 1
        assert result.mean_score >= 0.0

    def test_individual_runner_exposes_only_game_action_tool(self, monkeypatch):
        monkeypatch.setattr(
            TwentyFortyEightAgent,
            "generate_params",
            lambda self, config_id: TwentyFortyEightParams(max_tile=8),
        )
        tool_names_by_call = []
        system_prompts = []

        def recording_chat(config, messages, tools=None) -> ChatResult:
            tool_names_by_call.append([tool.function.name for tool in tools or []])
            system_prompts.append(messages[0].content)
            return _first_legal_chat(config, messages, tools)

        result = run_individual_open_spiel_eval(
            env_name=EnvironmentName.TWENTY_FORTY_EIGHT,
            chat_fn=recording_chat,
            config=_config(),
            num_games=1,
            base_seed=1,
        )

        assert result.num_games == 1
        assert tool_names_by_call
        assert all(tool_names == ["game_action"] for tool_names in tool_names_by_call)
        assert all("memory" not in (prompt or "").lower() for prompt in system_prompts)

    def test_individual_runner_truncates_episode_at_action_cap(self):
        calls = 0

        def counting_chat(config, messages, tools=None) -> ChatResult:
            nonlocal calls
            calls += 1
            return _first_legal_chat(config, messages, tools)

        result = run_individual_open_spiel_eval(
            env_name=EnvironmentName.TWENTY_FORTY_EIGHT,
            chat_fn=counting_chat,
            config=_config(),
            num_games=1,
            base_seed=1,
            max_player_actions_per_episode=1,
        )

        assert result.num_games == 1
        assert calls == 1
        assert result.mean_score >= 0.0

    def test_individual_runner_honors_total_time_budget(self):
        result = run_individual_open_spiel_eval(
            env_name=EnvironmentName.TWENTY_FORTY_EIGHT,
            chat_fn=_first_legal_chat,
            config=_config(),
            num_games=3,
            base_seed=1,
            time_budget_seconds=0.0,
        )

        assert result.num_games == 0
