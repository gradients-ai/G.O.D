"""Tests for the Goofspiel PvP agent."""

import importlib.util
import random

import pytest


try:
    if importlib.util.find_spec("pyspiel") is None:
        raise ImportError

    import numpy as np
    import pyspiel
    from open_spiel.python.algorithms import evaluate_bots

    from core.constants.environments import ENVIRONMENT_CONFIGS
    from core.constants.environments import EnvironmentName
    from core.constants.environments import EvalType
    from core.pvp.agents import GoofspielAgent
    from core.pvp.game_eval import _AGENT_REGISTRY
    from core.pvp.game_eval import config_id_for_seed
    from validator.evaluation.pvp.models import GameInstance
    from validator.evaluation.pvp.models import GoofspielParams

    HAS_PYSPIEL = True
except ImportError:
    HAS_PYSPIEL = False

needs_pyspiel = pytest.mark.skipif(not HAS_PYSPIEL, reason="pyspiel not installed")


@needs_pyspiel
class TestGoofspielGameConstruction:
    def test_load_game_is_sequential(self):
        agent = GoofspielAgent()
        game = agent.load_game(agent.generate_params(0))
        assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL

    def test_load_game_is_zero_sum_win_loss(self):
        agent = GoofspielAgent()
        game = agent.load_game(agent.generate_params(0))
        assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
        assert game.min_utility() == -1.0
        assert game.max_utility() == 1.0

    def test_game_name_and_rules_key(self):
        agent = GoofspielAgent()
        assert agent.game_name == "goofspiel"
        assert agent.rules_key == "goofspiel_rules"


@needs_pyspiel
class TestGoofspielParams:
    def test_params_have_imp_info_and_win_loss(self):
        params = GoofspielAgent().generate_params(0)
        assert params.imp_info is True
        assert params.returns_type == "win_loss"
        assert params.players == 2

    def test_num_cards_varies_with_config_id(self):
        agent = GoofspielAgent()
        sizes = {agent.generate_params(config_id).num_cards for config_id in range(50)}
        assert len(sizes) > 1

    def test_all_num_cards_options_load(self):
        agent = GoofspielAgent()
        for config_id in range(50):
            game = agent.load_game(agent.generate_params(config_id))
            assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL

    def test_config_id_for_seed_drives_variety(self):
        agent = GoofspielAgent()
        env_config = ENVIRONMENT_CONFIGS[EnvironmentName.GOOFSPIEL]
        sizes = {agent.generate_params(config_id_for_seed(seed, env_config)).num_cards for seed in range(200)}
        assert len(sizes) > 1


@needs_pyspiel
class TestGoofspielStateFormatting:
    def _fresh_state(self, agent, seed=1):
        game = agent.load_game(agent.generate_params(0))
        state = game.new_initial_state()
        rng = random.Random(seed)
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(rng.choices(actions, weights=probs)[0])
        return state

    def test_format_state_hides_opponent_hand(self):
        agent = GoofspielAgent()
        rendered = agent.format_state(self._fresh_state(agent), player_id=0)
        assert "P1 hand" not in rendered
        assert "P0 hand" in rendered or "your hand" in rendered.lower()

    def test_format_state_shows_point_card(self):
        agent = GoofspielAgent()
        rendered = agent.format_state(self._fresh_state(agent), player_id=0)
        assert "point card" in rendered.lower()


@needs_pyspiel
class TestGoofspielPlaythrough:
    def test_random_selfplay_terminates_win_loss(self):
        agent = GoofspielAgent()
        game = agent.load_game(agent.generate_params(0))
        state = game.new_initial_state()
        rng = random.Random(42)
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                state.apply_action(rng.choices(actions, weights=probs)[0])
            else:
                player = state.current_player()
                state.apply_action(rng.choice(state.legal_actions(player)))
        returns = state.returns()
        assert set(returns) <= {-1.0, 0.0, 1.0}
        assert returns[0] == -returns[1]

    def test_plays_through_evaluate_bots(self):
        agent = GoofspielAgent()
        game = agent.load_game(agent.generate_params(0))
        bots = [pyspiel.make_uniform_random_bot(player, 123 + player) for player in range(2)]
        returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random.RandomState(7))
        assert len(returns) == 2
        assert set(returns) <= {-1.0, 0.0, 1.0}


@needs_pyspiel
class TestGoofspielRegistration:
    def test_registered_as_pvp_env(self):
        assert EnvironmentName.GOOFSPIEL in _AGENT_REGISTRY
        assert _AGENT_REGISTRY[EnvironmentName.GOOFSPIEL] is GoofspielAgent
        assert ENVIRONMENT_CONFIGS[EnvironmentName.GOOFSPIEL].eval_type == EvalType.PVP

    def test_system_prompt_includes_rules(self):
        assert "goofspiel" in GoofspielAgent().generate_system_prompt().lower()


@needs_pyspiel
class TestGameParamsRoundTrip:
    def test_goofspiel_params_round_trip_preserves_subclass(self):
        inst = GameInstance(
            game_name="goofspiel",
            game_params=GoofspielParams(num_cards=13),
            model_a_player_id=0,
            seed=1,
            is_zero_sum=True,
            min_utility=-1.0,
            max_utility=1.0,
        )
        restored = GameInstance.model_validate_json(inst.model_dump_json())
        assert isinstance(restored.game_params, GoofspielParams)
        assert restored.game_params.num_cards == 13
        assert restored.game_params.to_pyspiel() == {
            "players": 2,
            "num_cards": 13,
            "imp_info": True,
            "points_order": "random",
            "returns_type": "win_loss",
        }
