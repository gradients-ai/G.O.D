"""Tests for the Goofspiel PvP agent.

Goofspiel is the only PvP game whose native OpenSpiel dynamics are SIMULTANEOUS,
so the agent must wrap it via convert_to_turn_based to play under the sequential
LLMBot/evaluate_bots harness. These tests pin that contract: the agent yields a
sequential, zero-sum game, varies num_cards per game, hides the opponent's hand
(imp_info), and plays out to a {-1, 0, 1} win_loss return.
"""

import importlib.util
import random

import pytest


try:
    if importlib.util.find_spec("pyspiel") is None:
        raise ImportError

    import numpy as np
    import pyspiel
    from open_spiel.python.algorithms import evaluate_bots

    from core.constants import EnvironmentName
    from core.constants import EvalType
    from core.constants import ENVIRONMENT_CONFIGS
    from core.models.pvp_models import GameInstance
    from core.models.pvp_models import GoofspielParams
    from core.pvp.agents import GoofspielAgent
    from core.pvp.game_eval import _AGENT_REGISTRY
    from core.pvp.game_eval import config_id_for_seed

    HAS_PYSPIEL = True
except ImportError:
    HAS_PYSPIEL = False

needs_pyspiel = pytest.mark.skipif(not HAS_PYSPIEL, reason="pyspiel not installed")


@needs_pyspiel
class TestGoofspielGameConstruction:
    def test_load_game_is_sequential(self):
        """The agent must convert the simultaneous game to turn-based."""
        agent = GoofspielAgent()
        game = agent.load_game(agent.generate_params(0))
        assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL

    def test_load_game_is_zero_sum_win_loss(self):
        """win_loss returns are zero-sum in [-1, 1] so determine_outcome maps cleanly."""
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
        agent = GoofspielAgent()
        params = agent.generate_params(0)
        assert params.imp_info is True
        assert params.returns_type == "win_loss"
        assert params.players == 2

    def test_num_cards_varies_with_config_id(self):
        """Different config ids select different deck sizes (per-game variety)."""
        agent = GoofspielAgent()
        sizes = {agent.generate_params(cid).num_cards for cid in range(50)}
        assert len(sizes) > 1, "num_cards should vary across config ids"

    def test_all_num_cards_options_load(self):
        """Every deck size the agent can pick must load and convert cleanly."""
        agent = GoofspielAgent()
        for cid in range(50):
            game = agent.load_game(agent.generate_params(cid))
            assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL

    def test_config_id_for_seed_drives_variety(self):
        """Seeds map through config_id_for_seed to a spread of deck sizes."""
        agent = GoofspielAgent()
        env_config = ENVIRONMENT_CONFIGS[EnvironmentName.GOOFSPIEL]
        sizes = {
            agent.generate_params(config_id_for_seed(seed, env_config)).num_cards
            for seed in range(200)
        }
        assert len(sizes) > 1


@needs_pyspiel
class TestGoofspielStateFormatting:
    def _fresh_state(self, agent, seed=1):
        game = agent.load_game(agent.generate_params(0))
        state = game.new_initial_state()
        rng = random.Random(seed)
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            state.apply_action(rng.choices([o for o, _ in outcomes], [p for _, p in outcomes])[0])
        return state

    def test_format_state_hides_opponent_hand(self):
        """imp_info=True: player 0's view must not reveal player 1's hand."""
        agent = GoofspielAgent()
        state = self._fresh_state(agent)
        rendered = agent.format_state(state, player_id=0)
        assert "P1 hand" not in rendered
        assert "P0 hand" in rendered or "your hand" in rendered.lower()

    def test_format_state_shows_point_card(self):
        agent = GoofspielAgent()
        state = self._fresh_state(agent)
        rendered = agent.format_state(state, player_id=0)
        assert "point card" in rendered.lower()


@needs_pyspiel
class TestGoofspielPlaythrough:
    def test_random_selfplay_terminates_win_loss(self):
        """A full random game reaches terminal with returns in {-1, 0, 1}."""
        agent = GoofspielAgent()
        game = agent.load_game(agent.generate_params(0))
        state = game.new_initial_state()
        rng = random.Random(42)
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                state.apply_action(rng.choices([o for o, _ in outcomes], [p for _, p in outcomes])[0])
            else:
                player = state.current_player()
                state.apply_action(rng.choice(state.legal_actions(player)))
        returns = state.returns()
        assert set(returns) <= {-1.0, 0.0, 1.0}
        assert returns[0] == -returns[1]

    def test_plays_through_evaluate_bots(self):
        """The wrapped game runs end-to-end under OpenSpiel's evaluate_bots."""
        agent = GoofspielAgent()
        game = agent.load_game(agent.generate_params(0))
        bots = [pyspiel.make_uniform_random_bot(p, 123 + p) for p in range(2)]
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
        prompt = GoofspielAgent().generate_system_prompt()
        assert "goofspiel" in prompt.lower()


@needs_pyspiel
class TestGameParamsRoundTrip:
    """The discriminated union on GameInstance.game_params must survive JSON.

    GameInstance only flows in-process today, so this is a safety property — but
    pinning it means a future rename of the `game` tag can't silently degrade a
    GoofspielParams to the base type without a test failing.
    """

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
