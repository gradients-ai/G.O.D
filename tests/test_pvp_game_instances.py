"""Tests for PvP game instance generation: position swapping, seed determinism,
config ID variation, and tally correctness.
"""

import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.constants import EnvironmentName
from core.models.pvp_models import GameInstance
from core.models.pvp_models import GameOutcome
from core.models.pvp_models import PvPEnvironmentResult


try:
    if importlib.util.find_spec("pyspiel") is None:
        raise ImportError

    import pyspiel

    from validator.evaluation.pvp.agents import GinRummyAgent
    from validator.evaluation.pvp.agents import LeducPokerAgent
    from validator.evaluation.pvp.agents import LiarsDiceAgent
    from validator.evaluation.pvp.agents import OthelloAgent
    from validator.evaluation.pvp.game_runner import PlayedGame
    from validator.evaluation.pvp.game_runner import _build_instances
    from validator.evaluation.pvp.game_runner import _execute_matchup
    from validator.evaluation.pvp.game_runner import _tally

    HAS_PYSPIEL = True
except ImportError:
    HAS_PYSPIEL = False

needs_pyspiel = pytest.mark.skipif(not HAS_PYSPIEL, reason="pyspiel not installed")


# --- 3a: Position swap invariant ---


@needs_pyspiel
class TestPositionSwap:
    def test_double_instances_per_seed(self):
        """N seeds → exactly 2N instances."""
        agent = LiarsDiceAgent()
        instances = _build_instances(EnvironmentName.LIARS_DICE, agent, num_games=5, base_seed=42)
        assert len(instances) == 10

    def test_each_seed_has_both_positions(self):
        """Every seed appears with model_a_player_id 0 and 1."""
        agent = LiarsDiceAgent()
        instances = _build_instances(EnvironmentName.LIARS_DICE, agent, num_games=3, base_seed=42)

        seeds = {}
        for inst in instances:
            seeds.setdefault(inst.seed, set()).add(inst.model_a_player_id)

        for seed, positions in seeds.items():
            assert positions == {0, 1}, f"Seed {seed} missing position: has {positions}"


# --- 3b: Seed determinism ---


@needs_pyspiel
class TestSeedDeterminism:
    def test_same_inputs_same_output(self):
        agent = LiarsDiceAgent()
        a = _build_instances(EnvironmentName.LIARS_DICE, agent, num_games=5, base_seed=42)
        b = _build_instances(EnvironmentName.LIARS_DICE, agent, num_games=5, base_seed=42)

        for ia, ib in zip(a, b):
            assert ia.seed == ib.seed
            assert ia.model_a_player_id == ib.model_a_player_id
            assert ia.game_params == ib.game_params

    def test_different_base_seed_different_output(self):
        agent = LiarsDiceAgent()
        a = _build_instances(EnvironmentName.LIARS_DICE, agent, num_games=5, base_seed=42)
        b = _build_instances(EnvironmentName.LIARS_DICE, agent, num_games=5, base_seed=99)

        a_seeds = [inst.seed for inst in a]
        b_seeds = [inst.seed for inst in b]
        assert a_seeds != b_seeds


# --- 3c: Config ID variation (GinRummy has variable params) ---


@needs_pyspiel
class TestConfigIdVariation:
    def test_gin_rummy_params_vary_by_config(self):
        """Different config_ids should produce different hand_size/knock_card."""
        agent = GinRummyAgent()
        params_set = set()
        for config_id in range(9):  # 3×3 = 9 unique combos from the formula
            p = agent.generate_params(config_id)
            params_set.add((p["hand_size"], p["knock_card"]))

        assert len(params_set) > 1, "All config_ids produced identical params"

    def test_gin_rummy_hand_size_range(self):
        agent = GinRummyAgent()
        for config_id in range(100):
            p = agent.generate_params(config_id)
            assert 7 <= p["hand_size"] <= 9
            assert 8 <= p["knock_card"] <= 10

    def test_liars_dice_params_constant(self):
        """Liar's dice has fixed params regardless of config_id."""
        agent = LiarsDiceAgent()
        p0 = agent.generate_params(0)
        p1 = agent.generate_params(99)
        assert p0 == p1 == {"players": 2, "numdice": 5}

    def test_leduc_poker_params_constant(self):
        agent = LeducPokerAgent()
        p0 = agent.generate_params(0)
        p1 = agent.generate_params(99)
        assert p0 == p1 == {"players": 2}

    def test_othello_params_empty(self):
        """Othello takes no pyspiel parameters regardless of config_id."""
        agent = OthelloAgent()
        assert agent.generate_params(0) == agent.generate_params(99) == {}


# --- 3c': Othello seeded opening plies (deterministic game needs injected variety) ---


@needs_pyspiel
class TestOthelloOpeningPlies:
    def _opened_board(self, seed: int) -> str:
        """Fresh othello state advanced by the agent's seeded opening plies."""
        agent = OthelloAgent()
        state = pyspiel.load_game("othello").new_initial_state()
        agent.setup_initial_state(state, seed)
        return state.observation_string(0)

    def test_same_seed_same_opening(self):
        assert self._opened_board(42) == self._opened_board(42)

    def test_format_state_names_the_players_colour(self):
        agent = OthelloAgent()
        state = pyspiel.load_game("othello").new_initial_state()
        assert "You play x (Black)" in agent.format_state(state, 0)
        assert "You play o (White)" in agent.format_state(state, 1)

    def test_different_seed_different_opening(self):
        """Across a spread of seeds, openings should diverge (not all identical)."""
        boards = {self._opened_board(seed) for seed in range(20)}
        assert len(boards) > 1, "All seeds produced the identical opening board"

    def test_opening_leaves_playable_nonterminal_state(self):
        agent = OthelloAgent()
        for seed in range(20):
            state = pyspiel.load_game("othello").new_initial_state()
            agent.setup_initial_state(state, seed)
            assert not state.is_terminal()
            assert state.current_player() in (0, 1)
            assert state.legal_actions()

    def test_opening_diverges_from_fresh_board(self):
        """At least some seeds move the board off the standard starting position."""
        fresh = pyspiel.load_game("othello").new_initial_state().observation_string(0)
        assert any(self._opened_board(seed) != fresh for seed in range(20))


# --- 3d: _tally correctness ---


@needs_pyspiel
class TestTally:
    def _fresh_result(self) -> PvPEnvironmentResult:
        return PvPEnvironmentResult()

    def test_win_increments_model_a(self):
        r = self._fresh_result()
        _tally(r, GameOutcome.WIN)
        assert r.model_a_wins == 1
        assert r.model_b_wins == 0
        assert r.draws == 0
        assert r.total_games == 1

    def test_loss_increments_model_b(self):
        r = self._fresh_result()
        _tally(r, GameOutcome.LOSS)
        assert r.model_b_wins == 1
        assert r.model_a_wins == 0

    def test_draw_increments_draws(self):
        r = self._fresh_result()
        _tally(r, GameOutcome.DRAW)
        assert r.draws == 1
        assert r.model_a_wins == 0
        assert r.model_b_wins == 0

    def test_total_always_increments(self):
        r = self._fresh_result()
        _tally(r, GameOutcome.WIN)
        _tally(r, GameOutcome.LOSS)
        _tally(r, GameOutcome.DRAW)
        assert r.total_games == 3
        assert r.model_a_wins + r.model_b_wins + r.draws == 3


# --- 3e: matchup-level forfeit shortcut ---


def _make_test_instances(count: int) -> list[GameInstance]:
    return [
        GameInstance(
            game_name="leduc_poker",
            game_params={"players": 2},
            model_a_player_id=i % 2,
            seed=i,
            is_zero_sum=True,
            min_utility=-1.0,
            max_utility=1.0,
        )
        for i in range(count)
    ]


@needs_pyspiel
class TestEpisodeForfeitLimit:
    def test_ten_non_consecutive_model_a_forfeits_awards_remaining_games_to_model_b(self):
        instances = _make_test_instances(24)
        played_games = []
        for _ in range(9):
            played_games.append(PlayedGame(outcome=GameOutcome.LOSS, forfeiting_model="a"))
            played_games.append(PlayedGame(outcome=GameOutcome.DRAW))
        played_games.append(PlayedGame(outcome=GameOutcome.LOSS, forfeiting_model="a"))

        with patch("validator.evaluation.pvp.game_runner._play_game", side_effect=played_games) as play_game:
            result = _execute_matchup(
                env_name=EnvironmentName.LEDUC_POKER,
                instances=instances,
                player_a=SimpleNamespace(config=SimpleNamespace(inference_model="test", tokenizer_repo=None)),
                player_b=SimpleNamespace(config=SimpleNamespace(inference_model="test", tokenizer_repo=None)),
                agent=object(),
            )

        assert play_game.call_count == 19
        assert result.model_a_wins == 0
        assert result.model_b_wins == 15
        assert result.draws == 9
        assert result.total_games == 24


# --- PvP env / agent registry exhaustiveness (import-time guard in game_eval) ---


@needs_pyspiel
class TestAgentRegistryCoversPvpEnvs:
    def test_every_pvp_env_has_an_agent(self):
        from core.constants import ENVIRONMENT_CONFIGS
        from core.constants import EvalType
        from core.pvp.game_eval import _AGENT_REGISTRY

        pvp_envs = {n for n, c in ENVIRONMENT_CONFIGS.items() if c.eval_type == EvalType.PVP}
        assert set(_AGENT_REGISTRY) == pvp_envs
