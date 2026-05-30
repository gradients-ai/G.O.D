"""Tests for PvP game instance generation: position swapping, seed determinism,
config ID variation, and tally correctness.
"""

import pytest

try:
    import pyspiel
    from validator.evaluation.pvp.game_runner import _build_instances, _tally
    from validator.evaluation.pvp.agents import LiarsDiceAgent, LeducPokerAgent, GinRummyAgent
    HAS_PYSPIEL = True
except ImportError:
    HAS_PYSPIEL = False

from core.constants import EnvironmentName
from core.models.pvp_models import GameOutcome, PvPEnvironmentResult

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


# --- 3d: _tally correctness ---


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
