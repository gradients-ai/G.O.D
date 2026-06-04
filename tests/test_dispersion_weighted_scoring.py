"""Tests for dispersion-weighted environment scoring.

Covers the new group-stage aggregation that replaced the flat 3/1/0 pairwise
accumulator: per-env win-rate / normalized score, combined with each env's
influence scaled by configured weight x dispersion (stdev) of its scores.
"""

import pytest

from core.constants import EnvironmentName
from core.models.pvp_models import PvPEnvironmentResult
from core.models.pvp_models import PvPEvalMetadata
from core.models.pvp_models import PvPGroupResults
from core.models.pvp_models import PvPPairResult
from core.models.scoring_models import EnvironmentWeight
from validator.evaluation.tournament_scoring import dispersion_weighted_standings
from validator.evaluation.tournament_scoring import pvp_results_to_winrates


LD = EnvironmentName.LIARS_DICE
IC = EnvironmentName.INTERCODE


def _group(hotkeys: list[str], pairs: list[tuple[str, str, int, int]], env=LD) -> PvPGroupResults:
    """Build PvPGroupResults from (hotkey_a, hotkey_b, a_wins, b_wins) tuples (no draws)."""
    pair_results = [
        PvPPairResult(
            hotkey_a=a,
            hotkey_b=b,
            results={env: PvPEnvironmentResult(model_a_wins=aw, model_b_wins=bw, draws=0, total_games=aw + bw)},
        )
        for a, b, aw, bw in pairs
    ]
    return PvPGroupResults(
        base_model="base",
        hotkeys=hotkeys,
        pair_results=pair_results,
        metadata=PvPEvalMetadata(seed=42, temperature=0.0),
    )


def _winner(standings) -> str:
    return standings[0].hotkey


def _rank(standings) -> list[str]:
    return [s.hotkey for s in standings]


# --- pvp_results_to_winrates ---


def test_winrate_preserves_margin():
    g = _group(["a", "b", "c"], [("a", "b", 200, 0), ("a", "c", 100, 100), ("b", "c", 0, 200)])
    wr = pvp_results_to_winrates(g)[LD]
    # a: 200/200 vs b + 100/200 vs c = 300/400 = 0.75
    assert wr["a"] == pytest.approx(0.75)
    # b: 0 vs a + 0 vs c = 0/400
    assert wr["b"] == pytest.approx(0.0)
    # c: 100 vs a + 200 vs b = 300/400 = 0.75
    assert wr["c"] == pytest.approx(0.75)


def test_winrate_counts_draws_as_half():
    g = _group(["a", "b"], [("a", "b", 0, 0)], env=LD)
    g.pair_results[0].results[LD] = PvPEnvironmentResult(model_a_wins=50, model_b_wins=50, draws=100, total_games=200)
    wr = pvp_results_to_winrates(g)[LD]
    # (50 + 0.5*100)/200 = 0.5 each
    assert wr["a"] == pytest.approx(0.5)
    assert wr["b"] == pytest.approx(0.5)


# --- dispersion_weighted_standings ---


def test_decisive_env_dominates_flat_env():
    """A clustered env must not override a decisive one under equal configured weights."""
    hotkeys = ["winner", "loser"]
    env_scores = {
        LD: {"winner": 1.0, "loser": 0.0},      # decisive: stdev 0.5
        IC: {"winner": 0.70, "loser": 0.72},     # nearly flat, slight reverse edge: stdev 0.01
    }
    weights = [EnvironmentWeight(environment=LD, weight=1.0), EnvironmentWeight(environment=IC, weight=1.0)]
    standings = dispersion_weighted_standings(env_scores, hotkeys, weights=weights)
    # liars_dice (stdev 0.5) ~50x the influence of intercode (stdev 0.01)
    assert _winner(standings) == "winner"


def test_flat_env_contributes_nothing():
    hotkeys = ["a", "b", "c"]
    env_scores = {
        LD: {"a": 0.9, "b": 0.5, "c": 0.1},  # decisive
        IC: {"a": 0.5, "b": 0.5, "c": 0.5},  # perfectly flat -> stdev 0 -> zero influence
    }
    standings = dispersion_weighted_standings(env_scores, hotkeys)
    assert _rank(standings) == ["a", "b", "c"]
    # ranking is fully determined by the only env with signal
    assert standings[0].points == pytest.approx(0.9)


def test_missing_miner_scored_zero():
    hotkeys = ["a", "b", "ghost"]
    env_scores = {LD: {"a": 0.8, "b": 0.4}}  # ghost absent (e.g. eval never produced games)
    standings = dispersion_weighted_standings(env_scores, hotkeys)
    assert _rank(standings) == ["a", "b", "ghost"]
    assert standings[-1].hotkey == "ghost"
    assert standings[-1].points == pytest.approx(0.0)


def test_all_flat_falls_back_to_mean_without_crashing():
    hotkeys = ["a", "b"]
    env_scores = {LD: {"a": 0.5, "b": 0.5}, IC: {"a": 0.3, "b": 0.3}}
    standings = dispersion_weighted_standings(env_scores, hotkeys)
    # No env separates anyone -> equal points, no division by zero
    assert standings[0].points == pytest.approx(standings[1].points)
    assert standings[0].points == pytest.approx(0.4)  # mean of normalized (0.5, 0.3)


def test_single_miner_no_crash():
    standings = dispersion_weighted_standings({LD: {"solo": 0.9}}, ["solo"])
    assert _winner(standings) == "solo"


def test_score_ref_max_normalizes_scale():
    """An env scored 0-100 must not swamp a [0,1] env purely due to scale."""
    hotkeys = ["a", "b"]
    env_scores = {
        LD: {"a": 1.0, "b": 0.0},        # ref 1.0 -> normalized spread 1.0
        IC: {"a": 60.0, "b": 40.0},      # ref 100 -> normalized 0.6/0.4, spread 0.2
    }
    ref = {LD: 1.0, IC: 100.0}
    standings = dispersion_weighted_standings(env_scores, hotkeys, score_ref_max=ref)
    # combined points stay in [0,1] after normalization
    assert all(0.0 <= s.points <= 1.0 for s in standings)
    assert _winner(standings) == "a"


# --- Backtest against real round-1 group 001 (tourn_33c30659e3c920d5_20260601) ---

# liars_dice pair results (hotkey_a, hotkey_b, a_wins, b_wins), 200 games each
_G1_PAIRS = [
    ("CMZQwPd", "CRwFq2Q", 0, 200),
    ("CMZQwPd", "EF2nASX", 0, 200),
    ("CMZQwPd", "FpdSckw", 0, 200),
    ("CMZQwPd", "GNP9XWd", 2, 198),
    ("CMZQwPd", "HWPK9f6", 111, 89),
    ("CRwFq2Q", "EF2nASX", 1, 199),
    ("CRwFq2Q", "FpdSckw", 1, 199),
    ("CRwFq2Q", "GNP9XWd", 35, 165),
    ("CRwFq2Q", "HWPK9f6", 138, 62),
    ("EF2nASX", "FpdSckw", 90, 110),
    ("EF2nASX", "GNP9XWd", 134, 66),
    ("EF2nASX", "HWPK9f6", 200, 0),
    ("FpdSckw", "GNP9XWd", 129, 71),
    ("FpdSckw", "HWPK9f6", 198, 2),
    ("GNP9XWd", "HWPK9f6", 200, 0),
]
_G1_HOTKEYS = ["CMZQwPd", "CRwFq2Q", "EF2nASX", "FpdSckw", "GNP9XWd", "HWPK9f6"]
_G1_INTERCODE = {
    "CMZQwPd": 0.791, "CRwFq2Q": 0.749, "EF2nASX": 0.729,
    "GNP9XWd": 0.721, "HWPK9f6": 0.719, "FpdSckw": 0.709,
}


def test_real_group001_flips_to_dice_champion():
    """Under the new scheme the 5-0 liars_dice champion wins group 001,
    instead of the intercode leader who went 1-4 head-to-head."""
    winrates = pvp_results_to_winrates(_group(_G1_HOTKEYS, _G1_PAIRS))[LD]
    env_scores = {LD: winrates, IC: _G1_INTERCODE}
    weights = [EnvironmentWeight(environment=LD, weight=1.0), EnvironmentWeight(environment=IC, weight=1.0)]
    standings = dispersion_weighted_standings(env_scores, _G1_HOTKEYS, weights=weights)

    # FpdSckw was 5-0 at liars_dice; CMZQwPd was 1-4 but topped intercode (and won under the old 3/1/0 scheme).
    assert _winner(standings) == "FpdSckw"
    # The old winner collapses: CMZQwPd was swept 0-200 four times (win-rate 0.113, the worst),
    # and its marginal intercode lead can't survive liars_dice getting ~11x the influence.
    assert _rank(standings)[-1] == "CMZQwPd"
    # Margin-aware win-rate correctly ranks CMZQwPd below HWPK9f6 (0-5 but more games won).
    assert _rank(standings).index("HWPK9f6") < _rank(standings).index("CMZQwPd")
