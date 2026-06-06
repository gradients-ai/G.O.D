"""Tests for rank-normalized environment combination (rank_weighted_standings).

Covers the two properties that motivated replacing the dispersion-weighted scheme:
  1. A tightly-clustered env (e.g. intercode ~[0.7, 0.8]) keeps its configured say
     instead of being drowned out by a wide-spread PvP env.
  2. A single per-env failure (score 0.0 in a cluster) is a bounded last-place penalty,
     not a weight explosion that lets the failing env dominate.
"""

import pytest

from core.constants import EnvironmentName
from core.models.pvp_models import PvPEvalMetadata
from core.models.pvp_models import PvPGroupResults
from core.models.pvp_models import PvPPairResult
from core.models.pvp_models import PvPEnvironmentResult
from core.models.scoring_models import EnvironmentWeight
from core.models.scoring_models import EnvMinerScores
from validator.evaluation.tournament_scoring import _rank_quantiles
from validator.evaluation.tournament_scoring import pvp_results_to_winrates
from validator.evaluation.tournament_scoring import rank_weighted_standings


INTERCODE = EnvironmentName.INTERCODE
LIARS = EnvironmentName.LIARS_DICE


def _points(standings):
    return {s.hotkey: s.points for s in standings}


class TestRankQuantiles:
    def test_spreads_evenly(self) -> None:
        q = _rank_quantiles({"a": 0.1, "b": 0.2, "c": 0.3}, ["a", "b", "c"])
        assert q == {"a": 0.0, "b": 0.5, "c": 1.0}

    def test_clustering_is_irrelevant(self) -> None:
        """A 0.001-wide cluster ranks identically to a full-range spread."""
        tight = _rank_quantiles({"a": 0.700, "b": 0.701, "c": 0.702}, ["a", "b", "c"])
        wide = _rank_quantiles({"a": 0.0, "b": 0.5, "c": 1.0}, ["a", "b", "c"])
        assert tight == wide

    def test_ties_share_mean_quantile(self) -> None:
        q = _rank_quantiles({"a": 0.5, "b": 0.5, "c": 0.9}, ["a", "b", "c"])
        assert q["a"] == q["b"] == 0.25  # mean of ranks 0 and 1, over (n-1)=2
        assert q["c"] == 1.0

    def test_all_tied_is_neutral_half(self) -> None:
        q = _rank_quantiles({"a": 0.0, "b": 0.0, "c": 0.0}, ["a", "b", "c"])
        assert q == {"a": 0.5, "b": 0.5, "c": 0.5}

    def test_single_miner(self) -> None:
        assert _rank_quantiles({"a": 0.42}, ["a"]) == {"a": 0.5}

    def test_missing_hotkey_sorts_last(self) -> None:
        q = _rank_quantiles({"a": 0.7, "b": 0.8}, ["a", "b", "missing"])
        assert q["missing"] == 0.0
        assert q["b"] == 1.0


class TestRankWeightedStandings:
    def test_clustered_env_still_counts(self) -> None:
        """Equal-weighted clustered intercode + wide liars: a miner that loses liars
        but wins intercode is pulled up, not erased (unlike dispersion weighting)."""
        hotkeys = ["x", "y"]
        env_scores = [
            EnvMinerScores(environment=INTERCODE, scores_by_hotkey={"x": 0.80, "y": 0.70}),
            EnvMinerScores(environment=LIARS, scores_by_hotkey={"x": 0.10, "y": 0.90}),
        ]
        pts = _points(rank_weighted_standings(env_scores, hotkeys))
        # x is rank-1 in intercode (q=1) but rank-0 in liars (q=0): mean 0.5.
        # y is the mirror: also 0.5. Equal weight -> exact tie, intercode NOT drowned out.
        assert pts["x"] == pytest.approx(0.5)
        assert pts["y"] == pytest.approx(0.5)

    def test_dispersion_would_have_let_liars_dominate(self) -> None:
        """Same data, but make intercode the tie-breaker via a stronger configured weight.
        Under rank-normalization the configured weight actually bites; the wide spread of
        liars cannot override it the way raw-score dispersion weighting did."""
        hotkeys = ["x", "y"]
        env_scores = [
            EnvMinerScores(environment=INTERCODE, scores_by_hotkey={"x": 0.80, "y": 0.70}),
            EnvMinerScores(environment=LIARS, scores_by_hotkey={"x": 0.10, "y": 0.90}),
        ]
        weights = [
            EnvironmentWeight(environment=INTERCODE, weight=2.0),
            EnvironmentWeight(environment=LIARS, weight=1.0),
        ]
        pts = _points(rank_weighted_standings(env_scores, hotkeys, weights))
        # x: (2*1 + 1*0)/3 = 0.667 ; y: (2*0 + 1*1)/3 = 0.333
        assert pts["x"] == pytest.approx(2 / 3)
        assert pts["y"] == pytest.approx(1 / 3)

    def test_failure_is_bounded_not_a_weight_explosion(self) -> None:
        """One miner fails intercode (0.0) inside a [0.7,0.8] cluster. It must land last
        in intercode, and the standings must still be driven by genuine performance —
        the failure does not hand intercode outsized influence."""
        hotkeys = ["good", "mid", "failed"]
        env_scores = [
            EnvMinerScores(environment=INTERCODE, scores_by_hotkey={"good": 0.75, "mid": 0.72, "failed": 0.0}),
            EnvMinerScores(environment=LIARS, scores_by_hotkey={"good": 0.9, "mid": 0.5, "failed": 0.4}),
        ]
        standings = rank_weighted_standings(env_scores, hotkeys)
        order = [s.hotkey for s in standings]
        assert order == ["good", "mid", "failed"]
        pts = _points(standings)
        # 'failed' is last in both envs -> q=0 both -> exactly 0.0, a bounded floor.
        assert pts["failed"] == pytest.approx(0.0)

    def test_empty_envs(self) -> None:
        assert rank_weighted_standings([], ["a", "b"]) == sorted(
            rank_weighted_standings([], ["a", "b"]), key=lambda s: s.points, reverse=True
        )
        pts = _points(rank_weighted_standings([], ["a", "b"]))
        assert pts == {"a": 0.0, "b": 0.0}


class TestPvPResultsToWinrates:
    def test_winrate_with_draws(self) -> None:
        group = PvPGroupResults(
            base_model="m",
            hotkeys=["a", "b"],
            pair_results=[
                PvPPairResult(
                    hotkey_a="a",
                    hotkey_b="b",
                    results={LIARS: PvPEnvironmentResult(model_a_wins=120, model_b_wins=60, draws=20, total_games=200)},
                )
            ],
            metadata=PvPEvalMetadata(seed=0, temperature=0.0, wall_time_seconds=0),
        )
        out = {e.environment: e.scores_by_hotkey for e in pvp_results_to_winrates(group)}
        # a: (120 + 0.5*20)/200 = 0.65 ; b: (60 + 0.5*20)/200 = 0.35
        assert out[LIARS]["a"] == pytest.approx(0.65)
        assert out[LIARS]["b"] == pytest.approx(0.35)
