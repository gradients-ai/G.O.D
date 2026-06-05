"""Rank-normalized environment combination tests."""

import pytest

from core.constants.environments import EnvironmentName
from validator.evaluation.pvp.models import PvPEnvironmentResult
from validator.evaluation.pvp.models import PvPEvalMetadata
from validator.evaluation.pvp.models import PvPGroupResults
from validator.evaluation.pvp.models import PvPPairResult
from validator.scoring.models import EnvironmentWeight
from validator.scoring.models import EnvMinerScores
from validator.scoring.tournaments import _rank_quantiles
from validator.scoring.tournaments import pvp_results_to_winrates
from validator.scoring.tournaments import rank_weighted_standings


INTERCODE = EnvironmentName.INTERCODE
LIARS = EnvironmentName.LIARS_DICE


def _points(standings):
    return {s.hotkey: s.points for s in standings}


class TestRankQuantiles:
    def test_spreads_evenly(self) -> None:
        q = _rank_quantiles({"a": 0.1, "b": 0.2, "c": 0.3}, ["a", "b", "c"])
        assert q == {"a": 0.0, "b": 0.5, "c": 1.0}

    def test_clustering_is_irrelevant(self) -> None:
        tight = _rank_quantiles({"a": 0.700, "b": 0.701, "c": 0.702}, ["a", "b", "c"])
        wide = _rank_quantiles({"a": 0.0, "b": 0.5, "c": 1.0}, ["a", "b", "c"])
        assert tight == wide

    def test_ties_share_mean_quantile(self) -> None:
        q = _rank_quantiles({"a": 0.5, "b": 0.5, "c": 0.9}, ["a", "b", "c"])
        assert q["a"] == q["b"] == 0.25
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
        hotkeys = ["x", "y"]
        env_scores = [
            EnvMinerScores(environment=INTERCODE, scores_by_hotkey={"x": 0.80, "y": 0.70}),
            EnvMinerScores(environment=LIARS, scores_by_hotkey={"x": 0.10, "y": 0.90}),
        ]
        pts = _points(rank_weighted_standings(env_scores, hotkeys))
        assert pts["x"] == pytest.approx(0.5)
        assert pts["y"] == pytest.approx(0.5)

    def test_configured_weight_flips_clustered_env_winner(self) -> None:
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
        assert pts["x"] == pytest.approx(2 / 3)
        assert pts["y"] == pytest.approx(1 / 3)

    def test_failure_is_bounded_not_a_weight_explosion(self) -> None:
        hotkeys = ["good", "mid", "failed"]
        env_scores = [
            EnvMinerScores(environment=INTERCODE, scores_by_hotkey={"good": 0.75, "mid": 0.72, "failed": 0.0}),
            EnvMinerScores(environment=LIARS, scores_by_hotkey={"good": 0.9, "mid": 0.5, "failed": 0.4}),
        ]
        standings = rank_weighted_standings(env_scores, hotkeys)
        assert [s.hotkey for s in standings] == ["good", "mid", "failed"]
        assert _points(standings)["failed"] == pytest.approx(0.0)

    def test_empty_envs(self) -> None:
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
                    results={
                        LIARS: PvPEnvironmentResult(
                            model_a_wins=120,
                            model_b_wins=60,
                            draws=20,
                            total_games=200,
                        )
                    },
                )
            ],
            metadata=PvPEvalMetadata(seed=0, temperature=0.0, wall_time_seconds=0),
        )
        out = {env.environment: env.scores_by_hotkey for env in pvp_results_to_winrates(group)}
        assert out[LIARS]["a"] == pytest.approx(0.65)
        assert out[LIARS]["b"] == pytest.approx(0.35)
