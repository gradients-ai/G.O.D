"""Unit tests for the pure text-tournament scoring (validator/tournament/text_scoring.py)."""

import pytest

from core.constants import TrainingStartPoint
from core.models.tournament_models import BossScenario
from core.models.tournament_models import CompetitorScore
from core.models.tournament_models import DatasetEvalResult
from core.models.utility_models import TaskType
from validator.tournament.text_scoring import _challenger_beats_boss_on_dataset
from validator.tournament.text_scoring import _rank_dataset
from validator.tournament.text_scoring import boss_is_best
from validator.tournament.text_scoring import boss_round_outcome
from validator.tournament.text_scoring import round_decay_weights
from validator.tournament.text_scoring import weighted_rank_scores


INSTRUCT = TaskType.INSTRUCTTEXTTASK
DPO = TaskType.DPOTASK
GRPO = TaskType.GRPOTASK


def res(hotkey: str, dataset_id: str, source_round: int, task_type: TaskType, score: float | None) -> DatasetEvalResult:
    return DatasetEvalResult(
        hotkey=hotkey, dataset_id=dataset_id, source_round=source_round, task_type=task_type, score=score
    )


def weights_by_round(rounds, **kwargs) -> dict[int, float]:
    return {w.source_round: w.weight for w in round_decay_weights(rounds, **kwargs)}


def ranks_by_hotkey(rows, competitors) -> dict[str, float]:
    return {r.hotkey: r.rank for r in _rank_dataset(rows, competitors)}


class TestRoundDecayWeights:
    def test_single_round_takes_all(self):
        weights = round_decay_weights([1])
        assert len(weights) == 1
        assert weights[0].source_round == 1
        assert weights[0].weight == 1.0

    def test_two_rounds(self):
        w = weights_by_round([1, 2])
        assert w[1] == pytest.approx(1 / 3)
        assert w[2] == pytest.approx(2 / 3)

    def test_three_rounds(self):
        w = weights_by_round([1, 2, 3])
        assert w[1] == pytest.approx(0.25 / 1.75)
        assert w[2] == pytest.approx(0.50 / 1.75)
        assert w[3] == pytest.approx(1.00 / 1.75)

    def test_four_rounds(self):
        w = weights_by_round([1, 2, 3, 4])
        assert [w[r] for r in (1, 2, 3, 4)] == pytest.approx(
            [0.125 / 1.875, 0.25 / 1.875, 0.5 / 1.875, 1.0 / 1.875]
        )

    def test_sums_to_one_and_increases_toward_current(self):
        for n in range(1, 6):
            weights = round_decay_weights(list(range(1, n + 1)))
            assert sum(w.weight for w in weights) == pytest.approx(1.0)
            in_round_order = [w.weight for w in sorted(weights, key=lambda w: w.source_round)]
            assert in_round_order == sorted(in_round_order)

    def test_lower_base_makes_current_round_more_dominant(self):
        steep = weights_by_round([1, 2, 3], base=0.3)
        flat = weights_by_round([1, 2, 3], base=0.7)
        assert steep[3] > flat[3]


class TestRankWithTies:
    def test_lower_is_better_for_loss(self):
        rows = [res("a", "d", 1, INSTRUCT, 0.1), res("b", "d", 1, INSTRUCT, 0.2), res("c", "d", 1, INSTRUCT, 0.3)]
        assert ranks_by_hotkey(rows, ["a", "b", "c"]) == {"a": 1.0, "b": 2.0, "c": 3.0}

    def test_higher_is_better_for_grpo(self):
        rows = [res("a", "d", 1, GRPO, 0.9), res("b", "d", 1, GRPO, 0.5), res("c", "d", 1, GRPO, 0.1)]
        assert ranks_by_hotkey(rows, ["a", "b", "c"]) == {"a": 1.0, "b": 2.0, "c": 3.0}

    def test_ties_get_average_rank(self):
        rows = [res("a", "d", 1, INSTRUCT, 0.1), res("b", "d", 1, INSTRUCT, 0.1), res("c", "d", 1, INSTRUCT, 0.3)]
        assert ranks_by_hotkey(rows, ["a", "b", "c"]) == {"a": 1.5, "b": 1.5, "c": 3.0}

    def test_missing_scores_share_the_bottom(self):
        rows = [res("a", "d", 1, INSTRUCT, 0.1), res("b", "d", 1, INSTRUCT, 0.2)]
        ranks = ranks_by_hotkey(rows, ["a", "b", "c", "d", "e"])
        assert ranks["a"] == 1.0
        assert ranks["b"] == 2.0
        assert ranks["c"] == ranks["d"] == ranks["e"] == 4.0  # avg of positions 3, 4, 5

    def test_all_failed_is_inert(self):
        rows = [res(h, "d", 1, INSTRUCT, None) for h in ["a", "b", "c", "d", "e"]]
        ranks = ranks_by_hotkey(rows, ["a", "b", "c", "d", "e"])
        assert all(rank == 3.0 for rank in ranks.values())  # (1 + 2 + 3 + 4 + 5) / 5

    def test_nan_treated_as_missing(self):
        rows = [res("a", "d", 1, INSTRUCT, 0.1), res("b", "d", 1, INSTRUCT, float("nan"))]
        ranks = ranks_by_hotkey(rows, ["a", "b"])
        assert ranks["a"] == 1.0
        assert ranks["b"] == 2.0


class TestWeightedRankScores:
    def worked_example(self) -> list[DatasetEvalResult]:
        # Ranks reproduce the design's worked R3 example (A best, then B, then Boss).
        return [
            res("A", "r1i", 1, INSTRUCT, 0.1), res("B", "r1i", 1, INSTRUCT, 0.2), res("Boss", "r1i", 1, INSTRUCT, 0.3),
            res("A", "r2d", 2, DPO, 0.2), res("B", "r2d", 2, DPO, 0.1), res("Boss", "r2d", 2, DPO, 0.3),
            res("A", "r2g", 2, GRPO, 0.9), res("B", "r2g", 2, GRPO, 0.1), res("Boss", "r2g", 2, GRPO, 0.5),
            res("A", "r3i", 3, INSTRUCT, 0.1), res("B", "r3i", 3, INSTRUCT, 0.2), res("Boss", "r3i", 3, INSTRUCT, 0.3),
            res("A", "r3d", 3, DPO, 0.2), res("B", "r3d", 3, DPO, 0.1), res("Boss", "r3d", 3, DPO, 0.3),
            res("A", "r3g", 3, GRPO, 0.9), res("B", "r3g", 3, GRPO, 0.5), res("Boss", "r3g", 3, GRPO, 0.1),
        ]

    def test_matches_worked_example(self):
        scores = {c.hotkey: c.score for c in weighted_rank_scores(self.worked_example(), ["A", "B", "Boss"])}
        # Geometric R3 weights [0.142857, 0.285714, 0.571429] over round-average ranks.
        assert scores["A"] == pytest.approx(1.333333, abs=1e-5)
        assert scores["B"] == pytest.approx(1.809524, abs=1e-5)
        assert scores["Boss"] == pytest.approx(2.857143, abs=1e-5)

    def test_sorted_best_first(self):
        ordered = [c.hotkey for c in weighted_rank_scores(self.worked_example(), ["A", "B", "Boss"])]
        assert ordered == ["A", "B", "Boss"]

    def test_competitor_with_no_results_is_worst(self):
        ordered = weighted_rank_scores(self.worked_example(), ["A", "B", "Boss", "Ghost"])
        assert ordered[-1].hotkey == "Ghost"


class TestBossIsBest:
    def test_boss_strictly_lowest(self):
        scores = [CompetitorScore(hotkey="boss", score=1.0), CompetitorScore(hotkey="x", score=1.5)]
        assert boss_is_best(scores, "boss")

    def test_boss_ties_lowest(self):
        scores = [CompetitorScore(hotkey="boss", score=1.5), CompetitorScore(hotkey="x", score=1.5)]
        assert boss_is_best(scores, "boss")

    def test_challenger_lower_beats_boss(self):
        scores = [CompetitorScore(hotkey="boss", score=2.0), CompetitorScore(hotkey="x", score=1.5)]
        assert not boss_is_best(scores, "boss")

    def test_boss_absent(self):
        scores = [CompetitorScore(hotkey="x", score=1.5)]
        assert not boss_is_best(scores, "boss")


class TestHandicap:
    def test_lower_is_better_needs_margin(self):
        # boss=1.0, threshold=0.05 -> challenger must reach <= 0.95
        assert _challenger_beats_boss_on_dataset(1.0, 0.94, False, 0.05)
        assert not _challenger_beats_boss_on_dataset(1.0, 0.96, False, 0.05)

    def test_higher_is_better_needs_margin(self):
        # boss=1.0, threshold=0.05 -> challenger must reach >= 1.05
        assert _challenger_beats_boss_on_dataset(1.0, 1.06, True, 0.05)
        assert not _challenger_beats_boss_on_dataset(1.0, 1.04, True, 0.05)

    def test_challenger_failure_loses(self):
        assert not _challenger_beats_boss_on_dataset(1.0, None, False, 0.05)

    def test_boss_failure_loses_to_valid_challenger(self):
        assert _challenger_beats_boss_on_dataset(None, 0.5, False, 0.05)

    def test_both_failed_boss_retains(self):
        assert not _challenger_beats_boss_on_dataset(None, None, False, 0.05)


class TestBossRoundOutcome:
    def one_dataset_scenario(self, scenario: TrainingStartPoint, boss: float, challenger: float) -> BossScenario:
        return BossScenario(
            scenario=scenario,
            results=[res("boss", "d", 1, INSTRUCT, boss), res("chal", "d", 1, INSTRUCT, challenger)],
        )

    def test_two_of_three_dethrones(self):
        scenarios = [
            self.one_dataset_scenario(TrainingStartPoint.FROM_SCRATCH, 1.0, 0.5),
            self.one_dataset_scenario(TrainingStartPoint.CONTINUATION, 1.0, 0.5),
            self.one_dataset_scenario(TrainingStartPoint.PREVIOUS_WINNER, 1.0, 2.0),
        ]
        outcome = boss_round_outcome(scenarios, "boss", "chal", threshold=0.05)
        assert outcome.scenarios_won_by_challenger == 2
        assert outcome.challenger_dethrones

    def test_one_of_three_retains(self):
        scenarios = [
            self.one_dataset_scenario(TrainingStartPoint.FROM_SCRATCH, 1.0, 0.5),
            self.one_dataset_scenario(TrainingStartPoint.CONTINUATION, 1.0, 2.0),
            self.one_dataset_scenario(TrainingStartPoint.PREVIOUS_WINNER, 1.0, 2.0),
        ]
        outcome = boss_round_outcome(scenarios, "boss", "chal", threshold=0.05)
        assert outcome.scenarios_won_by_challenger == 1
        assert not outcome.challenger_dethrones

    def test_scenario_needs_more_than_half_of_a_round(self):
        # Two datasets in one round: winning exactly one is 50%, which is not > 50%.
        scenario = BossScenario(
            scenario=TrainingStartPoint.FROM_SCRATCH,
            results=[
                res("boss", "d1", 1, INSTRUCT, 1.0), res("chal", "d1", 1, INSTRUCT, 0.5),  # challenger wins d1
                res("boss", "d2", 1, INSTRUCT, 1.0), res("chal", "d2", 1, INSTRUCT, 2.0),  # challenger loses d2
            ],
        )
        outcome = boss_round_outcome([scenario], "boss", "chal", threshold=0.05)
        assert outcome.scenarios[0].challenger_share == pytest.approx(0.5)
        assert not outcome.scenarios[0].challenger_won

    def test_recent_round_outweighs_older_round(self):
        # Challenger loses the older round (r1) but wins the current round (r2); r2 carries more
        # weight, so the challenger takes >50% and wins the scenario.
        scenario = BossScenario(
            scenario=TrainingStartPoint.CONTINUATION,
            results=[
                res("boss", "d1", 1, INSTRUCT, 1.0), res("chal", "d1", 1, INSTRUCT, 2.0),  # lose old round
                res("boss", "d2", 2, INSTRUCT, 1.0), res("chal", "d2", 2, INSTRUCT, 0.5),  # win current round
            ],
        )
        outcome = boss_round_outcome([scenario], "boss", "chal", threshold=0.05)
        assert outcome.scenarios[0].challenger_share == pytest.approx(2 / 3)  # current round weight
        assert outcome.scenarios[0].challenger_won
