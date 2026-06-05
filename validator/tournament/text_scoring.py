import math
from collections.abc import Iterable
from collections.abc import Sequence
from statistics import mean
from typing import TypeGuard

from scipy.stats import rankdata

from core.models.tournament_models import BossRoundOutcome
from core.models.tournament_models import BossScenario
from core.models.tournament_models import BossScenarioOutcome
from core.models.tournament_models import CompetitorScore
from core.models.tournament_models import TaskEvalResult
from core.models.tournament_models import RoundRank
from core.models.tournament_models import RoundValue
from core.models.tournament_models import RoundWeight
from core.models.utility_models import TaskType
from validator.tournament import constants as t_cst


def _higher_is_better(task_type: TaskType) -> bool:
    # GRPO is scored by a reward (higher is better); instruct/DPO by a loss (lower is better).
    return task_type == TaskType.GRPOTASK


def _is_present(score: float | None) -> TypeGuard[float]:
    return score is not None and not math.isnan(score)


def _score_for(rows: Sequence[TaskEvalResult], hotkey: str) -> float | None:
    return next((row.score for row in rows if row.hotkey == hotkey), None)


def round_decay_weights(rounds: Sequence[int], base: float = t_cst.TEXT_ROUND_DECAY_BASE) -> list[RoundWeight]:
    """Geometric decay toward the most recent round, normalised to sum to 1.

    A round k positions older than the newest gets raw weight ``base ** k`` (newest = 1.0), so
    each older round counts ``base`` as much as the one after it. Older rounds shrink but never
    vanish.
    """
    unique = sorted(set(rounds))
    newest_index = len(unique) - 1
    raw = [(source_round, base ** (newest_index - i)) for i, source_round in enumerate(unique)]
    total = sum(weight for _, weight in raw)
    return [RoundWeight(source_round=source_round, weight=weight / total) for source_round, weight in raw]


def _weighted_round_average(values: Sequence[RoundValue]) -> float:
    """Average the values within each source round, then combine the rounds with the decay
    weights. Each round contributes its decay weight regardless of how many values it holds."""
    weights = round_decay_weights(sorted({value.source_round for value in values}))
    return sum(
        weight.weight * mean([value.value for value in values if value.source_round == weight.source_round])
        for weight in weights
    )


def _rank_dataset(rows: Sequence[TaskEvalResult], competitors: Sequence[str]) -> list[RoundRank]:
    """Tie-averaged ranks (1 = best) for one dataset over all competitors; missing scores rank
    last. Reuses scipy's average-method rankdata over an order key that ascends in "worse"."""
    higher_is_better = _higher_is_better(rows[0].task_type)
    source_round = rows[0].source_round
    scores = [_score_for(rows, hotkey) for hotkey in competitors]
    order_keys = [(-s if higher_is_better else s) if _is_present(s) else math.inf for s in scores]
    ranks = rankdata(order_keys, method="average")
    return [
        RoundRank(hotkey=hotkey, source_round=source_round, rank=float(rank))
        for hotkey, rank in zip(competitors, ranks)
    ]


def weighted_rank_scores(
    results: Iterable[TaskEvalResult], competitors: Sequence[str]
) -> list[CompetitorScore]:
    """Per-competitor weighted-rank score, best first (lowest = best).

    Every competitor is ranked on every dataset (no row for a dataset = failure = last); ranks
    are averaged within each source round, then combined with the round-decay weights.
    """
    results = list(results)
    constituent_task_ids = list(dict.fromkeys(result.constituent_task_id for result in results))

    round_ranks: list[RoundRank] = []
    for constituent_task_id in constituent_task_ids:
        rows = [result for result in results if result.constituent_task_id == constituent_task_id]
        round_ranks.extend(_rank_dataset(rows, competitors))

    scores = [
        CompetitorScore(
            hotkey=hotkey,
            score=_weighted_round_average(
                [RoundValue(source_round=rank.source_round, value=rank.rank) for rank in round_ranks if rank.hotkey == hotkey]
            ),
        )
        for hotkey in competitors
    ]
    return sorted(scores, key=lambda competitor: competitor.score)


def boss_is_best(scores: Sequence[CompetitorScore], boss_hotkey: str) -> bool:
    """Early stop: the boss retains if it holds the best — lowest — weighted-rank score.
    The boss wins ties (incumbent edge)."""
    boss_score = next((competitor.score for competitor in scores if competitor.hotkey == boss_hotkey), None)
    if boss_score is None:
        return False
    challenger_scores = [competitor.score for competitor in scores if competitor.hotkey != boss_hotkey]
    return not challenger_scores or boss_score <= min(challenger_scores)


def _challenger_beats_boss_on_dataset(
    boss_score: float | None,
    challenger_score: float | None,
    higher_is_better: bool,
    threshold: float,
) -> bool:
    """Per-dataset boss-vs-challenger with the boss's progressive threshold handicap.

    The challenger must clear the boss by the handicap margin. Ties and challenger failures
    favour the boss (incumbent); a boss failure against a valid challenger is a challenger win.
    """
    if not _is_present(challenger_score):
        return False
    if not _is_present(boss_score):
        return True
    if higher_is_better:
        return challenger_score >= boss_score * (1 + threshold)
    return challenger_score <= boss_score * (1 - threshold)


def _scenario_outcome(
    scenario: BossScenario,
    boss_hotkey: str,
    challenger_hotkey: str,
    threshold: float,
) -> BossScenarioOutcome:
    constituent_task_ids = list(dict.fromkeys(result.constituent_task_id for result in scenario.results))

    wins: list[RoundValue] = []
    for constituent_task_id in constituent_task_ids:
        rows = [result for result in scenario.results if result.constituent_task_id == constituent_task_id]
        challenger_won = _challenger_beats_boss_on_dataset(
            _score_for(rows, boss_hotkey),
            _score_for(rows, challenger_hotkey),
            _higher_is_better(rows[0].task_type),
            threshold,
        )
        wins.append(RoundValue(source_round=rows[0].source_round, value=1.0 if challenger_won else 0.0))

    challenger_share = _weighted_round_average(wins)
    return BossScenarioOutcome(
        scenario=scenario.scenario,
        challenger_share=challenger_share,
        challenger_won=challenger_share > 0.5,
    )


def boss_round_outcome(
    scenarios: Sequence[BossScenario],
    boss_hotkey: str,
    challenger_hotkey: str,
    threshold: float,
) -> BossRoundOutcome:
    """The challenger dethrones the boss by winning a majority of scenarios (2 of 3 for the
    standard boss round). It wins a scenario by taking >50% of that scenario's decay-weighted
    dataset mass, with the boss's threshold handicap applied per dataset."""
    outcomes = [_scenario_outcome(scenario, boss_hotkey, challenger_hotkey, threshold) for scenario in scenarios]
    scenarios_won = sum(outcome.challenger_won for outcome in outcomes)
    return BossRoundOutcome(
        scenarios=outcomes,
        scenarios_won_by_challenger=scenarios_won,
        challenger_dethrones=scenarios_won * 2 > len(outcomes),
    )
