"""What counts as a narrow loss on each compression task, for the boss-battle analytics, and how a six-task round is judged.

Crowning needs five wins of six and never reads the narrowness of a loss; the analytics still report it
(within_allowed_loss). Fixed (paired test) is narrow unless the boss would have beaten the challenger under the same paired
bar with the sides swapped, a draw included; freeform is narrow at 1.01x the boss's effective bytes. The crown rule itself
is in test_quant_boss_crown_rule.py.
"""

import inspect
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

import validator.tournament.performance_calculator as performance_calculator
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from core.models.tournament_models import TournamentType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.scoring.quant_ranking import QuantReferenceUnavailable
from validator.tournament import constants as t_cst
from validator.tournament import round_results
from validator.tournament import thresholds
from validator.tournament.models import RoundStatus
from validator.tournament.models import RoundType
from validator.tournament.models import SeedRole
from validator.tournament.models import TaskPerformanceDifference
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.thresholds import challenger_within_freeform_allowed_loss
from validator.tournament.thresholds import challenger_within_paired_allowed_loss
from validator.tournament.thresholds import compare_paired_losses
from validator.tournament.thresholds import compare_paired_losses_swapped


BOSS = EMISSION_BURN_HOTKEY
CHALLENGER = "challenger_hotkey"
FIXED = TaskType.QUANTFIXEDTASK
FREEFORM = TaskType.QUANTFREEFORMTASK


def _freeform(hotkey: str, total_bytes: float | None, *, passed: bool | None = True, failure: str | None = None):
    artifact = None if total_bytes is None else int(total_bytes)
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=1.0,
        synth_loss=1.0,
        is_finetune=True,
        task_type=FREEFORM,
        artifact_bytes=artifact,
        decoder_bytes=None if artifact is None else 0,
        passed_bar=passed,
        quant_failure=failure,
    )


def _text(hotkey: str, loss: float | None, task_type: TaskType = FIXED):
    return MinerResultsText(
        hotkey=hotkey, test_loss=1.0, synth_loss=1.0, is_finetune=True, task_type=task_type, adjusted_loss=loss
    )


class TestFreeformWithin:
    def test_exactly_one_percent_more_bytes_is_within(self):
        assert challenger_within_freeform_allowed_loss(_freeform(BOSS, 10_000), _freeform(CHALLENGER, 10_100))

    def test_just_above_is_not_within(self):
        assert not challenger_within_freeform_allowed_loss(_freeform(BOSS, 10_000), _freeform(CHALLENGER, 10_101))

    def test_fewer_bytes_is_within(self):
        assert challenger_within_freeform_allowed_loss(_freeform(BOSS, 10_000), _freeform(CHALLENGER, 9_000))

    def test_over_the_cap_is_not_within(self):
        assert not challenger_within_freeform_allowed_loss(
            _freeform(BOSS, 10_000), _freeform(CHALLENGER, 10_000, passed=False)
        )

    def test_scored_fail_is_not_within(self):
        challenger = _freeform(CHALLENGER, 10_000, failure="bad decode")
        assert not challenger_within_freeform_allowed_loss(_freeform(BOSS, 10_000), challenger)

    def test_missing_bytes_is_not_within(self):
        assert not challenger_within_freeform_allowed_loss(_freeform(BOSS, None), _freeform(CHALLENGER, 10_000))
        assert not challenger_within_freeform_allowed_loss(_freeform(BOSS, 10_000), _freeform(CHALLENGER, None))

    def test_missing_results_are_not_within(self):
        assert not challenger_within_freeform_allowed_loss(None, _freeform(CHALLENGER, 10_000))
        assert not challenger_within_freeform_allowed_loss(_freeform(BOSS, 10_000), None)

    def test_non_finite_bytes_are_not_within(self):
        challenger = _freeform(CHALLENGER, 10_000)
        challenger.freeform_penalty = math.inf
        assert not challenger_within_freeform_allowed_loss(_freeform(BOSS, 10_000), challenger)


def _vectors(n: int = 800, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.5, 1.5, n)


class TestPairedWithin:
    def test_a_boss_clearly_better_with_low_noise_is_not_within(self):
        boss = _vectors()
        challenger = boss + 0.1 + np.random.default_rng(4).normal(0, 0.01, boss.size)
        within, swapped = challenger_within_paired_allowed_loss(list(boss), list(challenger), boss.mean(), challenger.mean())
        assert within is False
        assert swapped.challenger_won is True  # in the swapped call "challenger" is the boss

    def test_a_noisy_loss_with_no_significant_boss_win_is_within(self):
        boss = _vectors()
        challenger = boss + np.random.default_rng(5).normal(0, 0.3, boss.size)
        within, _ = challenger_within_paired_allowed_loss(list(boss), list(challenger), boss.mean(), challenger.mean())
        assert within is True

    def test_identical_vectors_are_within(self):
        boss = list(_vectors())
        within, swapped = challenger_within_paired_allowed_loss(boss, boss, 1.0, 1.0)
        assert within is True
        assert swapped.is_draw is True  # nothing decided in the swapped comparison counts as within

    def test_the_swap_is_deterministic(self):
        boss = _vectors()
        challenger = boss + np.random.default_rng(6).normal(0.02, 0.2, boss.size)
        runs = [
            compare_paired_losses_swapped(list(boss), list(challenger)).model_dump() for _ in range(3)
        ]
        assert runs[0] == runs[1] == runs[2]

    def test_the_swapped_call_uses_the_same_constants_as_the_forward_one(self, monkeypatch):
        calls = []

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return compare_paired_losses(*args, **kwargs)

        monkeypatch.setattr(thresholds, "compare_paired_losses", spy)
        compare_paired_losses_swapped([1.0, 2.0], [1.5, 2.5])

        (args, kwargs), = calls
        assert args == ()
        # Only the two vectors are passed, so every bar is the function's own default, as in the forward call.
        assert set(kwargs) == {"boss_losses", "challenger_losses"}
        assert kwargs["boss_losses"] == [1.5, 2.5] and kwargs["challenger_losses"] == [1.0, 2.0]
        defaults = {name: p.default for name, p in inspect.signature(compare_paired_losses).parameters.items()}
        assert defaults["deadzone_nats"] == t_cst.BOSS_ROUND_TIE_DEADZONE_NATS
        assert defaults["min_win_rate"] == t_cst.BOSS_ROUND_MIN_WIN_RATE
        assert defaults["min_mean_gap_nats"] == t_cst.BOSS_ROUND_MIN_MEAN_GAP_NATS

    def test_the_kl_scalar_gate_is_mirrored(self):
        # Boss wins the vectors clearly but carries a worse ranking scalar (a larger KL penalty): the boss does
        # not take the task under the mirrored gate, so the challenger is within.
        boss = _vectors()
        challenger = boss + 0.1
        within, _ = challenger_within_paired_allowed_loss(list(boss), list(challenger), 9.0, 1.0)
        assert within is True

    def test_mean_gap_bar_scales_on_the_first_argument_which_is_the_challenger_in_the_swap(self):
        # The bar is max(floor, 1% of |first-arg mean|). In the swap that mean is the challenger's, the larger
        # one when the boss is better, so the bar is a hair stricter on the boss: slightly more lenient on the challenger.
        boss = np.full(800, 10.0)
        challenger = boss + 0.105  # boss better by 0.105 nats; forward-style bar would be 0.1, swapped is 0.10105
        swapped = compare_paired_losses_swapped(list(boss), list(challenger))
        assert swapped.required_mean_gap_nats == pytest.approx(0.01 * 10.105)

# --- get_knockout_winners integration -------------------------------------------------------------------------


def _task_row(i: int) -> TournamentTask:
    return TournamentTask(
        tournament_id="tourn_1",
        round_id="tourn_1_round_004",
        task_id=f"00000000-0000-0000-0000-{i:012d}",
        pair_id="tourn_1_round_004_pair_001",
    )


def _final_round() -> TournamentRoundData:
    return TournamentRoundData(
        round_id="tourn_1_round_004",
        tournament_id="tourn_1",
        round_number=4,
        round_type=RoundType.KNOCKOUT,
        is_final_round=True,
        status=RoundStatus.PENDING,
    )


class Scenario:
    """One boss round: per task, a type, the ranked results (or None for no results) and the verdict the resolver gives."""

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.rows: list[TournamentTask] = []
        self.task_types: dict[str, TaskType] = {}
        self.results: dict[str, list | None] = {}
        self.verdicts: dict[str, round_results.BossRoundTaskVerdict] = {}
        self.determine_calls: list[dict] = []
        self.score_updates = AsyncMock()
        self.dead_seeds: set[str] = set()  # tasks whose boss-round seed died on the boss's side after attach

    def add(self, task_type: TaskType, results: list | None, verdict=None) -> str:
        row = _task_row(len(self.rows))
        self.rows.append(row)
        self.task_types[row.task_id] = task_type
        self.results[row.task_id] = results
        if verdict is not None:
            self.verdicts[row.task_id] = verdict
        return row.task_id

    async def run(self):
        mp = self.monkeypatch

        async def tournament(_tid, _db):
            return TournamentData(tournament_id="tourn_1", tournament_type=TournamentType.QUANT)

        async def resolver(**kwargs):
            return self.verdicts[kwargs["task_id"]]

        real_determine = round_results.determine_boss_round_winner

        def spy(*args, **kwargs):
            self.determine_calls.append(kwargs)
            return real_determine(*args, **kwargs)

        async def get_task(task_id, _db):
            return SimpleNamespace(
                task_id=task_id, task_type=self.task_types[task_id], quant_spec=None, status=TaskStatus.SUCCESS
            )

        async def results(task_id, _db):
            return self.results[task_id]

        async def dead_seed(task_id, _status, _db):
            return SeedRole.BOSS_CHAINED_FIXED if task_id in self.dead_seeds else None

        mp.setattr(round_results, "boss_side_seed_failure", dead_seed)
        mp.setattr(
            round_results,
            "get_nodes_assigned_to_task",
            AsyncMock(return_value=[SimpleNamespace(hotkey=BOSS), SimpleNamespace(hotkey=CHALLENGER)]),
        )
        mp.setattr(round_results, "get_tournament", tournament)
        mp.setattr(round_results, "count_hotkeys_assigned_to_task", AsyncMock(return_value=2))
        mp.setattr(round_results, "get_task", get_task)
        mp.setattr(round_results, "get_task_results_for_ranking", results)
        mp.setattr(round_results, "calculate_miner_ranking_and_scores", lambda ranked: ranked)
        mp.setattr(round_results, "_resolve_boss_round_task_winner", resolver)
        mp.setattr(round_results, "update_threshold_adjusted_quality_scores_for_task", self.score_updates)
        mp.setattr(round_results, "determine_boss_round_winner", spy)
        return await round_results.get_knockout_winners(_final_round(), self.rows, psql_db=None, config=None)


def _pair(task_type: TaskType, boss_loss: float, challenger_loss: float):
    return [_text(BOSS, boss_loss, task_type), _text(CHALLENGER, challenger_loss, task_type)]


def _won(winner: str = CHALLENGER):
    return round_results.BossRoundTaskVerdict(winner_hotkey=winner, decided_by="x")


def _kept(within: bool):
    return round_results.BossRoundTaskVerdict(winner_hotkey=BOSS, decided_by="x", challenger_within_allowed_loss=within)


def _add_wins(scenario: Scenario, freeform: int, fixed: int) -> None:
    """Tasks the challenger wins: freeform on 10% fewer bytes, fixed on a won verdict."""
    for _ in range(freeform):
        scenario.add(FREEFORM, [_freeform(BOSS, 10_000), _freeform(CHALLENGER, 9_000)])
    for _ in range(fixed):
        scenario.add(FIXED, _pair(FIXED, 1.0, 0.9), _won())


async def test_the_quant_round_pins_the_built_size_of_six(monkeypatch):
    scenario = Scenario(monkeypatch)
    _add_wins(scenario, freeform=3, fixed=2)
    scenario.add(FIXED, _pair(FIXED, 1.0, 1.001), _kept(True))

    outcome = await scenario.run()

    (call,) = scenario.determine_calls
    assert call["expected_task_count"] == t_cst.FINAL_ROUND_QUANT_TASKS == 6
    assert "challenger_losses" not in call
    assert outcome.winners == [CHALLENGER]


async def test_a_lost_freeform_task_does_not_block_the_crown_narrow_or_clear(monkeypatch):
    # The challenger is under the cap with 0.5% more bytes (narrow) or 5% more (clear): not 3% fewer, so the boss keeps it.
    for challenger_bytes in (10_050, 10_500):
        scenario = Scenario(monkeypatch)
        _add_wins(scenario, freeform=2, fixed=3)
        scenario.add(FREEFORM, [_freeform(BOSS, 10_000), _freeform(CHALLENGER, challenger_bytes)])
        assert (await scenario.run()).winners == [CHALLENGER]


async def test_a_task_with_no_results_is_the_bosss_and_a_second_miss_blocks(monkeypatch):
    scenario = Scenario(monkeypatch)
    _add_wins(scenario, freeform=3, fixed=2)
    scenario.add(FIXED, None)
    assert (await scenario.run()).winners == [CHALLENGER]

    second = Scenario(monkeypatch)
    _add_wins(second, freeform=3, fixed=1)
    second.add(FIXED, None)
    second.add(FIXED, None)
    assert (await second.run()).winners == [BOSS]


async def test_a_drawn_fixed_is_not_a_win_and_never_queues_a_replacement(monkeypatch):
    draw = round_results.BossRoundTaskVerdict(
        winner_hotkey=BOSS, decided_by="a draw", is_draw=True, challenger_within_allowed_loss=True
    )
    scenario = Scenario(monkeypatch)
    _add_wins(scenario, freeform=3, fixed=2)
    scenario.add(FIXED, _pair(FIXED, 1.0, 1.0), draw)
    outcome = await scenario.run()
    assert outcome.winners == [CHALLENGER]
    assert outcome.draw_resolution is None and outcome.deferred_reason is None

    second = Scenario(monkeypatch)
    _add_wins(second, freeform=2, fixed=2)
    second.add(FIXED, _pair(FIXED, 1.0, 1.0), draw)
    second.add(FREEFORM, [_freeform(BOSS, 10_000), _freeform(CHALLENGER, 10_050)])
    outcome = await second.run()
    assert outcome.winners == [BOSS] and outcome.draw_resolution is None


async def test_the_quant_round_pins_the_built_size_even_when_tasks_are_missing(monkeypatch):
    scenario = Scenario(monkeypatch)
    _add_wins(scenario, freeform=0, fixed=1)

    outcome = await scenario.run()

    assert scenario.determine_calls[0]["expected_task_count"] == 6
    assert outcome.winners == [BOSS]  # a lone decided task, won, is one win of six

    four = Scenario(monkeypatch)
    _add_wins(four, freeform=2, fixed=2)
    assert (await four.run()).winners == [BOSS]  # four attached cannot reach five wins

    five = Scenario(monkeypatch)
    _add_wins(five, freeform=3, fixed=2)
    assert (await five.run()).winners == [CHALLENGER]  # five attached must all be won


# --- _resolve_boss_round_task_winner: the reversed paired bar on the aligned vectors --------------------------


def _install_vectors(monkeypatch, boss, challenger):
    vectors = {BOSS: (list(boss), "fp"), CHALLENGER: (None if challenger is None else list(challenger), "fp")}
    monkeypatch.setattr(round_results, "get_per_example_losses", AsyncMock(side_effect=lambda _t, hk, _db: vectors[hk]))
    monkeypatch.setattr(performance_calculator, "get_per_example_losses", AsyncMock(side_effect=lambda _t, hk, _db: vectors[hk]))
    aligned = AsyncMock(side_effect=lambda _tt, _tid, _a, _o, losses, _db, _sp=None: losses)
    monkeypatch.setattr(round_results, "sku_aligned_losses", aligned)
    monkeypatch.setattr(performance_calculator, "sku_aligned_losses", aligned)


async def _verdict(task_type, boss_mean, challenger_mean):
    return await round_results._resolve_boss_round_task_winner(
        "task",
        SimpleNamespace(task_type=task_type, training_start_point=None),
        BOSS,
        CHALLENGER,
        boss_mean,
        challenger_mean,
        0.01,
        None,
    )


async def test_a_paired_loss_with_a_clear_boss_win_is_not_within(monkeypatch):
    boss = _vectors()
    challenger = boss + 0.1
    _install_vectors(monkeypatch, boss, challenger)

    verdict = await _verdict(FIXED, boss.mean(), challenger.mean())

    assert verdict.winner_hotkey == BOSS and verdict.challenger_within_allowed_loss is False


async def test_a_paired_loss_without_a_significant_boss_win_is_within(monkeypatch):
    boss = _vectors()
    challenger = boss + np.random.default_rng(8).normal(0.0, 0.3, boss.size)
    _install_vectors(monkeypatch, boss, challenger)

    verdict = await _verdict(FIXED, boss.mean(), challenger.mean() + 0.1)  # scalar says the boss; the vectors cannot tell

    assert verdict.winner_hotkey == BOSS and verdict.challenger_within_allowed_loss is True


async def test_missing_vectors_are_not_within(monkeypatch):
    boss = _vectors()
    _install_vectors(monkeypatch, boss, None)

    verdict = await _verdict(FIXED, 1.0, 1.1)

    assert verdict.winner_hotkey == BOSS and verdict.challenger_within_allowed_loss is False


async def test_unpairable_vectors_are_not_within(monkeypatch):
    boss = _vectors()
    _install_vectors(monkeypatch, boss, boss[:-1])

    verdict = await _verdict(FIXED, 1.0, 1.1)

    assert verdict.challenger_within_allowed_loss is False


async def test_no_same_sku_reference_is_not_within(monkeypatch):
    boss = _vectors()
    _install_vectors(monkeypatch, boss, boss + 0.1)
    unavailable = AsyncMock(side_effect=QuantReferenceUnavailable("no reference"))
    monkeypatch.setattr(round_results, "sku_aligned_losses", unavailable)

    verdict = await _verdict(FIXED, 1.0, 1.1)

    assert verdict.challenger_within_allowed_loss is False


async def test_fixed_uses_the_sku_aligned_vectors_for_the_swap(monkeypatch):
    boss = _vectors()
    raw_challenger = boss + 0.1  # on another SKU this looks clearly worse
    aligned_challenger = boss + np.random.default_rng(9).normal(0.0, 0.3, boss.size)  # the same model once aligned
    _install_vectors(monkeypatch, boss, raw_challenger)
    monkeypatch.setattr(round_results, "sku_aligned_losses", AsyncMock(return_value=list(aligned_challenger)))

    verdict = await _verdict(FIXED, boss.mean(), boss.mean() + 0.1)

    assert verdict.winner_hotkey == BOSS and verdict.challenger_within_allowed_loss is True


# --- analytics agrees with crowning ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task_type,noise,expected_won,expected_within",
    [
        (FIXED, -0.1, True, None),  # challenger clearly better: won
        (FIXED, 0.0, False, True),  # identical: a draw is a within miss
        (FIXED, 0.1, False, False),  # boss clearly better: out
    ],
)
async def test_analytics_within_matches_crowning_on_paired_tasks(monkeypatch, task_type, noise, expected_won, expected_within):
    boss = _vectors()
    challenger = boss + noise
    _install_vectors(monkeypatch, boss, challenger)
    boss_result, challenger_result = _text(BOSS, boss.mean(), task_type), _text(CHALLENGER, challenger.mean(), task_type)

    analytics = await performance_calculator._quant_challenger_won_task(
        "task",
        SimpleNamespace(task_type=task_type, training_start_point=None),
        CHALLENGER,
        boss.mean(),
        challenger.mean(),
        0.01,
        None,
        boss_result=boss_result,
        challenger_result=challenger_result,
    )
    verdict = await _verdict(task_type, boss.mean(), challenger.mean())

    assert analytics.challenger_won == (verdict.winner_hotkey == CHALLENGER)
    assert analytics.challenger_won is expected_won
    if expected_won:
        assert analytics.within_allowed_loss is None
    else:
        assert analytics.within_allowed_loss == verdict.challenger_within_allowed_loss == expected_within


async def test_analytics_within_for_a_within_case_matches_crowning(monkeypatch):
    boss = _vectors()
    challenger = boss + np.random.default_rng(8).normal(0.0, 0.3, boss.size)
    _install_vectors(monkeypatch, boss, challenger)
    boss_scalar, challenger_scalar = boss.mean(), challenger.mean() + 0.1

    analytics = await performance_calculator._quant_challenger_won_task(
        "task",
        SimpleNamespace(task_type=FIXED, training_start_point=None),
        CHALLENGER,
        boss_scalar,
        challenger_scalar,
        0.01,
        None,
        boss_result=_text(BOSS, boss_scalar),
        challenger_result=_text(CHALLENGER, challenger_scalar),
    )
    verdict = await _verdict(FIXED, boss_scalar, challenger_scalar)

    assert analytics.within_allowed_loss is True and verdict.challenger_within_allowed_loss is True


@pytest.mark.parametrize(
    "challenger_bytes,expected_won,expected_within",
    [(9_700, True, None), (10_050, False, True), (10_500, False, False)],
)
async def test_analytics_freeform_uses_effective_bytes_like_crowning(challenger_bytes, expected_won, expected_within):
    boss_result, challenger_result = _freeform(BOSS, 10_000), _freeform(CHALLENGER, challenger_bytes)

    analytics = await performance_calculator._quant_challenger_won_task(
        "task",
        SimpleNamespace(task_type=FREEFORM, training_start_point=None),
        CHALLENGER,
        math.inf,  # adjusted_loss of a freeform scored fail is inf; the byte verdict must not read it
        1.0,
        0.01,
        None,
        boss_result=boss_result,
        challenger_result=challenger_result,
    )
    verdict = round_results.resolve_freeform_boss_task(boss_result, challenger_result, t_cst.QUANT_FREEFORM_BOSS_BYTE_MARGIN)

    assert analytics.challenger_won is expected_won is (verdict.winner_hotkey == CHALLENGER)
    assert analytics.within_allowed_loss is expected_within


def test_task_performance_difference_carries_the_field():
    diff = TaskPerformanceDifference(
        task_id="a", task_type="x", boss_score=1.0, challenger_score=1.0, threshold_used=0.01,
        performance_difference=0.0, challenger_won=False, within_allowed_loss=True,
    )
    assert diff.within_allowed_loss is True


async def test_analytics_failed_challenger_score_is_not_within(monkeypatch):
    """The 'challenger failed evaluation' path agrees with crowning: a failed score is never within."""
    tournament = TournamentData(tournament_id="t", tournament_type=TournamentType.QUANT)
    task_row = SimpleNamespace(task_id="task-1")
    for name, value in (
        ("get_tournament", tournament),
        ("get_final_round_id", "round"),
        ("get_tournament_tasks", [task_row]),
        ("get_task", SimpleNamespace(task_type=FIXED, training_start_point=None)),
        ("get_task_results_for_ranking", [SimpleNamespace()]),
    ):
        monkeypatch.setattr(performance_calculator, name, AsyncMock(return_value=value))
    monkeypatch.setattr(
        performance_calculator,
        "calculate_miner_ranking_and_scores",
        lambda _results: [_text(BOSS, 1.0, FIXED), _text(CHALLENGER, None, FIXED)],
    )

    (diff,) = await performance_calculator.calculate_boss_round_performance_differences("t", None)

    assert diff.challenger_won is False and diff.challenger_score is None
    assert diff.within_allowed_loss is False


async def test_a_boss_side_seed_failure_is_the_challengers_slot(monkeypatch, caplog):
    scenario = Scenario(monkeypatch)
    scenario.dead_seeds.add(scenario.add(FIXED, None))  # first in the round, so the challenger is found from the nodes
    _add_wins(scenario, freeform=3, fixed=1)
    scenario.add(FREEFORM, [_freeform(BOSS, 10_000), _freeform(CHALLENGER, 10_050)])

    with caplog.at_level("ERROR"):
        outcome = await scenario.run()

    assert outcome.winners == [CHALLENGER]  # the dead slot is the fifth win
    assert "boss-side seed boss_chained_fixed failed prep after attach" in caplog.text


async def test_a_boss_side_seed_failure_persists_the_challengers_score_row(monkeypatch):
    scenario = Scenario(monkeypatch)
    dead_id = scenario.add(FIXED, None)
    scenario.dead_seeds.add(dead_id)
    scenario.add(FREEFORM, [_freeform(BOSS, 10_000), _freeform(CHALLENGER, 9_000)])

    await scenario.run()

    (call,) = [c for c in scenario.score_updates.await_args_list if c.kwargs["task_id"] == dead_id]
    assert call.kwargs["winner_hotkey"] == CHALLENGER
    assert call.kwargs["compared_hotkeys"] == [BOSS, CHALLENGER]
    assert "boss-side seed failure" in call.kwargs["basis"]
