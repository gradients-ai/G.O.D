"""Quant boss-round task resolution: fixed on the paired gate, freeform on the cap and effective bytes."""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from pydantic import BaseModel

import validator.tournament.performance_calculator as performance_calculator
import validator.tournament.round_results as round_results
from core.constants.environments import TrainingStartPoint
from core.models.quant_models import QuantEvalGpu
from core.models.task_models import TaskType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.tournament import constants as t_cst
from validator.tournament import thresholds
from validator.tournament.crown import HeadToHeadOutcome
from validator.tournament.models import TrainingStatus
from validator.tournament.thresholds import resolve_freeform_boss_task


BOSS = "5BossHotkey"
CHALLENGER = "5ChallengerHotkey"
TASK_ID = "11111111-2222-3333-4444-555555555555"
SXM = QuantEvalGpu(name="NVIDIA H100 80GB HBM3", sm_count=132, sku="H100 SXM")


class StubTask(BaseModel):
    task_type: TaskType
    training_start_point: TrainingStartPoint = TrainingStartPoint.DEFAULT


@pytest.fixture
def stub_vectors(monkeypatch):
    """Stub the per-example vector lookup, keyed by hotkey, and put both sides on one SKU."""

    def _stub(vectors: dict[str, tuple[list[float] | None, str | None]]) -> None:
        async def fake_get_per_example_losses(task_id, hotkey, psql_db):
            return vectors.get(hotkey, (None, None))

        monkeypatch.setattr(round_results, "get_per_example_losses", fake_get_per_example_losses)
        monkeypatch.setattr(thresholds, "get_task_node_eval_gpus", AsyncMock(return_value={BOSS: SXM, CHALLENGER: SXM}))

    return _stub


async def _resolve(task_type: TaskType, boss_loss: float, challenger_loss: float) -> str:
    verdict = await round_results._resolve_boss_round_task_winner(
        task_id=TASK_ID,
        task_object=StubTask(task_type=task_type),
        boss_hotkey=BOSS,
        opponent_hotkey=CHALLENGER,
        boss_loss=boss_loss,
        opponent_loss=challenger_loss,
        threshold_percentage=0.01,
        psql_db=None,
    )
    return verdict.winner_hotkey


def _paired(gap: float, n: int = 1000, seed: int = 0, noise: float = 0.0):
    """Boss and challenger vectors where the challenger is better by `gap` nats per example."""
    rng = np.random.default_rng(seed)
    boss = rng.gamma(2.0, 0.5, n)
    challenger = boss - gap + (rng.normal(0, noise, n) if noise else 0.0)
    return list(boss), list(challenger)


@pytest.mark.asyncio
async def test_all_in_dead_zone_is_a_draw(stub_vectors):
    """The defender holds the score row; the verdict is a draw, which is not a win for the challenger."""
    boss = np.random.default_rng(30).uniform(0.02, 0.03, 1000)
    challenger = boss - 0.002  # every gap inside the dead zone
    stub_vectors({BOSS: (list(boss), "fp"), CHALLENGER: (list(challenger), "fp")})

    verdict = await round_results._resolve_boss_round_task_winner(
        task_id=TASK_ID,
        task_object=StubTask(task_type=TaskType.QUANTFIXEDTASK),
        boss_hotkey=BOSS,
        opponent_hotkey=CHALLENGER,
        boss_loss=float(np.mean(boss)),
        opponent_loss=float(np.mean(challenger)),
        threshold_percentage=0.01,
        psql_db=None,
    )

    assert verdict.winner_hotkey == BOSS
    assert verdict.is_draw is True
    assert verdict.challenger_within_allowed_loss is True
    assert "draw" in verdict.decided_by.lower()



@pytest.mark.asyncio
async def test_fixed_quant_task_uses_the_paired_gate(stub_vectors, monkeypatch):
    boss, challenger = _paired(gap=0.08)
    stub_vectors({BOSS: (boss, "fp"), CHALLENGER: (challenger, "fp")})
    compared = []
    real_compare = round_results.compare_paired_losses

    def spy(boss_losses, challenger_losses):
        compared.append(len(boss_losses))
        return real_compare(boss_losses, challenger_losses)

    monkeypatch.setattr(round_results, "compare_paired_losses", spy)
    # Both scored on one SKU, so the vectors are compared as they are (test_quant_sku_paired_gates covers two).
    sxm = QuantEvalGpu(name="NVIDIA H100 80GB HBM3", sm_count=132, sku="H100 SXM")
    monkeypatch.setattr(thresholds, "get_task_node_eval_gpus", AsyncMock(return_value={BOSS: sxm, CHALLENGER: sxm}))

    assert await _resolve(TaskType.QUANTFIXEDTASK, float(np.mean(boss)), float(np.mean(challenger))) == CHALLENGER
    assert compared == [len(boss)]



MARGIN = t_cst.QUANT_FREEFORM_BOSS_BYTE_MARGIN


def _freeform(
    hotkey: str,
    total_bytes: int | None,
    passed_bar: bool = True,
    quant_failure: str | None = None,
    freeform_penalty: float | None = None,
):
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=math.nan if quant_failure else 1.0,
        synth_loss=1.0,
        is_finetune=True,
        task_type=TaskType.QUANTFREEFORMTASK,
        artifact_bytes=total_bytes,
        decoder_bytes=None if total_bytes is None else 0,
        passed_bar=passed_bar,
        freeform_penalty=freeform_penalty,
        quant_failure=quant_failure,
    )


def _boss(total_bytes: int | None = 2000, **kwargs):
    return _freeform(EMISSION_BURN_HOTKEY, total_bytes, **kwargs)


@pytest.mark.parametrize(
    "boss,challenger,expected",
    [
        pytest.param(_boss(), _freeform(CHALLENGER, 1940), CHALLENGER, id="3% smaller wins"),
        pytest.param(_boss(), _freeform(CHALLENGER, 1960), EMISSION_BURN_HOTKEY, id="2% smaller loses"),
        pytest.param(_boss(), _freeform(CHALLENGER, 2000), EMISSION_BURN_HOTKEY, id="equal bytes go to the boss"),
        pytest.param(_boss(passed_bar=False), _freeform(CHALLENGER, 9000), CHALLENGER, id="boss fails the bar"),
        pytest.param(None, _freeform(CHALLENGER, 9000), CHALLENGER, id="boss has no result"),
        pytest.param(_boss(), _freeform(CHALLENGER, 10, passed_bar=False), EMISSION_BURN_HOTKEY, id="challenger fails bar"),
        pytest.param(_boss(), _freeform(CHALLENGER, 10, quant_failure="did not load"), EMISSION_BURN_HOTKEY, id="scored fail"),
        pytest.param(_boss(), _freeform(CHALLENGER, None), EMISSION_BURN_HOTKEY, id="challenger bytes missing"),
        pytest.param(_boss(passed_bar=False), _freeform(CHALLENGER, 10, passed_bar=False), EMISSION_BURN_HOTKEY, id="both fail"),
        pytest.param(_boss(), None, EMISSION_BURN_HOTKEY, id="no challenger result"),
        pytest.param(_boss(), _freeform(CHALLENGER, 1940), CHALLENGER, id="exactly 0.97 of S wins"),
        pytest.param(_boss(), _freeform(CHALLENGER, 1941), EMISSION_BURN_HOTKEY, id="just above 0.97 of S loses"),
        pytest.param(
            _boss(), _freeform(CHALLENGER, 1960, freeform_penalty=0.09), EMISSION_BURN_HOTKEY, id="2% fewer bytes but 0.006 worse"
        ),
        pytest.param(
            _boss(freeform_penalty=0.09), _freeform(CHALLENGER, 2000), CHALLENGER, id="equal bytes, better quality wins"
        ),
        pytest.param(_boss(None), _freeform(CHALLENGER, 10), EMISSION_BURN_HOTKEY, id="boss bytes missing"),
    ],
)
def test_freeform_boss_task_is_decided_on_bar_then_bytes(boss, challenger, expected):
    verdict = resolve_freeform_boss_task(boss, challenger, MARGIN)

    assert verdict.winner_hotkey == expected
    assert verdict.is_draw is False
    assert verdict.decided_by


def test_freeform_byte_verdict_names_the_rule():
    verdict = resolve_freeform_boss_task(_boss(), _freeform(CHALLENGER, 1234), MARGIN)

    assert verdict.decided_by == f"effective bytes 1234 vs 2000 ({MARGIN * 100:g}% margin)"


def test_freeform_verdict_reports_effective_bytes_with_the_penalty():
    verdict = resolve_freeform_boss_task(_boss(freeform_penalty=0.09), _freeform(CHALLENGER, 2000), MARGIN)

    assert verdict.decided_by == f"effective bytes 2000 vs 2188 ({MARGIN * 100:g}% margin)"


def test_freeform_cap_verdicts_name_the_rule():
    over_cap = resolve_freeform_boss_task(_boss(), _freeform(CHALLENGER, 10, passed_bar=False), MARGIN)
    boss_out = resolve_freeform_boss_task(_boss(passed_bar=False), _freeform(CHALLENGER, 9000), MARGIN)
    no_bytes = resolve_freeform_boss_task(_boss(), _freeform(CHALLENGER, None), MARGIN)

    assert over_cap.decided_by == "cap: challenger did not pass"
    assert boss_out.decided_by == "cap: boss failed, challenger passed"
    assert no_bytes.decided_by == "bytes: missing byte count"


@pytest.mark.asyncio
async def test_freeform_boss_task_skips_the_loss_path(monkeypatch):
    """A scored-fail challenger has a NaN loss but is still a result: it is beyond both allowances on the cap."""
    results = [_boss(), _freeform(CHALLENGER, 10, quant_failure="did not load")]

    async def fail_loss_path(*_args, **_kwargs):
        raise AssertionError("freeform must not reach the paired comparison")

    monkeypatch.setattr(round_results, "get_task_results_for_ranking", AsyncMock(return_value=results))
    monkeypatch.setattr(round_results, "calculate_miner_ranking_and_scores", lambda ranked: ranked)
    monkeypatch.setattr(round_results, "get_training_status_for_task_and_hotkeys", AsyncMock(return_value={}))
    monkeypatch.setattr(round_results, "get_per_example_losses", fail_loss_path)
    task_object = SimpleNamespace(task_type=TaskType.QUANTFREEFORMTASK, training_start_point=None)

    result = await round_results.quant_task_head_to_head(TASK_ID, "x", task_object, CHALLENGER, None)

    assert result.outcome == HeadToHeadOutcome.BEYOND and result.reason == "no valid challenger entry"


def _quant_result(hotkey: str, loss: float | None, task_type: TaskType) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey, test_loss=1.0, synth_loss=1.0, is_finetune=True, task_type=task_type, adjusted_loss=loss
    )


async def _boss_round_slot(monkeypatch, task_type, results, training_statuses):
    """One boss-round slot decided against the boss from its stored results (no per-example vectors)."""
    monkeypatch.setattr(round_results, "get_task_results_for_ranking", AsyncMock(return_value=results))
    monkeypatch.setattr(round_results, "calculate_miner_ranking_and_scores", lambda ranked: ranked)
    monkeypatch.setattr(round_results, "get_training_status_for_task_and_hotkeys", AsyncMock(return_value=training_statuses))
    monkeypatch.setattr(round_results, "get_per_example_losses", AsyncMock(return_value=(None, None)))
    task_object = SimpleNamespace(task_type=task_type, training_start_point=None)
    return await round_results.quant_task_head_to_head(TASK_ID, "x", task_object, CHALLENGER, None)


SUCCESS, FAILURE = TrainingStatus.SUCCESS, TrainingStatus.FAILURE


@pytest.mark.asyncio
async def test_the_challenger_takes_a_fixed_compression_whose_boss_entry_failed_training(monkeypatch):
    results = [_quant_result(CHALLENGER, 0.9, TaskType.QUANTFIXEDTASK)]

    statuses = {EMISSION_BURN_HOTKEY: FAILURE, CHALLENGER: SUCCESS}

    result = await _boss_round_slot(monkeypatch, TaskType.QUANTFIXEDTASK, results, statuses)

    assert result.is_win


@pytest.mark.asyncio
async def test_a_missing_challenger_score_is_a_boss_win(monkeypatch):
    fixed = TaskType.QUANTFIXEDTASK
    results = [_quant_result(EMISSION_BURN_HOTKEY, 0.9, fixed), _quant_result(CHALLENGER, None, fixed)]
    statuses = {EMISSION_BURN_HOTKEY: SUCCESS, CHALLENGER: SUCCESS}

    result = await _boss_round_slot(monkeypatch, fixed, results, statuses)

    assert result.outcome == HeadToHeadOutcome.BEYOND


# A scored fail carries adjusted_loss = inf and has no vector. It must never turn into nan, and a valid challenger takes it.
@pytest.mark.asyncio
async def test_valid_challenger_takes_a_task_the_boss_scored_fail_on(stub_vectors):
    stub_vectors({})
    assert await _resolve(TaskType.QUANTFIXEDTASK, math.inf, 1.5) == CHALLENGER


@pytest.mark.asyncio
async def test_scored_fail_challenger_never_beats_a_valid_boss(stub_vectors):
    stub_vectors({})
    assert await _resolve(TaskType.QUANTFIXEDTASK, 1.5, math.inf) == BOSS
    assert await _resolve(TaskType.QUANTFIXEDTASK, 1.5, math.nan) == BOSS


@pytest.mark.asyncio
async def test_both_scored_fail_stays_with_the_boss(stub_vectors):
    stub_vectors({})
    assert await _resolve(TaskType.QUANTFIXEDTASK, math.inf, math.inf) == BOSS


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_quant_challenger_beats_boss_guards_non_finite_scores(bad):
    assert thresholds.quant_challenger_beats_boss(bad, 1.0, 0.01) is True
    assert thresholds.quant_challenger_beats_boss(1.0, bad, 0.01) is False
    assert thresholds.quant_challenger_beats_boss(bad, bad, 0.01) is False


def test_freeform_boss_scored_fail_loses_the_task_to_a_passing_challenger():
    boss = _boss(quant_failure="did not load")
    assert resolve_freeform_boss_task(boss, _freeform(CHALLENGER, 9000), MARGIN).winner_hotkey == CHALLENGER
    assert resolve_freeform_boss_task(_boss(), _freeform(CHALLENGER, 10, quant_failure="x"), MARGIN).winner_hotkey == (
        EMISSION_BURN_HOTKEY
    )


# --- the fixed boss task: the paired test AND a scalar loss QUANT_FIXED_BOSS_MIN_IMPROVEMENT better -------------------


async def _fixed_verdict(boss_loss: float, challenger_loss: float):
    return await round_results._resolve_boss_round_task_winner(
        task_id=TASK_ID,
        task_object=StubTask(task_type=TaskType.QUANTFIXEDTASK),
        boss_hotkey=BOSS,
        opponent_hotkey=CHALLENGER,
        boss_loss=boss_loss,
        opponent_loss=challenger_loss,
        threshold_percentage=0.01,
        psql_db=None,
    )


def test_the_fixed_minimum_improvement_is_one_percent_and_matches_the_boss_round_margin():
    assert t_cst.QUANT_FIXED_BOSS_MIN_IMPROVEMENT == 0.01
    assert t_cst.BOSS_ROUND_WIN_MARGIN == t_cst.QUANT_FIXED_BOSS_MIN_IMPROVEMENT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "challenger_loss,wins",
    [
        (0.90, True),  # 10% better
        (0.99, True),  # exactly 1% better
        (0.9901, False),  # 0.99% better
        (0.995, False),  # 0.5% better
        (1.0, False),  # level
    ],
)
async def test_a_paired_win_also_needs_one_percent_on_the_scalar(stub_vectors, challenger_loss, wins):
    boss, challenger = _paired(gap=0.08)  # the paired test is won clearly in every row
    stub_vectors({BOSS: (boss, "fp"), CHALLENGER: (challenger, "fp")})

    verdict = await _fixed_verdict(1.0, challenger_loss)

    assert (verdict.winner_hotkey == CHALLENGER) is wins
    assert verdict.is_draw is False


@pytest.mark.asyncio
async def test_a_paired_win_that_misses_the_scalar_says_so_and_by_how_much(stub_vectors):
    boss, challenger = _paired(gap=0.08)
    stub_vectors({BOSS: (boss, "fp"), CHALLENGER: (challenger, "fp")})

    verdict = await _fixed_verdict(1.0, 0.995)

    assert verdict.winner_hotkey == BOSS
    assert "0.50%" in verdict.decided_by and "1% minimum" in verdict.decided_by


@pytest.mark.asyncio
async def test_a_paired_loss_is_the_bosss_however_good_the_scalar(stub_vectors):
    boss, challenger = _paired(gap=-0.08)  # the challenger is clearly worse per example
    stub_vectors({BOSS: (boss, "fp"), CHALLENGER: (challenger, "fp")})

    assert (await _fixed_verdict(1.0, 0.95)).winner_hotkey == BOSS  # a 5% better scalar does not rescue it


@pytest.mark.asyncio
async def test_a_paired_draw_is_the_bosss_and_never_a_win_however_good_the_scalar(stub_vectors):
    boss = np.random.default_rng(31).uniform(0.02, 0.03, 1000)
    stub_vectors({BOSS: (list(boss), "fp"), CHALLENGER: (list(boss - 0.002), "fp")})

    verdict = await _fixed_verdict(1.0, 0.9)

    assert verdict.winner_hotkey == BOSS and verdict.is_draw is True


@pytest.mark.asyncio
@pytest.mark.parametrize("challenger_loss", [0.5, 0.99, 0.9901])
@pytest.mark.parametrize(
    "vectors",
    [
        {},  # neither side
        {BOSS: ([0.1] * 10, "fp")},  # only the boss
        {CHALLENGER: ([0.1] * 10, "fp")},  # only the challenger
    ],
)
async def test_without_vectors_on_both_sides_the_boss_keeps_the_task_however_good_the_scalar(
    stub_vectors, vectors, challenger_loss
):
    stub_vectors(vectors)

    verdict = await _fixed_verdict(1.0, challenger_loss)

    assert verdict.winner_hotkey == BOSS and verdict.decided_by == "no per-example losses to pair"
    assert verdict.is_draw is False


@pytest.mark.asyncio
@pytest.mark.parametrize("vectors", [{}, {BOSS: ([0.1] * 10, "fp")}, {CHALLENGER: ([0.1] * 10, "fp")}])
async def test_the_analytics_mirror_crowning_when_a_vector_is_absent(stub_vectors, monkeypatch, vectors):
    stub_vectors(vectors)
    monkeypatch.setattr(
        performance_calculator, "get_per_example_losses", AsyncMock(side_effect=lambda _t, hk, _db: vectors.get(hk, (None, None)))
    )

    analytics = await performance_calculator._quant_challenger_won_task(
        TASK_ID, StubTask(task_type=TaskType.QUANTFIXEDTASK), CHALLENGER, 1.0, 0.5, 0.01, None
    )
    verdict = await _fixed_verdict(1.0, 0.5)

    assert analytics.challenger_won is False and verdict.winner_hotkey == BOSS


@pytest.mark.asyncio
async def test_a_non_finite_boss_scalar_satisfies_the_improvement_but_the_paired_test_is_still_needed(stub_vectors):
    boss, challenger = _paired(gap=0.08)
    stub_vectors({BOSS: (boss, "fp"), CHALLENGER: (challenger, "fp")})
    assert (await _fixed_verdict(math.inf, 1.5)).winner_hotkey == CHALLENGER

    stub_vectors({BOSS: (boss, "fp"), CHALLENGER: (boss, "fp")})  # identical vectors: a draw
    assert (await _fixed_verdict(math.inf, 1.5)).winner_hotkey == BOSS


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [math.inf, math.nan])
async def test_a_non_finite_challenger_scalar_never_wins_a_paired_task(stub_vectors, bad):
    boss, challenger = _paired(gap=0.08)
    stub_vectors({BOSS: (boss, "fp"), CHALLENGER: (challenger, "fp")})

    assert (await _fixed_verdict(1.0, bad)).winner_hotkey == BOSS
