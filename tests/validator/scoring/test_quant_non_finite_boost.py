"""A failed-but-scored quant entry has an infinite loss. It must never reach the emission boost, and the quant champion
gets no loss-ratio boost at all."""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import validator.scoring.constants as cts
import validator.tournament.performance_calculator as performance_calculator
from core.models.task_models import TaskType
from core.models.tournament_models import TournamentType
from validator.scoring.models import MinerResultsText
from validator.scoring.weights import calculate_emission_boost_from_perf
from validator.scoring.weights import calculate_tournament_weight
from validator.scoring.weights import get_tournament_burn_details
from validator.tournament.models import TaskPerformanceDifference
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentStatus


BOSS = cts.EMISSION_BURN_HOTKEY
CHALLENGER = "challenger_hotkey"
FIXED = TaskType.QUANTFIXEDTASK
TEXT, QUANT, IMAGE = TournamentType.TEXT, TournamentType.QUANT, TournamentType.IMAGE
# A fixed participation-driven split (sums to 1.0), so these tests do not depend on how the split is derived.
BASE = {TEXT: 0.3, QUANT: 0.4, IMAGE: 0.3}


# --- calculate_boss_round_performance_differences / calculate_performance_difference ---


def _result(hotkey: str, loss: float | None) -> MinerResultsText:
    return MinerResultsText(hotkey=hotkey, test_loss=1.0, synth_loss=1.0, is_finetune=True, task_type=FIXED, adjusted_loss=loss)


async def _boss_round_diffs(monkeypatch, boss_loss: float, challenger_loss: float, challenger_won: bool = True):
    for name, value in (
        ("get_tournament", TournamentData(tournament_id="t", tournament_type=QUANT)),
        ("get_final_round_id", "round"),
        ("get_tournament_tasks", [SimpleNamespace(task_id="task-1")]),
        ("get_task", SimpleNamespace(task_type=FIXED, training_start_point=None)),
        ("get_task_results_for_ranking", [SimpleNamespace()]),
    ):
        monkeypatch.setattr(performance_calculator, name, AsyncMock(return_value=value))
    monkeypatch.setattr(
        performance_calculator,
        "calculate_miner_ranking_and_scores",
        lambda _r: [_result(BOSS, boss_loss), _result(CHALLENGER, challenger_loss)],
    )
    analytics = SimpleNamespace(challenger_won=challenger_won, paired_comparison=None, reason="x", within_allowed_loss=None)
    monkeypatch.setattr(performance_calculator, "_quant_challenger_won_task", AsyncMock(return_value=analytics))
    return await performance_calculator.calculate_boss_round_performance_differences("t", None)


async def test_boss_inf_challenger_finite_has_no_performance_difference(monkeypatch):
    (diff,) = await _boss_round_diffs(monkeypatch, math.inf, 1.0, challenger_won=True)

    assert diff.performance_difference is None
    assert diff.challenger_won is True  # who won is decided as before


async def test_boss_finite_challenger_inf_has_no_performance_difference(monkeypatch):
    (diff,) = await _boss_round_diffs(monkeypatch, 1.0, math.inf, challenger_won=False)

    assert diff.performance_difference is None
    assert diff.challenger_won is False


async def test_finite_scores_keep_their_performance_difference(monkeypatch):
    (diff,) = await _boss_round_diffs(monkeypatch, 1.2, 1.0)

    assert diff.performance_difference == pytest.approx(0.2)


def _row(perf: float | None) -> TaskPerformanceDifference:
    return TaskPerformanceDifference(
        task_id="a", task_type="x", boss_score=1.0, challenger_score=1.0, threshold_used=0.01,
        performance_difference=perf, challenger_won=True,
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([math.inf], 0.0),
        ([math.nan], 0.0),
        ([-math.inf, math.inf], 0.0),
        ([math.inf, 0.2, 0.4], 0.3),
        ([math.nan, None, 0.5], 0.5),
    ],
)
async def test_median_is_never_non_finite_and_is_stored_finite(monkeypatch, values, expected):
    monkeypatch.setattr(
        performance_calculator, "calculate_boss_round_performance_differences", AsyncMock(return_value=[_row(v) for v in values])
    )
    store = AsyncMock()
    monkeypatch.setattr(performance_calculator, "update_tournament_winning_performance", store)

    median = await performance_calculator.calculate_performance_difference("t", None)

    assert math.isfinite(median) and median == pytest.approx(expected)
    assert store.await_args.args[1] == median


async def test_non_finite_row_that_slips_past_the_row_guard_is_still_ignored(monkeypatch):
    # model_construct skips validation, standing in for any row built elsewhere with a non-finite number.
    bad = TaskPerformanceDifference.model_construct(**{**_row(None).model_dump(), "performance_difference": math.inf})
    monkeypatch.setattr(performance_calculator, "calculate_boss_round_performance_differences", AsyncMock(return_value=[bad]))
    monkeypatch.setattr(performance_calculator, "update_tournament_winning_performance", AsyncMock())

    assert await performance_calculator.calculate_performance_difference("t", None) == 0.0


# --- get_tournament_burn_details ---


def _completed(tournament_type: TournamentType, winner: str, base_winner: str = "champ") -> TournamentData:
    return TournamentData(
        tournament_id=f"{tournament_type.value}_1",
        tournament_type=tournament_type,
        status=TournamentStatus.COMPLETED,
        base_winner_hotkey=base_winner,
        winner_hotkey=winner,
    )


class _Env:
    """Fake DB layer for get_tournament_burn_details: the latest completed tournament per type, perf diffs per type."""

    def __init__(self, latest: dict, perf: dict, stored: dict | None = None):
        self.latest = latest
        self.perf = perf
        self.stored = stored or {}
        self.perf_calls: list[str] = []
        self.stored_calls: list[TournamentType] = []

    async def get_latest(self, _db, tournament_type, exclude_tournament_id=None):
        return None if exclude_tournament_id is not None else self.latest.get(tournament_type)

    async def calc_perf(self, tournament_id, _db):
        self.perf_calls.append(tournament_id)
        return self.perf[tournament_id]

    async def first_won(self, _db, tournament_type, _hotkey):
        self.stored_calls.append(tournament_type)
        return self.stored.get(tournament_type)

    async def run(self):
        with (
            patch("validator.scoring.weights.get_latest_completed_tournament", self.get_latest),
            patch("validator.scoring.weights.calculate_performance_difference", self.calc_perf),
            patch("validator.scoring.weights.get_tournament_where_champion_first_won", self.first_won),
            patch("validator.scoring.weights.get_dynamic_base_weights", AsyncMock(return_value=dict(BASE))),
        ):
            return await get_tournament_burn_details(MagicMock())


def _expected(text_perf: float | None, image_perf: float | None):
    """The text and image formula, base-only for quant, then the scale-down."""
    base = BASE
    max_by_type = {
        TEXT: cts.MAX_TEXT_TOURNAMENT_WEIGHT,
        QUANT: cts.MAX_QUANT_TOURNAMENT_WEIGHT,
        IMAGE: cts.MAX_IMAGE_TOURNAMENT_WEIGHT,
    }
    raw = {
        TEXT: min(base[TEXT] + calculate_emission_boost_from_perf(text_perf), max_by_type[TEXT]),
        QUANT: min(base[QUANT], max_by_type[QUANT]),
        IMAGE: min(base[IMAGE] + calculate_emission_boost_from_perf(image_perf), max_by_type[IMAGE]),
    }
    total = sum(raw.values())
    return base, {t: w / total for t, w in raw.items()}


def _assert_finite(burn):
    for value in (
        burn.text_tournament_weight, burn.quant_tournament_weight, burn.image_tournament_weight,
        burn.text_burn_proportion, burn.quant_burn_proportion, burn.image_burn_proportion,
    ):
        assert math.isfinite(value)


@pytest.mark.asyncio
async def test_new_quant_winner_gets_base_share_only_and_its_perf_is_not_even_computed():
    env = _Env(latest={QUANT: _completed(QUANT, "champ")}, perf={"quant_1": math.inf})

    burn = await env.run()

    base, expected = _expected(None, None)
    assert env.perf_calls == []
    assert burn.quant_performance_diff is None
    assert burn.quant_tournament_weight == pytest.approx(base[QUANT])  # no text/image boost, so no scale-down
    assert burn.quant_tournament_weight == pytest.approx(expected[QUANT])
    _assert_finite(burn)


@pytest.mark.asyncio
async def test_burn_hotkey_quant_winner_does_not_trigger_the_full_boost_fallback():
    env = _Env(latest={QUANT: _completed(QUANT, BOSS)}, perf={})

    burn = await env.run()

    base, _ = _expected(None, None)
    assert burn.quant_performance_diff is None  # the burn-winner fallback would have made this 1.0
    assert burn.quant_tournament_weight == pytest.approx(base[QUANT])
    _assert_finite(burn)


@pytest.mark.asyncio
async def test_defending_quant_champion_with_a_stored_infinite_difference_is_not_boosted():
    stored = SimpleNamespace(tournament_id="old", winning_performance_difference=math.inf)
    env = _Env(latest={QUANT: _completed(QUANT, BOSS)}, perf={}, stored={QUANT: stored})

    burn = await env.run()

    base, _ = _expected(None, None)
    assert QUANT not in env.stored_calls  # the stored value is not even read
    assert burn.quant_performance_diff is None
    assert burn.quant_tournament_weight == pytest.approx(base[QUANT])
    _assert_finite(burn)


@pytest.mark.asyncio
async def test_text_and_image_boosts_are_computed_as_on_main_next_to_an_unboosted_quant():
    env = _Env(
        latest={TEXT: _completed(TEXT, "t_champ"), QUANT: _completed(QUANT, "champ"), IMAGE: _completed(IMAGE, "i_champ")},
        perf={"text_1": 0.4, "quant_1": math.inf, "image_1": 0.2},
    )

    burn = await env.run()

    _, expected = _expected(0.4, 0.2)
    assert env.perf_calls == ["text_1", "image_1"]
    assert burn.text_performance_diff == 0.4 and burn.image_performance_diff == 0.2
    assert burn.quant_performance_diff is None
    assert burn.text_tournament_weight == pytest.approx(expected[TEXT])
    assert burn.quant_tournament_weight == pytest.approx(expected[QUANT])
    assert burn.image_tournament_weight == pytest.approx(expected[IMAGE])
    assert burn.text_tournament_weight + burn.quant_tournament_weight + burn.image_tournament_weight == pytest.approx(1.0)
    _assert_finite(burn)


@pytest.mark.asyncio
async def test_a_non_finite_pool_refuses_to_produce_weights():
    # A nan boost survives the cap (min(nan, max) is nan); an inf boost is capped to a finite pool by the existing rule.
    env = _Env(latest={TEXT: _completed(TEXT, "t_champ")}, perf={"text_1": math.nan})

    with pytest.raises(ValueError, match="Non-finite"):
        await env.run()


def test_quant_tournament_weight_ignores_a_boost():
    boosted = calculate_tournament_weight(QUANT, base_weight=0.4, emission_boost=math.inf, max_weight=0.5)

    assert boosted == pytest.approx(0.4)
    assert calculate_tournament_weight(TEXT, base_weight=0.2, emission_boost=0.1, max_weight=0.5) == pytest.approx(0.3)
