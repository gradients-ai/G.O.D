"""Compression's part of the flex pool phases in: scaled by the ramp, the rest to text and image by their flex parts."""

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from itertools import product
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import validator.scoring.constants as cts
import validator.scoring.emission_balance as emission_balance
from core.models.tournament_models import TournamentType
from validator.scoring.emission_balance import compute_dynamic_base_weights
from validator.scoring.emission_balance import quant_ramp_factor
from validator.scoring.weights import get_tournament_burn_details


TEXT = TournamentType.TEXT
QUANT = TournamentType.QUANT
IMAGE = TournamentType.IMAGE
START = datetime(2026, 12, 1, tzinfo=timezone.utc)
DAYS = cts.QUANT_EMISSION_RAMP_DAYS
MINIMUM = {TEXT: cts.TOURNAMENT_TEXT_WEIGHT, QUANT: cts.TOURNAMENT_QUANT_WEIGHT, IMAGE: cts.TOURNAMENT_IMAGE_WEIGHT}
MAXIMUM = {TEXT: cts.MAX_TEXT_TOURNAMENT_WEIGHT, QUANT: cts.MAX_QUANT_TOURNAMENT_WEIGHT, IMAGE: cts.MAX_IMAGE_TOURNAMENT_WEIGHT}
SAMPLE = {TEXT: 20, QUANT: 3, IMAGE: 10}
ENTRANTS = [None, 3, 40, 1000]
RAMPS = [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.fixture
def ramp_start(monkeypatch):
    monkeypatch.setattr(cts, "QUANT_EMISSION_RAMP_START", START)


def _at(ramp: float) -> datetime:
    return START + ramp * timedelta(days=DAYS)


def _flex(weights: dict[TournamentType, float]) -> dict[TournamentType, float]:
    return {t: weights[t] - MINIMUM[t] for t in weights}


def test_no_start_set_gives_quant_its_minimum_only(monkeypatch):
    monkeypatch.setattr(cts, "QUANT_EMISSION_RAMP_START", None)
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)

    assert quant_ramp_factor(now) == 0.0
    weights = compute_dynamic_base_weights(SAMPLE, now)
    assert weights[QUANT] == cts.TOURNAMENT_QUANT_WEIGHT
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)


def test_a_naive_start_is_rejected(monkeypatch):
    monkeypatch.setattr(cts, "QUANT_EMISSION_RAMP_START", datetime(2026, 12, 1))
    with pytest.raises(ValueError):
        quant_ramp_factor(START)


@pytest.mark.usefixtures("ramp_start")
@pytest.mark.parametrize(
    "now,expected",
    [
        (START - timedelta(days=365), 0.0),
        (START - timedelta(seconds=1), 0.0),
        (START, 0.0),
        (START + timedelta(days=7), 0.25),
        (START + timedelta(days=14), 0.5),
        (START + timedelta(days=DAYS), 1.0),
        (START + timedelta(days=365), 1.0),
    ],
)
def test_ramp_factor_is_linear_and_clamped(now, expected):
    assert quant_ramp_factor(now) == pytest.approx(expected)


@pytest.mark.usefixtures("ramp_start")
@pytest.mark.parametrize("ramp", RAMPS)
def test_quant_keeps_its_minimum_and_its_unearned_flex_goes_to_text_and_image_by_their_flex(ramp):
    full = _flex(compute_dynamic_base_weights(SAMPLE, _at(1.0)))
    weights = compute_dynamic_base_weights(SAMPLE, _at(ramp))
    flex = _flex(weights)
    unearned = (1.0 - ramp) * full[QUANT]

    assert weights[QUANT] == pytest.approx(cts.TOURNAMENT_QUANT_WEIGHT + ramp * full[QUANT])
    assert flex[TEXT] == pytest.approx(full[TEXT] + unearned * full[TEXT] / (full[TEXT] + full[IMAGE]))
    assert flex[IMAGE] == pytest.approx(full[IMAGE] + unearned * full[IMAGE] / (full[TEXT] + full[IMAGE]))
    assert flex[TEXT] / flex[IMAGE] == pytest.approx(full[TEXT] / full[IMAGE])


@pytest.mark.usefixtures("ramp_start")
@pytest.mark.parametrize("text,quant,image", list(product(ENTRANTS, repeat=3)))
def test_full_ramp_is_the_unramped_split_exactly(text, quant, image, monkeypatch):
    entrants = {TEXT: text, QUANT: quant, IMAGE: image}
    ramped = compute_dynamic_base_weights(entrants, _at(1.0))
    later = compute_dynamic_base_weights(entrants, START + timedelta(days=10 * DAYS))
    monkeypatch.setattr(emission_balance, "_phase_in_quant_flex", lambda desirability, ramp: desirability)

    assert ramped == later == compute_dynamic_base_weights(entrants, START)


@pytest.mark.usefixtures("ramp_start")
@pytest.mark.parametrize("ramp", RAMPS)
@pytest.mark.parametrize("text,quant,image", list(product(ENTRANTS + [0.5, 1e6], repeat=3)))
def test_bases_sum_to_one_and_stay_within_minimum_and_maximum(text, quant, image, ramp):
    weights = compute_dynamic_base_weights({TEXT: text, QUANT: quant, IMAGE: image}, _at(ramp))

    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)
    for tournament_type in (TEXT, QUANT, IMAGE):
        assert MINIMUM[tournament_type] - 1e-12 <= weights[tournament_type] <= MAXIMUM[tournament_type] + 1e-12


@pytest.mark.asyncio
@pytest.mark.usefixtures("ramp_start")
@pytest.mark.parametrize("ramp", [0.0, 0.5, 1.0])
async def test_burn_details_use_the_phased_pools(ramp):
    with (
        patch("validator.scoring.emission_balance.get_recent_completed_entrant_counts", AsyncMock(return_value=[])),
        patch("validator.scoring.emission_balance.datetime") as clock,
    ):
        clock.now.return_value = _at(ramp)
        burn = await get_tournament_burn_details(MagicMock())

    full = _flex(compute_dynamic_base_weights({TEXT: None, QUANT: None, IMAGE: None}, _at(1.0)))
    assert burn.quant_tournament_weight == pytest.approx(cts.TOURNAMENT_QUANT_WEIGHT + ramp * full[QUANT])
    assert burn.text_tournament_weight + burn.quant_tournament_weight + burn.image_tournament_weight == pytest.approx(1.0)
    for proportion in (burn.text_burn_proportion, burn.quant_burn_proportion, burn.image_burn_proportion):
        assert 0.0 <= proportion <= 1.0
