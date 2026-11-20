"""Compression's emission share phases in: scaled by the ramp, the rest to text and image by their minimums."""

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from itertools import product
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import validator.scoring.constants as cts
from core.models.tournament_models import TournamentType
import validator.scoring.emission_balance as emission_balance
from validator.scoring.emission_balance import compute_dynamic_base_weights
from validator.scoring.emission_balance import max_weight_by_type
from validator.scoring.emission_balance import quant_ramp_factor
from validator.scoring.weights import get_tournament_burn_details


TEXT = TournamentType.TEXT
QUANT = TournamentType.QUANT
IMAGE = TournamentType.IMAGE
START = datetime(2026, 12, 1, tzinfo=timezone.utc)
DAYS = cts.QUANT_EMISSION_RAMP_DAYS
BALANCED = {TEXT: 60, QUANT: 140, IMAGE: 100}
ENTRANTS = [None, 3, 40, 1000]
RAMPS = [0.0, 0.25, 0.5, 0.75, 1.0]
TEXT_SHARE_OF_UNEARNED = cts.TOURNAMENT_TEXT_WEIGHT / (cts.TOURNAMENT_TEXT_WEIGHT + cts.TOURNAMENT_IMAGE_WEIGHT)
IMAGE_SHARE_OF_UNEARNED = cts.TOURNAMENT_IMAGE_WEIGHT / (cts.TOURNAMENT_TEXT_WEIGHT + cts.TOURNAMENT_IMAGE_WEIGHT)


@pytest.fixture
def ramp_start(monkeypatch):
    monkeypatch.setattr(cts, "QUANT_EMISSION_RAMP_START", START)


def _at(ramp: float) -> datetime:
    return START + ramp * timedelta(days=DAYS)


def _full(entrants: dict[TournamentType, float | None]) -> dict[TournamentType, float]:
    return compute_dynamic_base_weights(entrants, START + timedelta(days=DAYS))


def test_no_start_set_means_no_quant_share(monkeypatch):
    monkeypatch.setattr(cts, "QUANT_EMISSION_RAMP_START", None)
    assert quant_ramp_factor(datetime(2030, 1, 1, tzinfo=timezone.utc)) == 0.0
    weights = compute_dynamic_base_weights(BALANCED, datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert weights[QUANT] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)


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
def test_at_zero_quant_has_nothing_and_text_and_image_split_the_rest_by_their_minimums():
    weights = compute_dynamic_base_weights(BALANCED, START)

    assert weights[QUANT] == 0.0
    assert weights[TEXT] + weights[IMAGE] == pytest.approx(1.0, abs=1e-12)
    # Balanced entrants: text and image already sit at their minimums' ratio, and the unearned share keeps it.
    assert weights[TEXT] / weights[IMAGE] == pytest.approx(cts.TOURNAMENT_TEXT_WEIGHT / cts.TOURNAMENT_IMAGE_WEIGHT)


@pytest.mark.usefixtures("ramp_start")
@pytest.mark.parametrize("ramp", RAMPS)
def test_unearned_quant_share_goes_to_text_and_image_by_their_minimums(ramp):
    entrants = {TEXT: 20, QUANT: 3, IMAGE: 10}
    full = _full(entrants)
    weights = compute_dynamic_base_weights(entrants, _at(ramp))
    unearned = (1.0 - ramp) * full[QUANT]

    assert weights[QUANT] == pytest.approx(ramp * full[QUANT])
    assert weights[TEXT] == pytest.approx(full[TEXT] + unearned * TEXT_SHARE_OF_UNEARNED)
    assert weights[IMAGE] == pytest.approx(full[IMAGE] + unearned * IMAGE_SHARE_OF_UNEARNED)


def test_a_naive_start_is_rejected(monkeypatch):
    monkeypatch.setattr(cts, "QUANT_EMISSION_RAMP_START", datetime(2026, 12, 1))
    with pytest.raises(ValueError):
        quant_ramp_factor(START)


@pytest.mark.usefixtures("ramp_start")
@pytest.mark.parametrize("text,quant,image", list(product(ENTRANTS, repeat=3)))
def test_full_ramp_is_the_unramped_split_exactly(text, quant, image, monkeypatch):
    entrants = {TEXT: text, QUANT: quant, IMAGE: image}
    ramped = compute_dynamic_base_weights(entrants, START + timedelta(days=DAYS))
    later = compute_dynamic_base_weights(entrants, START + timedelta(days=10 * DAYS))
    monkeypatch.setattr(emission_balance, "_phase_in_quant", lambda by_type, ramp: by_type)

    assert ramped == later == compute_dynamic_base_weights(entrants, START)


@pytest.mark.usefixtures("ramp_start")
@pytest.mark.parametrize("ramp", RAMPS)
@pytest.mark.parametrize("text,quant,image", list(product(ENTRANTS, repeat=3)))
def test_bases_sum_to_one_and_stay_within_their_phased_bounds(text, quant, image, ramp):
    now = _at(ramp)
    weights = compute_dynamic_base_weights({TEXT: text, QUANT: quant, IMAGE: image}, now)
    caps = max_weight_by_type(now)
    unearned_minimum = (1.0 - ramp) * cts.TOURNAMENT_QUANT_WEIGHT
    floors = {
        TEXT: cts.TOURNAMENT_TEXT_WEIGHT + unearned_minimum * TEXT_SHARE_OF_UNEARNED,
        QUANT: ramp * cts.TOURNAMENT_QUANT_WEIGHT,
        IMAGE: cts.TOURNAMENT_IMAGE_WEIGHT + unearned_minimum * IMAGE_SHARE_OF_UNEARNED,
    }

    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)
    for tournament_type in (TEXT, QUANT, IMAGE):
        assert floors[tournament_type] - 1e-12 <= weights[tournament_type] <= caps[tournament_type] + 1e-12


@pytest.mark.usefixtures("ramp_start")
def test_caps_phase_in_like_the_bases():
    assert max_weight_by_type(START + timedelta(days=DAYS)) == {
        TEXT: cts.MAX_TEXT_TOURNAMENT_WEIGHT,
        QUANT: cts.MAX_QUANT_TOURNAMENT_WEIGHT,
        IMAGE: cts.MAX_IMAGE_TOURNAMENT_WEIGHT,
    }
    caps = max_weight_by_type(_at(0.5))
    assert caps[QUANT] == pytest.approx(0.5 * cts.MAX_QUANT_TOURNAMENT_WEIGHT)
    assert caps[TEXT] == pytest.approx(cts.MAX_TEXT_TOURNAMENT_WEIGHT + 0.5 * cts.MAX_QUANT_TOURNAMENT_WEIGHT * TEXT_SHARE_OF_UNEARNED)
    assert caps[IMAGE] == pytest.approx(
        cts.MAX_IMAGE_TOURNAMENT_WEIGHT + 0.5 * cts.MAX_QUANT_TOURNAMENT_WEIGHT * IMAGE_SHARE_OF_UNEARNED
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("ramp_start")
@pytest.mark.parametrize("ramp", [0.0, 0.5, 1.0])
async def test_burn_details_use_the_phased_pools_and_caps(ramp):
    with patch("validator.scoring.emission_balance.get_recent_completed_entrant_counts", AsyncMock(return_value=[])):
        with patch("validator.scoring.weights.datetime") as clock:
            clock.now.return_value = _at(ramp)
            burn = await get_tournament_burn_details(MagicMock())

    full = _full({TEXT: None, QUANT: None, IMAGE: None})
    assert burn.quant_tournament_weight == pytest.approx(ramp * full[QUANT])
    assert burn.text_tournament_weight + burn.quant_tournament_weight + burn.image_tournament_weight == pytest.approx(1.0)
    for proportion in (burn.text_burn_proportion, burn.quant_burn_proportion, burn.image_burn_proportion):
        assert 0.0 <= proportion <= 1.0
    if ramp == 0.0:
        assert burn.quant_burn_proportion == 0.0
