
import pytest

import validator.scoring.constants as cts
from core.models.tournament_models import TournamentType
from validator.scoring.emission_balance import compute_dynamic_base_weights


TEXT = TournamentType.TEXT
QUANT = TournamentType.QUANT
IMAGE = TournamentType.IMAGE

ENTRANT_SCENARIOS = [
    (13, 21.75),
    (15, 15),
    (30, 15),
    (100, 15),
    (15, 100),
    (3, 1000),
    (1000, 3),
    (None, 15),
    (15, None),
    (None, None),
]


@pytest.mark.parametrize("text_entrants,image_entrants", ENTRANT_SCENARIOS)
def test_bases_stay_within_minimum_and_minimum_plus_flex(text_entrants, image_entrants):
    weights = compute_dynamic_base_weights({TEXT: text_entrants, QUANT: 15, IMAGE: image_entrants})

    assert weights[TEXT] + weights[QUANT] + weights[IMAGE] == pytest.approx(1.0)
    assert cts.TOURNAMENT_TEXT_WEIGHT <= weights[TEXT] <= cts.MAX_TEXT_TOURNAMENT_WEIGHT
    assert cts.TOURNAMENT_QUANT_WEIGHT <= weights[QUANT] <= cts.MAX_QUANT_TOURNAMENT_WEIGHT
    assert cts.TOURNAMENT_IMAGE_WEIGHT <= weights[IMAGE] <= cts.MAX_IMAGE_TOURNAMENT_WEIGHT


def test_equal_contender_counts_split_flex_equally():
    weights = compute_dynamic_base_weights({TEXT: 12, QUANT: 12, IMAGE: 12})

    third = cts.EMISSION_FLEX_POOL / 3
    assert weights[TEXT] == pytest.approx(0.25 + third)
    assert weights[QUANT] == pytest.approx(0.15 + third)
    assert weights[IMAGE] == pytest.approx(0.15 + third)


def test_flex_follows_raw_counts_not_the_minimums():
    # Image has more contenders than text, so it gets less flex, whatever the minimums are.
    weights = compute_dynamic_base_weights({TEXT: 10, QUANT: 20, IMAGE: 12})
    flex = {t: weights[t] - minimum for t, minimum in ((TEXT, 0.25), (QUANT, 0.15), (IMAGE, 0.15))}

    assert flex[TEXT] > flex[IMAGE] > flex[QUANT]
    assert flex[TEXT] / flex[IMAGE] == pytest.approx((12 / 10) ** cts.EMISSION_BALANCE_ALPHA)


def test_no_history_falls_back_to_an_equal_split():
    assert compute_dynamic_base_weights({TEXT: None, QUANT: None, IMAGE: None}) == pytest.approx(
        compute_dynamic_base_weights({TEXT: 7, QUANT: 7, IMAGE: 7})
    )


def test_a_type_with_no_history_gets_the_average_count():
    assert compute_dynamic_base_weights({TEXT: 10, QUANT: None, IMAGE: 20}) == pytest.approx(
        compute_dynamic_base_weights({TEXT: 10, QUANT: 15, IMAGE: 20})
    )


def test_starved_type_gains_flex():
    balanced = compute_dynamic_base_weights({TEXT: 100, QUANT: 60, IMAGE: 60})
    starved_text = compute_dynamic_base_weights({TEXT: 5, QUANT: 35, IMAGE: 25})
    crowded_text = compute_dynamic_base_weights({TEXT: 90, QUANT: 35, IMAGE: 25})

    assert starved_text[TEXT] > balanced[TEXT] > crowded_text[TEXT]
