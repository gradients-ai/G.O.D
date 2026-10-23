import pytest

import validator.scoring.constants as cts
from core.models.tournament_models import TournamentType
from validator.scoring.emission_balance import compute_dynamic_base_weights


TEXT = TournamentType.TEXT
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
    weights = compute_dynamic_base_weights({TEXT: text_entrants, IMAGE: image_entrants})

    assert weights[TEXT] + weights[IMAGE] == pytest.approx(1.0)
    assert cts.TOURNAMENT_TEXT_WEIGHT <= weights[TEXT] <= cts.MAX_TEXT_TOURNAMENT_WEIGHT
    assert cts.TOURNAMENT_IMAGE_WEIGHT <= weights[IMAGE] <= cts.MAX_IMAGE_TOURNAMENT_WEIGHT


def test_balanced_participation_splits_flex_at_minimums_ratio():
    # 50:30 entrants matches the minimums' ratio, so flex splits 62.5 / 37.5 of 0.20.
    weights = compute_dynamic_base_weights({TEXT: 50, IMAGE: 30})

    assert weights[TEXT] == pytest.approx(0.625)
    assert weights[IMAGE] == pytest.approx(0.375)


def test_no_history_falls_back_to_balanced_split():
    assert compute_dynamic_base_weights({TEXT: None, IMAGE: None}) == pytest.approx(
        compute_dynamic_base_weights({TEXT: 50, IMAGE: 30})
    )


def test_starved_type_gains_flex():
    starved_text = compute_dynamic_base_weights({TEXT: 10, IMAGE: 30})
    crowded_text = compute_dynamic_base_weights({TEXT: 90, IMAGE: 30})

    assert starved_text[TEXT] > 0.625 > crowded_text[TEXT]
