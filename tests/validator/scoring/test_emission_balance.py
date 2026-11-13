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


def test_balanced_participation_splits_flex_at_minimums_ratio():
    # 60:140:90 entrants matches the minimums' ratio (0.15:0.35:0.225), so flex splits 60/290, 140/290, 90/290 of 0.275.
    weights = compute_dynamic_base_weights({TEXT: 60, QUANT: 140, IMAGE: 90})

    flex = cts.EMISSION_FLEX_POOL
    assert weights[TEXT] == pytest.approx(0.15 + flex * 60 / 290)
    assert weights[QUANT] == pytest.approx(0.35 + flex * 140 / 290)
    assert weights[IMAGE] == pytest.approx(0.225 + flex * 90 / 290)


def test_no_history_falls_back_to_balanced_split():
    assert compute_dynamic_base_weights({TEXT: None, QUANT: None, IMAGE: None}) == pytest.approx(
        compute_dynamic_base_weights({TEXT: 60, QUANT: 140, IMAGE: 90})
    )


def test_starved_type_gains_flex():
    balanced = compute_dynamic_base_weights({TEXT: 60, QUANT: 140, IMAGE: 90})
    starved_text = compute_dynamic_base_weights({TEXT: 5, QUANT: 35, IMAGE: 25})
    crowded_text = compute_dynamic_base_weights({TEXT: 90, QUANT: 35, IMAGE: 25})

    assert starved_text[TEXT] > balanced[TEXT] > crowded_text[TEXT]
