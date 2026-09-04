"""Weight projections have no time decay: a champion's weight is constant across the
projection horizon, so cumulative alpha accrues linearly rather than integrating a curve.
"""

import pytest

import validator.scoring.constants as cts
import validator.tournament.constants as t_cst
from validator.tournament.models import TournamentType
from validator.tournament.performance_utils import calculate_tournament_projection


async def project(percentage_improvement: float = 10.0):
    from unittest.mock import patch

    with (
        patch("validator.tournament.performance_utils.get_latest_completed_tournament", return_value=None),
        patch("validator.tournament.performance_utils.get_active_tournament_participants", return_value=[]),
    ):
        return await calculate_tournament_projection(
            psql_db=None,
            tournament_type=TournamentType.TEXT,
            percentage_improvement=percentage_improvement,
            base_weight=cts.TOURNAMENT_TEXT_WEIGHT,
            max_weight=cts.MAX_TEXT_TOURNAMENT_WEIGHT,
        )


@pytest.mark.asyncio
async def test_champion_weight_is_constant_across_days():
    projection = await project()
    by_days = {p.days: p for p in projection.projections}

    assert by_days[7].weight == pytest.approx(projection.initial_weight)
    assert by_days[30].weight == pytest.approx(projection.initial_weight)
    assert by_days[90].weight == pytest.approx(projection.initial_weight)
    assert by_days[180].weight == pytest.approx(projection.initial_weight)


@pytest.mark.asyncio
async def test_total_alpha_accrues_linearly():
    projection = await project()
    by_days = {p.days: p for p in projection.projections}

    expected_90 = 90 * cts.DAILY_ALPHA_TO_MINERS * projection.initial_weight
    expected_180 = 180 * cts.DAILY_ALPHA_TO_MINERS * projection.initial_weight
    assert by_days[90].total_alpha == pytest.approx(expected_90)
    assert by_days[180].total_alpha == pytest.approx(expected_180)
    # No decay floor to hold at: alpha keeps growing at the same flat rate throughout.
    assert by_days[180].total_alpha - by_days[90].total_alpha == pytest.approx(expected_180 - expected_90)


@pytest.mark.asyncio
async def test_current_champion_decay_is_always_zero():
    """The field is kept for API stability but no longer means anything - decay is gone."""
    projection = await project()
    assert projection.current_champion_decay == 0.0


@pytest.mark.asyncio
async def test_runner_up_earns_only_until_next_tournament():
    # 0.5% improvement is below the 1% dethrone margin: boss defends, challenger is runner-up.
    projection = await project(percentage_improvement=0.5)
    assert projection.placement == "runner_up"
    by_days = {p.days: p for p in projection.projections}

    cutoff_alpha = t_cst.RUNNER_UP_EMISSION_DAYS * cts.DAILY_ALPHA_TO_MINERS * projection.initial_weight
    assert by_days[7].weight == pytest.approx(projection.initial_weight)
    assert by_days[7].total_alpha == pytest.approx(cutoff_alpha)
    for days in (30, 90, 180):
        assert by_days[days].weight == 0.0
        assert by_days[days].total_alpha == pytest.approx(cutoff_alpha)
