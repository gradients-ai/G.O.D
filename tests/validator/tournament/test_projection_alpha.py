"""Cumulative-alpha projections must integrate the piecewise decay curve.

A trapezoid between day 0 and the horizon would misstate the totals: the curve
decays steeply early, then holds at a floor from day 40 on, so alpha keeps
accruing past day 40 but at that flat floor rate.
"""

from unittest.mock import patch

import pytest

import validator.scoring.constants as cts
import validator.tournament.constants as t_cst
from validator.scoring.weights import emission_time_retention
from validator.tournament.models import TournamentType
from validator.tournament.performance_utils import calculate_tournament_projection


DECAY_FLOOR_DAY, DECAY_FLOOR_RETENTION = cts.EMISSION_TIME_DECAY_CURVE[-1]


async def project(percentage_improvement: float = 10.0):
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
async def test_weight_holds_at_floor_after_decay_settles():
    projection = await project()
    by_days = {p.days: p for p in projection.projections}

    floor_weight = projection.initial_weight * DECAY_FLOOR_RETENTION
    assert by_days[90].weight == pytest.approx(floor_weight)
    assert by_days[180].weight == pytest.approx(floor_weight)
    # Alpha keeps accruing past the floor day, at the flat floor rate.
    expected_extra = (180 - 90) * floor_weight * cts.DAILY_ALPHA_TO_MINERS
    assert by_days[180].total_alpha - by_days[90].total_alpha == pytest.approx(expected_extra, rel=1e-6)


@pytest.mark.asyncio
async def test_total_alpha_matches_curve_integral():
    projection = await project()
    by_days = {p.days: p for p in projection.projections}

    initial_weight = projection.initial_weight
    # Analytic area under the retention curve, in weight-days: the piecewise ramp
    # down to the floor, then a flat floor out to the horizon.
    curve = cts.EMISSION_TIME_DECAY_CURVE
    ramp_area = sum((d1 - d0) * (r0 + r1) / 2.0 for (d0, r0), (d1, r1) in zip(curve, curve[1:]))
    floor_area = (180 - DECAY_FLOOR_DAY) * DECAY_FLOOR_RETENTION
    expected = initial_weight * (ramp_area + floor_area) * cts.DAILY_ALPHA_TO_MINERS

    assert by_days[180].total_alpha == pytest.approx(expected, rel=1e-6)


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


@pytest.mark.asyncio
async def test_day7_weight_still_matches_retention_curve():
    projection = await project()
    by_days = {p.days: p for p in projection.projections}

    assert by_days[7].weight == pytest.approx(projection.initial_weight * emission_time_retention(7.0))
