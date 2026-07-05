"""Cumulative-alpha projections must integrate the piecewise decay curve.

The old trapezoid between day 0 and the horizon kept accruing alpha long after
the champion's weight hit the curve's zero-cliff (day 40), overstating 90/180
day totals several-fold.
"""

from unittest.mock import patch

import pytest

import validator.scoring.constants as cts
import validator.tournament.constants as t_cst
from validator.scoring.weights import emission_time_retention
from validator.tournament.models import TournamentType
from validator.tournament.performance_utils import calculate_tournament_projection


DECAY_ZERO_DAY = cts.EMISSION_TIME_DECAY_CURVE[-1][0]


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
async def test_no_alpha_accrues_after_decay_cliff():
    projection = await project()
    by_days = {p.days: p for p in projection.projections}

    assert by_days[90].weight == 0.0
    assert by_days[180].weight == 0.0
    # Weight is zero from day 40 on, so the 90 and 180 day totals must be equal.
    assert by_days[90].total_alpha == pytest.approx(by_days[180].total_alpha)
    assert by_days[90].total_alpha > by_days[30].total_alpha


@pytest.mark.asyncio
async def test_total_alpha_matches_curve_integral():
    projection = await project()
    by_days = {p.days: p for p in projection.projections}

    initial_weight = projection.initial_weight
    # Analytic area under the retention curve up to the zero-cliff, in weight-days.
    curve = cts.EMISSION_TIME_DECAY_CURVE
    area = sum((d1 - d0) * (r0 + r1) / 2.0 for (d0, r0), (d1, r1) in zip(curve, curve[1:]))
    expected = initial_weight * area * cts.DAILY_ALPHA_TO_MINERS

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
