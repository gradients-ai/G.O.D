"""QuantFreeformPenalty: the smooth size multiplier on a freeform entry's normalised loss."""

import math

import pytest
from pydantic import ValidationError

from core.models.quant_models import QuantFreeformPenalty


KNEE = 1.04
CURVATURE = 2500.0
CAP = 1.08
POLICY = QuantFreeformPenalty(knee=KNEE, curvature=CURVATURE, cap=CAP)


@pytest.mark.parametrize("r", [0.5, 0.99, 1.0, 1.039, KNEE])
def test_no_penalty_at_or_below_the_knee(r) -> None:
    assert POLICY.ln_penalty(r) == 0.0


def test_continuous_with_zero_slope_at_the_knee() -> None:
    assert POLICY.ln_penalty(KNEE + 1e-6) < 1e-8
    slopes = [(POLICY.ln_penalty(KNEE + h) - POLICY.ln_penalty(KNEE)) / h for h in (1e-2, 1e-3, 1e-4, 1e-5)]
    assert slopes == sorted(slopes, reverse=True)
    assert slopes[-1] < 0.1


def test_monotone_and_convex_above_the_knee() -> None:
    step = 0.004
    values = [POLICY.ln_penalty(KNEE + step * i) for i in range(1, 10)]
    assert all(b > a for a, b in zip(values, values[1:]))
    second_differences = [values[i + 1] - 2 * values[i] + values[i - 1] for i in range(1, len(values) - 1)]
    assert all(d > 0 for d in second_differences)


@pytest.mark.parametrize(
    ("excess", "multiplier"),
    [(0.001, 1.0025), (0.005, 1.0645), (0.010, 1.2840), (0.040, 54.598)],
)
def test_exact_multipliers(excess, multiplier) -> None:
    assert math.exp(POLICY.ln_penalty_under_cap(KNEE + excess)) == pytest.approx(multiplier, rel=1e-3)


def test_none_above_the_cap_and_defined_at_it() -> None:
    assert POLICY.ln_penalty(CAP + 1e-9) is None
    assert POLICY.ln_penalty(CAP) is not None


def test_cliff_is_zero_under_and_none_above() -> None:
    cliff = QuantFreeformPenalty.cliff(1.05)
    assert cliff.is_cliff
    assert cliff.ln_penalty(1.05) == 0.0
    assert cliff.ln_penalty(1.0) == 0.0
    assert cliff.ln_penalty(1.05 + 1e-9) is None
    assert not POLICY.is_cliff


def test_cap_below_knee_is_rejected() -> None:
    with pytest.raises(ValidationError):
        QuantFreeformPenalty(knee=1.05, curvature=100.0, cap=1.04)
