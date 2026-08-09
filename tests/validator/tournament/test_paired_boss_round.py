"""Tests for the paired per-example boss-round gate."""

import numpy as np
import pytest

import validator.tournament.constants as t_cst
from validator.tournament.thresholds import compare_paired_losses


def _losses(n: int = 1000, seed: int = 0) -> np.ndarray:
    """Plausible per-example losses: right-skewed, a few hard examples dominating."""
    return np.random.default_rng(seed).gamma(2.0, 0.5, n)


def test_identical_models_leave_everything_undecided():
    boss = _losses()
    result = compare_paired_losses(list(boss), list(boss))
    assert result.n_decided == 0
    assert result.challenger_won is False
    assert result.is_draw is True
    assert "Draw" in result.reason


def test_noise_without_a_real_edge_does_not_win():
    boss = _losses()
    challenger = boss + np.random.default_rng(1).normal(0, 0.05, boss.size)
    result = compare_paired_losses(list(boss), list(challenger))
    assert result.challenger_won is False
    assert 0.4 < result.win_rate < 0.6


def test_uniformly_better_challenger_wins():
    boss = _losses()
    result = compare_paired_losses(list(boss), list(boss - 0.05))
    assert result.challenger_won is True
    assert result.win_rate == 1.0


def test_winning_hairs_while_losing_big_is_rejected():
    """The case the mean-gap condition exists for: a 92% win rate that is still a worse model."""
    boss = _losses()
    challenger = boss - 0.012
    challenger[:80] = boss[:80] + 1.5

    result = compare_paired_losses(list(boss), list(challenger))

    assert result.win_rate > 0.9
    assert result.mean_gap_nats < 0
    assert result.challenger_won is False
    assert "mean gap" in result.reason


def test_saturated_task_is_a_draw_not_a_boss_win():
    """Every gap inside the dead zone - the models are indistinguishable, which is a draw. The
    defender still holds the task for the dethrone tally, but it is not recorded as a win."""
    boss = np.random.default_rng(2).uniform(0.018, 0.025, 1000)
    result = compare_paired_losses(list(boss), list(boss - 0.002))
    assert result.n_decided == 0
    assert result.challenger_won is False
    assert result.is_draw is True


def test_win_rate_just_under_the_bar_is_rejected():
    """A real but insufficient edge: better on ~52% of examples, short of the 55% required."""
    rng = np.random.default_rng(3)
    boss = _losses()
    challenger = boss.copy()
    idx = rng.permutation(boss.size)
    challenger[idx[:520]] -= 0.05
    challenger[idx[520:]] += 0.05

    result = compare_paired_losses(list(boss), list(challenger))

    assert 0.50 < result.win_rate < 0.55
    assert result.challenger_won is False
    assert "win rate" in result.reason


def test_too_few_decided_examples_goes_to_the_boss():
    """A large per-example edge on too small a decided set still fails - the gate is stricter
    when it has less to go on."""
    boss = _losses(n=1000, seed=4)
    challenger = boss.copy()
    challenger[:50] -= 0.5  # only 50 decided, below the 100 minimum

    result = compare_paired_losses(list(boss), list(challenger))

    assert result.n_decided == 50
    assert result.challenger_won is False
    assert "decided examples" in result.reason


def test_bootstrap_rejects_what_the_point_estimate_would_pass():
    """Few decided examples, all won: point estimate is 100% but the bound is what decides."""
    boss = _losses(n=200, seed=5)
    challenger = boss.copy()
    challenger[:120] -= 0.06

    result = compare_paired_losses(list(boss), list(challenger), min_decided=10)

    assert result.win_rate == 1.0
    assert result.win_rate_lower_bound == 1.0
    # Mean gap is diluted by the 80 undecided examples, so the gap condition is what bites.
    assert result.challenger_won is (result.mean_gap_lower_bound >= t_cst.BOSS_ROUND_MIN_MEAN_GAP_NATS)


def test_verdict_is_deterministic():
    """Two validators scoring the same boss round must reach the same answer."""
    boss = _losses(seed=6)
    challenger = boss - 0.03
    first = compare_paired_losses(list(boss), list(challenger))
    second = compare_paired_losses(list(boss), list(challenger))
    assert first.model_dump() == second.model_dump()


def test_non_finite_examples_are_dropped_from_both_sides():
    boss = _losses(n=500, seed=7)
    challenger = boss - 0.05
    challenger[:10] = np.nan
    boss[490:] = np.inf

    result = compare_paired_losses(list(boss), list(challenger))

    assert result.n_examples == 480
    assert result.challenger_won is True


def test_no_comparable_examples_is_not_a_draw():
    """Nothing to compare is a failure to evaluate, not a statement that the models are equal."""
    result = compare_paired_losses([float("nan")] * 10, [0.1] * 10)
    assert result.n_examples == 0
    assert result.challenger_won is False
    assert result.is_draw is False


def test_mismatched_vector_lengths_raise():
    with pytest.raises(ValueError, match="equal-length"):
        compare_paired_losses([0.1, 0.2], [0.1])


def test_ties_count_for_neither_side():
    """Gaps inside the dead zone are ties, not wins - otherwise the count is coin flips."""
    boss = _losses(n=1000, seed=8)
    challenger = boss.copy()
    challenger[:500] -= t_cst.BOSS_ROUND_TIE_DEADZONE_NATS / 2  # inside the dead zone
    challenger[500:] -= 0.5  # decisively better

    result = compare_paired_losses(list(boss), list(challenger))

    assert result.n_decided == 500
    assert result.challenger_example_wins == 500
    assert result.boss_example_wins == 0


def test_mean_gap_floor_scales_with_the_loss():
    """The floor is never weaker than the relative margin it replaces.

    A flat 0.02 nats is looser than 1% of the loss once the loss exceeds 2.0, which would have let
    a uniformly-but-slightly-better challenger take a high-loss task the old rule would have
    refused.
    """
    rng = np.random.default_rng(20)
    boss = rng.gamma(2.0, 5.0, 1000)  # mean loss ~10, so the floor scales to ~0.1 nats
    challenger = boss - 0.05  # clears the flat 0.02 but is under 1% of the loss

    result = compare_paired_losses(list(boss), list(challenger))

    assert result.win_rate == 1.0  # better on every single example
    assert result.mean_gap_nats > t_cst.BOSS_ROUND_MIN_MEAN_GAP_NATS
    assert result.challenger_won is False
    assert "mean gap" in result.reason


def test_flat_floor_still_applies_below_the_crossover():
    """Under a boss loss of 2.0 the scaling is a no-op and the flat 0.02 governs."""
    boss = _losses(n=1000, seed=21)  # mean ~1.0
    result = compare_paired_losses(list(boss), list(boss - 0.05))
    assert result.challenger_won is True


def test_a_draw_needs_as_much_evidence_as_a_difference():
    """All examples tied only means "equivalent" if there were enough of them for a real gap to
    have shown up. Below the floor it is the same cannot-tell case as too few decided."""
    for n_examples in (1, 20, 99):
        boss = np.random.default_rng(31).uniform(0.02, 0.03, n_examples)
        result = compare_paired_losses(list(boss), list(boss - 0.002))
        assert result.n_decided == 0
        assert result.is_draw is False, f"{n_examples} examples should be too few to call a draw"
        assert result.challenger_won is False

    boss = np.random.default_rng(31).uniform(0.02, 0.03, 100)
    result = compare_paired_losses(list(boss), list(boss - 0.002))
    assert result.is_draw is True
