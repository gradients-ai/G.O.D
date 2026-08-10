"""Tests for deciding knockout tasks on per-example win rate."""

import numpy as np

from validator.tournament.thresholds import paired_head_to_head_winner


A, B = "5MinerA", "5MinerB"


def _losses(n: int = 1000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).gamma(2.0, 0.5, n)


def test_majority_of_examples_wins():
    a = _losses()
    b = a + 0.05  # A better on every example
    winner, description = paired_head_to_head_winner(A, list(a), float(a.mean()), B, list(b), float(b.mean()))
    assert winner == A
    assert "1000/1000" in description


def test_bare_majority_is_enough():
    """No significance requirement - a knockout has to advance someone either way."""
    rng = np.random.default_rng(1)
    a = _losses()
    b = a.copy()
    idx = rng.permutation(a.size)
    b[idx[:530]] += 0.05  # A wins 530
    b[idx[530:]] -= 0.05  # B wins 470

    winner, _ = paired_head_to_head_winner(A, list(a), float(a.mean()), B, list(b), float(b.mean()))
    assert winner == A


def test_win_rate_beats_mean_loss_when_they_disagree():
    """A wins most examples; B has the better mean because A loses badly on a few."""
    a = _losses()
    b = a - 0.012
    b[:80] = a[:80] - 3.0  # B far better on 80 examples, dragging its mean down

    a_mean, b_mean = float(a.mean()), float(b.mean())
    assert b_mean < a_mean  # the old rule would have advanced B

    winner, _ = paired_head_to_head_winner(A, list(a), a_mean, B, list(b), b_mean)
    assert winner == B  # B still wins on examples here

    # Flip it: A better on most examples, B better on the mean.
    b2 = a - 0.02
    b2[:120] = a[:120] - 5.0
    a2 = a.copy()
    a2[120:] = a[120:] - 0.05  # A decisively better on the remaining 880
    winner, _ = paired_head_to_head_winner(A, list(a2), float(a2.mean()), B, list(b2), float(b2.mean()))
    assert float(b2.mean()) < float(a2.mean())  # B has the better mean
    assert winner == A  # but A won more examples


def test_equal_wins_break_on_mean_loss():
    a = _losses(n=1000, seed=2)
    b = a.copy()
    b[:100] += 0.05  # A wins 100
    b[100:200] -= 0.05  # B wins 100, rest tied

    winner, description = paired_head_to_head_winner(A, list(a), 0.40, B, list(b), 0.50)
    assert winner == A
    assert "broken on mean loss" in description

    winner, _ = paired_head_to_head_winner(A, list(a), 0.60, B, list(b), 0.50)
    assert winner == B


def test_nothing_decided_breaks_on_mean_loss():
    """Saturated task: every gap inside the dead zone, so the mean decides rather than nothing."""
    a = np.random.default_rng(3).uniform(0.02, 0.03, 500)
    b = a - 0.002

    winner, description = paired_head_to_head_winner(A, list(a), float(a.mean()), B, list(b), float(b.mean()))
    assert winner == B
    assert "broken on mean loss" in description


def test_non_finite_examples_are_dropped():
    a = _losses(n=500, seed=4)
    b = a + 0.05
    b[:10] = np.nan

    winner, description = paired_head_to_head_winner(A, list(a), float(a.mean()), B, list(b), float(np.nanmean(b)))
    assert winner == A
    assert "490/490" in description
