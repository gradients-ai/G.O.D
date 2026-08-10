"""Tests for the read-side per-example split reported to analytics."""

import numpy as np
import pytest

import validator.tournament.thresholds as thresholds
from core.models.task_models import TaskType
from validator.endpoints.tournament_analytics import _get_paired_example_wins
from validator.tournament.thresholds import count_example_wins
from validator.tournament.thresholds import summarise_paired_examples


A, B = "5MinerA", "5MinerB"
TASK_ID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def stub_vectors(monkeypatch):
    """Stub the per-example vector lookup, keyed by hotkey."""

    def _stub(vectors: dict[str, tuple[list[float] | None, str | None]]) -> None:
        async def fake_get_per_example_losses(task_id, hotkey, psql_db):
            return vectors.get(hotkey, (None, None))

        monkeypatch.setattr(thresholds, "get_per_example_losses", fake_get_per_example_losses)

    return _stub


def _losses(n: int = 1000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).gamma(2.0, 0.5, n)


def test_counts_split_wins_ties_and_comparable_examples():
    a = _losses(n=100)
    b = a.copy()
    b[:60] += 0.05  # A wins 60
    b[60:70] -= 0.05  # B wins 10, the remaining 30 sit in the dead zone

    assert count_example_wins(list(a), list(b)) == (60, 10, 100)


@pytest.mark.asyncio
async def test_summary_reports_the_split(stub_vectors):
    a = _losses()
    b = a + 0.05  # A better on every example
    stub_vectors({A: (list(a), "fp"), B: (list(b), "fp")})

    summary = await summarise_paired_examples(TASK_ID, A, B, psql_db=None)
    assert (summary.hotkey_a_wins, summary.hotkey_b_wins, summary.ties) == (1000, 0, 0)
    assert summary.n_examples == 1000


@pytest.mark.asyncio
async def test_unscorable_examples_drop_out_of_the_split(stub_vectors):
    """Nulls are stored for non-finite losses and read back as NaN, on either side.

    Those examples carry no comparison, so they must not land in the counts or the total - a
    reader dividing wins by n_examples has to get the same denominator the winner was decided on.
    """
    a = _losses(n=500)
    b = a + 0.05  # A better on every example it was scored on
    a[:10] = np.nan
    b[490:] = np.nan

    stub_vectors({A: (list(a), "fp"), B: (list(b), "fp")})

    summary = await summarise_paired_examples(TASK_ID, A, B, psql_db=None)
    assert summary.n_examples == 480
    assert (summary.hotkey_a_wins, summary.hotkey_b_wins, summary.ties) == (480, 0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vectors",
    [
        {},  # nothing stored: every task run before the paired rollout
        {A: ([0.1] * 10, "fp")},  # only one side scored
        {A: ([0.1] * 10, "fp"), B: ([0.1] * 9, "fp")},  # different eval-set sizes
        {A: ([0.1] * 10, "fp-a"), B: ([0.1] * 10, "fp-b")},  # scored on different data
    ],
)
async def test_unpairable_tasks_report_nothing(stub_vectors, vectors):
    """None is the signal to keep showing the mean losses, so it must cover every fallback case."""
    stub_vectors(vectors)
    assert await summarise_paired_examples(TASK_ID, A, B, psql_db=None) is None


class StubTaskDetails:
    def __init__(self, task_type: TaskType = TaskType.INSTRUCTTEXTTASK):
        self.task_type = task_type


def _score(hotkey: str, quality_score: float | None, test_loss: float | None = 0.4) -> dict:
    return {"hotkey": hotkey, "quality_score": quality_score, "test_loss": test_loss}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scores",
    [
        [_score(A, 0.0), _score(B, 3.0)],
        # A submission rejected as a non-finetune carries SCORE_PENALTY and cannot win, so the
        # contest is still between the other two - the same two the winner logic compared.
        [_score(A, 0.0), _score(B, 3.0), _score("5Penalised", -1.0)],
    ],
)
async def test_endpoint_pairs_the_two_eligible_competitors(stub_vectors, scores):
    stub_vectors({A: ([0.5] * 200, "fp"), B: ([0.4] * 200, "fp")})

    summary = await _get_paired_example_wins(TASK_ID, StubTaskDetails(), scores, psql_db=None)
    assert {summary.hotkey_a, summary.hotkey_b} == {A, B}
    assert summary.hotkey_b_wins == 200  # B better on every example


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scores",
    [
        # Only one side eligible: the winner logic falls back to the mean loss, so this must too.
        [_score(A, -1.0), _score(B, 3.0)],
        # Failed evaluation on one side.
        [_score(A, 0.0, test_loss=None), _score(B, 3.0)],
        # Three live competitors is not a two-way contest.
        [_score(A, 0.0), _score(B, 3.0), _score("5Third", 0.0)],
    ],
)
async def test_endpoint_reports_nothing_without_a_clean_pair(stub_vectors, scores):
    stub_vectors({A: ([0.5] * 200, "fp"), B: ([0.4] * 200, "fp")})
    assert await _get_paired_example_wins(TASK_ID, StubTaskDetails(), scores, psql_db=None) is None


@pytest.mark.asyncio
async def test_endpoint_reports_nothing_for_unpaired_task_types(stub_vectors):
    """GRPO and environment scores are reward signals with nothing comparable to pair."""
    stub_vectors({A: ([0.5] * 200, "fp"), B: ([0.4] * 200, "fp")})
    details = StubTaskDetails(TaskType.GRPOTASK)
    assert await _get_paired_example_wins(TASK_ID, details, [_score(A, 0.0), _score(B, 3.0)], psql_db=None) is None
