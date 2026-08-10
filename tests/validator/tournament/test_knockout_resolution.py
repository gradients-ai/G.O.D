"""Drives _resolve_knockout_task_winner end to end with stubbed DB reads."""
import numpy as np
import pytest
from pydantic import BaseModel

import validator.tournament.round_results as rr
from core.models.task_models import TaskType
from validator.scoring.models import MinerResultsText
from validator.tournament.models import TournamentTask

A, B = "5MinerA", "5MinerB"
TASK_ID = "11111111-2222-3333-4444-555555555555"


class StubTask(BaseModel):
    task_type: TaskType


def _task():
    return TournamentTask(tournament_id="t", round_id="r", task_id=TASK_ID)


@pytest.fixture
def wire(monkeypatch):
    def _wire(vectors, losses, task_type=TaskType.DPOTASK, eligible=None):
        async def fake_results(task_id, psql_db):
            return [
                MinerResultsText(hotkey=h, test_loss=l, synth_loss=l, is_finetune=True, task_type=task_type)
                for h, l in losses
            ]
        async def fake_eligible(task_id, psql_db):
            if eligible is not None:
                return set(eligible)
            # Mirror what _update_scores actually persists for a two-way contest: rank 1 gets
            # FIRST_PLACE_SCORE, the runner-up stays at 0.0, and both are eligible. Defaulting to
            # "everyone" here is what let a > 0 filter pass its tests while killing the feature.
            ranked = sorted(losses, key=lambda pair: pair[1])
            persisted = {hotkey: (3.0 if i == 0 else 0.0) for i, (hotkey, _) in enumerate(ranked)}
            if not any(score > 0 for score in persisted.values()):
                return set()
            return {hotkey for hotkey, score in persisted.items() if score >= 0}

        async def fake_vectors(task_id, hotkey, psql_db):
            return vectors.get(hotkey, (None, None))
        async def fake_task(task_id, psql_db):
            return StubTask(task_type=task_type)
        async def fake_fallback(task_id, psql_db):
            return "FALLBACK"
        monkeypatch.setattr(rr, "get_task_results_for_ranking", fake_results)
        monkeypatch.setattr(rr, "get_eligible_hotkeys_for_task", fake_eligible)
        monkeypatch.setattr(rr, "get_per_example_losses", fake_vectors)
        monkeypatch.setattr(rr, "get_task", fake_task)
        monkeypatch.setattr(rr, "get_task_winner", fake_fallback)

        persisted = {}

        async def fake_persist(task_id, winner_hotkey, threshold_percentage, psql_db, compared_hotkeys=None, basis=None):
            persisted["winner"] = winner_hotkey
            persisted["basis"] = basis

        monkeypatch.setattr(rr, "update_threshold_adjusted_quality_scores_for_task", fake_persist)
        return persisted
    return _wire


@pytest.mark.asyncio
async def test_paired_path_is_reached_for_a_two_way_contest(wire):
    """Regression: a score>0 filter dropped the runner-up and made this path dead code."""
    rng = np.random.default_rng(0)
    a = rng.gamma(2.0, 0.5, 1000)
    b = a + 0.05  # A better on every example
    wire({A: (list(a), "fp"), B: (list(b), "fp")}, [(A, float(a.mean())), (B, float(b.mean()))])

    assert await rr._resolve_knockout_task_winner(_task(), psql_db=None) == A


@pytest.mark.asyncio
async def test_paired_verdict_is_persisted(wire):
    """Audit data reads task_nodes, so a paired verdict that is not written back would contradict
    actual advancement."""
    rng = np.random.default_rng(4)
    a = rng.gamma(2.0, 0.5, 1000)
    b = a + 0.05
    persisted = wire({A: (list(a), "fp"), B: (list(b), "fp")}, [(A, float(a.mean())), (B, float(b.mean()))])

    await rr._resolve_knockout_task_winner(_task(), psql_db=None)

    assert persisted["winner"] == A
    assert "per-example win rate" in persisted["basis"]


@pytest.mark.asyncio
async def test_win_rate_decides_when_the_ranking_loss_agrees(wire):
    """The sample winner advances when it is not worse on the ranking loss."""
    rng = np.random.default_rng(1)
    a = rng.gamma(2.0, 0.5, 1000)
    b = a + 0.03  # A better on every example and on the mean
    wire({A: (list(a), "fp"), B: (list(b), "fp")}, [(A, float(a.mean())), (B, float(b.mean()))])

    assert await rr._resolve_knockout_task_winner(_task(), psql_db=None) == A


@pytest.mark.asyncio
async def test_sample_winner_that_loses_big_does_not_advance(wire):
    """Winning a majority of samples by hairs while losing the rest badly is not a better model,
    so the ranking loss vetoes it - the same pairing the boss round applies."""
    rng = np.random.default_rng(11)
    a = rng.gamma(2.0, 0.5, 1000)
    b = a - 0.012
    b[:80] = a[:80] + 1.5  # B wins most samples narrowly but is far worse on 80 of them
    wire({A: (list(a), "fp"), B: (list(b), "fp")}, [(A, float(a.mean())), (B, float(b.mean()))])

    assert float(b.mean()) > float(a.mean())
    assert await rr._resolve_knockout_task_winner(_task(), psql_db=None) == A


@pytest.mark.asyncio
async def test_missing_vectors_fall_back(wire):
    a = np.random.default_rng(2).gamma(2.0, 0.5, 100)
    wire({}, [(A, float(a.mean())), (B, float(a.mean()) + 0.1)])
    assert await rr._resolve_knockout_task_winner(_task(), psql_db=None) == "FALLBACK"


@pytest.mark.asyncio
async def test_grpo_falls_back(wire):
    a = np.random.default_rng(3).gamma(2.0, 0.5, 1000)
    wire({A: (list(a), "fp"), B: (list(a + 0.05), "fp")},
         [(A, float(a.mean())), (B, float(a.mean()) + 0.05)], task_type=TaskType.GRPOTASK)
    assert await rr._resolve_knockout_task_winner(_task(), psql_db=None) == "FALLBACK"


@pytest.mark.asyncio
async def test_penalised_submission_cannot_advance(wire):
    """A submission rejected as a non-finetune carries SCORE_PENALTY. It still has a real
    per-example vector, so without the eligibility filter it could win on samples and advance."""
    rng = np.random.default_rng(9)
    a = rng.gamma(2.0, 0.5, 1000)
    b = a + 0.05  # A wins every example, but A is the penalised one
    wire({A: (list(a), "fp"), B: (list(b), "fp")},
         [(A, float(a.mean())), (B, float(b.mean()))], eligible=[B])

    assert await rr._resolve_knockout_task_winner(_task(), psql_db=None) == "FALLBACK"


@pytest.mark.asyncio
async def test_sample_winner_worse_on_the_ranking_loss_does_not_advance(wire):
    """Knockout instruct tasks are KL-weighted from round 2: the vectors are raw CE but test_loss
    carries a per-model penalty, so a raw-CE sample win must not beat a worse ranking loss."""
    rng = np.random.default_rng(10)
    a = rng.gamma(2.0, 0.5, 1000)
    b = a + 0.05  # A wins on raw per-example CE...
    # ...but A's KL-weighted ranking loss is far worse than B's
    wire({A: (list(a), "fp"), B: (list(b), "fp")}, [(A, 1.30), (B, 1.02)])

    assert await rr._resolve_knockout_task_winner(_task(), psql_db=None) == B


@pytest.mark.asyncio
async def test_runner_up_at_zero_is_still_eligible(wire):
    """Regression: the runner-up of every two-way contest persists quality_score 0.0, so a > 0
    eligibility filter leaves one competitor and silently kills the paired path."""
    rng = np.random.default_rng(12)
    a = rng.gamma(2.0, 0.5, 1000)
    b = a + 0.05
    wire({A: (list(a), "fp"), B: (list(b), "fp")}, [(A, float(a.mean())), (B, float(b.mean()))])

    # Reaches the paired path rather than falling back to the mean-loss ranking.
    assert await rr._resolve_knockout_task_winner(_task(), psql_db=None) == A
