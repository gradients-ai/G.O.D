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
    def _wire(vectors, losses, task_type=TaskType.DPOTASK):
        async def fake_results(task_id, psql_db):
            return [
                MinerResultsText(hotkey=h, test_loss=l, synth_loss=l, is_finetune=True, task_type=task_type)
                for h, l in losses
            ]
        async def fake_vectors(task_id, hotkey, psql_db):
            return vectors.get(hotkey, (None, None))
        async def fake_task(task_id, psql_db):
            return StubTask(task_type=task_type)
        async def fake_fallback(task_id, psql_db):
            return "FALLBACK"
        monkeypatch.setattr(rr, "get_task_results_for_ranking", fake_results)
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
async def test_win_rate_overrides_the_mean_loss_ranking(wire):
    """B has the better mean, A wins more examples - A must advance, not FALLBACK."""
    rng = np.random.default_rng(1)
    a = rng.gamma(2.0, 0.5, 1000)
    b = a - 0.02
    b[:120] = a[:120] - 5.0  # B far better on a few, dragging its mean below A's
    a2 = a.copy()
    a2[120:] = a[120:] - 0.05  # A decisively better on the other 880

    assert float(b.mean()) < float(a2.mean())
    wire({A: (list(a2), "fp"), B: (list(b), "fp")}, [(A, float(a2.mean())), (B, float(b.mean()))])

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
