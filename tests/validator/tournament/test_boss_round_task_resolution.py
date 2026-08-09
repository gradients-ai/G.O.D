"""Tests for how a boss-round task picks between the paired gate and the relative margin."""

import numpy as np
import pytest
from pydantic import BaseModel

import validator.tournament.round_results as round_results
from core.models.task_models import TaskType


BOSS = "5BossHotkey"
CHALLENGER = "5ChallengerHotkey"
TASK_ID = "11111111-2222-3333-4444-555555555555"


class StubTask(BaseModel):
    task_type: TaskType


@pytest.fixture
def stub_vectors(monkeypatch):
    """Stub the per-example vector lookup, keyed by hotkey."""

    def _stub(vectors: dict[str, tuple[list[float] | None, str | None]]) -> None:
        async def fake_get_per_example_losses(task_id, hotkey, psql_db):
            return vectors.get(hotkey, (None, None))

        monkeypatch.setattr(round_results, "get_per_example_losses", fake_get_per_example_losses)

    return _stub


async def _resolve(task_type: TaskType, boss_loss: float, challenger_loss: float) -> str:
    return await round_results._resolve_boss_round_task_winner(
        task_id=TASK_ID,
        task_object=StubTask(task_type=task_type),
        boss_hotkey=BOSS,
        opponent_hotkey=CHALLENGER,
        boss_loss=boss_loss,
        opponent_loss=challenger_loss,
        threshold_percentage=0.01,
        psql_db=None,
    )


def _paired(gap: float, n: int = 1000, seed: int = 0, noise: float = 0.0):
    """Boss and challenger vectors where the challenger is better by `gap` nats per example."""
    rng = np.random.default_rng(seed)
    boss = rng.gamma(2.0, 0.5, n)
    challenger = boss - gap + (rng.normal(0, noise, n) if noise else 0.0)
    return list(boss), list(challenger)


@pytest.mark.asyncio
async def test_clear_paired_win_takes_the_task(stub_vectors):
    boss, challenger = _paired(gap=0.08)
    stub_vectors({BOSS: (boss, "fp"), CHALLENGER: (challenger, "fp")})

    assert await _resolve(TaskType.DPOTASK, float(np.mean(boss)), float(np.mean(challenger))) == CHALLENGER


@pytest.mark.asyncio
async def test_margin_win_that_the_paired_gate_rejects(stub_vectors):
    """The point of the change: the old relative margin passes this, the paired gate does not.

    A saturated task where the challenger is ahead on the mean by well over 1% but the per-example
    advantage is inside the tie dead zone, so nothing is actually decided.
    """
    rng = np.random.default_rng(1)
    boss = rng.uniform(0.02, 0.03, 1000)
    challenger = boss - 0.002

    boss_mean, challenger_mean = float(np.mean(boss)), float(np.mean(challenger))
    # The old rule would have crowned the challenger here.
    assert challenger_mean <= boss_mean - abs(boss_mean) * 0.01

    stub_vectors({BOSS: (list(boss), "fp"), CHALLENGER: (list(challenger), "fp")})

    assert await _resolve(TaskType.DPOTASK, boss_mean, challenger_mean) == BOSS


@pytest.mark.asyncio
async def test_missing_vectors_fall_back_to_the_margin(stub_vectors):
    """Rollout and pre-existing tasks: no vectors yet, so the old margin still decides."""
    stub_vectors({})

    assert await _resolve(TaskType.INSTRUCTTEXTTASK, 1.0, 0.5) == CHALLENGER
    assert await _resolve(TaskType.INSTRUCTTEXTTASK, 1.0, 0.999) == BOSS


@pytest.mark.asyncio
async def test_one_sided_vector_falls_back_to_the_margin(stub_vectors):
    boss, _ = _paired(gap=0.08)
    stub_vectors({BOSS: (boss, "fp")})

    assert await _resolve(TaskType.DPOTASK, 1.0, 0.5) == CHALLENGER


@pytest.mark.asyncio
async def test_fingerprint_mismatch_awards_the_task_to_the_boss(stub_vectors):
    """The eval row set depends on the candidate's max_position_embeddings, so a challenger can
    change what it is scored on. Falling back to the weaker margin would reward that."""
    boss, challenger = _paired(gap=0.08)
    stub_vectors({BOSS: (boss, "fp-a"), CHALLENGER: (challenger, "fp-b")})

    # Scalars the margin would have crowned - unpairable still loses.
    assert await _resolve(TaskType.DPOTASK, 1.0, 0.5) == BOSS


@pytest.mark.asyncio
async def test_length_mismatch_awards_the_task_to_the_boss(stub_vectors):
    boss, challenger = _paired(gap=0.08)
    stub_vectors({BOSS: (boss, "fp"), CHALLENGER: (challenger[:-5], "fp")})

    assert await _resolve(TaskType.DPOTASK, 1.0, 0.5) == BOSS


@pytest.mark.asyncio
async def test_grpo_never_uses_the_paired_gate(stub_vectors):
    """GRPO rewards are higher-is-better on an arbitrary scale and keep the relative margin."""
    boss, challenger = _paired(gap=0.08)
    stub_vectors({BOSS: (boss, "fp"), CHALLENGER: (challenger, "fp")})

    # Higher is better here, so a lower challenger score must lose despite the paired vectors.
    assert await _resolve(TaskType.GRPOTASK, 1.0, 0.5) == BOSS
    assert await _resolve(TaskType.GRPOTASK, 1.0, 1.5) == CHALLENGER


@pytest.mark.asyncio
async def test_continuous_sft_chat_tasks_use_the_paired_gate(stub_vectors):
    """Continuous-SFT boss tasks are ChatRawTask and carry the win-them-all dethrone gate, so they
    must not be left on the relative margin."""
    rng = np.random.default_rng(11)
    boss = rng.uniform(0.02, 0.03, 1000)
    challenger = boss - 0.002
    boss_mean, challenger_mean = float(np.mean(boss)), float(np.mean(challenger))
    assert challenger_mean <= boss_mean - abs(boss_mean) * 0.01  # old rule would crown

    stub_vectors({BOSS: (list(boss), "fp"), CHALLENGER: (list(challenger), "fp")})

    assert await _resolve(TaskType.CHATTASK, boss_mean, challenger_mean) == BOSS
