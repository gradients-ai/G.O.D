"""Tests for the degenerate-dataset filter used at synthetic task creation time."""

import pytest

import validator.tasks.synthetics.constants as synth_cst
import validator.tasks.synthetics.scheduler as scheduler
import validator.tournament  # noqa: F401  (import-order: tournament package must init before scheduler)
from core.models.task_models import TaskType


@pytest.fixture
def stub_losses(monkeypatch):
    """Stub the DB lookup so the filter can be exercised on fixed loss histories."""

    def _stub(losses: list[float]) -> None:
        async def fake_get_dataset_test_losses(ds_name: str, psql_db, lookback_days: int) -> list[float]:
            assert lookback_days == synth_cst.DEGENERATE_LOSS_LOOKBACK_DAYS
            return losses

        monkeypatch.setattr(scheduler, "get_dataset_test_losses", fake_get_dataset_test_losses)

    return _stub


async def _is_degenerate(task_type: TaskType) -> bool:
    return await scheduler._is_dataset_degenerate("some/dataset", task_type, psql_db=None)


@pytest.mark.asyncio
async def test_no_history_is_allowed(stub_losses):
    stub_losses([])
    assert await _is_degenerate(TaskType.DPOTASK) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("best_loss", [0.0212, 0.0215, 0.0491])
async def test_dpo_near_zero_loss_is_rejected(stub_losses, best_loss):
    """Real boss-round offenders bottomed out at 0.02-0.05, above the generic 0.01 floor."""
    stub_losses([best_loss, 1.4, 2.2])
    assert await _is_degenerate(TaskType.DPOTASK) is True


@pytest.mark.asyncio
async def test_dpo_healthy_loss_is_allowed(stub_losses):
    stub_losses([0.5756, 0.59, 1.1])
    assert await _is_degenerate(TaskType.DPOTASK) is False


@pytest.mark.asyncio
async def test_instruct_keeps_the_tighter_collapse_floor(stub_losses):
    """0.02 is a plausible instruct loss; only DPO gets the 0.05 floor."""
    stub_losses([0.02, 0.4])
    assert await _is_degenerate(TaskType.INSTRUCTTEXTTASK) is False

    stub_losses([0.005, 0.4])
    assert await _is_degenerate(TaskType.INSTRUCTTEXTTASK) is True


@pytest.mark.asyncio
async def test_dpo_noise_band_uses_best_not_mean(stub_losses):
    """Nobody beat ln(2) -> reject, even though weak submissions pull the mean out of the band."""
    stub_losses([0.6931, 1.9, 3.4])
    assert await _is_degenerate(TaskType.DPOTASK) is True


@pytest.mark.asyncio
async def test_dpo_beating_noise_is_allowed(stub_losses):
    """A single miner well under ln(2) means the dataset is learnable, not random."""
    stub_losses([0.35, 0.6931, 1.9])
    assert await _is_degenerate(TaskType.DPOTASK) is False


@pytest.mark.asyncio
async def test_instruct_unlearnable_data_is_rejected(stub_losses):
    stub_losses([2.5, 3.1])
    assert await _is_degenerate(TaskType.INSTRUCTTEXTTASK) is True


@pytest.mark.asyncio
async def test_db_failure_allows_the_dataset(monkeypatch):
    async def boom(ds_name: str, psql_db, lookback_days: int) -> list[float]:
        raise RuntimeError("connection reset")

    monkeypatch.setattr(scheduler, "get_dataset_test_losses", boom)
    assert await _is_degenerate(TaskType.DPOTASK) is False
