"""Instruct-to-fixed boss tasks: both players compress the shared input, so the raw-loss paired gate decides.

Both sides are scored on one SKU here, where the per-SKU references cancel exactly; the two-SKU case is in
test_quant_sku_paired_gates.py.
"""

from unittest.mock import AsyncMock

import numpy as np
import pytest
from pydantic import BaseModel

import validator.tournament.round_results as round_results
from core.constants.environments import TrainingStartPoint
from core.models.quant_models import QuantEvalGpu
from core.models.task_models import TaskType
from validator.tournament import constants as t_cst
from validator.tournament import thresholds


BOSS = "5BossHotkey"
CHALLENGER = "5ChallengerHotkey"
TASK_ID = "11111111-2222-3333-4444-555555555555"
FP = "fp"
SXM = QuantEvalGpu(name="NVIDIA H100 80GB HBM3", sm_count=132, sku="H100 SXM")

Vector = tuple[list[float] | None, str | None]


class StubTask(BaseModel):
    task_type: TaskType
    training_start_point: TrainingStartPoint = TrainingStartPoint.DEFAULT


@pytest.fixture
def stub_db(monkeypatch):
    """Stub the per-example read, keyed by hotkey; records the dead zone each paired comparison ran with."""
    compared: list[float] = []
    real_compare = round_results.compare_paired_losses

    def install(quant: dict[str, Vector]) -> list[float]:
        async def fake_quant(task_id, hotkey, psql_db):
            return quant.get(hotkey, (None, None))

        def spy(boss_losses, challenger_losses, **kwargs):
            compared.append(kwargs.get("deadzone_nats", t_cst.BOSS_ROUND_TIE_DEADZONE_NATS))
            return real_compare(boss_losses, challenger_losses, **kwargs)

        monkeypatch.setattr(round_results, "get_per_example_losses", fake_quant)
        monkeypatch.setattr(round_results, "compare_paired_losses", spy)
        monkeypatch.setattr(thresholds, "get_task_node_eval_gpus", AsyncMock(return_value={BOSS: SXM, CHALLENGER: SXM}))
        return compared

    return install


async def _verdict(quant: dict[str, Vector]):
    boss_loss, challenger_loss = (float(np.nanmean(quant[hotkey][0])) for hotkey in (BOSS, CHALLENGER))
    return await round_results._resolve_boss_round_task_winner(
        task_id=TASK_ID,
        task_object=StubTask(task_type=TaskType.QUANTFIXEDTASK),
        boss_hotkey=BOSS,
        opponent_hotkey=CHALLENGER,
        boss_loss=boss_loss,
        opponent_loss=challenger_loss,
        threshold_percentage=0.01,
        psql_db=None,
    )


def _shared_input_losses(boss_gap: float, challenger_gap: float, n: int = 1000) -> dict[str, Vector]:
    """Both players' quantised losses on one shared input, each `gap` nats above its bf16 loss."""
    reference = np.random.default_rng(0).uniform(1.5, 3.0, n)
    return {BOSS: (list(reference + boss_gap), FP), CHALLENGER: (list(reference + challenger_gap), FP)}


async def test_lower_raw_loss_takes_the_task_on_the_nat_scale_gate(stub_db) -> None:
    quant = _shared_input_losses(boss_gap=0.10, challenger_gap=0.02)
    compared = stub_db(quant)

    verdict = await _verdict(quant)

    assert verdict.winner_hotkey == CHALLENGER
    assert verdict.decided_by.startswith("per-example win rate (")
    assert compared == [t_cst.BOSS_ROUND_TIE_DEADZONE_NATS]


async def test_higher_raw_loss_keeps_the_boss(stub_db) -> None:
    quant = _shared_input_losses(boss_gap=0.02, challenger_gap=0.10)
    stub_db(quant)

    verdict = await _verdict(quant)

    assert verdict.winner_hotkey == BOSS
    assert not verdict.is_draw


async def test_equal_cost_inside_the_dead_zone_is_a_draw(stub_db) -> None:
    quant = _shared_input_losses(boss_gap=0.030, challenger_gap=0.032)
    stub_db(quant)

    verdict = await _verdict(quant)

    assert verdict.is_draw is True
    assert verdict.winner_hotkey == BOSS


async def test_unpairable_vectors_retain_the_boss(stub_db) -> None:
    quant = _shared_input_losses(boss_gap=0.10, challenger_gap=0.02)
    quant[CHALLENGER] = (quant[CHALLENGER][0], "other")
    compared = stub_db(quant)

    verdict = await _verdict(quant)

    assert verdict.winner_hotkey == BOSS
    assert verdict.decided_by == "unpairable per-example vectors"
    assert compared == []
