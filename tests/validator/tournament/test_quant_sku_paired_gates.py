"""Paired quant gates across SKUs: each side is a multiple of its own SKU's bf16 reference, example by example.

The boss round's fixed and chained fixed tasks share _resolve_boss_round_task_winner; knockouts and the
emission projection pair the same way. A uniform kernel drift on one SKU scales that side's served losses
and its reference alike, so every verdict must be the one the undrifted losses give.
"""

import math
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from pydantic import BaseModel

import validator.tournament.performance_calculator as performance_calculator
import validator.tournament.round_results as round_results
from core.constants.environments import TrainingStartPoint
from core.models.quant_models import QuantEvalGpu
from core.models.task_models import TaskType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.scoring.models import QuantSkuReference
from validator.scoring.quant_ranking import QuantReferenceUnavailable
from validator.tournament import thresholds
from validator.tournament.models import TournamentTask
from validator.tournament.thresholds import align_onto_sku
from validator.tournament.thresholds import sku_aligned_losses


BOSS = EMISSION_BURN_HOTKEY
CHALLENGER = "5ChallengerHotkey"
TASK_ID = "11111111-2222-3333-4444-555555555555"
FP = "fp"
SXM = QuantEvalGpu(name="NVIDIA H100 80GB HBM3", sm_count=132, sku="H100 SXM")
PCIE = QuantEvalGpu(name="NVIDIA H100 PCIe", sm_count=114, sku="H100 PCIe")
DRIFT = 1.05  # large on purpose: enough to flip every raw-loss verdict below


class StubTask(BaseModel):
    task_type: TaskType
    training_start_point: TrainingStartPoint = TrainingStartPoint.DEFAULT


def _reference(gpu: QuantEvalGpu, losses: np.ndarray, fingerprint: str = FP) -> QuantSkuReference:
    return QuantSkuReference(
        task_id=uuid.UUID(TASK_ID),
        sku=gpu.reference_key,
        model="org/shared-input",
        loss=float(np.mean(losses)),
        per_example_losses=list(losses),
        eval_set_fingerprint=fingerprint,
        eval_gpu=gpu,
    )


class Scenario(BaseModel):
    """Boss on SXM, challenger on PCIe; every PCIe number scaled by `drift`."""

    boss: list[float]
    challenger: list[float]
    references: dict[tuple[str, str], QuantSkuReference]
    boss_scalar: float
    challenger_scalar: float


def _scenario(boss_gap: float, challenger_gap: float, drift: float, n: int = 1000) -> Scenario:
    reference = np.random.default_rng(0).uniform(1.5, 3.0, n)
    boss, challenger = reference + boss_gap, (reference + challenger_gap) * drift
    references = {("", "H100 SXM"): _reference(SXM, reference), ("", "H100 PCIe"): _reference(PCIE, reference * drift)}
    return Scenario(
        boss=list(boss),
        challenger=list(challenger),
        references=references,
        # adjusted_loss: each scalar over its own SKU's reference loss.
        boss_scalar=float(np.mean(boss) / np.mean(reference)),
        challenger_scalar=float(np.mean(challenger) / np.mean(reference * drift)),
    )


@pytest.fixture
def skus(monkeypatch):
    """Stub the eval-GPU and reference reads behind sku_aligned_losses."""

    def install(references: dict[tuple[str, str], QuantSkuReference], gpus=None):
        gpus = gpus if gpus is not None else {BOSS: SXM, CHALLENGER: PCIE}
        monkeypatch.setattr(thresholds, "get_task_node_eval_gpus", AsyncMock(return_value=gpus))
        monkeypatch.setattr(thresholds, "get_quant_sku_references", AsyncMock(return_value=references))

    return install


class TestAlignOntoSku:
    def test_scales_each_example_by_anchor_over_own(self) -> None:
        assert align_onto_sku([2.0, 3.0], own_reference=[1.0, 2.0], anchor_reference=[1.5, 1.0]) == [3.0, 1.5]

    def test_an_unusable_own_reference_drops_the_example(self) -> None:
        aligned = align_onto_sku([2.0, 3.0, 4.0], own_reference=[0.0, math.nan, 2.0], anchor_reference=[1.0, 1.0, 1.0])
        assert math.isnan(aligned[0]) and math.isnan(aligned[1]) and aligned[2] == 2.0


class TestSkuAlignedLosses:
    async def test_non_quant_vectors_pass_through_without_a_read(self) -> None:
        assert await sku_aligned_losses(TaskType.INSTRUCTTEXTTASK, TASK_ID, BOSS, CHALLENGER, [1.0], None) == [1.0]

    async def test_one_sku_needs_no_reference(self, skus) -> None:
        skus({}, gpus={BOSS: PCIE, CHALLENGER: PCIE})
        assert await sku_aligned_losses(TaskType.QUANTFIXEDTASK, TASK_ID, BOSS, CHALLENGER, [1.0], None) == [1.0]

    async def test_a_drifted_side_is_brought_back_onto_the_anchor_sku(self, skus) -> None:
        scenario = _scenario(boss_gap=0.1, challenger_gap=0.02, drift=DRIFT)
        skus(scenario.references)
        undrifted = _scenario(boss_gap=0.1, challenger_gap=0.02, drift=1.0).challenger

        aligned = await sku_aligned_losses(TaskType.QUANTFIXEDTASK, TASK_ID, BOSS, CHALLENGER, scenario.challenger, None)

        assert aligned == pytest.approx(undrifted)

    @pytest.mark.parametrize("case", ["no-gpu", "uncached-sku", "other-eval-set"])
    async def test_without_matching_references_it_refuses(self, skus, case) -> None:
        scenario = _scenario(boss_gap=0.1, challenger_gap=0.02, drift=DRIFT)
        references = dict(scenario.references)
        gpus = {BOSS: SXM, CHALLENGER: PCIE}
        if case == "no-gpu":
            gpus = {BOSS: SXM}
        elif case == "uncached-sku":
            references.pop(("", "H100 PCIe"))
        else:
            references[("", "H100 PCIe")] = references[("", "H100 PCIe")].model_copy(update={"eval_set_fingerprint": "other"})
        skus(references, gpus=gpus)

        with pytest.raises(QuantReferenceUnavailable):
            await sku_aligned_losses(TaskType.QUANTFIXEDTASK, TASK_ID, BOSS, CHALLENGER, scenario.challenger, None)


class TestOwnModelSkuAlignedLosses:
    """On an own-model task each player has its own per-hotkey reference; there is no task-common fallback."""

    OWN = TrainingStartPoint.OWN_UPSTREAM_MODEL

    @staticmethod
    def _own_references(boss_ref: np.ndarray, challenger_ref: np.ndarray, gpus=(SXM, PCIE)):
        return {
            (BOSS, gpus[0].reference_key): _reference(gpus[0], boss_ref),
            (CHALLENGER, gpus[1].reference_key): _reference(gpus[1], challenger_ref),
        }

    async def test_same_sku_is_still_aligned_by_each_players_own_reference(self, skus) -> None:
        boss_ref, challenger_ref = np.full(4, 2.0), np.full(4, 4.0)
        skus(self._own_references(boss_ref, challenger_ref, gpus=(PCIE, PCIE)), gpus={BOSS: PCIE, CHALLENGER: PCIE})

        aligned = await sku_aligned_losses(
            TaskType.QUANTFIXEDTASK, TASK_ID, BOSS, CHALLENGER, [4.0, 8.0, 4.0, 2.0], None, self.OWN
        )

        assert aligned == pytest.approx([2.0, 4.0, 2.0, 1.0])  # x * anchor_ref / own_ref = x * 2 / 4

    async def test_cross_sku_uses_the_per_hotkey_references(self, skus) -> None:
        boss_ref, challenger_ref = np.full(4, 2.0), np.full(4, 5.0)
        skus(self._own_references(boss_ref, challenger_ref))

        aligned = await sku_aligned_losses(TaskType.QUANTFIXEDTASK, TASK_ID, BOSS, CHALLENGER, [5.0] * 4, None, self.OWN)

        assert aligned == pytest.approx([2.0] * 4)

    async def test_the_task_common_reference_is_not_a_fallback(self, skus) -> None:
        common = {("", "H100 SXM"): _reference(SXM, np.full(4, 2.0)), ("", "H100 PCIe"): _reference(PCIE, np.full(4, 2.0))}
        skus(common)

        with pytest.raises(QuantReferenceUnavailable):
            await sku_aligned_losses(TaskType.QUANTFIXEDTASK, TASK_ID, BOSS, CHALLENGER, [1.0] * 4, None, self.OWN)

    async def test_a_missing_own_reference_raises(self, skus) -> None:
        references = self._own_references(np.full(4, 2.0), np.full(4, 5.0))
        references.pop((CHALLENGER, "H100 PCIe"))
        skus(references)

        with pytest.raises(QuantReferenceUnavailable):
            await sku_aligned_losses(TaskType.QUANTFIXEDTASK, TASK_ID, BOSS, CHALLENGER, [1.0] * 4, None, self.OWN)

    async def test_a_task_common_start_point_keeps_the_same_sku_shortcut(self, skus) -> None:
        skus({}, gpus={BOSS: PCIE, CHALLENGER: PCIE})

        out = await sku_aligned_losses(
            TaskType.QUANTFIXEDTASK, TASK_ID, BOSS, CHALLENGER, [1.0], None, TrainingStartPoint.DEFAULT
        )

        assert out == [1.0]


async def _boss_round_verdict(monkeypatch, scenario: Scenario):
    vectors = {BOSS: (scenario.boss, FP), CHALLENGER: (scenario.challenger, FP)}
    monkeypatch.setattr(round_results, "get_per_example_losses", AsyncMock(side_effect=lambda _t, hk, _db: vectors[hk]))
    return await round_results._resolve_boss_round_task_winner(
        task_id=TASK_ID,
        task_object=StubTask(task_type=TaskType.QUANTFIXEDTASK),
        boss_hotkey=BOSS,
        opponent_hotkey=CHALLENGER,
        boss_loss=scenario.boss_scalar,
        opponent_loss=scenario.challenger_scalar,
        threshold_percentage=0.01,
        psql_db=None,
    )


class TestBossRoundGate:
    @pytest.mark.parametrize(
        ("boss_gap", "challenger_gap", "winner"),
        [(0.10, 0.02, "challenger"), (0.02, 0.10, "boss"), (0.030, 0.032, "draw")],
    )
    async def test_a_uniform_sku_drift_leaves_the_verdict_unchanged(
        self, monkeypatch, skus, boss_gap, challenger_gap, winner
    ) -> None:
        undrifted = _scenario(boss_gap, challenger_gap, drift=1.0)
        drifted = _scenario(boss_gap, challenger_gap, drift=DRIFT)

        skus(undrifted.references)
        expected = await _boss_round_verdict(monkeypatch, undrifted)
        skus(drifted.references)
        verdict = await _boss_round_verdict(monkeypatch, drifted)

        assert (verdict.winner_hotkey, verdict.is_draw) == (expected.winner_hotkey, expected.is_draw)
        outcome = "draw" if verdict.is_draw else ("challenger" if verdict.winner_hotkey == CHALLENGER else "boss")
        assert outcome == winner

    async def test_raw_losses_would_have_flipped_it(self) -> None:
        """Guards the scenario: without the alignment the drifted challenger loses a task it wins."""
        drifted = _scenario(boss_gap=0.10, challenger_gap=0.02, drift=DRIFT)
        assert not thresholds.compare_paired_losses(drifted.boss, drifted.challenger).challenger_won

    async def test_no_same_sku_reference_keeps_the_boss(self, monkeypatch, skus) -> None:
        scenario = _scenario(boss_gap=0.10, challenger_gap=0.02, drift=DRIFT)
        skus({("", "H100 SXM"): scenario.references[("", "H100 SXM")]})

        verdict = await _boss_round_verdict(monkeypatch, scenario)

        assert verdict.winner_hotkey == BOSS
        assert verdict.decided_by == "no same-SKU reference to pair on"


class TestEmissionProjection:
    async def test_mirrors_the_crowning_verdict_across_skus(self, monkeypatch, skus) -> None:
        scenario = _scenario(boss_gap=0.10, challenger_gap=0.02, drift=DRIFT)
        skus(scenario.references)
        vectors = {BOSS: (scenario.boss, FP), CHALLENGER: (scenario.challenger, FP)}
        monkeypatch.setattr(
            performance_calculator, "get_per_example_losses", AsyncMock(side_effect=lambda _t, hk, _db: vectors[hk])
        )

        analytics = await performance_calculator._quant_challenger_won_task(
            TASK_ID,
            StubTask(task_type=TaskType.QUANTFIXEDTASK),
            CHALLENGER,
            scenario.boss_scalar,
            scenario.challenger_scalar,
            0.01,
            None,
        )

        assert analytics.challenger_won is True and analytics.paired_comparison.challenger_won is True


class TestKnockout:
    @pytest.fixture
    def knockout(self, monkeypatch, skus):
        """A two-way quant fixed knockout: first is the SXM side, second the drifted PCIe side."""

        def install(scenario: Scenario):
            skus(scenario.references, gpus={"on_sxm": SXM, "on_pcie": PCIE})
            sxm_reference, pcie_reference = scenario.references[("", "H100 SXM")], scenario.references[("", "H100 PCIe")]
            results = [
                MinerResultsText(
                    hotkey=hotkey,
                    test_loss=float(np.mean(losses)),
                    synth_loss=float(np.mean(losses)),
                    is_finetune=True,
                    task_type=TaskType.QUANTFIXEDTASK,
                    reference_loss=reference.loss,
                    eval_gpu=reference.eval_gpu,
                )
                for hotkey, losses, reference in (
                    ("on_sxm", scenario.boss, sxm_reference),
                    ("on_pcie", scenario.challenger, pcie_reference),
                )
            ]
            vectors = {"on_sxm": (scenario.boss, FP), "on_pcie": (scenario.challenger, FP)}
            monkeypatch.setattr(round_results, "get_task", AsyncMock(return_value=StubTask(task_type=TaskType.QUANTFIXEDTASK)))
            monkeypatch.setattr(round_results, "get_eligible_hotkeys_for_task", AsyncMock(return_value={"on_sxm", "on_pcie"}))
            monkeypatch.setattr(round_results, "get_task_results_for_ranking", AsyncMock(return_value=results))
            monkeypatch.setattr(
                round_results, "get_per_example_losses", AsyncMock(side_effect=lambda _t, hk, _db: vectors[hk])
            )
            monkeypatch.setattr(round_results, "update_threshold_adjusted_quality_scores_for_task", AsyncMock())

        return install

    async def test_the_better_model_advances_whichever_sku_it_drew(self, knockout) -> None:
        # The PCIe side is better by 0.08 nats, but its raw losses are ~5% higher.
        knockout(_scenario(boss_gap=0.10, challenger_gap=0.02, drift=DRIFT))
        task = TournamentTask(tournament_id="t", round_id="r", task_id=TASK_ID)

        assert await round_results._resolve_knockout_task_winner(task, psql_db=SimpleNamespace()) == "on_pcie"
