"""Quant rank keys, the freeform bar, and quant scoring through calculate_miner_ranking_and_scores."""

import math
import uuid

import numpy as np
import pytest

from core.models.payload_models import EvaluationResultText
from core.models.quant_models import QuantEvalGpu
from core.models.quant_models import QuantFreeformPenalty
from core.models.task_models import TaskType
from validator.scoring import constants as scoring_cst
from validator.scoring.models import MinerResultsImage
from validator.scoring.models import MinerResultsText
from validator.scoring.models import QuantSkuReference
from validator.scoring.quant_ranking import SCORED_FAIL_REASON_PREFIX
from validator.scoring.quant_ranking import QuantReferenceUnavailable
from validator.scoring.quant_ranking import effective_bytes
from validator.scoring.quant_ranking import freeform_bar
from validator.scoring.quant_ranking import normalised_loss
from validator.scoring.quant_ranking import passes_bar
from validator.scoring.quant_ranking import quant_rank_key
from validator.scoring.quant_ranking import rank_metric_for
from validator.scoring.quant_ranking import rank_quant_results
from validator.scoring.quant_ranking import reference_on
from validator.scoring.tasks import RANKED_BELOW_REASON_PREFIX
from validator.scoring.tasks import RANKED_FIRST_REASON_PREFIX
from validator.scoring.tasks import _create_failed_miner_result
from validator.scoring.tasks import calculate_miner_ranking_and_scores


# Reference 2.0 on every row unless a test says otherwise, so the bar is 2.1 and normalised = loss / 2.
REFERENCE = 2.0
BAR_CAP = QuantFreeformPenalty.cliff(1.05)
BAR = 2.1
SXM = QuantEvalGpu(name="NVIDIA H100 80GB HBM3", sm_count=132, sku="H100 SXM")
PCIE = QuantEvalGpu(name="NVIDIA H100 PCIe", sm_count=114, sku="H100 PCIe")


def _freeform(hotkey, loss, artifact_bytes=1000, decoder_bytes=0, quant_failure=None, reference=REFERENCE, gpu=SXM):
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=loss,
        synth_loss=loss,
        is_finetune=True,
        task_type=TaskType.QUANTFREEFORMTASK,
        artifact_bytes=artifact_bytes,
        decoder_bytes=decoder_bytes,
        quant_failure=quant_failure,
        reference_loss=reference,
        eval_gpu=gpu,
    )


def _fixed(hotkey, loss, quant_failure=None, reference=REFERENCE, gpu=SXM):
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=loss,
        synth_loss=loss,
        is_finetune=True,
        task_type=TaskType.QUANTFIXEDTASK,
        quant_failure=quant_failure,
        reference_loss=reference,
        eval_gpu=gpu,
    )


def _hotkeys(results):
    return [result.hotkey for result in results]


class TestFreeformBar:
    def test_bar_is_cap_over_reference(self) -> None:
        assert freeform_bar(2.0, 0.05) == pytest.approx(2.1)

    @pytest.mark.parametrize("reference_loss", [0.0, -1.0, math.inf, math.nan])
    def test_rejects_unusable_reference(self, reference_loss) -> None:
        with pytest.raises(ValueError):
            freeform_bar(reference_loss, 0.05)

    def test_loss_exactly_on_the_bar_passes(self) -> None:
        assert passes_bar(BAR, BAR)
        assert not passes_bar(BAR + 1e-9, BAR)
        assert not passes_bar(math.nan, BAR)


class TestFreeformRankKey:
    def test_passes_by_bytes_then_bar_fails_by_loss_then_scored_fails(self) -> None:
        results = [
            _freeform("scored_fail", math.nan, quant_failure="decoder crashed"),
            _freeform("bar_fail_far", 3.0, artifact_bytes=10),
            _freeform("pass_big", 2.0, artifact_bytes=5000),
            _freeform("bar_fail_close", 2.2, artifact_bytes=10),
            _freeform("pass_small", 2.05, artifact_bytes=800, decoder_bytes=100),
        ]
        assert _hotkeys(rank_quant_results(results, BAR_CAP)) == [
            "pass_small",
            "pass_big",
            "bar_fail_close",
            "bar_fail_far",
            "scored_fail",
        ]

    def test_loss_on_the_bar_ranks_as_a_pass(self) -> None:
        on_bar = _freeform("on_bar", BAR, artifact_bytes=900)
        assert quant_rank_key(on_bar, BAR_CAP) == (0.0, 900.0, BAR / REFERENCE)

    def test_byte_tie_broken_by_lower_loss(self) -> None:
        results = [_freeform("worse", 2.0, artifact_bytes=900), _freeform("better", 1.9, artifact_bytes=900)]
        assert _hotkeys(rank_quant_results(results, BAR_CAP)) == ["better", "worse"]

    def test_nobody_passes_lowest_loss_wins_not_smallest(self) -> None:
        results = [_freeform("tiny_garbage", 9.0, artifact_bytes=1), _freeform("closest", 2.2, artifact_bytes=10**9)]
        assert _hotkeys(rank_quant_results(results, BAR_CAP)) == ["closest", "tiny_garbage"]

    def test_pass_without_byte_count_is_a_scored_fail(self) -> None:
        unmeasured = _freeform("unmeasured", 1.0, decoder_bytes=None)
        assert quant_rank_key(unmeasured, BAR_CAP) == (2.0, math.inf, math.inf)
        assert rank_metric_for(unmeasured, BAR_CAP) == math.inf

    def test_rank_metric_is_bytes_for_passes_and_loss_for_fails(self) -> None:
        assert rank_metric_for(_freeform("pass", 2.0, artifact_bytes=700, decoder_bytes=50), BAR_CAP) == 750
        assert rank_metric_for(_freeform("bar_fail", 2.5), BAR_CAP) == 2.5 / REFERENCE

    def test_persisted_passed_bar_used_without_a_bar(self) -> None:
        result = _freeform("persisted", 2.0, artifact_bytes=700)
        result.passed_bar = True
        assert quant_rank_key(result, None) == (0.0, 700.0, 1.0)
        result.passed_bar = None
        with pytest.raises(ValueError):
            quant_rank_key(result, None)


class TestFixedRankKey:
    def test_ranks_by_loss_with_fails_last(self) -> None:
        results = [_fixed("fail", math.nan, quant_failure="does not load"), _fixed("b", 1.2), _fixed("a", 1.1)]
        assert _hotkeys(rank_quant_results(results, None)) == ["a", "b", "fail"]
        assert quant_rank_key(results[0], None) == (1.0, math.inf)
        assert quant_rank_key(results[2], None) == (0.0, 1.1 / REFERENCE)


class TestFreeformScoring:
    def test_smallest_pass_wins_and_scored_fails_rank_last(self) -> None:
        results = [
            _freeform("pass_big", 2.0, artifact_bytes=5000),
            _freeform("pass_small", 2.1, artifact_bytes=900),
            _freeform("bar_fail", 2.3, artifact_bytes=10),
            _freeform("scored_fail", math.nan, artifact_bytes=None, decoder_bytes=None, quant_failure="does not load"),
            _create_failed_miner_result("no_show", "Evaluation failed", TaskType.QUANTFREEFORMTASK),
        ]
        scored = {r.hotkey: r for r in calculate_miner_ranking_and_scores(results, policy=BAR_CAP)}

        assert scored["pass_small"].score == scoring_cst.FIRST_PLACE_SCORE
        assert scored["pass_small"].score_reason == f"{RANKED_FIRST_REASON_PREFIX}quant penalised bytes under cap"
        assert scored["pass_big"].score_reason == f"{RANKED_BELOW_REASON_PREFIX}quant penalised bytes under cap"
        assert [scored[hk].rank for hk in ("pass_small", "pass_big", "bar_fail", "scored_fail")] == [1, 2, 3, 4]
        assert scored["pass_small"].rank_metric == 900
        assert scored["bar_fail"].rank_metric == 2.3 / REFERENCE
        assert [scored[hk].passed_bar for hk in ("pass_small", "pass_big", "bar_fail", "scored_fail", "no_show")] == [
            True,
            True,
            False,
            False,
            False,
        ]

        fail = scored["scored_fail"]
        assert fail.is_finetune is True
        assert fail.score == scoring_cst.SCORE_PENALTY
        assert fail.adjusted_loss == math.inf
        assert fail.rank_metric == math.inf
        assert fail.score_reason == f"{SCORED_FAIL_REASON_PREFIX}does not load"

        no_show = scored["no_show"]
        assert no_show.is_finetune is False
        assert no_show.score_reason == "Evaluation failed"
        assert no_show.rank is None

    def test_scored_fails_share_the_last_rank(self) -> None:
        results = [
            _freeform("pass", 2.0),
            _freeform("fail_a", math.nan, quant_failure="does not load"),
            _freeform("fail_b", math.nan, quant_failure="structural check"),
        ]
        scored = {r.hotkey: r for r in calculate_miner_ranking_and_scores(results, policy=BAR_CAP)}
        assert scored["fail_a"].rank == scored["fail_b"].rank == 2


class TestFixedScoring:
    def test_ranks_by_loss_with_scored_fails_last(self) -> None:
        results = [_fixed("fail", math.nan, quant_failure="does not load"), _fixed("b", 1.2), _fixed("a", 1.1)]
        scored = {r.hotkey: r for r in calculate_miner_ranking_and_scores(results)}

        assert scored["a"].score == scoring_cst.FIRST_PLACE_SCORE
        assert scored["a"].score_reason == f"{RANKED_FIRST_REASON_PREFIX}quant test_loss / same-SKU reference"
        assert [scored[hk].rank for hk in ("a", "b", "fail")] == [1, 2, 3]
        assert scored["b"].rank_metric == scored["b"].adjusted_loss == 1.2 / REFERENCE
        assert scored["fail"].score == scoring_cst.SCORE_PENALTY
        assert all(result.passed_bar is None for result in scored.values())


# A uniform kernel drift on one SKU: every served loss and its bf16 reference scale by the same factor.
DRIFT = 1.013


def _stored_reference(gpu: QuantEvalGpu, loss: float, hotkey: str = "") -> QuantSkuReference:
    return QuantSkuReference(
        task_id=uuid.uuid4(),
        hotkey=hotkey,
        sku=gpu.reference_key,
        model="org/base",
        loss=loss,
        per_example_losses=[loss],
        eval_set_fingerprint="fp",
        eval_gpu=gpu,
    )


class TestReferenceOn:
    def test_a_miners_own_reference_wins_over_the_task_common_one(self) -> None:
        common, own = _stored_reference(SXM, 2.0), _stored_reference(SXM, 3.0, hotkey="a")
        references = {common.key: common, own.key: own}

        assert reference_on("a", SXM, references) is own
        assert reference_on("b", SXM, references) is common

    def test_falls_back_from_the_hotkey_to_the_task_common_row(self) -> None:
        common = _stored_reference(SXM, 2.0)

        assert reference_on("a", SXM, {common.key: common}) is common
        assert reference_on("", SXM, {common.key: common}) is common

    def test_a_miners_reference_on_another_sku_is_not_borrowed(self) -> None:
        own = _stored_reference(SXM, 3.0, hotkey="a")

        assert reference_on("a", PCIE, {own.key: own}) is None
        assert reference_on("b", SXM, {own.key: own}) is None
        assert reference_on("a", None, {own.key: own}) is None


class TestSkuNormalisation:
    def test_normalised_loss_is_loss_over_its_own_reference(self) -> None:
        assert normalised_loss(_fixed("a", 1.5, reference=1.2)) == pytest.approx(1.25)
        assert normalised_loss(_fixed("nan", math.nan)) == math.inf

    @pytest.mark.parametrize("reference", [None, 0.0, -1.0, math.inf, math.nan])
    def test_a_loss_without_a_usable_reference_is_never_compared(self, reference) -> None:
        with pytest.raises(QuantReferenceUnavailable, match="H100 SXM"):
            normalised_loss(_fixed("a", 1.5, reference=reference))

    def test_rows_are_normalised_by_their_own_miners_reference(self) -> None:
        references = {
            ("a", SXM.reference_key): _stored_reference(SXM, 2.0, hotkey="a"),
            ("b", SXM.reference_key): _stored_reference(SXM, 4.0, hotkey="b"),
        }
        a = _fixed("a", 3.0, reference=reference_on("a", SXM, references).loss)
        b = _fixed("b", 3.0, reference=reference_on("b", SXM, references).loss)

        assert normalised_loss(a) == pytest.approx(1.5)
        assert normalised_loss(b) == pytest.approx(0.75)
        assert [r.hotkey for r in rank_quant_results([a, b], None)] == ["b", "a"]

    def test_freeform_bar_is_per_row_on_its_own_sku(self) -> None:
        # 2.12 is over the SXM bar (2.1) but under the PCIe bar (2.0 x 1.013 x 1.05 = 2.127).
        sxm = _freeform("sxm", 2.12)
        pcie = _freeform("pcie", 2.12, reference=REFERENCE * DRIFT, gpu=PCIE)
        scored = {r.hotkey: r for r in calculate_miner_ranking_and_scores([sxm, pcie], policy=BAR_CAP)}

        assert (scored["sxm"].passed_bar, scored["pcie"].passed_bar) == (False, True)

    @pytest.mark.parametrize("track", ["fixed", "freeform"])
    def test_two_skus_with_a_uniform_drift_rank_as_without_it(self, track) -> None:
        """One submission per SKU, the PCIe side drifted. Ranking must not move."""
        make = _fixed if track == "fixed" else _freeform
        # Losses chosen so the raw comparison flips under the drift: 2.04 x 1.013 > 2.06.
        losses = {"on_sxm": 2.06, "on_pcie": 2.04}

        def scored(drift: float):
            results = [
                make("on_sxm", losses["on_sxm"], gpu=SXM),
                make("on_pcie", losses["on_pcie"] * drift, reference=REFERENCE * drift, gpu=PCIE),
            ]
            return {r.hotkey: r for r in calculate_miner_ranking_and_scores(results, policy=BAR_CAP)}

        undrifted, drifted = scored(1.0), scored(DRIFT)

        assert losses["on_pcie"] * DRIFT > losses["on_sxm"], "the drift must flip a raw-loss comparison"
        for hotkey in losses:
            assert drifted[hotkey].rank == undrifted[hotkey].rank
            assert drifted[hotkey].score == undrifted[hotkey].score
            assert drifted[hotkey].passed_bar == undrifted[hotkey].passed_bar
            assert drifted[hotkey].adjusted_loss == pytest.approx(undrifted[hotkey].adjusted_loss)
        assert drifted["on_pcie"].rank == 1

    def test_freeform_byte_tie_is_broken_on_normalised_loss_across_skus(self) -> None:
        results = [
            _freeform("sxm", 2.06, artifact_bytes=900),
            _freeform("pcie", 2.04 * DRIFT, artifact_bytes=900, reference=REFERENCE * DRIFT, gpu=PCIE),
        ]
        assert _hotkeys(rank_quant_results(results, BAR_CAP)) == ["pcie", "sxm"]

    def test_freeform_bytes_still_decide_passes_regardless_of_sku(self) -> None:
        results = [
            _freeform("sxm_big", 1.9, artifact_bytes=5000),
            _freeform("pcie_small", 2.1 * DRIFT, artifact_bytes=900, reference=REFERENCE * DRIFT, gpu=PCIE),
        ]
        assert _hotkeys(rank_quant_results(results, BAR_CAP)) == ["pcie_small", "sxm_big"]


class TestExistingTypesAreRankedAsBefore:
    def test_instruct_scores_unchanged_and_ranked(self) -> None:
        results = [
            MinerResultsText(hotkey=hk, test_loss=loss, synth_loss=loss, is_finetune=True, task_type=TaskType.INSTRUCTTEXTTASK)
            for hk, loss in (("b", 0.2), ("a", 0.1), ("nan", np.nan))
        ]
        scored = {r.hotkey: r for r in calculate_miner_ranking_and_scores(results)}

        assert (scored["a"].score, scored["a"].score_reason) == (scoring_cst.FIRST_PLACE_SCORE, "Ranked 1st by test_loss")
        assert (scored["b"].score, scored["b"].score_reason) == (0.0, "Ranked below top 1 by test_loss")
        assert (scored["nan"].score, scored["nan"].score_reason) == (scoring_cst.SCORE_PENALTY, "Invalid test loss")
        assert all(result.rank is None and result.rank_metric is None for result in scored.values())

    def test_grpo_ranks_descending(self) -> None:
        results = [
            MinerResultsText(hotkey=hk, test_loss=loss, synth_loss=loss, is_finetune=True, task_type=TaskType.GRPOTASK)
            for hk, loss in (("low", 0.2), ("high", 0.9))
        ]
        scored = {r.hotkey: r for r in calculate_miner_ranking_and_scores(results)}
        assert scored["high"].score_reason == "Ranked 1st by GRPO score (bigger is better)"
        assert scored["high"].rank is None and scored["low"].rank is None

    def test_image_ranked_by_loss(self) -> None:
        results = [
            MinerResultsImage(hotkey=hk, test_loss=loss, synth_loss=loss, is_finetune=True)
            for hk, loss in (("b", 0.5), ("a", 0.4))
        ]
        scored = {r.hotkey: r for r in calculate_miner_ranking_and_scores(results)}
        assert scored["a"].score_reason == "Ranked 1st by test_loss"
        assert scored["a"].rank is None


class TestEvaluationResultText:
    def test_missing_loss_requires_quant_failure(self) -> None:
        with pytest.raises(ValueError):
            EvaluationResultText(is_finetune=True, eval_loss=None)

    def test_accepts_scorer_success_dict(self) -> None:
        result = EvaluationResultText.model_validate(
            {
                "is_finetune": True,
                "eval_loss": 1.93,
                "artifact_bytes": 4_000_000_000,
                "decoder_bytes": 12_000,
                "kernel_classes": {"model.layers.0.mlp": "MarlinLinearKernel"},
            }
        )
        assert result.quant_failure is None
        assert result.artifact_bytes == 4_000_000_000

    def test_accepts_scorer_scored_fail_dict(self) -> None:
        result = EvaluationResultText.model_validate(
            {"is_finetune": True, "eval_loss": None, "quant_failure": "vLLM failed to load the artifact"}
        )
        assert result.eval_loss is None
        assert result.quant_failure == "vLLM failed to load the artifact"


# Reference 2.0, policy knee 1.04 / cap 1.08: the knee is loss 2.08, the cap loss 2.16.
PENALTY = QuantFreeformPenalty(knee=1.04, curvature=2500.0, cap=1.08)


def _ratio(hotkey, r, artifact_bytes=1000):
    return _freeform(hotkey, r * REFERENCE, artifact_bytes=artifact_bytes)


class TestFreeformPenaltyRanking:
    def test_ten_percent_smaller_artifact_loses_when_well_past_the_knee(self) -> None:
        # r = 1.052 is 0.012 over the knee: x1.434, so 900 bytes is worth 1290.
        results = [_ratio("small_lossy", 1.052, artifact_bytes=900), _ratio("at_knee", 1.04, artifact_bytes=1000)]
        assert _hotkeys(rank_quant_results(results, PENALTY)) == ["at_knee", "small_lossy"]

    def test_ten_percent_smaller_artifact_wins_just_past_the_knee(self) -> None:
        # r = 1.042 is 0.002 over the knee: x1.010, so 900 bytes is worth 909.
        results = [_ratio("at_knee", 1.04, artifact_bytes=1000), _ratio("small_lossy", 1.042, artifact_bytes=900)]
        assert _hotkeys(rank_quant_results(results, PENALTY)) == ["small_lossy", "at_knee"]

    def test_key_is_effective_bytes_then_ratio(self) -> None:
        result = _ratio("x", 1.052, artifact_bytes=900)
        r = normalised_loss(result)
        assert quant_rank_key(result, PENALTY) == (0.0, 900.0 * math.exp(2500.0 * (r - 1.04) ** 2), r)

    def test_over_cap_orders_by_ratio_after_every_under_cap_entry(self) -> None:
        results = [
            _freeform("scored_fail", math.nan, quant_failure="does not load"),
            _ratio("out_far", 1.20, artifact_bytes=1),
            _ratio("in_big", 1.0, artifact_bytes=10**9),
            _ratio("out_close", 1.09, artifact_bytes=5),
        ]
        assert _hotkeys(rank_quant_results(results, PENALTY)) == ["in_big", "out_close", "out_far", "scored_fail"]

    def test_ratio_exactly_at_the_cap_is_in(self) -> None:
        on_cap = _freeform("on_cap", 1.08 * REFERENCE)
        assert quant_rank_key(on_cap, PENALTY)[0] == 0.0
        assert quant_rank_key(_freeform("past", 1.08 * REFERENCE + 1e-9), PENALTY)[0] == 1.0

    def test_equal_effective_size_falls_to_lower_ratio(self) -> None:
        results = [_ratio("worse", 1.03, artifact_bytes=900), _ratio("better", 1.0, artifact_bytes=900)]
        assert _hotkeys(rank_quant_results(results, PENALTY)) == ["better", "worse"]

    def test_rank_metric_is_effective_bytes(self) -> None:
        result = _ratio("x", 1.05, artifact_bytes=800)
        assert rank_metric_for(result, PENALTY) == effective_bytes(result, PENALTY)
        assert rank_metric_for(result, PENALTY) > 800


CLIFF_CASES = [
    ("below", 2.0, 5000, 0, None),
    ("on_bar", 2.1, 700, 50, None),
    ("just_over", 2.1000001, 10, 0, None),
    ("far_over", 9.0, 1, 0, None),
    ("scored_fail", math.nan, None, None, "decoder crashed"),
    ("missing_bytes", 1.0, 100, None, None),
]


class TestCliffMatchesTheOldKey:
    @staticmethod
    def _old_key(loss, total_bytes, quant_failure):
        if quant_failure or (math.isfinite(loss) and total_bytes is None and loss <= 2.1):
            return (2.0, math.inf, math.inf)
        if not (math.isfinite(loss) and loss <= 2.1):
            return (1.0, loss / 2.0, math.inf)
        return (0.0, float(total_bytes), loss / 2.0)

    @pytest.mark.parametrize(("hotkey", "loss", "artifact_bytes", "decoder_bytes", "failure"), CLIFF_CASES)
    def test_cliff_key_equals_the_old_key(self, hotkey, loss, artifact_bytes, decoder_bytes, failure) -> None:
        result = _freeform(hotkey, loss, artifact_bytes=artifact_bytes, decoder_bytes=decoder_bytes, quant_failure=failure)
        assert quant_rank_key(result, QuantFreeformPenalty.cliff(1.05)) == self._old_key(loss, result.total_bytes, failure)

    def test_persisted_penalty_prices_the_bytes_without_a_policy(self) -> None:
        result = _freeform("persisted", 2.0, artifact_bytes=700)
        result.passed_bar = True
        result.freeform_penalty = 0.25
        assert quant_rank_key(result, None) == (0.0, 700.0 * math.exp(0.25), 1.0)

    def test_row_without_penalty_ranks_on_exact_bytes(self) -> None:
        result = _freeform("unpenalised", 2.0, artifact_bytes=700)
        result.passed_bar = True
        assert result.freeform_penalty is None
        assert quant_rank_key(result, None) == (0.0, 700.0, 1.0)


class TestFreeformPenaltyScoring:
    def test_sets_penalty_verdict_metric_and_ranking_type(self) -> None:
        results = [
            _ratio("at_knee", 1.03, artifact_bytes=1000),
            _ratio("past_knee", 1.05, artifact_bytes=900),
            _ratio("over_cap", 1.2, artifact_bytes=1),
        ]
        scored = {r.hotkey: r for r in calculate_miner_ranking_and_scores(results, policy=PENALTY)}

        assert scored["at_knee"].freeform_penalty == 0.0
        assert scored["past_knee"].freeform_penalty == pytest.approx(2500.0 * (1.05 - 1.04) ** 2)
        assert scored["over_cap"].freeform_penalty is None
        assert [scored[hk].passed_bar for hk in ("at_knee", "past_knee", "over_cap")] == [True, True, False]
        assert scored["past_knee"].rank_metric == pytest.approx(900 * math.exp(0.25))
        assert scored["past_knee"].rank_metric == effective_bytes(scored["past_knee"], PENALTY)
        assert [scored[hk].rank for hk in ("at_knee", "past_knee", "over_cap")] == [1, 2, 3]
        assert scored["at_knee"].score_reason == f"{RANKED_FIRST_REASON_PREFIX}quant penalised bytes under cap"
