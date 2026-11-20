"""Rank keys and the freeform loss penalty for quant tasks.

Every loss comparison is on the normalised loss, r = test_loss / reference_loss, where reference_loss is the
row's own bf16 reference scored on the same GPU SKU as the row (quant_sku_references): its miner's own
model on an own-model task, the task-common model otherwise. Kernel drift between
SKUs scales the served and the reference loss alike, so it cancels to first order and rows scored on
different SKUs compare.

Fixed track: lower normalised loss wins; scored fails rank last.
Freeform track: an entry at or under the cap (cap x its own SKU's reference loss) ranks by effective size,
total bytes x exp(curvature x max(0, r - knee)^2), then r. Bytes are free up to the knee and priced
smoothly above it. Entries over the cap rank after every under-cap entry by r, so when nobody is under the
cap the closest entry wins rather than the smallest garbage. A cliff policy (knee == cap, no curvature) is
a plain pass/fail bar. Scored fails (the submission did not load or failed the structural check) rank
after everything.
"""

import math

from core.models.quant_models import QuantEvalGpu
from core.models.quant_models import QuantFreeformPenalty
from core.models.task_models import TaskType
from validator.scoring.models import MinerResultsText
from validator.scoring.models import QuantSkuReference
from validator.scoring.models import QuantSkuReferences


SCORED_FAIL_REASON_PREFIX = "Scored fail: "
MISSING_BYTE_COUNT_REASON = "missing byte count"


# The freeform task's penalty policy; None re-ranks on each row's persisted passed_bar and freeform_penalty instead.
QuantFreeformPolicy = QuantFreeformPenalty | None


class QuantReferenceUnavailable(ValueError):
    """A quant loss has no bf16 reference on its SKU to be normalised by."""


def eval_gpu_from_column(value: dict | None) -> QuantEvalGpu | None:
    """A task_nodes.eval_gpu value (QuantEvalGpu JSON) as the model; None for a row scored without one."""
    return QuantEvalGpu.model_validate(value) if value is not None else None


def reference_on(hotkey: str, eval_gpu: QuantEvalGpu | None, references: QuantSkuReferences) -> QuantSkuReference | None:
    """The bf16 reference on the SKU a row was scored on, or None when that SKU has none cached.

    hotkey is the row's reference_hotkey_for(task, miner): its own reference first, then the task-common one.
    """
    if eval_gpu is None:
        return None
    return references.get((hotkey, eval_gpu.reference_key)) or references.get(("", eval_gpu.reference_key))


def freeform_bar(reference_loss: float, bar_cap: float) -> float:
    if not math.isfinite(reference_loss) or reference_loss <= 0:
        raise ValueError(f"Reference loss must be finite and positive, got {reference_loss}")
    return (1 + bar_cap) * reference_loss


def passes_bar(test_loss: float, bar: float) -> bool:
    return math.isfinite(test_loss) and test_loss <= bar


def _usable_reference(result: MinerResultsText) -> float:
    reference_loss = result.reference_loss
    if reference_loss is None or not math.isfinite(reference_loss) or reference_loss <= 0:
        sku = result.eval_gpu.reference_key if result.eval_gpu else None
        raise QuantReferenceUnavailable(f"Quant result {result.hotkey} has no usable reference on SKU {sku}: {reference_loss}")
    return reference_loss


def normalised_loss(result: MinerResultsText) -> float:
    """test_loss as a multiple of the bf16 reference on the row's own SKU; inf for a non-finite loss."""
    if not math.isfinite(result.test_loss):
        return math.inf
    return result.test_loss / _usable_reference(result)


def row_bar(result: MinerResultsText, policy: QuantFreeformPenalty) -> float:
    """The loss above which this row is out: the policy cap x the reference loss on its SKU."""
    return policy.cap * _usable_reference(result)


def freeform_passed(result: MinerResultsText, policy: QuantFreeformPolicy) -> bool:
    """Under-cap status from the row's own bar, or from the persisted passed_bar when re-ranking without a policy."""
    if policy is not None:
        return passes_bar(result.test_loss, row_bar(result, policy))
    if result.passed_bar is None:
        raise ValueError(f"Freeform result {result.hotkey} has no policy and no persisted passed_bar")
    return result.passed_bar


def freeform_ln_penalty(result: MinerResultsText, policy: QuantFreeformPolicy) -> float:
    """ln P for an under-cap row: from the policy, or the persisted value (0.0 when none is persisted)."""
    if policy is not None:
        return policy.ln_penalty_under_cap(normalised_loss(result))
    return result.freeform_penalty or 0.0


def effective_bytes(result: MinerResultsText, policy: QuantFreeformPolicy) -> float:
    """S = total_bytes x P for an under-cap row; lower wins."""
    return float(result.total_bytes) * math.exp(freeform_ln_penalty(result, policy))


def scored_fail_reason(result: MinerResultsText, policy: QuantFreeformPolicy) -> str | None:
    """Why a quant result is a scored fail, or None when it is scored on its loss."""
    if result.quant_failure:
        return result.quant_failure
    is_freeform = result.task_type == TaskType.QUANTFREEFORMTASK
    if is_freeform and math.isfinite(result.test_loss) and result.total_bytes is None and freeform_passed(result, policy):
        return MISSING_BYTE_COUNT_REASON
    return None


def quant_rank_key(result: MinerResultsText, policy: QuantFreeformPolicy) -> tuple[float, ...]:
    """Sort key, lower first."""
    is_scored_fail = scored_fail_reason(result, policy) is not None
    if result.task_type == TaskType.QUANTFIXEDTASK:
        return (1.0, math.inf) if is_scored_fail else (0.0, normalised_loss(result))
    if is_scored_fail:
        return (2.0, math.inf, math.inf)
    if not freeform_passed(result, policy):
        return (1.0, normalised_loss(result), math.inf)
    return (0.0, effective_bytes(result, policy), normalised_loss(result))


def rank_quant_results(results: list[MinerResultsText], policy: QuantFreeformPolicy) -> list[MinerResultsText]:
    return sorted(results, key=lambda result: quant_rank_key(result, policy))


def rank_metric_for(result: MinerResultsText, policy: QuantFreeformPolicy) -> float:
    """The value the rank key ordered on: effective bytes for an under-cap freeform entry, the normalised
    loss otherwise, inf for a scored fail."""
    if scored_fail_reason(result, policy) is not None:
        return math.inf
    if result.task_type == TaskType.QUANTFREEFORMTASK and freeform_passed(result, policy):
        return effective_bytes(result, policy)
    return normalised_loss(result)
