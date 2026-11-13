from __future__ import annotations

import asyncio
import math
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Awaitable
from typing import Callable
from typing import TypeVar

import numpy as np
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

import validator.db.constants as db_cst
import validator.evaluation.constants as eval_cst
import validator.infrastructure.service_constants as service_cst
import validator.scoring.constants as scoring_cst
import validator.tournament.constants as t_cst
from core.constants.credentials import HUGGINGFACE_TOKEN
from core.logging import LogContext
from core.logging import get_logger
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import DpoDatasetType
from core.models.dataset_models import EnvironmentDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import GrpoDatasetType
from core.models.dataset_models import InstructTextDatasetType
from core.models.dataset_models import TextDatasetType
from core.models.payload_models import DiffusionLosses
from core.models.payload_models import EvaluationResultImage
from core.models.payload_models import EvaluationResultText
from core.models.quant_models import QuantFixedSpec
from core.models.quant_models import QuantFormat
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TEXT_TASK_TYPES
from core.models.task_models import TaskType
from core.models.task_models import higher_is_better
from validator.db.sql import tournaments as tournament_sql
from validator.db.sql.quant_references import add_quant_sku_reference
from validator.db.sql.quant_references import get_quant_sku_references
from validator.db.sql.submissions_and_scoring import add_submission
from validator.db.sql.submissions_and_scoring import get_quant_decode_rows
from validator.db.sql.submissions_and_scoring import get_task_node_losses
from validator.db.sql.submissions_and_scoring import set_task_node_losses
from validator.db.sql.submissions_and_scoring import set_task_node_quality_score
from validator.db.sql.tasks import get_expected_repo_name
from validator.db.sql.tasks import get_nodes_assigned_to_task
from validator.db.sql.tournament_performance import is_paired_comparison_task
from validator.evaluation.docker_evaluation import decoded_model_urls_for
from validator.evaluation.docker_evaluation import run_evaluation_basilica_image
from validator.evaluation.docker_evaluation import run_evaluation_basilica_quant
from validator.evaluation.docker_evaluation import run_evaluation_basilica_text
from validator.evaluation.errors import EvaluationInfraError
from validator.evaluation.errors import EvaluationRetryableError
from validator.evaluation.errors import RepoEvaluationFailed
from validator.evaluation.errors import as_infra_error
from validator.evaluation.errors import describe_error
from validator.evaluation.errors import is_infra_error
from validator.evaluation.notifications import notify_evaluation_exception
from validator.evaluation.notifications import task_deployment_ids_for_hotkeys
from validator.evaluation.quant_artifacts import HubArtifactListing
from validator.evaluation.quant_artifacts import artifact_bytes_from_hub
from validator.evaluation.quant_artifacts import artifact_listing_from_hub
from validator.evaluation.quant_structure import StructuralCheckUnavailable
from validator.evaluation.quant_structure import TensorSet
from validator.evaluation.quant_structure import expected_tensor_set_for
from validator.evaluation.quant_structure import fixed_submission_failure
from validator.infrastructure.minio_client import async_minio_client
from validator.scoring.models import EvalHotkeyResults
from validator.scoring.models import MinerResults
from validator.scoring.models import MinerResultsImage
from validator.scoring.models import MinerResultsText
from validator.scoring.models import QuantSkuReference
from validator.scoring.models import QuantSkuReferences
from validator.scoring.models import Submission
from validator.scoring.quant_ranking import SCORED_FAIL_REASON_PREFIX
from validator.scoring.quant_ranking import QuantFreeformPolicy
from validator.scoring.quant_ranking import eval_gpu_from_column
from validator.scoring.quant_ranking import freeform_ln_penalty
from validator.scoring.quant_ranking import freeform_passed
from validator.scoring.quant_ranking import normalised_loss
from validator.scoring.quant_ranking import rank_metric_for
from validator.scoring.quant_ranking import rank_quant_results
from validator.scoring.quant_ranking import reference_on
from validator.scoring.quant_ranking import scored_fail_reason
from validator.tasks.models import AnyTypeRawTask
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import freeform_penalty_policy
from validator.tournament.quant_reference import NO_REFERENCE_MODEL_FAILURE
from validator.tournament.quant_reference import is_own_model_start_point
from validator.tournament.quant_reference import quant_reference_model
from validator.tournament.quant_reference import reference_hotkey_for


if TYPE_CHECKING:
    from fiber.chain.models import Node

    from validator.app.config import Config


logger = get_logger(__name__)


# Score reasons produced by ranking below. Unlike failure/exclusion reasons (e.g. "Evaluation
# failed", "Non-finetuned submission"), these are recomputed on every finalize from the current
# pool, so they must not be treated as pre-set reasons when rebuilding results from persisted rows.
RANKED_FIRST_REASON_PREFIX = "Ranked 1st by "
RANKED_BELOW_REASON_PREFIX = "Ranked below top 1 by "
BOTTOM_PENALTY_REASON_PREFIX = "Bottom 25% ranked by "
NON_FINETUNED_REASON = "Non-finetuned submission"
INVALID_TEST_LOSS_REASON = "Invalid test loss"


def _is_ranking_derived_reason(reason: str | None) -> bool:
    """True for score reasons assigned by calculate_miner_ranking_and_scores from the ranking.

    These reasons encode a prior ranking, not an intrinsic reason to exclude a miner. When a task is
    re-finalized (e.g. after re-evaluating one hotkey), rebuilding results must drop these so every
    valid miner re-enters the ranking pool. Otherwise calculate_miner_ranking_and_scores skips every
    already-scored miner and crowns whichever hotkey was freshly evaluated.
    """
    if not reason:
        return False
    return reason.startswith(
        (RANKED_FIRST_REASON_PREFIX, RANKED_BELOW_REASON_PREFIX, BOTTOM_PENALTY_REASON_PREFIX, SCORED_FAIL_REASON_PREFIX)
    )


_RANKING_TYPES = {
    TaskType.GRPOTASK: "GRPO score (bigger is better)",
    TaskType.ENVIRONMENTTASK: "Environment score (bigger is better)",
    TaskType.QUANTFIXEDTASK: "quant test_loss / same-SKU reference",
    TaskType.QUANTFREEFORMTASK: "quant penalised bytes under cap",
}
# A freeform row over the cap is out of the bytes race and ranks after every under-cap row, closest normalised loss first.
FREEFORM_OVER_CAP_RANKING_TYPE = "quant normalised loss, over cap"


def _ranking_type_for(result: MinerResults, ranking_type: str) -> str:
    """What this row was actually ranked on: the task's ranking type, except for a freeform row over the cap
    (passed_bar is set by _set_freeform_verdict before ranking)."""
    if result.task_type == TaskType.QUANTFREEFORMTASK and result.passed_bar is False:
        return FREEFORM_OVER_CAP_RANKING_TYPE
    return ranking_type


def _quant_scored_fail_reason(result: MinerResults, policy: QuantFreeformPolicy) -> str | None:
    return scored_fail_reason(result, policy) if result.task_type in QUANT_TASK_TYPES else None


def _order_valid_results(valid_results: list[MinerResults], policy: QuantFreeformPolicy) -> list[tuple[MinerResults, float]]:
    """Best first, each paired with the metric it was ranked on."""
    task_type = valid_results[0].task_type
    if task_type in QUANT_TASK_TYPES:
        return [(result, rank_metric_for(result, policy)) for result in rank_quant_results(valid_results, policy)]

    ranked_results = [(result, result.test_loss) for result in valid_results]
    if higher_is_better(task_type):
        ranked_results.sort(key=lambda x: float("-inf") if math.isnan(x[1]) else -x[1])
    else:
        ranked_results.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])
    return ranked_results


def _set_freeform_verdict(miner_results: list[MinerResults], policy: QuantFreeformPolicy) -> None:
    """Record the cap verdict and ln P for every scored freeform loss, so a row under the cap scored-failed on
    bytes still reads as under it. ln P is None over the cap; a persisted value is kept when there is no policy."""
    for result in miner_results:
        if result.task_type != TaskType.QUANTFREEFORMTASK:
            continue
        result.passed_bar = math.isfinite(result.test_loss) and freeform_passed(result, policy)
        if policy is not None and math.isfinite(result.test_loss):
            result.freeform_penalty = freeform_ln_penalty(result, policy) if result.passed_bar else None


def _freeform_log_suffix(result: MinerResults, metric: float) -> str:
    """Why a freeform entry ranked where it did: r, effective size S and ln P; empty for other rows."""
    if result.task_type != TaskType.QUANTFREEFORMTASK:
        return ""
    ln_penalty = f"{result.freeform_penalty:.4f}" if result.freeform_penalty is not None else "n/a"
    return f" r={normalised_loss(result):.4f} S={metric:.0f} lnP={ln_penalty}"


def _score_quant_scored_fails(scored_fails: list[MinerResults], rank: int) -> None:
    """Scored fails rank after every valid result and take the penalty; they are not errors."""
    for result in scored_fails:
        with LogContext(miner_hotkey=result.hotkey):
            result.adjusted_loss = math.inf
            result.rank_metric = math.inf
            result.rank = rank
            result.score = scoring_cst.SCORE_PENALTY
            logger.info(f"Miner {result.hotkey}: {result.score_reason}, score={result.score}")


def _calculate_quant_ranking_and_scores(
    miner_results: list[MinerResultsText | MinerResultsImage],
    policy: QuantFreeformPolicy = None,
) -> list[MinerResultsText | MinerResultsImage]:
    """Rank and score one task's results. A quant task ranks on the normalised loss (adjusted_loss is
    test_loss / the same-SKU reference); policy prices and caps each freeform row, else its persisted
    passed_bar and freeform_penalty count."""
    logger.info("Beginning score calculation...")

    valid_results = []
    scored_fails = []
    # Initialize all scores to 0.0 and set appropriate reasons
    for result in miner_results:
        with LogContext(miner_hotkey=result.hotkey):
            result.score = 0.0
            # atp, we only set score_reason in these cases (all are invalid and is_finetune == False):
            # "Invalid/No repo submitted", "Evaluation failed", "Duplicated submission"
            if result.score_reason:
                continue
            elif not result.is_finetune:
                result.score_reason = NON_FINETUNED_REASON
                logger.info(f"Miner {result.hotkey}: Non-finetuned, score initialized to 0.0")
            elif fail_reason := _quant_scored_fail_reason(result, policy):
                result.score_reason = f"{SCORED_FAIL_REASON_PREFIX}{fail_reason}"
                scored_fails.append(result)
            elif np.isnan(result.test_loss):
                result.score_reason = INVALID_TEST_LOSS_REASON
                logger.info(f"Miner {result.hotkey}: Invalid test loss, score initialized to 0.0")
            else:
                valid_results.append(result)

    _set_freeform_verdict(miner_results, policy)

    if not valid_results:
        logger.warning("No valid finetuned submissions found. All scores set to 0.0")
        return miner_results

    _score_quant_scored_fails(scored_fails, rank=len(valid_results) + 1)

    task_type = valid_results[0].task_type
    ranking_type = _RANKING_TYPES.get(task_type, "test_loss")
    logger.info(f"Processing {task_type} - ranking by {ranking_type}")

    for result in valid_results:
        result.adjusted_loss = normalised_loss(result) if task_type in QUANT_TASK_TYPES else result.test_loss
        logger.info(f"Miner {result.hotkey}: test_loss {result.test_loss:.6f} adjusted_loss {result.adjusted_loss:.6f}")

    ranked_results = _order_valid_results(valid_results, policy)
    for position, (result, metric) in enumerate(ranked_results, start=1):
        result.rank = position
        result.rank_metric = metric

    if ranked_results:
        top_result, top_metric = ranked_results[0]
        with LogContext(miner_hotkey=top_result.hotkey):
            top_result.score = scoring_cst.FIRST_PLACE_SCORE
            top_result.score_reason = f"{RANKED_FIRST_REASON_PREFIX}{_ranking_type_for(top_result, ranking_type)}"
            logger.info(
                f"Miner {top_result.hotkey} (finetuned):"
                f" test_loss={top_result.test_loss:.4f}"
                f" {_ranking_type_for(top_result, ranking_type)}={top_metric:.4f}"
                f"{_freeform_log_suffix(top_result, top_metric)}"
                f" score={top_result.score:.4f}"
                f" score_reason={top_result.score_reason}"
            )

    total_valid_miners = len(valid_results)
    if total_valid_miners > scoring_cst.MIN_IDEAL_NUM_MINERS_IN_POOL:
        penalty_count = max(1, int(total_valid_miners * 0.25))
        penalty_start_idx = total_valid_miners - penalty_count

        for result, metric in ranked_results[1:penalty_start_idx]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score_reason = f"{RANKED_BELOW_REASON_PREFIX}{_ranking_type_for(result, ranking_type)}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" {_ranking_type_for(result, ranking_type)}={metric:.4f}"
                    f"{_freeform_log_suffix(result, metric)}"
                    f" score=0.0"
                    f" score_reason={result.score_reason}"
                )

        for result, metric in ranked_results[penalty_start_idx:]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score = scoring_cst.SCORE_PENALTY
                result.score_reason = f"{BOTTOM_PENALTY_REASON_PREFIX}{_ranking_type_for(result, ranking_type)}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" {_ranking_type_for(result, ranking_type)}={metric:.4f}"
                    f"{_freeform_log_suffix(result, metric)}"
                    f" score={result.score:.4f}"
                    f" score_reason={result.score_reason}"
                )
    else:
        for result, metric in ranked_results[1:]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score_reason = f"{RANKED_BELOW_REASON_PREFIX}{_ranking_type_for(result, ranking_type)}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" {_ranking_type_for(result, ranking_type)}={metric:.4f}"
                    f"{_freeform_log_suffix(result, metric)}"
                    f" score=0.0"
                    f" score_reason={result.score_reason}"
                )

    # Apply penalty scores to failed submissions when valid submissions exist
    if valid_results:
        for result in miner_results:
            # Find failed submissions that haven't been scored yet
            if (not result.is_finetune or np.isnan(result.test_loss)) and result.score == 0.0:
                result.score = scoring_cst.SCORE_PENALTY
                logger.info(
                    f"Miner {result.hotkey}: Failed submission ({result.score_reason}), "
                    f"applying penalty score {scoring_cst.SCORE_PENALTY}"
                )

    return miner_results


def calculate_miner_ranking_and_scores(
    miner_results: list[MinerResultsText | MinerResultsImage],
    policy: QuantFreeformPolicy = None,
) -> list[MinerResultsText | MinerResultsImage]:
    if any(isinstance(result, MinerResultsText) and result.task_type in QUANT_TASK_TYPES for result in miner_results):
        return _calculate_quant_ranking_and_scores(miner_results, policy)

    logger.info("Beginning score calculation...")

    valid_results = []
    # Initialize all scores to 0.0 and set appropriate reasons
    for result in miner_results:
        with LogContext(miner_hotkey=result.hotkey):
            result.score = 0.0
            # atp, we only set score_reason in these cases (all are invalid and is_finetune == False):
            # "Invalid/No repo submitted", "Evaluation failed", "Duplicated submission"
            if result.score_reason:
                continue
            elif not result.is_finetune:
                result.score_reason = NON_FINETUNED_REASON
                logger.info(f"Miner {result.hotkey}: Non-finetuned, score initialized to 0.0")
            elif np.isnan(result.test_loss):
                result.score_reason = INVALID_TEST_LOSS_REASON
                logger.info(f"Miner {result.hotkey}: Invalid test loss, score initialized to 0.0")
            else:
                valid_results.append(result)

    if not valid_results:
        logger.warning("No valid finetuned submissions found. All scores set to 0.0")
        return miner_results

    is_grpo_task = False
    if valid_results and isinstance(valid_results[0], MinerResultsText):
        is_grpo_task = valid_results[0].task_type == TaskType.GRPOTASK
        if is_grpo_task:
            logger.info("Processing GRPO task - higher loss is better")
        else:
            logger.info(f"Processing {valid_results[0].task_type} - using test_loss for ranking")

    is_env_task = False
    if valid_results and isinstance(valid_results[0], MinerResultsText):
        is_env_task = valid_results[0].task_type == TaskType.ENVIRONMENTTASK
        if is_env_task:
            logger.info("Processing Env task - higher score is better")
        else:
            logger.info(f"Processing {valid_results[0].task_type} - using test_loss for ranking")

    logger.info("Using test loss for ranking")
    ranked_results = []
    for result in valid_results:
        result.adjusted_loss = result.test_loss
        ranked_results.append((result, result.test_loss))
        logger.info(f"Miner {result.hotkey}: test_loss {result.test_loss:.6f}")

    if is_grpo_task:
        # For GRPO, sort in reverse order (higher value is better)
        ranked_results.sort(key=lambda x: float("-inf") if math.isnan(x[1]) else -x[1])
        ranking_type = "GRPO score (bigger is better)"
    elif is_env_task:
        # For Env taks, sort in reverse order (higher value is better)
        ranked_results.sort(key=lambda x: float("-inf") if math.isnan(x[1]) else -x[1])
        ranking_type = "Environment score (bigger is better)"
    else:
        # For other tasks, sort normally (lower loss is better)
        ranked_results.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])
        ranking_type = "test_loss"

    if ranked_results:
        top_result, top_metric = ranked_results[0]
        with LogContext(miner_hotkey=top_result.hotkey):
            top_result.score = scoring_cst.FIRST_PLACE_SCORE
            top_result.score_reason = f"{RANKED_FIRST_REASON_PREFIX}{ranking_type}"
            logger.info(
                f"Miner {top_result.hotkey} (finetuned):"
                f" test_loss={top_result.test_loss:.4f}"
                f" {ranking_type}={top_metric:.4f}"
                f" score={top_result.score:.4f}"
                f" score_reason={top_result.score_reason}"
            )

    total_valid_miners = len(valid_results)
    if total_valid_miners > scoring_cst.MIN_IDEAL_NUM_MINERS_IN_POOL:
        penalty_count = max(1, int(total_valid_miners * 0.25))
        penalty_start_idx = total_valid_miners - penalty_count

        for result, metric in ranked_results[1:penalty_start_idx]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score_reason = f"{RANKED_BELOW_REASON_PREFIX}{ranking_type}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" {ranking_type}={metric:.4f}"
                    f" score=0.0"
                    f" score_reason={result.score_reason}"
                )

        for result, metric in ranked_results[penalty_start_idx:]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score = scoring_cst.SCORE_PENALTY
                result.score_reason = f"{BOTTOM_PENALTY_REASON_PREFIX}{ranking_type}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" {ranking_type}={metric:.4f}"
                    f" score={result.score:.4f}"
                    f" score_reason={result.score_reason}"
                )
    else:
        for result, metric in ranked_results[1:]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score_reason = f"{RANKED_BELOW_REASON_PREFIX}{ranking_type}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" {ranking_type}={metric:.4f}"
                    f" score=0.0"
                    f" score_reason={result.score_reason}"
                )

    # Apply penalty scores to failed submissions when valid submissions exist
    if valid_results:
        for result in miner_results:
            # Find failed submissions that haven't been scored yet
            if (not result.is_finetune or np.isnan(result.test_loss)) and result.score == 0.0:
                result.score = scoring_cst.SCORE_PENALTY
                logger.info(
                    f"Miner {result.hotkey}: Failed submission ({result.score_reason}), "
                    f"applying penalty score {scoring_cst.SCORE_PENALTY}"
                )

    return miner_results


def _get_dataset_type(task: AnyTypeRawTask) -> TextDatasetType | None:
    if task.task_type == TaskType.INSTRUCTTEXTTASK:
        return InstructTextDatasetType(
            field_system=task.field_system,
            field_instruction=task.field_instruction,
            field_input=task.field_input,
            field_output=task.field_output,
            format=task.format,
            no_input_format=task.no_input_format,
        )
    elif task.task_type == TaskType.IMAGETASK:
        return None
    elif task.task_type == TaskType.DPOTASK:
        return DpoDatasetType(
            field_prompt=task.field_prompt,
            field_system=task.field_system,
            field_chosen=task.field_chosen,
            field_rejected=task.field_rejected,
            prompt_format=task.prompt_format,
            chosen_format=task.chosen_format,
            rejected_format=task.rejected_format,
        )
    elif task.task_type == TaskType.GRPOTASK:
        return GrpoDatasetType(
            field_prompt=task.field_prompt,
            reward_functions=task.reward_functions,
            extra_column=task.extra_column,
        )
    elif task.task_type == TaskType.ENVIRONMENTTASK:
        env_names = getattr(task, "environment_names", [])
        return EnvironmentDatasetType(
            environment_names=env_names or None
        )
    elif task.task_type == TaskType.CHATTASK:
        return ChatTemplateDatasetType(
            chat_template=task.chat_template,
            chat_column=task.chat_column,
            chat_role_field=task.chat_role_field,
            chat_content_field=task.chat_content_field,
            chat_user_reference=task.chat_user_reference,
            chat_assistant_reference=task.chat_assistant_reference,
        )
    elif task.task_type in QUANT_TASK_TYPES:
        return task.dataset_type
    else:
        raise ValueError(f"Unknown task type: {task.task_type}")


def _create_failed_miner_result(hotkey: str, score_reason: str, task_type: TaskType) -> MinerResults:
    """Create a result object for failed miner submissions with initial score of 0.0.
    The score may later be adjusted to a penalty if valid submissions exist."""
    if task_type in TEXT_TASK_TYPES:
        return MinerResultsText(
            hotkey=hotkey,
            test_loss=np.nan,
            synth_loss=np.nan,
            is_finetune=False,
            score=0.0,
            score_reason=score_reason,
            task_type=task_type,
        )
    else:
        return MinerResultsImage(
            hotkey=hotkey, test_loss=np.nan, synth_loss=np.nan, is_finetune=False, score=0.0, score_reason=score_reason
        )


def _calculate_weighted_loss_for_image_eval(eval_result: EvaluationResultImage) -> float:
    """The task's weighted image loss: DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT * mean(text-guided) + the rest * mean(no-text).

    NaN when the result carries no per-image losses or either loss vector is empty: nothing was measured, and a
    missing mean is never a loss of 0.
    """
    losses = eval_result.eval_loss
    if not isinstance(losses, DiffusionLosses) or not losses.text_guided_losses or not losses.no_text_losses:
        return math.nan

    text_guided_avg = sum(losses.text_guided_losses) / len(losses.text_guided_losses)
    no_text_avg = sum(losses.no_text_losses) / len(losses.no_text_losses)
    return (
        eval_cst.DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT * text_guided_avg
        + (1 - eval_cst.DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT) * no_text_avg
    )


def _per_image_losses(eval_result: EvaluationResultImage) -> list[float] | None:
    """The weighted loss of each test image, in test-image order; its mean is the task's weighted loss.

    None unless both loss vectors are non-empty and hold one value per test image (the same length).
    """
    losses = eval_result.eval_loss
    if not isinstance(losses, DiffusionLosses) or not losses.text_guided_losses:
        return None
    if len(losses.text_guided_losses) != len(losses.no_text_losses):
        return None
    weight = eval_cst.DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT
    return [
        weight * text_guided + (1 - weight) * no_text
        for text_guided, no_text in zip(losses.text_guided_losses, losses.no_text_losses)
    ]


def _image_miner_result(hotkey: str, eval_result: EvaluationResultImage, submission: Submission | None) -> MinerResultsImage:
    """One image entry's result. An entry that is not a fine-tune, or whose losses are empty, has no valid loss:
    NaN, with its reason, so it is never stored or ranked as a score."""
    weighted_loss = _calculate_weighted_loss_for_image_eval(eval_result)
    if not eval_result.is_finetune:
        return MinerResultsImage(
            hotkey=hotkey,
            test_loss=math.nan,
            synth_loss=math.nan,
            is_finetune=False,
            submission=submission,
            score_reason=NON_FINETUNED_REASON,
        )
    if math.isnan(weighted_loss):
        return MinerResultsImage(
            hotkey=hotkey,
            test_loss=math.nan,
            synth_loss=math.nan,
            is_finetune=True,
            submission=submission,
            score_reason=INVALID_TEST_LOSS_REASON,
        )
    return MinerResultsImage(
        hotkey=hotkey,
        test_loss=weighted_loss,
        synth_loss=weighted_loss,
        is_finetune=True,
        submission=submission,
        per_example_losses=_per_image_losses(eval_result),
    )


def _base_model_submission_result(task_type: TaskType) -> EvaluationResultText:
    """Submitting the base model unchanged: non-finetuned for training tasks, a scored fail for quant."""
    if task_type in QUANT_TASK_TYPES:
        return EvaluationResultText(
            is_finetune=True, eval_loss=None, quant_failure="submission is the uncompressed base model"
        )
    return EvaluationResultText(is_finetune=False, eval_loss=0.0)


def _quant_format_for(quant_spec: QuantFixedSpec | None) -> QuantFormat | None:
    return next((quant_format for quant_format in QuantFormat if quant_format.spec() == quant_spec), None)


T = TypeVar("T")


async def _read_hub_with_retries(read: Callable[[], Awaitable[T]], what: str) -> T:
    """Run one structural-check hub read, retrying StructuralCheckUnavailable in place with doubling backoff.

    When the hub is still unreadable after the last attempt this raises EvaluationInfraError: the caller's
    evaluation cycle resets the rows to pending and tries again next cycle. Hub trouble is never the miner's fault.
    """
    for attempt in range(1, scoring_cst.STRUCTURAL_CHECK_READ_ATTEMPTS + 1):
        try:
            return await read()
        except StructuralCheckUnavailable as e:
            if attempt == scoring_cst.STRUCTURAL_CHECK_READ_ATTEMPTS:
                raise EvaluationInfraError(f"structural check unavailable for {what}: {describe_error(e)}") from e
            delay = scoring_cst.STRUCTURAL_CHECK_READ_BACKOFF_SECONDS * 2 ** (attempt - 1)
            logger.warning(f"Structural check read failed for {what} (attempt {attempt}), retrying in {delay:.0f}s: {e}")
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def _structurally_checked_repos(
    task: AnyTypeRawTask,
    repos: list[str],
    reference_models: dict[str, str],
    results: dict[str, EvaluationResultText | Exception],
) -> list[str]:
    """Fixed-track repos whose tensor set matches what their reference model implies under the task's spec.

    A mismatch is a scored fail and is never dispatched, so it fails before a GPU-minute is spent. An unreadable
    hub is retried in place, then raises EvaluationInfraError (the whole evaluation is re-queued, nothing has
    been dispatched yet); it is never a scored fail. A repo's reference is its reference_models entry (its own
    model on an own-model task), else the entrants' common input.
    """
    expected_by_model: dict[str, TensorSet] = {}
    unsupported: dict[str, str] = {}
    passing = []
    for repo in repos:
        reference_model = reference_models.get(repo) or task.augmented_model_id or task.model_id
        if reference_model not in expected_by_model and reference_model not in unsupported:
            try:
                expected_by_model[reference_model] = await _read_hub_with_retries(
                    lambda: expected_tensor_set_for(reference_model, task.quant_spec),
                    f"base {reference_model} (task {task.task_id})",
                )
            except ValueError as e:
                # Our bug or a base we cannot derive a tensor set for: never the miner's fault, so an evaluation error.
                logger.error(f"Structural check cannot run for task {task.task_id}: unsupported base {reference_model}: {e}")
                unsupported[reference_model] = str(e)
        if reference_model in unsupported:
            reason = unsupported[reference_model]
            results[repo] = EvaluationInfraError(f"structural check cannot run: unsupported base {reference_model}: {reason}")
            continue
        expected = expected_by_model[reference_model]
        failure = await _read_hub_with_retries(lambda: fixed_submission_failure(repo, expected), repo)
        if failure is None:
            passing.append(repo)
        else:
            logger.info(f"Repository {repo} failed the structural check: {failure}")
            results[repo] = EvaluationResultText(is_finetune=True, eval_loss=None, quant_failure=failure)
    return passing


async def _evaluate_quant_submissions(
    task: AnyTypeRawTask,
    repos_to_evaluate: list[str],
    results: dict[str, EvaluationResultText | Exception],
    num_gpus: int,
    config: "Config | None",
    decoded_model_urls: dict[str, dict[str, str]] | None,
    repo_to_hotkey: dict[str, str],
) -> dict[str, EvaluationResultText | Exception]:
    """Score quant repos in vLLM. A freeform repo is scored from its decoded model in object storage;
    one whose decoded model is missing is an evaluation error (EvaluationInfraError: the miner did nothing wrong), not sent.

    A repo with no bf16 reference model is not sent: on an own-model task that is a miner without a starting model, its
    scored fail (NO_REFERENCE_MODEL_FAILURE, as finalize records it); on any other task the common input is missing, an
    evaluation error. A fixed repo is then held to the structural check (_structurally_checked_repos) and only the
    passing ones are dispatched."""
    assert task.test_data is not None, "Test data shouldn't be none for quant tasks"
    if task.task_type == TaskType.QUANTFREEFORMTASK:
        decoded_model_urls = decoded_model_urls or {}
        sendable = [repo for repo in repos_to_evaluate if repo in decoded_model_urls]
        for repo in [repo for repo in repos_to_evaluate if repo not in sendable]:
            logger.error(f"Repository {repo} has no decoded model in object storage - not evaluating it")
            results[repo] = EvaluationInfraError("decoded model not available in object storage")
        repos_to_evaluate = sendable

    reference_models, cached_references = await _quant_reference_inputs(task, repos_to_evaluate, repo_to_hotkey, config)
    if config is not None:
        for repo in [repo for repo in repos_to_evaluate if repo not in reference_models]:
            if is_own_model_start_point(task.training_start_point):
                results[repo] = EvaluationResultText(is_finetune=True, eval_loss=None, quant_failure=NO_REFERENCE_MODEL_FAILURE)
            else:
                results[repo] = EvaluationInfraError(f"no bf16 reference model for task {task.task_id}")
        repos_to_evaluate = [repo for repo in repos_to_evaluate if repo in reference_models]
    if task.task_type == TaskType.QUANTFIXEDTASK:
        passing = await _structurally_checked_repos(task, repos_to_evaluate, reference_models, results)
        repos_to_evaluate = [repo for repo in repos_to_evaluate if repo in passing]
    if not repos_to_evaluate:
        return results

    quant_spec = task.quant_spec if task.task_type == TaskType.QUANTFIXEDTASK else None
    logger.info(f"Starting quant test evaluation (bf16 references {reference_models}, cached {cached_references})")
    test_results = await run_evaluation_basilica_quant(
        dataset=task.test_data,
        models=repos_to_evaluate,
        original_model=task.augmented_model_id or task.model_id,
        dataset_type=task.dataset_type,
        file_format=FileFormat.JSON,
        num_gpus=num_gpus,
        quant_spec=quant_spec,
        quant_format=_quant_format_for(quant_spec),
        task_id=task.task_id,
        psql_db=config.psql_db if config is not None else None,
        decoded_model_urls=decoded_model_urls,
        reference_models=reference_models,
        cached_references=cached_references,
    )
    task.model_params_count = test_results.base_model_params_count
    for repo in repos_to_evaluate:
        results[repo] = test_results.results.get(repo, EvaluationInfraError(f"no evaluation result for {repo}"))
    if config is not None:
        await _record_quant_references(task, results, repo_to_hotkey, config)
    return results


async def _quant_reference_inputs(
    task: AnyTypeRawTask, repos: list[str], repo_to_hotkey: dict[str, str], config: "Config | None"
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Each repo's bf16 reference model and the loss already cached per reference_key for it, for the dispatch env.

    A repo with no reference model (an own-model miner without a starting model) is left out and logged:
    no deployment can measure its reference.
    """
    if config is None:
        return {}, {}
    references = await get_quant_sku_references(task.task_id, config.psql_db)
    reference_models: dict[str, str] = {}
    cached_references: dict[str, dict[str, float]] = {}
    for repo in repos:
        hotkey = repo_to_hotkey[repo]
        reference_model = await quant_reference_model(task, hotkey, config.psql_db)
        if reference_model is None:
            logger.error(f"Task {task.task_id}: no bf16 reference model for {hotkey} ({repo}); its loss cannot be normalised")
            continue
        reference_models[repo] = reference_model
        reference_hotkey = reference_hotkey_for(task, hotkey)
        cached_references[repo] = {
            sku: reference.loss for (ref_hotkey, sku), reference in references.items() if ref_hotkey == reference_hotkey
        }
    return reference_models, cached_references


async def _record_quant_references(
    task: AnyTypeRawTask,
    results: dict[str, EvaluationResultText | Exception],
    repo_to_hotkey: dict[str, str],
    config: Config,
) -> None:
    """Cache each same-deployment reference, first writer wins.

    Two deployments on one SKU can both measure before either is cached. Per-SKU kernels are deterministic,
    so the second must match the stored row exactly; a mismatch keeps the stored row and alerts.
    """
    for repo, result in results.items():
        if not isinstance(result, EvaluationResultText) or result.quant_reference is None:
            continue
        reference = result.quant_reference
        measured = QuantSkuReference(
            task_id=task.task_id,
            hotkey=reference_hotkey_for(task, repo_to_hotkey[repo]),
            sku=reference.eval_gpu.reference_key,
            model=reference.model,
            loss=reference.eval_loss,
            per_example_losses=reference.per_example_losses,
            eval_set_fingerprint=reference.eval_set_fingerprint,
            eval_gpu=reference.eval_gpu,
        )
        stored = await add_quant_sku_reference(measured, config.psql_db)
        if stored.matches(measured):
            logger.info(f"Task {task.task_id}: bf16 reference {measured.model} on {measured.sku} is {stored.loss:.6f}")
            continue
        message = (
            f"Task {task.task_id}: bf16 reference re-measured on {measured.sku} in the deployment for {repo} does not "
            f"match the cached one ({measured.model} {measured.loss!r} vs {stored.model} {stored.loss!r}); keeping the cached one"
        )
        logger.error(message)
        await notify_evaluation_exception(
            config,
            task_id=str(task.task_id),
            task_type=task.task_type,
            context="Quant reference re-measurement mismatch",
            error=ValueError(message),
            repos=[repo],
        )


async def _evaluate_submissions(
    task: AnyTypeRawTask,
    submission_repos: list[str],
    num_gpus: int,
    dataset_type: TextDatasetType | None = None,
    config: "Config | None" = None,
    decoded_model_urls: dict[str, dict[str, str]] | None = None,
    repo_to_hotkey: dict[str, str] | None = None,
) -> dict[str, EvaluationResultText | EvaluationResultImage | Exception]:
    """decoded_model_urls (freeform quant only): repo -> {filename: presigned url} of its decoded model.
    repo_to_hotkey (quant only): each repo's miner, whose bf16 reference its loss is normalised by."""
    unique_repos = list(set(submission_repos))
    if len(unique_repos) != len(submission_repos):
        logger.warning(
            f"Found duplicate repos. Deduplicating {len(submission_repos)} repos to {len(unique_repos)} unique repos"
        )

    if task.task_type in TEXT_TASK_TYPES:
        results: dict[str, EvaluationResultText | Exception] = {}
        repos_to_evaluate = []
        base_model = task.augmented_model_id or task.model_id
        for repo in unique_repos:
            if repo == base_model:
                logger.warning(f"Repository {repo} matches base model ID - marking as non-finetuned")
                results[repo] = _base_model_submission_result(task.task_type)
            else:
                repos_to_evaluate.append(repo)

        if not repos_to_evaluate:
            return results

        assert task.test_data is not None, "Test data shouldn't be none for text tasks"
        if task.task_type in QUANT_TASK_TYPES:
            return await _evaluate_quant_submissions(
                task, repos_to_evaluate, results, num_gpus, config, decoded_model_urls, repo_to_hotkey or {}
            )

        eval_seed = None
        use_kl, kl_coef = (task.use_kl, task.kl_coef) if isinstance(task, InstructTextRawTask) else (False, None)
        continuous_sft_tokenizer_repo = t_cst.continuous_sft_seed_repo_for_ds(task.ds)

        # Per-example loss vectors are only read by the head-to-head comparisons (boss and knockout
        # rounds), and computing them costs an extra pass over the eval set - four extra forward
        # passes per pair for DPO - so only ask for them where they are used.
        emit_per_example_losses = (
            task.task_type in (*t_cst.PAIRED_BOSS_ROUND_TASK_TYPES, *t_cst.QUANT_PAIRED_TASK_TYPES)
            and config is not None
            and task.task_id is not None
            and await is_paired_comparison_task(task.task_id, config.psql_db)
        )
        if emit_per_example_losses:
            logger.info(f"Head-to-head task {task.task_id}: requesting per-example losses for the paired comparison")

        evaluation_params = {
            "file_format": FileFormat.JSON,
            "original_model": base_model,
            "models": repos_to_evaluate,
            "dataset_type": dataset_type,
            "num_gpus": num_gpus,
            "eval_seed": eval_seed,
            "task_id": task.task_id,
            "psql_db": config.psql_db if config is not None else None,
            "use_kl": use_kl,
            "kl_coef": kl_coef,
            "continuous_sft_tokenizer_repo": continuous_sft_tokenizer_repo,
            "emit_per_example_losses": emit_per_example_losses,
        }

        logger.info("Starting test evaluation")
        test_results = await run_evaluation_basilica_text(dataset=task.test_data, **evaluation_params)

        test_eval_results = test_results.results
        task.model_params_count = test_results.base_model_params_count

        for repo in repos_to_evaluate:
            if isinstance(test_eval_results.get(repo), Exception):
                results[repo] = test_eval_results[repo]
            else:
                test_result = test_eval_results[repo]
                results[repo] = test_result

    elif task.task_type == TaskType.IMAGETASK:
        results: dict[str, EvaluationResultImage | Exception] = {}
        repos_to_evaluate = []
        base_model = task.augmented_model_id or task.model_id
        for repo in unique_repos:
            if repo == base_model:
                logger.warning(f"Repository {repo} matches base model ID - marking as non-finetuned")
                results[repo] = EvaluationResultImage(
                    eval_loss=DiffusionLosses(text_guided_losses=[0], no_text_losses=[0]), is_finetune=False
                )
            else:
                repos_to_evaluate.append(repo)

        if not repos_to_evaluate:
            return results

        evaluation_params = {
            "test_split_url": task.test_data,
            "original_model_repo": base_model,
            "models": repos_to_evaluate,
            "model_type": task.model_type,
            "training_mode": task.training_mode,
            "trigger_word": task.trigger_word,
            "num_gpus": num_gpus,
            "task_id": task.task_id,
            "psql_db": config.psql_db if config is not None else None,
        }

        assert task.test_data is not None, "Test data shouldn't be none for image tasks"
        logger.info("Starting image model evaluation")
        image_results = await run_evaluation_basilica_image(**evaluation_params)
        image_eval_results = image_results.results
        task.model_params_count = image_results.base_model_params_count
        for repo in repos_to_evaluate:
            results[repo] = image_eval_results[repo]

    for repo in unique_repos:
        if repo not in results:
            results[repo] = (
                EvaluationInfraError(f"no evaluation result for {repo}")
                if task.task_type in QUANT_TASK_TYPES
                else Exception("Evaluation failed to complete")
            )

    return results


async def _clear_up_s3(file_paths: list[str]) -> None:
    for file_path in file_paths:
        try:
            logger.info(f"files = {file_paths} and bucket is {service_cst.BUCKET_NAME}")
            object_name = file_path.split(service_cst.BUCKET_NAME + "/")[-1]
            logger.info(f"Deleting file {object_name} from MinIO bucket {service_cst.BUCKET_NAME}")
            await async_minio_client.delete_file(service_cst.BUCKET_NAME, object_name)
        except Exception as e:
            logger.error(f"Failed to delete file {file_path} from MinIO: {e}")


async def _update_scores(task: AnyTypeRawTask, task_results: list[MinerResultsText | MinerResultsImage], psql_db) -> None:
    assert task.task_id is not None, "task id needs to be set to update scores"
    for result in task_results:
        with LogContext(miner_hotkey=result.hotkey):
            if result.score is None:
                continue

            await set_task_node_quality_score(
                task_id=task.task_id,
                hotkey=result.hotkey,
                quality_score=float(result.score),
                test_loss=result.test_loss,
                synth_loss=result.synth_loss,
                score_reason=result.score_reason,
                psql_db=psql_db,
                passed_bar=result.passed_bar,
                freeform_penalty=result.freeform_penalty,
                quant_failure=result.quant_failure,
            )

            if result.submission:
                result.submission.score = result.score
                await add_submission(result.submission, psql_db)


async def _persist_raw_task_results(
    task: AnyTypeRawTask,
    task_results: list[MinerResultsText | MinerResultsImage],
    psql_db,
) -> None:
    assert task.task_id is not None, "task id needs to be set to persist losses"

    # The paired comparison reads the per-example vectors of head-to-head tasks, so those are stored. An image
    # tournament task also stores its per-image vector; no verdict reads it. Checked once per task, not per miner.
    keep_per_example_losses = await is_paired_comparison_task(task.task_id, psql_db) or (
        task.task_type == TaskType.IMAGETASK
        and await tournament_sql.get_tournament_task_by_task_id(str(task.task_id), psql_db) is not None
    )

    for result in task_results:
        with LogContext(miner_hotkey=result.hotkey):
            test_loss = None if np.isnan(result.test_loss) else float(result.test_loss)
            synth_loss = None if np.isnan(result.synth_loss) else float(result.synth_loss)
            await set_task_node_losses(
                task_id=task.task_id,
                hotkey=result.hotkey,
                test_loss=test_loss,
                synth_loss=synth_loss,
                score_reason=result.score_reason,
                psql_db=psql_db,
                per_example_losses=result.per_example_losses if keep_per_example_losses else None,
                eval_set_fingerprint=result.eval_set_fingerprint if keep_per_example_losses else None,
                artifact_bytes=result.artifact_bytes,
                decoder_bytes=result.decoder_bytes,
                quant_failure=result.quant_failure,
                eval_gpu=result.eval_gpu,
                artifact_files=result.artifact_files,
                artifact_revision=result.artifact_revision,
            )

            if result.submission:
                await add_submission(result.submission, psql_db)


def _scored_fail_from_persisted_row(
    task: AnyTypeRawTask, hotkey: str, row: dict, submission: Submission | None = None
) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=np.nan,
        synth_loss=np.nan,
        is_finetune=True,
        submission=submission,
        task_type=task.task_type,
        quant_failure=row[db_cst.QUANT_FAILURE],
        artifact_bytes=row.get(db_cst.ARTIFACT_BYTES),
        decoder_bytes=row.get(db_cst.DECODER_BYTES),
        freeform_penalty=row.get(db_cst.FREEFORM_PENALTY),
        eval_gpu=eval_gpu_from_column(row.get(db_cst.EVAL_GPU)),
    )


def _result_from_persisted_row(
    task: AnyTypeRawTask, hotkey: str, row: dict | None, references: QuantSkuReferences | None = None
) -> MinerResultsText | MinerResultsImage:
    """references (quant only) supplies the reference loss on the row's own SKU."""
    if row and row.get("quant_failure"):
        return _scored_fail_from_persisted_row(task, hotkey, row)

    score_reason = row.get("score_reason") if row else None
    test_loss = row.get("test_loss") if row else None
    synth_loss = row.get("synth_loss") if row else None

    if test_loss is None or synth_loss is None:
        return _create_failed_miner_result(
            hotkey,
            score_reason=score_reason or "Evaluation failed",
            task_type=task.task_type,
        )

    # Drop reasons that merely encode a prior ranking so this valid submission re-enters the pool on
    # re-finalize. Genuine exclusion reasons (e.g. "Non-finetuned submission") are preserved because
    # calculate_miner_ranking_and_scores keys off score_reason to keep such rows out of the ranking.
    reason_for_ranking = None if _is_ranking_derived_reason(score_reason) else score_reason

    eval_gpu = eval_gpu_from_column(row.get(db_cst.EVAL_GPU))

    if task.task_type == TaskType.IMAGETASK:
        return MinerResultsImage(
            hotkey=hotkey,
            test_loss=float(test_loss),
            synth_loss=float(synth_loss),
            is_finetune=True,
            score_reason=reason_for_ranking,
        )

    reference = None
    if task.task_type in QUANT_TASK_TYPES:
        reference = reference_on(reference_hotkey_for(task, hotkey), eval_gpu, references or {})
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=float(test_loss),
        synth_loss=float(synth_loss),
        is_finetune=True,
        score_reason=reason_for_ranking,
        task_type=task.task_type,
        artifact_bytes=row.get("artifact_bytes"),
        decoder_bytes=row.get("decoder_bytes"),
        passed_bar=row.get("passed_bar"),
        freeform_penalty=row.get(db_cst.FREEFORM_PENALTY),
        eval_gpu=eval_gpu,
        reference_loss=reference.loss if reference else None,
    )


def group_by_losses(task_results: list[MinerResults]) -> dict[float, list[tuple[str, str]]]:
    loss_groups: dict[float, list[tuple[str, str]]] = {}

    for result in task_results:
        if result.submission and not np.isnan(result.test_loss):
            loss = float(result.test_loss)
            if loss not in loss_groups:
                loss_groups[loss] = []
            loss_groups[loss].append((result.hotkey, result.submission.repo))

    return loss_groups


def get_hf_upload_timestamp(repo_url: str) -> datetime | None:
    try:
        repo_path = repo_url.replace("https://huggingface.co/", "").split("/tree/")[0]
        api = HfApi()

        model_info = api.model_info(repo_path, timeout=5.0)
        if model_info and model_info.lastModified:
            return model_info.lastModified

    except Exception as e:
        logger.error(f"Failed to get upload timestamp for {repo_url}: {e}")
    return None


def _hub_artifact_bytes(repo: str) -> int | None:
    """The counted artifact size, measured from what the trainer uploaded; None when HF cannot say.

    An unreachable hub raises EvaluationInfraError: a missing byte count makes an under-cap freeform row a scored
    fail, which hub trouble must never cause."""
    try:
        return artifact_bytes_from_hub(repo, token=HUGGINGFACE_TOKEN)
    except Exception as e:
        if is_infra_error(e):
            raise as_infra_error(e, f"artifact bytes of {repo}") from e
        logger.error(f"Could not measure artifact bytes for {repo}: {describe_error(e)}")
        return None


def _hub_artifact_listing(repo: str) -> HubArtifactListing | None:
    """Freeform: the counted bytes, file hashes and revision from one hub listing; None when HF cannot say.

    An unreachable hub raises EvaluationInfraError, as for _hub_artifact_bytes."""
    try:
        return artifact_listing_from_hub(repo, token=HUGGINGFACE_TOKEN)
    except Exception as e:
        if is_infra_error(e):
            raise as_infra_error(e, f"artifact listing of {repo}") from e
        logger.error(f"Could not list artifact files for {repo}: {describe_error(e)}")
        return None


def _new_submission(task: AnyTypeRawTask, hotkey: str, repo: str) -> Submission:
    return Submission(
        task_id=task.task_id,
        hotkey=hotkey,
        repo=repo,
        created_on=datetime.now(),
        updated_on=datetime.now(),
    )


async def _persisted_quant_rows(task: AnyTypeRawTask, config: Config) -> dict[str, dict]:
    """task_nodes rows by hotkey for a quant task (decode-job bytes and scored fails); empty otherwise."""
    if task.task_type not in QUANT_TASK_TYPES:
        return {}
    return {row[db_cst.HOTKEY]: row for row in await get_task_node_losses(task.task_id, config.psql_db)}


async def process_miners_pool(
    miners: list[Node],
    task: AnyTypeRawTask,
    config: Config,
    num_gpus: int,
    dataset_type: TextDatasetType | None = None,
) -> list[MinerResultsText | MinerResultsImage]:
    assert task.task_id is not None, "We should have a task id when processing miners"

    # Environment tournaments were sunset: their evaluation runtime no longer exists. Fail closed
    # if a legacy ENVIRONMENTTASK still reaches scoring so we never dispatch a deleted eval path.
    if task.task_type == TaskType.ENVIRONMENTTASK:
        logger.warning(
            f"Skipping scoring for environment task {task.task_id}: environment tournaments are no longer supported"
        )
        return []

    is_quant = task.task_type in QUANT_TASK_TYPES
    miner_repos: dict[str, str] = {}
    results = []
    persisted = await _persisted_quant_rows(task, config)

    for miner in miners:
        with LogContext(miner_hotkey=miner.hotkey):
            expected_name = await get_expected_repo_name(task.task_id, miner.hotkey, config.psql_db)

            if not expected_name:
                if is_quant:
                    # We assign the repo name when the miner is attached, so its absence is our bug, not the miner's.
                    raise EvaluationInfraError(f"No expected repo name found for miner {miner.hotkey} on task {task.task_id}")
                logger.error(f"No expected repo name found for miner {miner.hotkey} on task {task.task_id}")
                results.append(
                    _create_failed_miner_result(
                        miner.hotkey, score_reason="No expected repo name found", task_type=task.task_type
                    )
                )
                continue

            repo = f"{service_cst.RAYONLABS_HF_USERNAME}/{expected_name}"
            persisted_row = persisted.get(miner.hotkey)
            if persisted_row and persisted_row.get(db_cst.QUANT_FAILURE):
                # The decode job already scored this entry as a fail: ranked last, never sent to a GPU.
                logger.info(f"Miner {miner.hotkey}: persisted quant scored fail, skipping evaluation")
                submission = _new_submission(task, miner.hotkey, repo)
                results.append(_scored_fail_from_persisted_row(task, miner.hotkey, persisted_row, submission))
                continue
            try:
                HfApi(token=HUGGINGFACE_TOKEN if is_quant else None).repo_info(repo, timeout=30)
            except Exception as e:
                if is_quant and not isinstance(e, RepositoryNotFoundError):
                    # Only an answer that the repo does not exist is the miner's: a timeout, a 5xx or anything else is the
                    # hub or us, and must never score the miner 0.
                    raise as_infra_error(e, f"repo_info for {repo}") from e
                logger.warning(f"Repo {repo} not found for miner {miner.hotkey} — scoring 0")
                results.append(
                    _create_failed_miner_result(
                        miner.hotkey, score_reason="Model repo not found on HuggingFace", task_type=task.task_type
                    )
                )
                continue
            logger.info(f"Constructed repo {repo} for miner {miner.hotkey}")
            miner_repos[miner.hotkey] = repo

    if miner_repos:
        try:
            decoded_model_urls = None
            if task.task_type == TaskType.QUANTFREEFORMTASK:
                decoded_model_urls = await decoded_model_urls_for(task.task_id, miner_repos, bucket=service_cst.BUCKET_NAME)
            eval_results = await _evaluate_submissions(
                task=task,
                submission_repos=list(miner_repos.values()),
                num_gpus=num_gpus,
                dataset_type=dataset_type or None,
                config=config,
                decoded_model_urls=decoded_model_urls,
                repo_to_hotkey={repo: hotkey for hotkey, repo in miner_repos.items()},
            )

            # Only the evaluator's own error report about a repo (RepoEvaluationFailed) is the miner's failure. Any
            # other exception in the results is infrastructure or our code: the whole call is re-queued, never scored.
            not_miners_fault = {
                hotkey: describe_error(eval_results[repo])
                for hotkey, repo in miner_repos.items()
                if is_quant
                and isinstance(eval_results.get(repo), Exception)
                and not isinstance(eval_results.get(repo), RepoEvaluationFailed)
            }
            if not_miners_fault:
                raise EvaluationInfraError(
                    "Evaluation errors that are not the miner's fault: "
                    + "; ".join(f"{hotkey}: {error}" for hotkey, error in not_miners_fault.items())
                )

            for miner in miners:
                with LogContext(miner_hotkey=miner.hotkey):
                    if miner.hotkey not in miner_repos:
                        continue

                    repo = miner_repos[miner.hotkey]
                    eval_result = eval_results.get(repo)

                    if isinstance(eval_result, Exception):
                        logger.error(f"Evaluation failed for miner {miner.hotkey}: {eval_result}")
                        await notify_evaluation_exception(
                            config,
                            task_id=str(task.task_id),
                            task_type=task.task_type,
                            context="Miner evaluation result exception",
                            error=eval_result,
                            hotkeys=[miner.hotkey],
                            repos=[repo],
                            deployment_ids=await task_deployment_ids_for_hotkeys(
                                task.task_id,
                                config,
                                [miner.hotkey],
                            ),
                        )
                        results.append(
                            _create_failed_miner_result(
                                miner.hotkey,
                                score_reason=f"Evaluation failed: {str(eval_result)[:350]}",
                                task_type=task.task_type,
                            )
                        )
                        continue
                    elif task.task_type in TEXT_TASK_TYPES or task.task_type == TaskType.IMAGETASK:
                        test_result = eval_result
                    else:
                        raise ValueError(f"Unknown task type: {task.task_type}")

                    submission = _new_submission(task, miner.hotkey, repo)

                if task.task_type in TEXT_TASK_TYPES:
                    # Only the instruct and DPO evaluators emit per-example losses; every other
                    # text evaluator leaves these None and the boss round falls back to the
                    # scalar margin.
                    is_text_result = isinstance(test_result, EvaluationResultText)
                    if is_text_result and test_result.per_example_losses_failed:
                        logger.error(
                            "Repo %s: per-example loss extraction failed after eval_loss; "
                            "boss-round paired comparison unavailable for hotkey %s",
                            repo,
                            miner.hotkey,
                        )
                    # A quant scored fail carries no loss; NaN keeps it out of the loss-based paths.
                    test_loss = np.nan if is_quant and test_result.eval_loss is None else float(test_result.eval_loss)
                    artifact_bytes = test_result.artifact_bytes if is_text_result else None
                    decoder_bytes = test_result.decoder_bytes if is_text_result else None
                    artifact_listing = None
                    if task.task_type == TaskType.QUANTFREEFORMTASK:
                        # One hub call yields the bytes, the file hashes and the revision for the scorecard.
                        artifact_listing = await asyncio.to_thread(_hub_artifact_listing, repo)
                        if artifact_bytes is None and artifact_listing is not None:
                            artifact_bytes = artifact_listing.bytes
                    if task.task_type in QUANT_TASK_TYPES:
                        if artifact_bytes is None and artifact_listing is None:
                            artifact_bytes = await asyncio.to_thread(_hub_artifact_bytes, repo)
                        if decoder_bytes is None:
                            # set_task_node_losses rewrites the bytes columns, so the decode job's count must be carried.
                            decoder_bytes = persisted.get(miner.hotkey, {}).get(db_cst.DECODER_BYTES)
                    results.append(
                        MinerResultsText(
                            hotkey=miner.hotkey,
                            test_loss=test_loss,
                            synth_loss=test_loss,
                            is_finetune=test_result.is_finetune,
                            submission=submission,
                            task_type=task.task_type,
                            per_example_losses=test_result.per_example_losses if is_text_result else None,
                            eval_set_fingerprint=test_result.eval_set_fingerprint if is_text_result else None,
                            per_example_losses_failed=(
                                test_result.per_example_losses_failed if is_text_result else False
                            ),
                            quant_failure=test_result.quant_failure if is_text_result else None,
                            artifact_bytes=artifact_bytes,
                            decoder_bytes=decoder_bytes,
                            artifact_files=artifact_listing.files if artifact_listing else None,
                            artifact_revision=artifact_listing.revision if artifact_listing else None,
                            eval_gpu=test_result.eval_gpu if is_text_result else None,
                        )
                    )
                elif task.task_type == TaskType.IMAGETASK:
                    results.append(_image_miner_result(miner.hotkey, test_result, submission))
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")

        except EvaluationRetryableError:
            raise
        except Exception as e:
            if is_quant:
                # An exception out of the evaluation call is infrastructure or our code, never a verdict on a repo: re-queue
                # (lifecycle counts, alerts and eventually holds the rows) rather than score every miner in the batch 0.
                logger.error(f"Error during batch evaluation: {describe_error(e)}", exc_info=True)
                raise as_infra_error(e, "Batch evaluation") from e
            logger.error(f"Error during batch evaluation: {e}", exc_info=True)
            await notify_evaluation_exception(
                config,
                task_id=str(task.task_id),
                task_type=task.task_type,
                context="Batch evaluation exception",
                error=e,
                hotkeys=[miner.hotkey for miner in miners],
                repos=list(miner_repos.values()),
                deployment_ids=await task_deployment_ids_for_hotkeys(
                    task.task_id,
                    config,
                    [miner.hotkey for miner in miners],
                ),
            )
            results.extend(
                [
                    _create_failed_miner_result(
                        miner.hotkey, score_reason=f"Evaluation failed: {str(e)[:350]}", task_type=task.task_type
                    )
                    for miner in miners
                    if miner.hotkey not in [r.hotkey for r in results]
                ]
            )

    return results


async def _undecoded_freeform_hotkeys(task: AnyTypeRawTask, hotkeys: list[str], config: Config) -> list[str]:
    """Freeform hotkeys that are not ready to evaluate yet; they stay pending and are retried next cycle.

    Tournament hotkeys are seeded for evaluation the moment training succeeds, which is before the trainer's
    decode job has settled. Scoring such a hotkey would persist an infra gap (an unfinished or timed-out decode
    upload) as a miner fail, so wait for a persisted decode outcome. A settled quant_failure is the miner's
    own scored fail and is evaluated as usual; a settled-ok decode whose decoded index is missing from object
    storage is our problem, so it waits too.
    """
    if task.task_type != TaskType.QUANTFREEFORMTASK:
        return []
    # Organic freeform decodes are never persisted (no participant to fault), so only tournament tasks are gated.
    if await tournament_sql.get_tournament_id_by_task_id(str(task.task_id), config.psql_db) is None:
        return []
    rows = await get_quant_decode_rows(task.task_id, config.psql_db)
    unsettled = [hotkey for hotkey in hotkeys if hotkey not in rows or not rows[hotkey].settled]
    decoded_ok = [hotkey for hotkey in hotkeys if hotkey in rows and rows[hotkey].settled and not rows[hotkey].quant_failure]
    indexed = await decoded_model_urls_for(
        task.task_id, {hotkey: hotkey for hotkey in decoded_ok}, bucket=service_cst.BUCKET_NAME
    )
    missing_index = [hotkey for hotkey in decoded_ok if hotkey not in indexed]
    if unsettled:
        logger.info(f"Task {task.task_id}: freeform decode not settled for {unsettled}; deferring their evaluation")
    if missing_index:
        logger.error(
            f"Task {task.task_id}: decode settled ok but no decoded model in object storage for {missing_index}; deferring"
        )
    return unsettled + missing_index


async def evaluate_and_score_hotkeys(
    task: AnyTypeRawTask,
    hotkeys: list[str],
    num_gpus: int,
    config: Config,
) -> EvalHotkeyResults:
    """Evaluate a subset of task hotkeys, persist raw losses, return results.

    A quant scored fail (quant_failure set) counts as evaluated, not failed: the submission was
    judged and ranks last with the penalty, so it never trips the 'no evaluation succeeded' path.
    """
    assert task.task_id is not None, "Task ID must be present"

    miner_pool = await get_nodes_assigned_to_task(str(task.task_id), config.psql_db)
    deferred = await _undecoded_freeform_hotkeys(task, hotkeys, config)
    miner_pool = [miner for miner in miner_pool if miner.hotkey in set(hotkeys) and miner.hotkey not in deferred]
    if not miner_pool:
        return EvalHotkeyResults(evaluated=[], deferred=deferred)

    dataset_type = _get_dataset_type(task)
    logger.info(f"Beginning evaluation for task {task.task_id} with {len(miner_pool)} miners")
    task_results = await process_miners_pool(miner_pool, task, config, num_gpus, dataset_type)

    failed = [r.hotkey for r in task_results if (not r.is_finetune) or (np.isnan(r.test_loss) and not r.quant_failure)]
    evaluated = [r.hotkey for r in task_results]
    await _persist_raw_task_results(task, task_results, config.psql_db)
    return EvalHotkeyResults(evaluated=evaluated, failed=failed, deferred=deferred)


async def finalize_task_scores_from_raw_losses(
    task: AnyTypeRawTask,
    hotkeys: list[str],
    config: Config,
) -> list[MinerResultsText | MinerResultsImage]:
    """Compute final rankings once all per-hotkey evaluations are terminal.

    A quant row is ranked on its loss over the bf16 reference cached for its SKU; one without a reference
    raises QuantReferenceUnavailable (lifecycle re-evaluates such rows before it gets here).
    """
    assert task.task_id is not None, "Task ID must be present"

    raw_rows = await get_task_node_losses(task.task_id, config.psql_db)
    raw_by_hotkey = {row.get("hotkey"): row for row in raw_rows}
    is_quant = task.task_type in QUANT_TASK_TYPES
    references = await get_quant_sku_references(task.task_id, config.psql_db) if is_quant else {}

    policy = freeform_penalty_policy(task) if task.task_type == TaskType.QUANTFREEFORMTASK else None

    final_results = [_result_from_persisted_row(task, hotkey, raw_by_hotkey.get(hotkey), references) for hotkey in hotkeys]

    final_results = calculate_miner_ranking_and_scores(final_results, policy=policy)
    await _update_scores(task, final_results, config.psql_db)
    return final_results


def has_disk_cache_error(task_results: list[MinerResultsText | MinerResultsImage]) -> bool:
    try:
        for result in task_results:
            if "Cannot find the requested files in the disk cache" in str(result.score_reason):
                return True
    except Exception as e:
        logger.error(f"Error checking for disk cache error: {e}")
        return False
    return False
