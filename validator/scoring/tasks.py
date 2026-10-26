from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from huggingface_hub import HfApi

import validator.evaluation.constants as eval_cst
import validator.infrastructure.service_constants as service_cst
import validator.scoring.constants as scoring_cst
import validator.tournament.constants as t_cst
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
from core.models.task_models import TaskType
from validator.db.sql.submissions_and_scoring import add_submission
from validator.db.sql.submissions_and_scoring import get_task_node_losses
from validator.db.sql.submissions_and_scoring import set_task_node_losses
from validator.db.sql.submissions_and_scoring import set_task_node_quality_score
from validator.db.sql.tasks import get_expected_repo_name
from validator.db.sql.tasks import get_nodes_assigned_to_task
from validator.db.sql.tournament_performance import is_paired_comparison_task
from validator.evaluation.basilica import EvaluationRetryableError
from validator.evaluation.docker_evaluation import run_evaluation_basilica_image
from validator.evaluation.docker_evaluation import run_evaluation_basilica_text
from validator.evaluation.notifications import notify_evaluation_exception
from validator.evaluation.notifications import task_deployment_ids_for_hotkeys
from validator.infrastructure.minio_client import async_minio_client
from validator.scoring.models import EvalHotkeyResults
from validator.scoring.models import MinerResults
from validator.scoring.models import MinerResultsImage
from validator.scoring.models import MinerResultsText
from validator.scoring.models import Submission
from validator.tasks.models import AnyTypeRawTask
from validator.tasks.models import InstructTextRawTask


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
        (RANKED_FIRST_REASON_PREFIX, RANKED_BELOW_REASON_PREFIX, BOTTOM_PENALTY_REASON_PREFIX)
    )


def calculate_miner_ranking_and_scores(
    miner_results: list[MinerResultsText | MinerResultsImage],
) -> list[MinerResultsText | MinerResultsImage]:
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
                result.score_reason = "Non-finetuned submission"
                logger.info(f"Miner {result.hotkey}: Non-finetuned, score initialized to 0.0")
            elif np.isnan(result.test_loss):
                result.score_reason = "Invalid test loss"
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
    else:
        raise ValueError(f"Unknown task type: {task.task_type}")


def _create_failed_miner_result(hotkey: str, score_reason: str, task_type: TaskType) -> MinerResults:
    """Create a result object for failed miner submissions with initial score of 0.0.
    The score may later be adjusted to a penalty if valid submissions exist."""
    if task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK, TaskType.ENVIRONMENTTASK]:
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
    if isinstance(eval_result.eval_loss, DiffusionLosses):
        text_guided_avg = (
            sum(eval_result.eval_loss.text_guided_losses) / len(eval_result.eval_loss.text_guided_losses)
            if eval_result.eval_loss.text_guided_losses
            else 0
        )

        no_text_avg = (
            sum(eval_result.eval_loss.no_text_losses) / len(eval_result.eval_loss.no_text_losses)
            if eval_result.eval_loss.no_text_losses
            else 0
        )

        weighted_loss = (
            eval_cst.DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT * text_guided_avg
            + (1 - eval_cst.DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT) * no_text_avg
        )
        return weighted_loss

    return None


async def _evaluate_submissions(
    task: AnyTypeRawTask,
    submission_repos: list[str],
    num_gpus: int,
    dataset_type: TextDatasetType | None = None,
    config: "Config | None" = None,
) -> dict[str, EvaluationResultText | EvaluationResultImage | Exception]:
    unique_repos = list(set(submission_repos))
    if len(unique_repos) != len(submission_repos):
        logger.warning(
            f"Found duplicate repos. Deduplicating {len(submission_repos)} repos to {len(unique_repos)} unique repos"
        )

    if task.task_type in [
        TaskType.INSTRUCTTEXTTASK,
        TaskType.DPOTASK,
        TaskType.GRPOTASK,
        TaskType.CHATTASK,
        TaskType.ENVIRONMENTTASK,
    ]:
        results: dict[str, EvaluationResultText | Exception] = {}
        repos_to_evaluate = []
        base_model = task.augmented_model_id or task.model_id
        for repo in unique_repos:
            if repo == base_model:
                logger.warning(f"Repository {repo} matches base model ID - marking as non-finetuned")
                results[repo] = EvaluationResultText(is_finetune=False, eval_loss=0.0)
            else:
                repos_to_evaluate.append(repo)

        if not repos_to_evaluate:
            return results

        assert task.test_data is not None, "Test data shouldn't be none for text tasks"

        eval_seed = None
        use_kl, kl_coef = (task.use_kl, task.kl_coef) if isinstance(task, InstructTextRawTask) else (False, None)
        continuous_sft_tokenizer_repo = t_cst.continuous_sft_seed_repo_for_ds(task.ds)

        # Per-example loss vectors are only read by the head-to-head comparisons (boss and knockout
        # rounds), and computing them costs an extra pass over the eval set - four extra forward
        # passes per pair for DPO - so only ask for them where they are used.
        emit_per_example_losses = (
            task.task_type in t_cst.PAIRED_BOSS_ROUND_TASK_TYPES
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
            results[repo] = Exception("Evaluation failed to complete")

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

    # The paired boss-round comparison is the only consumer of the per-example vectors, so they are
    # only stored for boss-round tasks. Checked once per task rather than once per miner.
    keep_per_example_losses = await is_paired_comparison_task(task.task_id, psql_db)

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
            )

            if result.submission:
                await add_submission(result.submission, psql_db)


def _result_from_persisted_row(task: AnyTypeRawTask, hotkey: str, row: dict | None) -> MinerResultsText | MinerResultsImage:
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

    if task.task_type == TaskType.IMAGETASK:
        return MinerResultsImage(
            hotkey=hotkey,
            test_loss=float(test_loss),
            synth_loss=float(synth_loss),
            is_finetune=True,
            score_reason=reason_for_ranking,
        )

    return MinerResultsText(
        hotkey=hotkey,
        test_loss=float(test_loss),
        synth_loss=float(synth_loss),
        is_finetune=True,
        score_reason=reason_for_ranking,
        task_type=task.task_type,
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

    miner_repos: dict[str, str] = {}
    results = []

    for miner in miners:
        with LogContext(miner_hotkey=miner.hotkey):
            expected_name = await get_expected_repo_name(task.task_id, miner.hotkey, config.psql_db)

            if not expected_name:
                logger.error(f"No expected repo name found for miner {miner.hotkey} on task {task.task_id}")
                results.append(
                    _create_failed_miner_result(
                        miner.hotkey, score_reason="No expected repo name found", task_type=task.task_type
                    )
                )
                continue

            repo = f"{service_cst.RAYONLABS_HF_USERNAME}/{expected_name}"
            try:
                HfApi().repo_info(repo, timeout=30)
            except Exception:
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
            eval_results = await _evaluate_submissions(
                task=task,
                submission_repos=list(miner_repos.values()),
                num_gpus=num_gpus,
                dataset_type=dataset_type or None,
                config=config,
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
                    elif task.task_type in [
                        TaskType.INSTRUCTTEXTTASK,
                        TaskType.DPOTASK,
                        TaskType.GRPOTASK,
                        TaskType.CHATTASK,
                        TaskType.ENVIRONMENTTASK,
                    ]:
                        test_result = eval_result
                    elif task.task_type == TaskType.IMAGETASK:
                        test_result = eval_result.model_copy(
                            update={"eval_loss": _calculate_weighted_loss_for_image_eval(eval_result)}
                        )
                    else:
                        raise ValueError(f"Unknown task type: {task.task_type}")

                    submission = Submission(
                        task_id=task.task_id,
                        hotkey=miner.hotkey,
                        repo=repo,
                        created_on=datetime.now(),
                        updated_on=datetime.now(),
                    )

                if task.task_type in [
                    TaskType.INSTRUCTTEXTTASK,
                    TaskType.DPOTASK,
                    TaskType.GRPOTASK,
                    TaskType.CHATTASK,
                    TaskType.ENVIRONMENTTASK,
                ]:
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
                    results.append(
                        MinerResultsText(
                            hotkey=miner.hotkey,
                            test_loss=float(test_result.eval_loss),
                            synth_loss=float(test_result.eval_loss),
                            is_finetune=test_result.is_finetune,
                            submission=submission,
                            task_type=task.task_type,
                            per_example_losses=test_result.per_example_losses if is_text_result else None,
                            eval_set_fingerprint=test_result.eval_set_fingerprint if is_text_result else None,
                            per_example_losses_failed=(
                                test_result.per_example_losses_failed if is_text_result else False
                            ),
                        )
                    )
                elif task.task_type == TaskType.IMAGETASK:
                    results.append(
                        MinerResultsImage(
                            hotkey=miner.hotkey,
                            test_loss=float(test_result.eval_loss),
                            synth_loss=float(test_result.eval_loss),
                            is_finetune=test_result.is_finetune,
                            submission=submission,
                        )
                    )
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")

        except EvaluationRetryableError:
            raise
        except Exception as e:
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


async def evaluate_and_score_hotkeys(
    task: AnyTypeRawTask,
    hotkeys: list[str],
    num_gpus: int,
    config: Config,
) -> EvalHotkeyResults:
    """Evaluate a subset of task hotkeys, persist raw losses, return results."""
    assert task.task_id is not None, "Task ID must be present"

    miner_pool = await get_nodes_assigned_to_task(str(task.task_id), config.psql_db)
    miner_pool = [miner for miner in miner_pool if miner.hotkey in set(hotkeys)]

    dataset_type = _get_dataset_type(task)
    logger.info(f"Beginning evaluation for task {task.task_id} with {len(miner_pool)} miners")
    task_results = await process_miners_pool(miner_pool, task, config, num_gpus, dataset_type)

    failed = [r.hotkey for r in task_results if (not r.is_finetune) or np.isnan(r.test_loss)]
    evaluated = [r.hotkey for r in task_results]
    await _persist_raw_task_results(task, task_results, config.psql_db)
    return EvalHotkeyResults(evaluated=evaluated, failed=failed)


async def finalize_task_scores_from_raw_losses(
    task: AnyTypeRawTask,
    hotkeys: list[str],
    config: Config,
) -> list[MinerResultsText | MinerResultsImage]:
    """Compute final rankings once all per-hotkey evaluations are terminal."""
    assert task.task_id is not None, "Task ID must be present"

    raw_rows = await get_task_node_losses(task.task_id, config.psql_db)
    raw_by_hotkey = {row.get("hotkey"): row for row in raw_rows}
    final_results = [_result_from_persisted_row(task, hotkey, raw_by_hotkey.get(hotkey)) for hotkey in hotkeys]

    final_results = calculate_miner_ranking_and_scores(final_results)
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
