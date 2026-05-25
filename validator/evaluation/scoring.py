import asyncio
import math
import os
from datetime import datetime

import numpy as np
from fiber.chain.models import Node
from huggingface_hub import HfApi

import validator.core.constants as cts
from core import constants as core_cst
from core.models.payload_models import DiffusionLosses
from core.models.payload_models import EvaluationResultImage
from core.models.payload_models import EvaluationResultText
from core.models.pvp_models import PvPEnvironmentResult, PvPEvalMetadata, PvPGroupModelSpec, PvPGroupResults, PvPIncompleteError, PvPPairDbRow, PvPPairResult, _canonical_pair_key

PairKey = str  # sorted "hotkey_a:hotkey_b"
from core.models.scoring_models import EvalHotkeyResults
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from core.models.utility_models import TextDatasetType
from core.models.utility_models import TrainingStatus
from core.utils import download_s3_file
from validator.core.config import Config
from validator.core.models import AnyTypeRawTask
from validator.core.models import MinerResults
from validator.core.models import MinerResultsImage
from validator.core.models import MinerResultsText
from validator.core.models import Submission
from validator.db.sql.submissions_and_scoring import add_submission
from validator.db.sql.submissions_and_scoring import get_task_node_losses
from validator.db.sql.submissions_and_scoring import set_task_node_losses
from validator.db.sql.submissions_and_scoring import set_task_node_quality_score
from validator.db.sql.tasks import get_env_task_eval_seed
from validator.db.sql.tasks import get_expected_repo_name
from validator.db.sql.tasks import get_nodes_assigned_to_task
from validator.db.sql import tournaments as tournament_sql
from validator.db.sql.tournaments import get_tournament_id_by_task_id
from validator.db.sql.tournaments import get_training_status_for_task_and_hotkeys
from validator.evaluation.docker_evaluation import run_evaluation_basilica_image
from validator.evaluation.docker_evaluation import run_evaluation_basilica_text
from validator.evaluation.docker_evaluation import run_evaluation_pvp_group
from validator.evaluation.docker_evaluation import run_evaluation_pvp_pair
from validator.evaluation.tournament_scoring import compute_pvp_tournament_points
from validator.utils.logging import LogContext
from validator.utils.logging import add_context_tag
from validator.utils.logging import get_logger
from validator.utils.minio import async_minio_client


logger = get_logger(__name__)

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
            top_result.score = cts.FIRST_PLACE_SCORE
            top_result.score_reason = f"Ranked 1st by {ranking_type}"
            logger.info(
                f"Miner {top_result.hotkey} (finetuned):"
                f" test_loss={top_result.test_loss:.4f}"
                f" {ranking_type}={top_metric:.4f}"
                f" score={top_result.score:.4f}"
                f" score_reason={top_result.score_reason}"
            )

    total_valid_miners = len(valid_results)
    if total_valid_miners > cts.MIN_IDEAL_NUM_MINERS_IN_POOL:
        penalty_count = max(1, int(total_valid_miners * 0.25))
        penalty_start_idx = total_valid_miners - penalty_count

        for result, metric in ranked_results[1:penalty_start_idx]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score_reason = f"Ranked below top 1 by {ranking_type}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" {ranking_type}={metric:.4f}"
                    f" score=0.0"
                    f" score_reason={result.score_reason}"
                )

        for result, metric in ranked_results[penalty_start_idx:]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score = cts.SCORE_PENALTY
                result.score_reason = f"Bottom 25% ranked by {ranking_type}"
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
                result.score_reason = f"Ranked below top 1 by {ranking_type}"
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
                result.score = cts.SCORE_PENALTY
                logger.info(
                    f"Miner {result.hotkey}: Failed submission ({result.score_reason}), "
                    f"applying penalty score {cts.SCORE_PENALTY}"
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
            cts.DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT * text_guided_avg + (1 - cts.DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT) * no_text_avg
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
        logger.warning(f"Found duplicate repos. Deduplicating {len(submission_repos)} repos to {len(unique_repos)} unique repos")

    if task.task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK, TaskType.ENVIRONMENTTASK]:
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

        if task.task_type != TaskType.ENVIRONMENTTASK:
            assert task.test_data is not None, "Test data shouldn't be none for text tasks"

        # Fetch eval_seed for environment tasks
        eval_seed = None
        if task.task_type == TaskType.ENVIRONMENTTASK and config is not None and task.task_id is not None:
            eval_seed = await get_env_task_eval_seed(task.task_id, config.psql_db)
            logger.info(f"Fetched eval_seed={eval_seed} for environment task {task.task_id}")

        evaluation_params = {
            "file_format": FileFormat.JSON,
            "original_model": base_model,
            "models": repos_to_evaluate,
            "dataset_type": dataset_type,
            "num_gpus": num_gpus,
            "eval_seed": eval_seed,
            "task_id": task.task_id,
            "psql_db": config.psql_db if config is not None else None,
        }

        logger.info("Starting test evaluation")
        if task.task_type != TaskType.ENVIRONMENTTASK:
            test_results = await run_evaluation_basilica_text(dataset=task.test_data, **evaluation_params)
        else:
            test_results = await run_evaluation_basilica_text(dataset="proxy", **evaluation_params)

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
                    eval_losses=DiffusionLosses(text_guided_losses=[0], no_text_losses=[0]), is_finetune=False
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
            logger.info(f"files = {file_paths} and bucket is {cts.BUCKET_NAME}")
            object_name = file_path.split(cts.BUCKET_NAME + "/")[-1]
            logger.info(f"Deleting file {object_name} from MinIO bucket {cts.BUCKET_NAME}")
            await async_minio_client.delete_file(cts.BUCKET_NAME, object_name)
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


async def _persist_raw_task_results(task: AnyTypeRawTask, task_results: list[MinerResultsText | MinerResultsImage], psql_db) -> None:
    assert task.task_id is not None, "task id needs to be set to persist losses"
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

    if task.task_type == TaskType.IMAGETASK:
        return MinerResultsImage(
            hotkey=hotkey,
            test_loss=float(test_loss),
            synth_loss=float(synth_loss),
            is_finetune=True,
            score_reason=score_reason,
        )

    return MinerResultsText(
        hotkey=hotkey,
        test_loss=float(test_loss),
        synth_loss=float(synth_loss),
        is_finetune=True,
        score_reason=score_reason,
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

            repo = f"{cts.RAYONLABS_HF_USERNAME}/{expected_name}"
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

    if miner_repos and should_use_pvp(task):
        try:
            results.extend(await _run_pvp_group_eval(task, miner_repos, config))
        except PvPIncompleteError:
            raise
        except Exception as e:
            logger.error(f"PvP group evaluation failed: {e}", exc_info=True)
            raise PvPIncompleteError(f"PvP eval failed, will retry: {e}") from e
    elif miner_repos:
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
                        results.append(
                            _create_failed_miner_result(
                                miner.hotkey,
                                score_reason=f"Evaluation failed: {str(eval_result)[:350]}",
                                task_type=task.task_type,
                            )
                        )
                        continue
                    elif task.task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK, TaskType.ENVIRONMENTTASK]:
                        test_result = eval_result
                    elif task.task_type == TaskType.IMAGETASK:
                        test_result = eval_result
                        test_result.eval_loss = _calculate_weighted_loss_for_image_eval(test_result)
                    else:
                        raise ValueError(f"Unknown task type: {task.task_type}")

                    submission = Submission(
                        task_id=task.task_id,
                        hotkey=miner.hotkey,
                        repo=repo,
                        created_on=datetime.now(),
                        updated_on=datetime.now(),
                    )

                if task.task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK, TaskType.ENVIRONMENTTASK]:
                    results.append(
                        MinerResultsText(
                            hotkey=miner.hotkey,
                            test_loss=float(test_result.eval_loss),
                            synth_loss=float(test_result.eval_loss),
                            is_finetune=test_result.is_finetune,
                            submission=submission,
                            task_type=task.task_type,
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

        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}", exc_info=True)
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


def should_use_pvp(task: AnyTypeRawTask) -> bool:
    """Check if this task should use PvP evaluation based on its games' eval_type."""
    if task.task_type != TaskType.ENVIRONMENTTASK:
        return False
    env_names = getattr(task, "environment_names", None)
    if not env_names:
        return False
    for name in env_names:
        env_config = core_cst.ENVIRONMENT_CONFIGS.get(name)
        if env_config and env_config.eval_type == core_cst.EvalType.PVP:
            return True
    return False


async def _run_pvp_group_eval(
    task: AnyTypeRawTask,
    miner_repos: dict[str, str],
    config: Config,
) -> list[MinerResultsText]:
    """Run PvP group eval and convert standings to MinerResultsText."""
    base_model = task.augmented_model_id or task.model_id
    environment_names = getattr(task, "environment_names", None) or list(core_cst.EnvironmentName)

    eval_seed = await get_env_task_eval_seed(task.task_id, config.psql_db)
    seed = eval_seed if eval_seed is not None else cts.ENV_EVAL_DEFAULT_SEED

    participants = [
        PvPGroupModelSpec(repo=repo, hotkey=hotkey)
        for hotkey, repo in miner_repos.items()
    ]

    logger.info(f"PvP group eval: task={task.task_id}, {len(participants)} participants, envs={environment_names}")

    # Check if all pairs already complete in DB — skip Basilica entirely
    all_hotkeys = list(miner_repos.keys())
    env_name_strs = [e.value for e in environment_names]
    db_rows = await tournament_sql.get_pvp_pair_results(str(task.task_id), config.psql_db)
    rows_by_pair = _group_db_rows_by_pair(db_rows)
    max_pair_attempts = 3

    required_pairs = set()
    for i, hk_a in enumerate(all_hotkeys):
        for hk_b in all_hotkeys[i + 1:]:
            required_pairs.add(_canonical_pair_key(hk_a, hk_b))

    all_pair_results: list[PvPPairResult] = []
    for pair_key in required_pairs:
        if pair_key in rows_by_pair:
            pr = _try_build_pair_result(pair_key, rows_by_pair[pair_key], env_name_strs, max_pair_attempts)
            if pr:
                all_pair_results.append(pr)

    if len(all_pair_results) == len(required_pairs):
        logger.info(f"All {len(required_pairs)} pairs already complete in DB — skipping Basilica")
        group_results = PvPGroupResults(
            base_model=base_model,
            hotkeys=all_hotkeys,
            pair_results=all_pair_results,
            metadata=PvPEvalMetadata(seed=seed, temperature=0.0, wall_time_seconds=0),
        )
    else:
        group_results = await run_evaluation_pvp_group(
            participants=participants,
            base_model=base_model,
            environment_names=environment_names,
            seed=seed,
            task_id=task.task_id,
            psql_db=config.psql_db,
        )

        # Persist group eval pair results to DB
        for pair_result in group_results.pair_results:
            for env_name, env_result in pair_result.results.items():
                await tournament_sql.save_pvp_pair_result(
                    task_id=str(task.task_id),
                    result=pair_result,
                    environment_name=env_name.value,
                    env_result=env_result,
                    psql_db=config.psql_db,
                )

        if group_results.full_weight_fallbacks:
            fallback_pair_results = await _run_full_weight_fallback(
                group_results, miner_repos, base_model, environment_names, seed,
                task_id=str(task.task_id), psql_db=config.psql_db,
            )
            group_results.pair_results.extend(fallback_pair_results)
            group_results.hotkeys.extend(group_results.full_weight_fallbacks.hotkeys)

    env_weights = getattr(task, "environment_weights", None) or None
    logger.info(
        f"Scoring: {len(group_results.pair_results)} pair_results, "
        f"{len(group_results.hotkeys)} hotkeys: {group_results.hotkeys}"
    )
    for pr in group_results.pair_results:
        for env, er in pr.results.items():
            logger.info(f"  {pr.hotkey_a[:8]} vs {pr.hotkey_b[:8]} {env.value}: a={er.model_a_wins} b={er.model_b_wins} d={er.draws}")
    standings = compute_pvp_tournament_points(group_results, weights=env_weights)
    points_by_hotkey = {s.hotkey: s.points for s in standings}
    logger.info(f"Standings: {[(s.hotkey[:8], s.points) for s in standings]}")

    return [
        MinerResultsText(
            hotkey=hotkey,
            test_loss=points_by_hotkey.get(hotkey, 0.0),
            synth_loss=points_by_hotkey.get(hotkey, 0.0),
            is_finetune=True,
            submission=Submission(
                task_id=task.task_id,
                hotkey=hotkey,
                repo=repo,
                created_on=datetime.now(),
                updated_on=datetime.now(),
            ),
            task_type=task.task_type,
        )
        for hotkey, repo in miner_repos.items()
    ]


def _group_db_rows_by_pair(rows: list[PvPPairDbRow]) -> dict[PairKey, list[PvPPairDbRow]]:
    grouped: dict[PairKey, list[PvPPairDbRow]] = {}
    for row in rows:
        grouped.setdefault(row.pair_key, []).append(row)
    return grouped


def _try_build_pair_result(
    pair_key: str,
    rows: list[PvPPairDbRow],
    required_envs: list[str],
    max_attempts: int,
) -> PvPPairResult | None:
    """Build a PvPPairResult if the pair is complete or exhausted retries."""
    complete_rows = {r.environment_name: r for r in rows if r.is_complete}
    if set(required_envs) <= set(complete_rows.keys()):
        results = {
            core_cst.EnvironmentName(env): PvPEnvironmentResult(
                total_games=complete_rows[env].total_games,
                model_a_wins=complete_rows[env].model_a_wins,
                model_b_wins=complete_rows[env].model_b_wins,
                draws=complete_rows[env].draws,
            )
            for env in required_envs
        }
        hk_a, hk_b = pair_key.split(":")
        logger.info(f"Pair {pair_key} complete in DB")
        return PvPPairResult(hotkey_a=hk_a, hotkey_b=hk_b, results=results)

    # Check if exhausted — any non-complete row at max attempts
    if any(r.n_attempts >= max_attempts and not r.is_complete for r in rows):
        logger.warning(f"Pair {pair_key} exhausted {max_attempts} attempts — scoring as 0-0 draw")
        hk_a, hk_b = pair_key.split(":")
        results = {
            core_cst.EnvironmentName(env): PvPEnvironmentResult()
            for env in required_envs
        }
        return PvPPairResult(hotkey_a=hk_a, hotkey_b=hk_b, results=results)

    return None


async def _run_full_weight_fallback(
    group_results: PvPGroupResults,
    miner_repos: dict[str, str],
    base_model: str,
    environment_names: list[core_cst.EnvironmentName],
    seed: int,
    task_id: str,
    psql_db,
) -> list[PvPPairResult]:
    """Run 1v1 pair evals for full-weight contestants with persistent result tracking."""
    fallback = group_results.full_weight_fallbacks
    if fallback is None:
        return []
    lora_hotkeys = [h for h in group_results.hotkeys if h not in set(fallback.hotkeys)]
    fw_repo_by_hotkey = dict(zip(fallback.hotkeys, fallback.repos))

    pair_args: dict[PairKey, dict] = {}
    for fw_hotkey in fallback.hotkeys:
        fw_repo = fw_repo_by_hotkey[fw_hotkey]
        for lora_hotkey in lora_hotkeys:
            key = _canonical_pair_key(fw_hotkey, lora_hotkey)
            pair_args[key] = dict(
                model_a_repo=fw_repo, model_b_repo=miner_repos[lora_hotkey],
                hotkey_a=fw_hotkey, hotkey_b=lora_hotkey,
            )
        for other_fw_hotkey in fallback.hotkeys:
            if other_fw_hotkey <= fw_hotkey:
                continue
            key = _canonical_pair_key(fw_hotkey, other_fw_hotkey)
            pair_args[key] = dict(
                model_a_repo=fw_repo, model_b_repo=fw_repo_by_hotkey[other_fw_hotkey],
                hotkey_a=fw_hotkey, hotkey_b=other_fw_hotkey,
            )

    # Ensure all required pair+env rows exist in DB
    env_name_strs = [e.value for e in environment_names]
    stub_pairs = [
        PvPPairResult(hotkey_a=k.split(":")[0], hotkey_b=k.split(":")[1], results={})
        for k in pair_args
    ]
    await tournament_sql.ensure_pvp_pairs_exist(str(task_id), stub_pairs, env_name_strs, psql_db)

    # Check what's already complete or exhausted
    db_rows = await tournament_sql.get_pvp_pair_results(str(task_id), psql_db)
    completed_keys: set[str] = set()
    all_pair_results: list[PvPPairResult] = []
    max_pair_attempts = 3

    rows_by_pair = _group_db_rows_by_pair(db_rows)
    for pair_key, rows in rows_by_pair.items():
        pair_result = _try_build_pair_result(pair_key, rows, env_name_strs, max_pair_attempts)
        if pair_result:
            completed_keys.add(pair_key)
            all_pair_results.append(pair_result)

    remaining_keys = [k for k in pair_args if k not in completed_keys]
    if not remaining_keys:
        logger.info("All pairs complete in DB")
        return all_pair_results

    # Fire off all missing pairs — each persists independently on completion
    failed_pairs: list[str] = []

    async def _run_and_persist(key: str):
        hk_a, hk_b = key.split(":")
        await tournament_sql.increment_pvp_pair_attempts(str(task_id), hk_a, hk_b, psql_db)
        try:
            pair_group = await run_evaluation_pvp_pair(
                **pair_args[key], base_model=base_model,
                environment_names=environment_names, seed=seed,
            )
            for pair_result in pair_group.pair_results:
                for env_name, env_result in pair_result.results.items():
                    await tournament_sql.save_pvp_pair_result(
                        task_id=str(task_id),
                        result=pair_result,
                        environment_name=env_name.value,
                        env_result=env_result,
                        psql_db=psql_db,
                    )
            logger.info(f"Pair {key} completed and persisted")
        except Exception as e:
            logger.error(f"Pair {key} failed: {e}")
            failed_pairs.append(key)

    logger.info(f"Dispatching {len(remaining_keys)} pairs in parallel")
    await asyncio.gather(*[_run_and_persist(k) for k in remaining_keys])

    if failed_pairs:
        logger.warning(f"{len(failed_pairs)}/{len(remaining_keys)} pairs failed: {failed_pairs}")

    # Re-read DB and build final results
    updated_rows = await tournament_sql.get_pvp_pair_results(str(task_id), psql_db)
    updated_by_pair = _group_db_rows_by_pair(updated_rows)

    for pair_key, rows in updated_by_pair.items():
        if pair_key in completed_keys:
            continue
        pair_result = _try_build_pair_result(pair_key, rows, env_name_strs, max_pair_attempts)
        if pair_result:
            completed_keys.add(pair_key)
            all_pair_results.append(pair_result)

    still_incomplete = [k for k in pair_args if k not in completed_keys]
    if still_incomplete:
        raise PvPIncompleteError(
            f"{len(still_incomplete)}/{len(pair_args)} pairs incomplete: {still_incomplete}"
        )

    return all_pair_results


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


