import math
import os
from datetime import datetime

import numpy as np
from fiber.chain.models import Node
from huggingface_hub import HfApi

import validator.core.constants as cts
from core.models.payload_models import DiffusionLosses
from core.models.payload_models import EvaluationResultImage
from core.models.payload_models import EvaluationResultText
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
from validator.db.sql.submissions_and_scoring import set_task_node_quality_score
from validator.db.sql.tasks import get_env_task_eval_seed
from validator.db.sql.tasks import get_expected_repo_name
from validator.db.sql.tasks import get_nodes_assigned_to_task
from validator.db.sql.tournaments import get_tournament_id_by_task_id
from validator.db.sql.tournaments import get_training_status_for_task_and_hotkeys
from validator.evaluation.docker_evaluation import run_evaluation_docker_image
from validator.evaluation.docker_evaluation import run_evaluation_docker_text
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
        return EnvironmentDatasetType(
            environment_name=task.environment_name
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
    gpu_ids: list[int],
    dataset_type: TextDatasetType | None = None,
    config: "Config | None" = None,
) -> dict[str, EvaluationResultText | EvaluationResultImage | Exception]:
    unique_repos = list(set(submission_repos))
    if len(unique_repos) != len(submission_repos):
        logger.warning(f"Found duplicate repos. Deduplicating {len(submission_repos)} repos to {len(unique_repos)} unique repos")

    if task.task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK, TaskType.ENVIRONMENTTASK]:
        results: dict[str, EvaluationResultText | Exception] = {}
        repos_to_evaluate = []
        for repo in unique_repos:
            if repo == task.model_id:
                logger.warning(f"Repository {repo} matches original model ID - marking as non-finetuned")
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
            "original_model": task.model_id,
            "models": repos_to_evaluate,
            "dataset_type": dataset_type,
            "gpu_ids": gpu_ids,
            "eval_seed": eval_seed,
        }

        logger.info("Starting test evaluation")
        if task.task_type != TaskType.ENVIRONMENTTASK:
            test_data_filepath = await download_s3_file(task.test_data)
            test_results = await run_evaluation_docker_text(dataset=test_data_filepath, **evaluation_params)
            try:
                os.remove(test_data_filepath)
            except Exception as e:
                logger.warning(f"Failed to remove test data file {test_data_filepath}: {e}")
        else:
            test_results = await run_evaluation_docker_text(dataset="proxy", **evaluation_params)
            test_eval_results = test_results.results

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
        for repo in unique_repos:
            if repo == task.model_id:
                logger.warning(f"Repository {repo} matches original model ID - marking as non-finetuned")
                results[repo] = EvaluationResultImage(
                    eval_losses=DiffusionLosses(text_guided_losses=[0], no_text_losses=[0]), is_finetune=False
                )
            else:
                repos_to_evaluate.append(repo)

        if not repos_to_evaluate:
            return results

        evaluation_params = {
            "test_split_url": task.test_data,
            "original_model_repo": task.model_id,
            "models": repos_to_evaluate,
            "model_type": task.model_type,
            "gpu_ids": gpu_ids,
        }

        assert task.test_data is not None, "Test data shouldn't be none for image tasks"
        logger.info("Starting image model evaluation")
        image_results = await run_evaluation_docker_image(**evaluation_params)
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
    gpu_ids: list[int],
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
            logger.info(f"Constructed repo {repo} for miner {miner.hotkey}")
            miner_repos[miner.hotkey] = repo

    if miner_repos:
        try:
            eval_results = await _evaluate_submissions(
                task=task, submission_repos=list(miner_repos.values()), gpu_ids=gpu_ids, dataset_type=dataset_type or None, config=config
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


def has_disk_cache_error(task_results: list[MinerResultsText | MinerResultsImage]) -> bool:
    try:
        for result in task_results:
            if "Cannot find the requested files in the disk cache" in str(result.score_reason):
                return True
    except Exception as e:
        logger.error(f"Error checking for disk cache error: {e}")
        return False
    return False


async def evaluate_and_score(task: AnyTypeRawTask, gpu_ids: list[int], config: Config) -> AnyTypeRawTask:
    assert task.task_id is not None, "Task ID must be present"
    assert task.test_data is not None, "Test data must be present"

    miner_pool = await get_nodes_assigned_to_task(str(task.task_id), config.psql_db)
    
    # For tournament tasks, only evaluate miners with training status "success"
    tournament_id = await get_tournament_id_by_task_id(str(task.task_id), config.psql_db)
    if tournament_id:
        logger.info(f"Task {task.task_id} is a tournament task (tournament_id: {tournament_id}), filtering to only evaluate miners with training status 'success'")
        hotkeys = [node.hotkey for node in miner_pool]
        training_statuses = await get_training_status_for_task_and_hotkeys(str(task.task_id), hotkeys, config.psql_db)
        
        # Filter to only include miners with SUCCESS training status
        filtered_miner_pool = [
            node for node in miner_pool 
            if training_statuses.get(node.hotkey) == TrainingStatus.SUCCESS.value
        ]
        
        skipped_count = len(miner_pool) - len(filtered_miner_pool)
        if skipped_count > 0:
            skipped_hotkeys = [
                node.hotkey for node in miner_pool 
                if training_statuses.get(node.hotkey) != TrainingStatus.SUCCESS.value
            ]
            logger.info(
                f"Skipping {skipped_count} miners without 'success' training status: {skipped_hotkeys}"
            )
        
        miner_pool = filtered_miner_pool
        logger.info(f"Filtered to {len(miner_pool)} miners with 'success' training status for tournament task evaluation")
    
    dataset_type = _get_dataset_type(task)

    logger.info(f"Beginning evaluation for task {task.task_id} with {len(miner_pool)} miners")
    task_results = await process_miners_pool(miner_pool, task, config, gpu_ids, dataset_type)

    if has_disk_cache_error(task_results):
        if task.n_eval_attempts < cts.MAX_EVAL_ATTEMPTS - 1:
            task.status = TaskStatus.PREEVALUATION
            add_context_tag("status", task.status.value)
            logger.info(f"Task {task.task_id} marked as pre-evaluation due to disk cache error")
            task.n_eval_attempts = (task.n_eval_attempts or 0) + 1
            return task
        else:
            logger.info(
                f"Task {task.task_id} has a disk cache error but has reached the maximum number of retries. "
                "Will let it continue with what we have."
            )

    logger.info("Calculating final scores...")
    task_results = calculate_miner_ranking_and_scores(task_results)
    await _update_scores(task, task_results, config.psql_db)
    all_scores_zero = all(result.score == 0.0 for result in task_results)

    if cts.DELETE_S3_AFTER_COMPLETE:
        if task.task_type in [
            TaskType.INSTRUCTTEXTTASK,
            TaskType.DPOTASK,
            TaskType.GRPOTASK,
            TaskType.CHATTASK,
            TaskType.IMAGETASK,
            TaskType.ENVIRONMENTTASK
        ]:
            files_to_delete = [task.training_data, task.test_data]
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")

    if all_scores_zero:
        if task.n_eval_attempts < cts.MAX_EVAL_ATTEMPTS - 1:
            task.status = TaskStatus.PREEVALUATION
            add_context_tag("status", task.status.value)
            logger.info(f"All scores are zero for task {task.task_id}, setting status to PREEVALUATION to re-evaluate")
        else:
            task.status = TaskStatus.FAILURE
            add_context_tag("status", task.status.value)
            logger.info(f"Task {task.task_id} marked as failure")
            await _clear_up_s3(files_to_delete)
    else:
        await _clear_up_s3(files_to_delete)
        task.status = TaskStatus.SUCCESS
        add_context_tag("status", task.status.value)
        logger.info(f"Task {task.task_id} completed successfully with non-zero scores")
    task.n_eval_attempts = (task.n_eval_attempts or 0) + 1
    return task
