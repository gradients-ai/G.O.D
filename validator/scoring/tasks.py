import asyncio
import math
from datetime import datetime
from uuid import UUID

import numpy as np
from fiber.chain.models import Node
from huggingface_hub import HfApi

import core.constants.environments as core_cst
import validator.evaluation.constants as eval_cst
import validator.infrastructure.service_constants as service_cst
import validator.scoring.constants as scoring_cst
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
from validator.app.config import Config
from validator.db.sql import tournaments as tournament_sql
from validator.db.sql.submissions_and_scoring import add_submission
from validator.db.sql.submissions_and_scoring import get_task_node_losses
from validator.db.sql.submissions_and_scoring import set_task_node_losses
from validator.db.sql.submissions_and_scoring import set_task_node_quality_score
from validator.db.sql.tasks import get_env_task_eval_seed
from validator.db.sql.tasks import get_expected_repo_name
from validator.db.sql.tasks import get_nodes_assigned_to_task
from validator.db.sql.tasks import get_starting_model_repo
from validator.evaluation.basilica import EvaluationRetryableError
from validator.evaluation.docker_evaluation import run_evaluation_basilica_image
from validator.evaluation.docker_evaluation import run_evaluation_basilica_text
from validator.evaluation.docker_evaluation import run_evaluation_individual
from validator.evaluation.docker_evaluation import run_evaluation_pvp_pair
from validator.evaluation.model_checks import check_for_lora
from validator.evaluation.notifications import notify_evaluation_exception
from validator.evaluation.notifications import task_deployment_ids_for_hotkeys
from validator.evaluation.pvp.models import PvPEnvironmentResult
from validator.evaluation.pvp.models import PvPEvalMetadata
from validator.evaluation.pvp.models import PvPGroupResults
from validator.evaluation.pvp.models import PvPIncompleteError
from validator.evaluation.pvp.models import PvPIndividualScoreDbRow
from validator.evaluation.pvp.models import PvPPairDbRow
from validator.evaluation.pvp.models import PvPPairResult
from validator.evaluation.pvp.models import _canonical_pair_key
from validator.infrastructure.minio_client import async_minio_client
from validator.scoring.models import EnvMinerScores
from validator.scoring.models import EvalHotkeyResults
from validator.scoring.models import GroupStagePoints
from validator.scoring.models import IndividualEvalResult
from validator.scoring.models import IndividualScoresByEnv
from validator.scoring.models import MinerRepos
from validator.scoring.models import MinerResults
from validator.scoring.models import MinerResultsImage
from validator.scoring.models import MinerResultsText
from validator.scoring.models import Submission
from validator.scoring.tournaments import pvp_results_to_winrates
from validator.scoring.tournaments import rank_weighted_standings
from validator.tasks.models import AnyTypeRawTask
from validator.tasks.models import EnvRawTask
from validator.tasks.models import InstructTextRawTask


logger = get_logger(__name__)

PairKey = str  # sorted "hotkey_a:hotkey_b"
TOURNAMENT_EVAL_TYPES = frozenset({core_cst.EvalType.PVP, core_cst.EvalType.INDIVIDUAL})


class PvPEvaluationExhaustedError(RuntimeError):
    """Raised when a PvP pair has consumed all eval attempts without a result."""

    def __init__(self, message: str, deployment_ids: list[str] | None = None):
        super().__init__(message)
        self.deployment_ids = deployment_ids or []


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
            top_result.score_reason = f"Ranked 1st by {ranking_type}"
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
                result.score = scoring_cst.SCORE_PENALTY
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

        if task.task_type != TaskType.ENVIRONMENTTASK:
            assert task.test_data is not None, "Test data shouldn't be none for text tasks"

        # Fetch eval_seed for environment tasks
        eval_seed = None
        if task.task_type == TaskType.ENVIRONMENTTASK and config is not None and task.task_id is not None:
            eval_seed = await get_env_task_eval_seed(task.task_id, config.psql_db)
            logger.info(f"Fetched eval_seed={eval_seed} for environment task {task.task_id}")

        use_kl, kl_coef = (task.use_kl, task.kl_coef) if isinstance(task, InstructTextRawTask) else (False, None)
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

    if miner_repos and should_use_tournament_eval(task):
        try:
            results.extend(await _run_env_tournament_eval(task, miner_repos, config))
        except PvPIncompleteError:
            raise
        except PvPEvaluationExhaustedError as e:
            logger.error(f"PvP pairwise evaluation exhausted attempts: {e}", exc_info=True)
            await notify_evaluation_exception(
                config,
                task_id=str(task.task_id),
                task_type=task.task_type,
                context="PvP tournament evaluation exhausted attempts",
                error=e,
                hotkeys=list(miner_repos.keys()),
                repos=list(miner_repos.values()),
                deployment_ids=e.deployment_ids,
            )
            results.extend(
                _create_failed_miner_result(
                    hotkey,
                    score_reason=f"Evaluation failed: {str(e)[:350]}",
                    task_type=task.task_type,
                )
                for hotkey in miner_repos
            )
        except Exception as e:
            logger.error(f"PvP pairwise evaluation failed: {e}", exc_info=True)
            await notify_evaluation_exception(
                config,
                task_id=str(task.task_id),
                task_type=task.task_type,
                context="PvP tournament evaluation exception",
                error=e,
                hotkeys=list(miner_repos.keys()),
                repos=list(miner_repos.values()),
                deployment_ids=await task_deployment_ids_for_hotkeys(
                    task.task_id,
                    config,
                    list(miner_repos.keys()),
                ),
            )
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

                if task.task_type in [
                    TaskType.INSTRUCTTEXTTASK,
                    TaskType.DPOTASK,
                    TaskType.GRPOTASK,
                    TaskType.CHATTASK,
                    TaskType.ENVIRONMENTTASK,
                ]:
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


def should_use_tournament_eval(task: AnyTypeRawTask) -> bool:
    """Check if this task should use tournament evaluation (PvP or individual envs)."""
    if task.task_type != TaskType.ENVIRONMENTTASK:
        return False
    env_names = getattr(task, "environment_names", None)
    if not env_names:
        return False
    for name in env_names:
        env_config = core_cst.ENVIRONMENT_CONFIGS.get(name)
        if env_config and env_config.eval_type in TOURNAMENT_EVAL_TYPES:
            return True
    return False



async def _run_env_tournament_eval(
    task: AnyTypeRawTask,
    miner_repos: dict[str, str],  # {hotkey: repo_id}
    config: Config,
) -> list[MinerResultsText]:
    """Run tournament eval with env partitioning. Delegates to focused sub-functions."""
    if not isinstance(task, EnvRawTask):
        raise TypeError(f"Expected EnvRawTask, got {type(task).__name__}")

    training_statuses = await tournament_sql.get_training_status_for_task(str(task.task_id), config.psql_db)
    if training_statuses:
        successful_hotkeys = {hotkey for hotkey, status in training_statuses.items() if status == "success"}
        skipped_hotkeys = sorted(set(miner_repos) - successful_hotkeys)
        if skipped_hotkeys:
            logger.info(f"Excluding non-successful training hotkeys from tournament eval: {skipped_hotkeys}")
        miner_repos = {hotkey: repo for hotkey, repo in miner_repos.items() if hotkey in successful_hotkeys}

    miners = MinerRepos(by_hotkey=miner_repos)
    base_model = task.augmented_model_id or task.model_id
    model_params = task.model_params_count or 0

    eval_seed = await get_env_task_eval_seed(task.task_id, config.psql_db)
    seed = eval_seed if eval_seed is not None else eval_cst.ENV_EVAL_DEFAULT_SEED

    pvp_envs = [e for e in task.environment_names if core_cst.ENVIRONMENT_CONFIGS[e].eval_type == core_cst.EvalType.PVP]
    individual_envs = [
        e for e in task.environment_names if core_cst.ENVIRONMENT_CONFIGS[e].eval_type == core_cst.EvalType.INDIVIDUAL
    ]

    logger.info(
        f"Tournament eval: task={task.task_id}, {len(miners)} miners, "
        f"pvp_envs={[e.value for e in pvp_envs]}, individual_envs={[e.value for e in individual_envs]}"
    )

    env_scores: list[EnvMinerScores] = []
    base_chains = await _get_continuation_base_chains(task, miners, base_model, config)

    if pvp_envs:
        env_scores.extend(await _eval_pvp_envs(
            task_id=str(task.task_id), pvp_envs=pvp_envs, miners=miners,
            base_model=base_model, seed=seed, config=config,
            base_chains=base_chains,
        ))

    if individual_envs:
        env_scores.extend(await _eval_individual_envs(
            task_id=task.task_id, individual_envs=individual_envs, miners=miners,
            base_model=base_model, model_params=model_params, seed=seed, config=config,
            base_chains=base_chains,
        ))

    standings = rank_weighted_standings(env_scores, miners.hotkeys, weights=task.environment_weights or None)
    return _standings_to_results(standings, miners, task)


def _standings_to_results(
    standings: list[GroupStagePoints],
    miners: MinerRepos,
    task: EnvRawTask,
) -> list[MinerResultsText]:
    """Convert point standings into MinerResultsText."""
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
                repo=miners.by_hotkey[hotkey],
                created_on=datetime.now(),
                updated_on=datetime.now(),
            ),
            task_type=task.task_type,
        )
        for hotkey in miners.hotkeys
    ]


# --- PvP env partition ---


def _get_shared_env_config(envs: list[core_cst.EnvironmentName]) -> core_cst.EnvironmentConfig:
    """Get config shared by all envs in a partition. Validates they share tournament_eval_image and gpu_multiplier."""
    configs = [core_cst.ENVIRONMENT_CONFIGS[e] for e in envs]
    image = configs[0].tournament_eval_image
    multiplier = configs[0].gpu_multiplier
    if not all(c.tournament_eval_image == image for c in configs):
        raise ValueError(
            f"All envs in partition must share tournament_eval_image, got: "
            f"{[(e.value, c.tournament_eval_image) for e, c in zip(envs, configs)]}"
        )
    if not all(c.gpu_multiplier == multiplier for c in configs):
        raise ValueError(
            f"All envs in partition must share gpu_multiplier, got: "
            f"{[(e.value, c.gpu_multiplier) for e, c in zip(envs, configs)]}"
        )
    return configs[0]


async def _get_continuation_base_chains(
    task: AnyTypeRawTask,
    miners: MinerRepos,
    base_model: str,
    config: Config,
) -> dict[str, list[str]]:
    """Return per-miner adapter lineage needed to serve continuation models."""
    base_chains: dict[str, list[str]] = {}
    for hotkey in miners.hotkeys:
        starting_repo = await get_starting_model_repo(str(task.task_id), hotkey, config.psql_db)
        if not starting_repo or starting_repo in (base_model, task.model_id):
            continue
        if not await asyncio.to_thread(check_for_lora, starting_repo, False):
            logger.info(f"Miner {hotkey}: starting repo {starting_repo} is not a LoRA adapter; serving on foundation")
            continue
        base_chains[hotkey] = [starting_repo]
    if base_chains:
        logger.info(f"Continuation base chains for {len(base_chains)} miners on task {task.task_id}")
    return base_chains


async def _eval_pvp_envs(
    task_id: str,
    pvp_envs: list[core_cst.EnvironmentName],
    miners: MinerRepos,
    base_model: str,
    seed: int,
    config: Config,
    base_chains: dict[str, list[str]] | None = None,
) -> list[EnvMinerScores]:
    """Run pairwise PvP eval for PVP-type environments, return per-env win rates."""
    env_config = _get_shared_env_config(pvp_envs)

    group_results = await _get_or_run_pvp_pairs(
        task_id=task_id, pvp_envs=pvp_envs, miners=miners,
        base_model=base_model, seed=seed,
        image=env_config.tournament_eval_image,
        gpu_count=eval_cst.PVP_BASILICA_GPU_COUNT, config=config,
        base_chains=base_chains,
    )

    return pvp_results_to_winrates(group_results)


async def _get_or_run_pvp_pairs(
    task_id: str,
    pvp_envs: list[core_cst.EnvironmentName],
    miners: MinerRepos,
    base_model: str,
    seed: int,
    image: str,
    gpu_count: int,
    config: Config,
    base_chains: dict[str, list[str]] | None = None,
) -> PvPGroupResults:
    """Check DB for complete pair results; if missing, run missing 1v1 pairs."""
    env_name_strs = [e.value for e in pvp_envs]
    max_pair_attempts = scoring_cst.MAX_TOURNAMENT_EVAL_ATTEMPTS
    task_uuid = UUID(task_id)
    all_hotkeys = miners.hotkeys
    base_chains = base_chains or {}

    db_rows = await tournament_sql.get_pvp_pair_results(task_id, config.psql_db)
    rows_by_pair = _group_db_rows_by_pair(db_rows)

    required_pairs: set[str] = set()
    for i, hk_a in enumerate(all_hotkeys):
        for hk_b in all_hotkeys[i + 1:]:
            required_pairs.add(_canonical_pair_key(hk_a, hk_b))

    stub_pairs = [
        PvPPairResult(hotkey_a=pair_key.split(":")[0], hotkey_b=pair_key.split(":")[1], results={})
        for pair_key in required_pairs
    ]
    await tournament_sql.ensure_pvp_pairs_exist(task_id, stub_pairs, env_name_strs, config.psql_db)

    complete_pairs: list[PvPPairResult] = []
    completed_keys: set[str] = set()
    for pair_key in required_pairs:
        if pair_key in rows_by_pair:
            pair_result = _try_build_pair_result(pair_key, rows_by_pair[pair_key], env_name_strs, max_pair_attempts)
            if pair_result:
                completed_keys.add(pair_key)
                complete_pairs.append(pair_result)

    remaining_keys = [pair_key for pair_key in required_pairs if pair_key not in completed_keys]
    if not remaining_keys:
        logger.info(f"All {len(required_pairs)} PvP pairs already complete in DB")
    else:
        failed_pairs: list[str] = []

        async def _run_and_persist(pair_key: str) -> None:
            hk_a, hk_b = pair_key.split(":")
            try:
                pair_group = await run_evaluation_pvp_pair(
                    model_a_repo=miners.by_hotkey[hk_a],
                    model_b_repo=miners.by_hotkey[hk_b],
                    hotkey_a=hk_a,
                    hotkey_b=hk_b,
                    base_model=base_model,
                    environment_names=pvp_envs,
                    seed=seed,
                    image=image,
                    gpu_count=gpu_count,
                    task_id=task_uuid,
                    psql_db=config.psql_db,
                    base_chain_a=base_chains.get(hk_a, []),
                    base_chain_b=base_chains.get(hk_b, []),
                )
            except EvaluationRetryableError as exc:
                # Transient infra failure (e.g. no eval GPU capacity) — not the pair's
                # fault. Do NOT consume a retry attempt; leave the pair pending so it is
                # retried next cycle instead of consuming an eval attempt.
                logger.info(f"Pair {pair_key} deferred, eval capacity unavailable: {exc}")
                failed_pairs.append(pair_key)
                return
            except Exception as exc:
                await tournament_sql.increment_pvp_pair_attempts(task_id, hk_a, hk_b, config.psql_db)
                logger.error(f"Pair {pair_key} failed: {exc}", exc_info=True)
                failed_pairs.append(pair_key)
                return

            await tournament_sql.increment_pvp_pair_attempts(task_id, hk_a, hk_b, config.psql_db)
            for pair_result in pair_group.pair_results:
                for env_name, env_result in pair_result.results.items():
                    await tournament_sql.save_pvp_pair_result(
                        task_id=task_id,
                        result=pair_result,
                        environment_name=env_name.value,
                        env_result=env_result,
                        psql_db=config.psql_db,
                    )
            logger.info(f"Pair {pair_key} completed and persisted")

        logger.info(f"Dispatching {len(remaining_keys)} PvP pairs in parallel")
        await asyncio.gather(*[_run_and_persist(pair_key) for pair_key in remaining_keys])

        if failed_pairs:
            logger.warning(f"{len(failed_pairs)}/{len(remaining_keys)} pairs failed: {failed_pairs}")

        updated_rows = await tournament_sql.get_pvp_pair_results(task_id, config.psql_db)
        updated_by_pair = _group_db_rows_by_pair(updated_rows)
        for pair_key, rows in updated_by_pair.items():
            if pair_key in completed_keys:
                continue
            pair_result = _try_build_pair_result(pair_key, rows, env_name_strs, max_pair_attempts)
            if pair_result:
                completed_keys.add(pair_key)
                complete_pairs.append(pair_result)

        still_incomplete = [pair_key for pair_key in required_pairs if pair_key not in completed_keys]
        if still_incomplete:
            raise PvPIncompleteError(
                f"{len(still_incomplete)}/{len(required_pairs)} pairs incomplete: {still_incomplete}"
            )

    return PvPGroupResults(
        base_model=base_model,
        hotkeys=all_hotkeys,
        pair_results=complete_pairs,
        metadata=PvPEvalMetadata(seed=seed, temperature=0.0, wall_time_seconds=0),
    )


# --- Individual env partition ---


async def _eval_individual_envs(
    task_id: UUID | None,
    individual_envs: list[core_cst.EnvironmentName],
    miners: MinerRepos,
    base_model: str,
    model_params: int,
    seed: int,
    config: Config,
    base_chains: dict[str, list[str]] | None = None,
) -> list[EnvMinerScores]:
    """Run per-miner containers for INDIVIDUAL-type envs, return per-env raw scores."""
    task_id_str = str(task_id)
    env_name_strs = [e.value for e in individual_envs]

    await tournament_sql.ensure_individual_scores_exist(
        task_id_str, miners.hotkeys, env_name_strs, config.psql_db,
    )

    db_scores = await tournament_sql.get_individual_scores(task_id_str, config.psql_db)
    scores = _build_scores_from_db(db_scores, individual_envs)

    for env in individual_envs:
        scores = await _dispatch_missing_individual(
            env=env, task_id=task_id, task_id_str=task_id_str,
            miners=miners, base_model=base_model, model_params=model_params,
            seed=seed, config=config, scores=scores, db_scores=db_scores,
            base_chains=base_chains,
        )

    # Re-fetch to get accurate n_attempts after dispatches
    db_scores = await tournament_sql.get_individual_scores(task_id_str, config.psql_db)
    scores = _build_scores_from_db(db_scores, individual_envs)

    incomplete = scores.missing(individual_envs, miners.hotkeys)
    if incomplete:
        # Check if any missing hotkeys are still retryable — if so, raise to retry next cycle
        still_retryable = False
        for env, missing_hks in incomplete:
            retryable = _filter_exhausted(
                missing_hks,
                env.value,
                db_scores,
                max_attempts=scoring_cst.MAX_TOURNAMENT_EVAL_ATTEMPTS,
            )
            if retryable:
                still_retryable = True
                break

        if still_retryable:
            raise PvPIncompleteError(f"Individual env scores incomplete: {incomplete}")

        # All missing hotkeys exhausted attempts — assign score=0
        for env, missing_hks in incomplete:
            if env not in scores.results:
                scores.results[env] = IndividualEvalResult(environment_name=env, scores_by_hotkey={})
            for hk in missing_hks:
                scores.results[env].scores_by_hotkey[hk] = 0.0
                await tournament_sql.save_individual_score(
                    task_id=task_id_str, hotkey=hk,
                    environment_name=env.value, score=0.0, psql_db=config.psql_db,
                )
            logger.warning(
                f"Individual eval {env.value}: assigned score=0 for exhausted hotkeys: {[hk[:8] for hk in missing_hks]}"
            )

    return [
        EnvMinerScores(environment=env, scores_by_hotkey=dict(scores.results[env].scores_by_hotkey))
        for env in individual_envs
    ]


def _build_scores_from_db(
    db_scores: list[PvPIndividualScoreDbRow],
    envs: list[core_cst.EnvironmentName],
) -> IndividualScoresByEnv:
    """Build IndividualScoresByEnv from complete DB rows."""
    results: dict[core_cst.EnvironmentName, IndividualEvalResult] = {}
    for env in envs:
        hotkey_scores = {row.hotkey: row.score for row in db_scores if row.environment_name == env.value and row.is_complete}
        if hotkey_scores:
            results[env] = IndividualEvalResult(environment_name=env, scores_by_hotkey=hotkey_scores)
    return IndividualScoresByEnv(results=results)


async def _dispatch_missing_individual(
    env: core_cst.EnvironmentName,
    task_id: UUID | None,
    task_id_str: str,
    miners: MinerRepos,
    base_model: str,
    model_params: int,
    seed: int,
    config: Config,
    scores: IndividualScoresByEnv,
    db_scores: list[PvPIndividualScoreDbRow],
    base_chains: dict[str, list[str]] | None = None,
) -> IndividualScoresByEnv:
    """Deploy containers for missing individual scores on a single env."""
    env_config = core_cst.ENVIRONMENT_CONFIGS[env]
    existing_hotkeys = set(scores.results[env].scores_by_hotkey.keys()) if env in scores.results else set()
    missing_hotkeys = [hk for hk in miners.hotkeys if hk not in existing_hotkeys]

    if not missing_hotkeys:
        return scores

    max_attempts = scoring_cst.MAX_TOURNAMENT_EVAL_ATTEMPTS
    to_run = _filter_exhausted(missing_hotkeys, env.value, db_scores, max_attempts)
    if not to_run:
        return scores

    # Individual env evals deploy one model per Basilica job.
    individual_gpu_count = eval_cst.INDIVIDUAL_BASILICA_GPU_COUNT

    eval_result = await run_evaluation_individual(
        miners=miners.subset(to_run),
        base_model=base_model,
        environment_name=env,
        seed=seed,
        image=env_config.tournament_eval_image,
        gpu_count=individual_gpu_count,
        task_id=task_id,
        psql_db=config.psql_db,
        base_chains=base_chains,
    )

    # Persist scores for hotkeys that succeeded
    for hotkey, score in eval_result.scores_by_hotkey.items():
        await tournament_sql.save_individual_score(
            task_id=task_id_str, hotkey=hotkey,
            environment_name=env.value, score=score, psql_db=config.psql_db,
        )

    # Increment attempts only for hotkeys that were dispatched but didn't produce a score
    failed_hotkeys = [hk for hk in to_run if hk not in eval_result.scores_by_hotkey]
    if failed_hotkeys:
        await notify_evaluation_exception(
            config,
            task_id=task_id_str,
            task_type=TaskType.ENVIRONMENTTASK,
            context=f"Individual tournament evaluation failed for {env.value}",
            error="Evaluation did not produce a score",
            hotkeys=failed_hotkeys,
            repos=[miners.by_hotkey[hk] for hk in failed_hotkeys if hk in miners.by_hotkey],
            deployment_ids=await task_deployment_ids_for_hotkeys(
                task_id,
                config,
                failed_hotkeys,
            ),
        )
    for hk in failed_hotkeys:
        await tournament_sql.increment_individual_score_attempts(task_id_str, hk, env.value, config.psql_db)

    if env not in scores.results:
        scores.results[env] = IndividualEvalResult(environment_name=env, scores_by_hotkey={})
    scores.results[env].scores_by_hotkey.update(eval_result.scores_by_hotkey)
    return scores


def _filter_exhausted(
    missing_hotkeys: list[str],
    env_value: str,
    db_scores: list[PvPIndividualScoreDbRow],
    max_attempts: int,
) -> list[str]:
    """Return hotkeys that haven't exhausted retry attempts."""
    to_run = []
    for hk in missing_hotkeys:
        row = next((r for r in db_scores if r.environment_name == env_value and r.hotkey == hk), None)
        if row and row.n_attempts >= max_attempts:
            logger.warning(f"Individual eval {env_value}: hotkey {hk[:8]} exhausted {max_attempts} attempts")
            continue
        to_run.append(hk)
    return to_run


def _group_db_rows_by_pair(rows: list[PvPPairDbRow]) -> dict[PairKey, list[PvPPairDbRow]]:
    grouped: dict[PairKey, list[PvPPairDbRow]] = {}
    for row in rows:
        grouped.setdefault(row.pair_key, []).append(row)
    return grouped


def _deployment_ids_from_pvp_rows(rows: list[PvPPairDbRow]) -> list[str]:
    return sorted({row.deployment_id for row in rows if row.deployment_id})


def _try_build_pair_result(
    pair_key: str,
    rows: list[PvPPairDbRow],
    required_envs: list[str],
    max_attempts: int,
) -> PvPPairResult | None:
    """Build a PvPPairResult if the pair is complete; raise if attempts are exhausted."""
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
        raise PvPEvaluationExhaustedError(
            f"PvP pair {pair_key} exhausted {max_attempts} attempts without producing complete results "
            f"for environments: {', '.join(required_envs)}",
            deployment_ids=_deployment_ids_from_pvp_rows(rows),
        )

    return None


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
