import asyncio
import json
import math
import os
import string
import uuid
from datetime import datetime
from datetime import timezone

import httpx
from dotenv import load_dotenv
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

import validator.tournament.constants as cst
from core.models.payload_models import DstackRunStatus
from core.models.tournament_models import GpuRequirement
from core.models.tournament_models import TournamentType
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from core.models.utility_models import TrainingStatus
from validator.core.config import Config
from validator.core.config import load_config
from validator.core.constants import DSTACK_RUNS_APPLY_ENDPOINT
from validator.core.constants import DSTACK_RUNS_GET_ENDPOINT
from validator.core.constants import EMISSION_BURN_HOTKEY
from validator.core.models import AnyTypeRawTask
from validator.db.sql import tasks as task_sql
from core.models.utility_models import Backend
from validator.db.sql import tournaments as tournament_sql
from validator.evaluation.scoring import _get_dataset_type
from validator.tournament.utils import get_tournament_gpu_requirement
from validator.utils.logging import LogContext
from validator.utils.logging import get_logger
from validator.utils.util import try_db_connections


logger = get_logger(__name__)


def load_dstack_config() -> dict:
    """Load dstack configuration from environment variables"""
    return {
        'url': os.getenv("DSTACK_URL"),
        'token': os.getenv("DSTACK_TOKEN"),
        'project': os.getenv("DSTACK_PROJECT")
    }


async def submit_dstack_run(task_config: dict) -> str:
    """
    Submit a run to dstack using runs/apply endpoint
    
    Args:
        task_config: Task configuration (plan structure)
    
    Returns:
        Run name from the response
    """
    dstack_config = load_dstack_config()
    dstack_url = dstack_config['url']
    dstack_token = dstack_config['token']
    dstack_project = dstack_config['project']
    
    request_url = f"{dstack_url}{DSTACK_RUNS_APPLY_ENDPOINT.format(project=dstack_project)}"
    
    headers = {
        "Authorization": f"Bearer {dstack_token}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(request_url, headers=headers, json=task_config)
        if response.status_code != 200:
            error_detail = response.text
            logger.error(f"dstack API error ({response.status_code}): {error_detail}")
            logger.error(f"Request payload: {task_config}")
            response.raise_for_status()
        result = response.json()
        run_name = result.get("run_spec", {}).get("run_name") or result.get("run_name")
        logger.info(f"Submitted dstack run: {run_name}")
        return run_name


async def get_dstack_run_status(run_name: str) -> DstackRunStatus:
    """
    Get status of a dstack run using runs/get endpoint
    
    Args:
        run_name: Name of the run to check
    
    Returns:
        DstackRunStatus: Parsed run status information
    """
    dstack_config = load_dstack_config()
    dstack_url = dstack_config['url']
    dstack_token = dstack_config['token']
    dstack_project = dstack_config['project']
    
    request_url = f"{dstack_url}{DSTACK_RUNS_GET_ENDPOINT.format(project=dstack_project)}"
    
    headers = {
        "Authorization": f"Bearer {dstack_token}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(request_url, headers=headers, json={"run_name": run_name})
        response.raise_for_status()
        response_data = response.json()
        return DstackRunStatus.model_validate(response_data)


async def fetch_organic_tasks_ready_to_train(config: Config):
    """
    Fill the `tournament_task_hotkey_trainings` table with organic task-hotkey pairs that haven't been trained yet.
    """
    while True:
        try:
            logger.info("Fetching organic tasks ready to train")
            await _fetch_organic_tasks_ready_to_train(config)
        except Exception as e:
            logger.error(f"Error in organic task orchestrator cycles: {str(e)}", exc_info=True)
        finally:
            await asyncio.sleep(cst.FETCH_TASKS_CYCLE_INTERVAL)


async def _fetch_organic_tasks_ready_to_train(config: Config):
    """
    Fetch organic (non-tournament, non-benchmark) tasks that are ready for training.
    These will be scheduled on dstack.
    """
    # Get organic tasks that are ready for training (filter by backend="runpod")
    organic_tasks = await task_sql.get_tasks_with_status(
        TaskStatus.READY, config.psql_db, tournament_filter="exclude", benchmark_filter="exclude", backend=Backend.RUNPOD.value
    )
    logger.info(f"Found {len(organic_tasks)} organic tasks ready for training")
    
    if not organic_tasks:
        return
    
    assignments = []
    tasks_to_update = []
    
    for task in organic_tasks:
        assignments.append(
            tournament_sql.TaskTrainingAssignment(
                task_id=str(task.task_id),
                hotkey=EMISSION_BURN_HOTKEY,
                created_at=task.created_at,
                priority=1,
                training_repo=None,
                training_commit_hash=None,
            )
        )
        tasks_to_update.append(task)
    
    if assignments:
        await tournament_sql.add_tournament_task_hotkey_pairs_for_training(assignments, config.psql_db)
        logger.info(f"Added {len(assignments)} organic task training assignments")
    
    for task in tasks_to_update:
        task.status = TaskStatus.TRAINING
        await task_sql.update_task(task, config.psql_db)
    
    logger.info(f"Moved {len(tasks_to_update)} organic tasks to TRAINING status")


async def process_pending_organic_tasks(config: Config):
    """
    Process organic tasks (priority 1) and schedule them on dstack.
    """
    while True:
        try:
            pending_training_tasks = await tournament_sql.get_tournament_training_tasks(
                config.psql_db,
                TrainingStatus.PENDING,
            )
            
            organic_tasks = [t for t in pending_training_tasks if t.priority == 1 and t.task.backend is not None and t.task.backend.value == Backend.RUNPOD.value]
            
            logger.info(f"Fetched {len(organic_tasks)} pending organic tasks")
            
            if not organic_tasks:
                await asyncio.sleep(cst.PROCESS_PENDING_TASKS_CYCLE_INTERVAL)
                continue
            
            await schedule_organic_tasks_for_dstack(organic_tasks, config)
        except Exception as e:
            logger.error(f"Error in process_pending_organic_tasks cycle: {str(e)}", exc_info=True)
            await asyncio.sleep(cst.PROCESS_PENDING_TASKS_CYCLE_INTERVAL)


async def schedule_organic_tasks_for_dstack(pending_training_tasks: list, config: Config):
    """
    Process organic tasks and schedule them on dstack.
    Only schedule tasks when we're ready to submit them.
    For failed tasks, wait 30 minutes before retrying (max 3 retries).
    """
    RETRY_DELAY_MINUTES = cst.DSTACK_RETRY_DELAY_MINUTES
    MAX_RETRIES = cst.DSTACK_MAX_RETRIES
    MAX_ATTEMPTS = MAX_RETRIES + 1
    
    while pending_training_tasks:
        oldest_task_training = pending_training_tasks[-1]
        task = oldest_task_training.task
        
        with LogContext(task_id=str(task.task_id)):
            if oldest_task_training.n_training_attempts >= MAX_ATTEMPTS:
                logger.warning(
                    f"Task {task.task_id} has exceeded max retries "
                    f"({oldest_task_training.n_training_attempts}), marking as failed"
                )
                await tournament_sql.update_tournament_task_training_status(
                    task.task_id, oldest_task_training.hotkey, TrainingStatus.FAILURE, config.psql_db
                )
                pending_training_tasks.pop()
                continue
            
            if oldest_task_training.n_training_attempts > 0:
                time_since_update = datetime.now(timezone.utc) - oldest_task_training.updated_at.replace(tzinfo=timezone.utc)
                minutes_since_update = time_since_update.total_seconds() / 60
                
                if minutes_since_update < RETRY_DELAY_MINUTES:
                    pending_training_tasks.insert(0, pending_training_tasks.pop())
                    await asyncio.sleep(60)
                    continue
            
            try:
                attempt_number = oldest_task_training.n_training_attempts
                run_name = _generate_dstack_run_name(str(task.task_id), attempt_number)
                
                dstack_config = await _create_dstack_request(
                    task, run_name, config
                )
                
                submitted_run_name = await submit_dstack_run(dstack_config)

                await tournament_sql.update_dstack_runname(
                    task.task_id, oldest_task_training.hotkey, run_name, config.psql_db
                )
                
                await tournament_sql.update_tournament_task_training_status(
                    task.task_id, oldest_task_training.hotkey, TrainingStatus.TRAINING, config.psql_db
                )
                
                pending_training_tasks.pop()
                logger.info(
                    f"Successfully scheduled task {task.task_id} "
                    f"for training on dstack with run name {run_name} "
                    f"(attempt {oldest_task_training.n_training_attempts + 1})"
                )
   
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Failed to schedule task {task.task_id} on dstack: {str(e)}")
                pending_training_tasks.insert(0, pending_training_tasks.pop())
                await asyncio.sleep(60) 
                continue
    
    logger.info(f"Completed scheduling cycle, {len(pending_training_tasks)} tasks remaining")


def _generate_dstack_run_name(task_id: str, attempt_number: int = 0) -> str:
    run_name = uuid.uuid4().hex
    if run_name[0].isdigit():
        replacement_char = string.ascii_lowercase[int(run_name[0])]
        run_name = f"{replacement_char}{run_name[1:]}"
    logger.debug(
        f"Generated run name {run_name} for task {task_id} (attempt {attempt_number})"
    )
    return run_name


async def _create_dstack_request(
    task: AnyTypeRawTask,
    run_name: str,
    config: Config,
) -> dict:
    """
    Create a dstack request configuration based on the task type.
    Uses static docker images from environment variables.
    
    Args:
        task: The task to create a request for
        run_name: The unique run name for this task
        config: Configuration object for database access
    
    Returns:
        dict: dstack task configuration dict
    """
    logger.info(f"Creating dstack request for task {task.task_id} with run name {run_name}")
    
    expected_repo_name = await task_sql.get_expected_repo_name(task.task_id, EMISSION_BURN_HOTKEY, config.psql_db)
    if not expected_repo_name:
        expected_repo_name = f"organic_{task.task_id}"
    
    required_gpus = get_tournament_gpu_requirement(task.task_type, task.model_params_count, task.model_id)
    
    if task.task_type == TaskType.IMAGETASK:
        gpu_name = "A100"
        gpu_count = _get_gpu_count_from_requirement(required_gpus)
        logger.info(f"Task {task.task_id} is IMAGETASK, using {gpu_count}x{gpu_name}")
    else:
        gpu_name = "H200"
        gpu_count = _get_h200_count_from_requirement(required_gpus)
        logger.info(f"Task {task.task_id} is text task (type={task.task_type}), using {gpu_count}x{gpu_name}")
    
    timeout_seconds = int(task.hours_to_complete * 3600) + 3600 # Add 1 hour for provisioning/download/upload
    
    task_env = {
        "TASK_ID": str(task.task_id),
        "MODEL": task.model_id,
        "TASK_TYPE": task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type),
        "EXPECTED_REPO_NAME": expected_repo_name,
        "HOURS_TO_COMPLETE": str(task.hours_to_complete),
    }
    
    if task.task_type == TaskType.IMAGETASK:
        task_env["DATASET_ZIP"] = task.training_data
        task_env["MODEL_TYPE"] = task.model_type
    else:
        task_env["DATASET"] = task.training_data
        dataset_type = _get_dataset_type(task)
        if dataset_type:
            dataset_type_dict = dataset_type.model_dump() if hasattr(dataset_type, 'model_dump') else dataset_type.dict()
            task_env["DATASET_TYPE"] = json.dumps(dataset_type_dict)
        task_env["FILE_FORMAT"] = "s3"
    
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    huggingface_username = os.getenv("HUGGINGFACE_USERNAME")
    if huggingface_token:
        task_env["HUGGINGFACE_TOKEN"] = huggingface_token
    if huggingface_username:
        task_env["HUGGINGFACE_USERNAME"] = huggingface_username
        
    if task.task_type == TaskType.IMAGETASK:
        docker_image = os.getenv("DSTACK_IMAGE_TASK_DOCKER_IMAGE", "diagonalge/image-winner-single:latest")
    else:
        docker_image = os.getenv("DSTACK_TEXT_TASK_DOCKER_IMAGE", "diagonalge/text-winner-single:latest")
    
    logger.info(f"Using docker image: {docker_image}")
    
    if task.task_type == TaskType.IMAGETASK:
        regions = cst.DSTACK_IMAGE_REGIONS
    else:
        regions = cst.DSTACK_TEXT_REGIONS
    
    # Build dstack task configuration
    task_config = {
        "plan": {
            "run_spec": {
                "run_name": run_name,
                "configuration": {
                    "type": "task",
                    "name": "organic",
                    "image": docker_image,
                    "env": task_env,
                    "resources": {
                        "gpu": {
                            "name": [gpu_name],
                            "count": {
                                "min": gpu_count,
                                "max": gpu_count
                            }
                        },
                        "disk": {
                            "size": "1000GB"
                        }
                    },
                    "regions": regions,
                    "max_duration": timeout_seconds
                }
            }
        },
        "force": False,
    }
    
    return task_config


def _get_gpu_count_from_requirement(requirement: GpuRequirement) -> int:
    """Get the number of GPUs required for a given GPU requirement (for A100)."""
    if requirement == GpuRequirement.A100:
        return 1
    elif requirement == GpuRequirement.H100_1X:
        return 1
    elif requirement == GpuRequirement.H100_2X:
        return 2
    elif requirement == GpuRequirement.H100_4X:
        return 4
    elif requirement == GpuRequirement.H100_8X:
        return 8
    return 1


def _get_h200_count_from_requirement(requirement: GpuRequirement) -> int:
    H100_VRAM_GB = 80
    H200_VRAM_GB = 141

    if requirement == GpuRequirement.A100:
        return 1
    elif requirement == GpuRequirement.H100_1X:
        total_vram_needed = 1 * H100_VRAM_GB 
    elif requirement == GpuRequirement.H100_2X:
        total_vram_needed = 2 * H100_VRAM_GB 
    elif requirement == GpuRequirement.H100_4X:
        total_vram_needed = 4 * H100_VRAM_GB
    elif requirement == GpuRequirement.H100_8X:
        total_vram_needed = 8 * H100_VRAM_GB 
    else:
        return 1
    
    h200_count = math.ceil(total_vram_needed / H200_VRAM_GB)
    return h200_count


async def monitor_dstack_tasks(config: Config):
    """
    Monitor dstack training tasks and update status based on completion.
    """
    while True:
        try:
            logger.info("Monitoring dstack training tasks")
            await _monitor_dstack_tasks(config)
        except Exception as e:
            logger.error(f"Error in monitor_dstack_tasks cycle: {str(e)}", exc_info=True)
        finally:
            await asyncio.sleep(cst.MONITOR_TRAINING_TASKS_CYCLE_INTERVAL)


async def _monitor_dstack_tasks(config: Config):
    """
    Monitor dstack training tasks and update status based on completion.
    Logic:
    - Submit task (usually succeeds)
    - Keep pinging for task details
    - If provisioning or running, do nothing
    - If failed, retry after 30 mins for max 3 retries
    - If done, mark as success
    """
    training_tasks = await tournament_sql.get_tournament_training_tasks(config.psql_db, TrainingStatus.TRAINING)
    organic_tasks = [t for t in training_tasks if t.priority == 1 and t.task.backend is not None and t.task.backend.value == Backend.RUNPOD.value]
    
    logger.info(f"Found {len(organic_tasks)} organic tasks currently in training on dstack")
    
    if not organic_tasks:
        logger.info("No organic tasks in training, skipping monitoring cycle")
        return
    
    for training_task in organic_tasks:
        with LogContext(task_id=str(training_task.task.task_id)):
            try:
                run_name = await tournament_sql.get_dstack_runname(
                    training_task.task.task_id, training_task.hotkey, config.psql_db
                )
                
                if not run_name:
                    logger.warning(
                        f"No dstack runname found for task {training_task.task.task_id}, "
                        f"hotkey {training_task.hotkey}, skipping"
                    )
                    continue
                
                run_status = await get_dstack_run_status(run_name)
                status_str = run_status.get_status().lower()
                
                logger.info(
                    f"Task {training_task.task.task_id} dstack run {run_name} status: {status_str}"
                )
                
                if run_status.is_provisioning() or run_status.is_running():
                    logger.debug(
                        f"Task {training_task.task.task_id} is {status_str}, continuing to monitor"
                    )
                    continue
                
                if run_status.is_done():
                    logger.info(
                        f"Task {training_task.task.task_id} "
                        f"completed successfully on dstack with run {run_name} "
                        f"(n_training_attempts={training_task.n_training_attempts})"
                    )
                    await tournament_sql.update_tournament_task_training_status(
                        training_task.task.task_id, training_task.hotkey, TrainingStatus.SUCCESS, config.psql_db
                    )
                    task = training_task.task
                    task.status = TaskStatus.PREEVALUATION
                    await task_sql.update_task(task, config.psql_db)
                    continue
                
                if run_status.is_failed():
                    logger.warning(
                        f"Task {training_task.task.task_id} "
                        f"failed on dstack with status {status_str}"
                    )
                    
                    max_retries = cst.DSTACK_MAX_RETRIES
                    max_attempts = max_retries + 1
                    if training_task.n_training_attempts >= max_attempts and not run_status.got_no_offers():
                        logger.error(
                            f"Task {training_task.task.task_id} has exceeded max retries "
                            f"({training_task.n_training_attempts}), marking as failed"
                        )
                        await tournament_sql.update_tournament_task_training_status(
                            training_task.task.task_id, training_task.hotkey, TrainingStatus.FAILURE, config.psql_db
                        )
                        task = training_task.task
                        task.status = TaskStatus.FAILURE
                        await task_sql.update_task(task, config.psql_db)
                    else:
                        logger.info(
                            f"Task {training_task.task.task_id} will be retried after 30 minutes "
                            f"(attempt {training_task.n_training_attempts + 1}/{max_retries + 1})"
                        )
                        await tournament_sql.update_tournament_task_training_status(
                            training_task.task.task_id, training_task.hotkey, TrainingStatus.PENDING, config.psql_db
                        )
                    continue
            
            except Exception as e:
                logger.error(
                    f"Error checking dstack task {training_task.task.task_id}: {str(e)}"
                )
                continue
    
    logger.info(f"Completed monitoring cycle, processed {len(organic_tasks)} tasks")


async def run_dstack_orchestrator_cycle():
    """
    Main entry point for dstack orchestrator cycle.
    """
    config = load_config()
    await try_db_connections(config)
    
    logger.info("Starting dstack orchestrator cycle for organic tasks")
    await asyncio.gather(
        fetch_organic_tasks_ready_to_train(config),
        process_pending_organic_tasks(config),
        monitor_dstack_tasks(config),
    )


if __name__ == "__main__":
    load_dotenv(".vali.env", override=True)
    asyncio.run(run_dstack_orchestrator_cycle())

