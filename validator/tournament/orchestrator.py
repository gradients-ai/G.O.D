import asyncio

import httpx
from dotenv import load_dotenv
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

import validator.tournament.constants as cst
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.tournament_models import GpuRequirement
from core.models.tournament_models import TaskTrainingAssignment
from core.models.tournament_models import TournamentTaskTraining
from core.models.tournament_models import TournamentType
from core.models.utility_models import Backend
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GPUInfo
from core.models.utility_models import GPUType
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from core.models.utility_models import TrainingStatus
from validator.core.config import Config
from validator.core.config import load_config
from validator.core.constants import EMISSION_BURN_HOTKEY
from validator.core.constants import GET_GPU_AVAILABILITY_ENDPOINT
from validator.core.constants import PROXY_TRAINING_IMAGE_ENDPOINT
from validator.core.constants import TASK_DETAILS_ENDPOINT
from validator.core.models import AnyTypeRawTask
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournaments as tournament_sql
from validator.db.sql.tournaments import get_tournament_id_by_task_id
from validator.evaluation.scoring import _get_dataset_type
from validator.tournament.utils import get_tournament_gpu_requirement
from validator.utils.logging import LogContext
from validator.utils.logging import get_logger
from validator.utils.util import try_db_connections


logger = get_logger(__name__)


simple_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=10),
    reraise=True,
)


@simple_retry
async def fetch_trainer_gpus(trainer_ip: str) -> list[GPUInfo]:
    """
    Fetch GPU availability information from a trainer.

    Args:
        trainer_ip: IP address of the trainer to contact

    Returns:
        List of GPUInfo objects from the trainer
    """
    async with httpx.AsyncClient(timeout=cst.TRAINER_HTTP_TIMEOUT) as client:
        # Default to port 8001 if no port is specified
        if ":" not in trainer_ip:
            trainer_ip_with_port = f"{trainer_ip}:8001"
        else:
            trainer_ip_with_port = trainer_ip

        url = f"http://{trainer_ip_with_port}{GET_GPU_AVAILABILITY_ENDPOINT}"
        logger.info(f"Fetching GPU availability from trainer at {url}")

        response = await client.get(url)
        response.raise_for_status()

        gpu_data = response.json()
        gpu_infos = [GPUInfo.model_validate(gpu_info) for gpu_info in gpu_data]

        logger.info(f"Retrieved {len(gpu_infos)} GPUs from trainer {trainer_ip}")
        return gpu_infos


@simple_retry
async def start_training_task(trainer_ip: str, training_request: TrainerProxyRequest) -> bool:
    """
    Ask trainer to start training.


    Args:
        trainer_ip: IP address of the trainer
        training_request: The training request to send


    Returns:
        bool: True if training started successfully, False otherwise
    """
    try:
        # Validate the request by converting to dict and back
        validated_request = TrainerProxyRequest.model_validate(training_request.model_dump())
        logger.info("Schema validation passed for training request")
    except Exception as e:
        logger.error(f"Schema validation failed for training request: {str(e)}")
        logger.error(f"Request payload: {training_request.model_dump()}")
        return False

    async with httpx.AsyncClient(timeout=cst.TRAINER_HTTP_TIMEOUT) as client:
        # Default to port 8001 if no port is specified
        if ":" not in trainer_ip:
            trainer_ip_with_port = f"{trainer_ip}:8001"
        else:
            trainer_ip_with_port = trainer_ip

        url = f"http://{trainer_ip_with_port}{PROXY_TRAINING_IMAGE_ENDPOINT}"
        logger.info(f"Requesting training from trainer at {url} with payload: {validated_request.model_dump()}")

        response = await client.post(url, json=validated_request.model_dump())
        response.raise_for_status()

        response_data = response.json()

        # Check for no retry flag
        if response_data.get("no_retry", False):
            logger.warning(
                f"Error cloning github repository for task {training_request.training_data.task_id} "
                f"with hotkey {training_request.hotkey}"
            )
            return cst.NO_RETRY_RESULT

        return response_data["message"] == cst.EXPECTED_TRAINING_START_MESSAGE


@simple_retry
async def get_training_task_details(trainer_ip: str, task_id: str, hotkey: str) -> TrainerTaskLog:
    """
    Get the details of a training task from a trainer.

    Args:
        trainer_ip: IP address of the trainer
        task_id: The task ID to get details for
        hotkey: The hotkey of the miner

    Returns:
        TrainerTaskLog: The task log from the trainer
    """
    async with httpx.AsyncClient(timeout=cst.TRAINER_HTTP_TIMEOUT) as client:
        # Default to port 8001 if no port is specified
        if ":" not in trainer_ip:
            trainer_ip_with_port = f"{trainer_ip}:8001"
        else:
            trainer_ip_with_port = trainer_ip

        url = f"http://{trainer_ip_with_port}{TASK_DETAILS_ENDPOINT.format(task_id=task_id)}"
        logger.debug(f"Getting task details from trainer at {url} for task {task_id}")

        response = await client.get(url, params={"hotkey": hotkey})
        response.raise_for_status()

        return TrainerTaskLog.model_validate(response.json())


async def fetch_tournament_tasks_ready_to_train(config: Config):
    """
    Fill the `tournament_task_hotkey_trainings` table with task-hotkey pairs that haven't been trained yet.
    """
    while True:
        try:
            logger.info("Fetching tournament tasks ready to train")
            await _fetch_tournament_tasks_ready_to_train(config)
        except Exception as e:
            logger.error(f"Error in tournament orchestrator cycles: {str(e)}", exc_info=True)
        finally:
            await asyncio.sleep(cst.FETCH_TASKS_CYCLE_INTERVAL)


async def _fetch_tournament_tasks_ready_to_train(config: Config):
    """
    Fetch tasks that are looking for nodes,
    then move them to training status and record the hotkey assignments.
    Process in priority order: organic (1), tournament (2), benchmark (3).
    Smart prioritization: fetch lower priority tasks when pending queue is low (per task type).
    """
    pending_training_tasks = await tournament_sql.get_tournament_training_tasks(
        config.psql_db,
        TrainingStatus.PENDING,
    )
    pending_count = len(pending_training_tasks)
    logger.info(f"Current pending training assignments: {pending_count}")

    # Count pending tasks by type
    pending_text_count = 0
    pending_image_count = 0
    for training_task in pending_training_tasks:
        if training_task.task.task_type == TaskType.IMAGETASK:
            pending_image_count += 1
        else:
            pending_text_count += 1

    logger.info(f"Pending by type - Text: {pending_text_count}, Image: {pending_image_count}")

    organic_tasks = await task_sql.get_tasks_with_status(
        TaskStatus.READY, config.psql_db, tournament_filter="exclude", benchmark_filter="exclude", backend=Backend.OBLIVUS.value
    )
    logger.info(f"Found {len(organic_tasks)} organic (non-tournament, non-benchmark) tasks ready for training")
    await _process_tasks_for_training(organic_tasks, config, priority=1)

    # Fetch tournament tasks by type
    all_tournament_tasks = await task_sql.get_tasks_with_status(
        TaskStatus.LOOKING_FOR_NODES, config.psql_db, tournament_filter="only", benchmark_filter="exclude"
    )

    text_tasks_to_process = []
    image_tasks_to_process = []

    for task in all_tournament_tasks:
        if task.task_type == TaskType.IMAGETASK:
            if pending_image_count < cst.PENDING_QUEUE_THRESHOLD_PER_TYPE:
                image_tasks_to_process.append(task)
        else:
            if pending_text_count < cst.PENDING_QUEUE_THRESHOLD_PER_TYPE:
                text_tasks_to_process.append(task)

    if text_tasks_to_process:
        logger.info(
            f"Pending text queue below {cst.PENDING_QUEUE_THRESHOLD_PER_TYPE}, "
            f"processing {len(text_tasks_to_process)} text tournament tasks"
        )
        await _process_tasks_for_training(text_tasks_to_process, config, priority=2)

    if image_tasks_to_process:
        logger.info(
            f"Pending image queue below {cst.PENDING_QUEUE_THRESHOLD_PER_TYPE}, "
            f"processing {len(image_tasks_to_process)} image tournament tasks"
        )
        await _process_tasks_for_training(image_tasks_to_process, config, priority=2)

    if pending_count < cst.PENDING_QUEUE_THRESHOLD_FOR_BENCHMARK:
        logger.info(f"Pending queue below {cst.PENDING_QUEUE_THRESHOLD_FOR_BENCHMARK}, fetching benchmark tasks")
        benchmark_tasks = await task_sql.get_tasks_with_status(
            TaskStatus.LOOKING_FOR_NODES, config.psql_db, tournament_filter="exclude", benchmark_filter="only"
        )
        logger.info(f"Found {len(benchmark_tasks)} benchmark tasks looking for nodes")
        await _process_tasks_for_training(benchmark_tasks, config, priority=3)


async def _process_tasks_for_training(tasks: list[AnyTypeRawTask], config: Config, priority: int):
    """
    Process a list of tasks for training with the specified priority.

    Args:
        tasks: List of tasks to process
        config: Configuration object
        priority: Training priority (1=organic, 2=tournament, 3=benchmark)
    """
    if not tasks:
        return

    assignments = []
    tasks_to_update = []
    tasks_without_nodes = []

    for task in tasks:
        # For tournament and benchmark tasks (priority 2 and 3), get actual assigned nodes
        # For organic tasks (priority 1), we'll use EMISSION_BURN_HOTKEY
        if priority in [2, 3]:
            logger.debug(f"Getting nodes for task {task.task_id} (type: {type(task.task_id)})")
            nodes = await task_sql.get_nodes_assigned_to_task(task.task_id, config.psql_db)
            logger.debug(f"Found {len(nodes)} nodes for task {task.task_id}: {[n.hotkey for n in nodes]}")
            hotkeys = [node.hotkey for node in nodes]

            if hotkeys:
                # Get tournament_id for this task
                tournament_id = await tournament_sql.get_tournament_id_by_task_id(task.task_id, config.psql_db)

                for hotkey in hotkeys:
                    if tournament_id:
                        training_repo, training_commit_hash = await tournament_sql.get_tournament_training_repo_and_commit(
                            hotkey, tournament_id, config.psql_db
                        )

                    assignments.append(
                        TaskTrainingAssignment(
                            task_id=str(task.task_id),
                            hotkey=hotkey,
                            created_at=task.created_at,
                            priority=priority,
                            training_repo=training_repo,
                            training_commit_hash=training_commit_hash,
                        )
                    )
                tasks_to_update.append(task)
            else:
                tasks_without_nodes.append(task)
        elif priority == 1:
            # Priority 1: Organic (non-tournament, non-benchmark) tasks
            # Use EMISSION_BURN_HOTKEY and get last tournament winner's repo
            if task.task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK]:
                tournament_type = TournamentType.TEXT
            elif task.task_type == TaskType.IMAGETASK:
                tournament_type = TournamentType.IMAGE
            elif task.task_type == TaskType.ENVIRONMENTTASK:
                tournament_type = TournamentType.ENVIRONMENT
            else:
                tournament_type = None

            # Get the last completed tournament winner's repo
            last_tournament = await tournament_sql.get_latest_completed_tournament(config.psql_db, tournament_type)
            if last_tournament and last_tournament.winner_hotkey:
                training_repo, training_commit_hash = await tournament_sql.get_tournament_training_repo_and_commit(
                    last_tournament.winner_hotkey, last_tournament.tournament_id, config.psql_db
                )

            assignments.append(
                TaskTrainingAssignment(
                    task_id=str(task.task_id),
                    hotkey=EMISSION_BURN_HOTKEY,
                    created_at=task.created_at,
                    priority=priority,
                    training_repo=training_repo,
                    training_commit_hash=training_commit_hash,
                )
            )
            tasks_to_update.append(task)

    if tasks_without_nodes:
        logger.warning(
            f"Found {len(tasks_without_nodes)} tasks with priority {priority} without assigned nodes: "
            f"{[str(t.task_id) for t in tasks_without_nodes]}"
        )

    if assignments:
        await tournament_sql.add_tournament_task_hotkey_pairs_for_training(assignments, config.psql_db)

    for task in tasks_to_update:
        task.status = TaskStatus.TRAINING
        await task_sql.update_task(task, config.psql_db)

    logger.info(f"Moved {len(tasks_to_update)} tasks with priority {priority} to TRAINING status")


async def process_pending_tournament_tasks(config: Config):
    while True:
        try:
            pending_training_tasks = await tournament_sql.get_tournament_training_tasks(
                config.psql_db,
                TrainingStatus.PENDING,
            )

            # Filter out tasks with backend="runpod" - those are handled by dstack orchestrator
            # Only process tasks with backend="oblivus" or backend IS NULL (for backward compatibility)
            tournament_tasks = [
                t for t in pending_training_tasks if t.task.backend is None or t.task.backend.value == Backend.OBLIVUS.value
            ]

            logger.info(f"Fetched {len(pending_training_tasks)} pending training tasks, {len(tournament_tasks)}")

            if not tournament_tasks:
                await asyncio.sleep(cst.PROCESS_PENDING_TASKS_CYCLE_INTERVAL)
                continue

            await schedule_tasks_for_training(tournament_tasks, config)
        except Exception as e:
            logger.error(f"Error in process_pending_tournament_tasks cycle: {str(e)}", exc_info=True)
            await asyncio.sleep(cst.PROCESS_PENDING_TASKS_CYCLE_INTERVAL)


async def schedule_tasks_for_training(pending_training_tasks: list[TournamentTaskTraining], config: Config):
    """
    Process tasks from the list and schedule them for training.
    Only pop tasks when we're 100% sure GPUs are available.
    """
    # Track failed attempts for this scheduling session
    failed_attempts = {}

    tasks_without_gpus = []

    while pending_training_tasks:
        oldest_task_training = pending_training_tasks[-1]
        tournament_id = await get_tournament_id_by_task_id(oldest_task_training.task.task_id, config.psql_db)
        with LogContext(
            task_id=oldest_task_training.task.task_id, hotkey=oldest_task_training.hotkey, tournament_id=tournament_id
        ):
            task = oldest_task_training.task
            task_key = f"{task.task_id}_{oldest_task_training.hotkey}"

            if oldest_task_training.n_training_attempts >= cst.MAX_TRAINING_ATTEMPTS:
                logger.warning(
                    f"Task {task.task_id} with hotkey {oldest_task_training.hotkey} has exceeded max attempts "
                    f"({oldest_task_training.n_training_attempts}), marking as failed"
                )

                await tournament_sql.update_tournament_task_training_status(
                    task.task_id, oldest_task_training.hotkey, TrainingStatus.FAILURE, config.psql_db
                )
                pending_training_tasks.pop()
                continue

            # Get training repo directly from the TournamentTaskTraining object
            training_repo = oldest_task_training.training_repo

            if training_repo is None:
                logger.error(
                    f"No training repository found for hotkey {oldest_task_training.hotkey} in tournament_participants table"
                )
                await tournament_sql.update_tournament_task_training_status(
                    task.task_id, oldest_task_training.hotkey, TrainingStatus.FAILURE, config.psql_db
                )
                pending_training_tasks.pop()
                continue

            # Determine required GPUs for this task
            required_gpus = get_tournament_gpu_requirement(task.task_type, task.model_params_count, task.model_id)
            logger.info(f"Task {task.task_id} requires {required_gpus.value}")
            await _update_all_trainers_gpu_availability(config)
            suitable_gpus_result = await _check_suitable_gpus(config, required_gpus)

            if not suitable_gpus_result:
                logger.info(
                    f"No suitable GPUs found for requirement {required_gpus.value}, skipping this task and continuing with next"
                )

                tasks_without_gpus.append(pending_training_tasks.pop())
                await asyncio.sleep(1)  # TODO: put in constant or even remove
                continue

            trainer_ip, gpu_ids = suitable_gpus_result

        try:
            training_task = pending_training_tasks[-1]
            tournament_id = await get_tournament_id_by_task_id(training_task.task.task_id, config.psql_db)
            with LogContext(task_id=str(training_task.task.task_id), hotkey=training_task.hotkey, tournament_id=tournament_id):
                training_request = await _create_training_request(
                    training_task.task,
                    training_task.hotkey,
                    gpu_ids,
                    training_task.training_repo,
                    training_task.training_commit_hash,
                    config,
                    n_training_attempts=training_task.n_training_attempts,
                )
                training_result = await start_training_task(trainer_ip, training_request)

                if training_result == cst.NO_RETRY_RESULT:
                    logger.warning(f"No retry failure for task {training_task.task.task_id} with hotkey {training_task.hotkey}")
                    await tournament_sql.update_tournament_task_training_status(
                        training_task.task.task_id, training_task.hotkey, TrainingStatus.FAILURE, config.psql_db
                    )
                    pending_training_tasks.pop()
                    continue
                elif training_result:
                    await tournament_sql.update_tournament_task_training_status(
                        training_task.task.task_id, training_task.hotkey, TrainingStatus.TRAINING, config.psql_db, trainer_ip
                    )
                    await tournament_sql.update_gpu_availability(
                        trainer_ip, gpu_ids, training_task.task.hours_to_complete, config.psql_db
                    )

                    pending_training_tasks.pop()
                    logger.info(
                        f"Successfully scheduled task {training_task.task.task_id} with hotkey {training_task.hotkey} "
                        f"for training on trainer {trainer_ip} with GPUs {gpu_ids} "
                        f"for {training_task.task.hours_to_complete} hours"
                    )

                    logger.info("Waiting 10 seconds before scheduling next task to avoid overwhelming trainers")
                    await asyncio.sleep(10)

                else:
                    logger.error(f"Failed to start training for task {training_task.task.task_id} on trainer {trainer_ip}")
                    # Track failed attempts for this scheduling session
                    failed_attempts[task_key] = failed_attempts.get(task_key, 0) + 1

                    if failed_attempts[task_key] >= 10 and failed_attempts[task_key] % 10 == 0:
                        logger.warning(
                            f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} has failed "
                            f"{failed_attempts[task_key]} scheduling attempts - this may indicate a persistent issue"
                        )
                    else:
                        logger.info(
                            f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} failed, scheduling attempt "
                            f"{failed_attempts[task_key]}, will retry"
                        )
                    await asyncio.sleep(cst.TRAINING_START_RETRY_INTERVAL)
                    continue
        except Exception as e:
            logger.error(f"Exception while scheduling training: {str(e)}")
            # Track failed attempts for this scheduling session
            failed_attempts[task_key] = failed_attempts.get(task_key, 0) + 1

            if failed_attempts[task_key] >= 10 and failed_attempts[task_key] % 10 == 0:
                logger.warning(
                    f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} has failed "
                    f"{failed_attempts[task_key]} scheduling attempts due to exception - this may indicate a persistent issue"
                )
            else:
                logger.info(
                    f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} failed due to exception, "
                    f"scheduling attempt {failed_attempts[task_key]}, will retry"
                )
            await asyncio.sleep(cst.TRAINING_START_RETRY_INTERVAL)
            continue

    if tasks_without_gpus:
        logger.info(
            f"Skipped {len(tasks_without_gpus)} tasks due to GPU unavailability. They will get picked up in the next cycle."
        )

    logger.info(f"Completed scheduling cycle, {len(pending_training_tasks)} tasks remaining")


async def _check_suitable_gpus(config: Config, required_gpus: GpuRequirement) -> tuple[str, list[int]] | None:
    """
    Check if there are any suitable GPUs across all trainers for the given GPU requirement.
    Optimizes allocation by selecting the trainer that maximizes the ratio:
    num_needed_GPUs_for_task / total_free_GPUs_in_trainer

    This ensures tasks are packed efficiently and smaller tasks don't occupy larger trainers
    unnecessarily.

    Args:
        config: Configuration object for database access
        required_gpus: Required GPU specification

    Returns:
        tuple[str, list[int]] | None: (trainer_ip, gpu_ids) if suitable GPUs found, None otherwise
    """
    try:
        trainers = await tournament_sql.get_trainers(config.psql_db)
        required_gpu_count = _get_gpu_count_from_requirement(required_gpus)

        best_trainer = None
        best_gpu_ids = None
        best_ratio = -1.0
        for trainer in trainers:
            gpu_ids = _trainer_has_sufficient_gpus(trainer.gpus, required_gpus)
            if gpu_ids:
                free_gpu_count = sum(1 for gpu in trainer.gpus if gpu.available)
                ratio = required_gpu_count / free_gpu_count
                logger.info(
                    f"Trainer {trainer.trainer_ip}: {len(gpu_ids)} GPUs available for requirement {required_gpus.value}, "
                    f"{free_gpu_count} total free GPUs, ratio: {ratio:.2f}"
                )

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_trainer = trainer.trainer_ip
                    best_gpu_ids = gpu_ids

        if best_trainer:
            logger.info(
                f"Selected trainer {best_trainer} with best utilization ratio {best_ratio:.2f} "
                f"for requirement {required_gpus.value}"
            )
            return best_trainer, best_gpu_ids

        logger.info(f"No suitable GPUs found for requirement {required_gpus.value}")
        return None

    except Exception as e:
        logger.error(f"Error checking suitable GPUs: {str(e)}")
        return None


def _get_gpu_count_from_requirement(requirement: GpuRequirement) -> int:
    """
    Get the number of GPUs required for a given GPU requirement.
    """
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

    # Default to 1 if unknown
    return 1


async def _create_training_request(
    task: AnyTypeRawTask, hotkey: str, available_gpu_ids: list[int], training_repo: str, training_commit_hash: str, config: Config, n_training_attempts: int = 0
) -> TrainerProxyRequest:
    """
    Create a TrainerProxyRequest based on the task type.

    Args:
        task: The task to create a training request for
        hotkey: The hotkey of the miner
        available_gpu_ids: List of available GPU IDs
        training_repo: The training repository URL
        training_commit_hash: The training repository commit hash
        config: Configuration object for database access
        n_training_attempts: Number of training attempts (for retry workarounds)

    Returns:
        TrainerProxyRequest: The training request
    """
    expected_repo_name = await task_sql.get_expected_repo_name(task.task_id, hotkey, config.psql_db)

    logger.info(f"Creating training request for hotkey {hotkey}, task {task.task_id}")
    logger.info(f"Expected repo name: {expected_repo_name}")
    logger.info(f"Training repo: {training_repo}")
    logger.info(f"Training commit hash: {training_commit_hash}")

    # Validate that training repo exists for this hotkey
    if training_repo is None:
        logger.error(f"No training repository found for hotkey {hotkey} in tournament_participants table")
        logger.error(
            "This hotkey may not be registered as a tournament participant or the training repo was not properly set "
            "during tournament registration"
        )
        raise ValueError(
            f"No training repository found for hotkey {hotkey}. This hotkey may not be registered as a tournament "
            f"participant or the training repo was not properly set during tournament registration."
        )

    if task.task_type == TaskType.IMAGETASK:
        training_data = TrainRequestImage(
            model=task.model_id,
            task_id=str(task.task_id),
            hours_to_complete=task.hours_to_complete,
            expected_repo_name=expected_repo_name,
            dataset_zip=task.training_data,
            model_type=task.model_type,
        )
    else:
        dataset_type = _get_dataset_type(task)
        
        # Workaround: For environment tasks on retry (n_training_attempts >= 1), change environment_name to "game"
        if task.task_type == TaskType.ENVIRONMENTTASK and n_training_attempts == 0:
            if isinstance(dataset_type, EnvironmentDatasetType):
                logger.info(f"Environment task retry detected (attempts={n_training_attempts}), changing environment_name to 'game'")
                dataset_type = EnvironmentDatasetType(environment_name="game")
        
        training_data = TrainRequestText(
            model=task.model_id,
            task_id=str(task.task_id),
            hours_to_complete=task.hours_to_complete,
            expected_repo_name=expected_repo_name,
            dataset=task.training_data,
            dataset_type=dataset_type,
            file_format=FileFormat.S3,  # always an S3 since we task prep
        )

    return TrainerProxyRequest(
        training_data=training_data,
        github_repo=training_repo,
        gpu_ids=available_gpu_ids,
        hotkey=hotkey,
        github_commit_hash=training_commit_hash,
    )


def _trainer_has_sufficient_gpus(trainer_gpus: list[GPUInfo], requirement: GpuRequirement) -> list[int]:
    """
    Check if a trainer has sufficient GPUs to meet the requirement.

    Args:
        trainer_gpus: List of GPUs on the trainer
        requirement: Required GPU specification

    Returns:
        list[int]: List of GPU IDs needed for the requirement, empty list if insufficient
    """
    available_h100s = [gpu for gpu in trainer_gpus if gpu.available and gpu.gpu_type == GPUType.H100]
    available_a100s = [gpu for gpu in trainer_gpus if gpu.available and gpu.gpu_type == GPUType.A100]

    if requirement == GpuRequirement.A100:
        return [available_a100s[0].gpu_id] if len(available_a100s) >= 1 else []
    elif requirement == GpuRequirement.H100_1X:
        return [available_h100s[0].gpu_id] if len(available_h100s) >= 1 else []
    elif requirement == GpuRequirement.H100_2X:
        return [gpu.gpu_id for gpu in available_h100s[:2]] if len(available_h100s) >= 2 else []
    elif requirement == GpuRequirement.H100_4X:
        return [gpu.gpu_id for gpu in available_h100s[:4]] if len(available_h100s) >= 4 else []
    elif requirement == GpuRequirement.H100_8X:
        return [gpu.gpu_id for gpu in available_h100s[:8]] if len(available_h100s) >= 8 else []

    return []


async def monitor_training_tasks(config: Config):
    """
    Monitor training tasks and update GPU availability based on completion status.
    """
    while True:
        try:
            logger.info("Monitoring training tasks")
            await _monitor_training_tasks(config)
        except Exception as e:
            logger.error(f"Error in monitor_training_tasks cycle: {str(e)}", exc_info=True)
        finally:
            await asyncio.sleep(cst.MONITOR_TRAINING_TASKS_CYCLE_INTERVAL)


async def _monitor_training_tasks(config: Config):
    """
    Monitor training tasks and update GPU availability based on completion status.
    """
    # Get all tasks currently in training status
    training_tasks = await tournament_sql.get_tournament_training_tasks(config.psql_db, TrainingStatus.TRAINING)

    # Filter out tasks with backend="runpod" - those are handled by dstack orchestrator
    # Only monitor tasks with backend="oblivus" or backend IS NULL (for backward compatibility)
    tournament_tasks = [t for t in training_tasks if t.task.backend is None or t.task.backend.value == Backend.OBLIVUS.value]

    logger.info(f"Found {len(training_tasks)} tasks in training, {len(tournament_tasks)}")

    if not tournament_tasks:
        logger.info("No tasks in training, skipping monitoring cycle")
        return

    # Track if any tasks completed to determine if we need to update GPU availability
    any_completed = False

    # Check each training task
    for training_task in tournament_tasks:
        tournament_id = await get_tournament_id_by_task_id(training_task.task.task_id, config.psql_db)
        if tournament_id is None:
            logger.warning(f"Task {training_task.task.task_id} not found in tournament_tasks table - no tournament_id available")
        with LogContext(task_id=str(training_task.task.task_id), hotkey=training_task.hotkey, tournament_id=tournament_id):
            try:
                # Only query the trainer where this task was scheduled
                trainer_ip = training_task.trainer_ip
                if not trainer_ip:
                    logger.warning(
                        f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} has no trainer_ip, "
                        f"querying all trainers as fallback"
                    )
                    trainers = await tournament_sql.get_trainers(config.psql_db)
                    trainer_ips = [trainer.trainer_ip for trainer in trainers]
                else:
                    trainer_ips = [trainer_ip]

                logger.info(
                    f"Checking task {training_task.task.task_id} with hotkey {training_task.hotkey} "
                    f"on trainer(s) {trainer_ips}"
                )
                responses = []
                for ip in trainer_ips:
                    try:
                        task_log = await get_training_task_details(ip, str(training_task.task.task_id), training_task.hotkey)
                        if task_log:
                            responses.append((ip, task_log))
                    except httpx.HTTPStatusError as e:
                        status_code = e.response.status_code
                        if 500 <= status_code < 600:
                            logger.error(f"Server error ({status_code}) from trainer {ip}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.info(f"Could not get task details from trainer {ip}: {str(e)}")
                        continue

                if not responses:
                    logger.warning(
                        f"Could not find trainer for task {training_task.task.task_id} with hotkey {training_task.hotkey}"
                    )
                    # Move task back to PENDING since trainer may have restarted or lost the task
                    await tournament_sql.update_tournament_task_training_status(
                        training_task.task.task_id, training_task.hotkey, TrainingStatus.PENDING, config.psql_db
                    )
                    logger.info(
                        f"Moved task {training_task.task.task_id} with hotkey {training_task.hotkey} back to PENDING status"
                    )
                    continue

                # Gather all statuses
                statuses = [task_log.status for _, task_log in responses]
                # Priority: SUCCESS > TRAINING > FAILURE
                if TaskStatus.SUCCESS in statuses:
                    any_completed = True
                    logger.info(
                        f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} completed with status SUCCESS "
                        f"(at least one trainer)"
                    )
                    await tournament_sql.update_tournament_task_training_status(
                        training_task.task.task_id, training_task.hotkey, TrainingStatus.SUCCESS, config.psql_db
                    )
                elif all(s == TaskStatus.FAILURE for s in statuses):
                    any_completed = True
                    logger.info(f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} failed on all trainers")
                    await tournament_sql.update_tournament_task_training_status(
                        training_task.task.task_id, training_task.hotkey, TrainingStatus.PENDING, config.psql_db
                    )
                else:
                    logger.info(
                        f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} is still training on at least "
                        f"one trainer"
                    )

            except Exception as e:
                logger.error(f"Error checking task {training_task.task.task_id} with hotkey {training_task.hotkey}: {str(e)}")
                continue

    # If any tasks completed, update all trainers' GPU availability
    if any_completed:
        logger.info("Found completed tasks, updating GPU availability across all trainers")
        await _update_all_trainers_gpu_availability(config)

    logger.info(f"Completed monitoring cycle, processed {len(training_tasks)} tasks")


async def _update_all_trainers_gpu_availability(config: Config):
    """
    Update GPU availability for all trainers by fetching current status and syncing with database.
    """
    try:
        # Get all trainers from database
        trainers = await tournament_sql.get_trainers(config.psql_db)

        for trainer in trainers:
            try:
                # Fetch current GPU availability from trainer
                current_gpus = await fetch_trainer_gpus(trainer.trainer_ip)

                # Find GPUs that are free according to trainer but marked as used in DB
                gpus_to_reset = []
                gpus_to_mark_unavailable = []
                for current_gpu in current_gpus:
                    if current_gpu.available:
                        # Check if this GPU is marked as used in our database
                        for db_gpu in trainer.gpus:
                            if db_gpu.gpu_id == current_gpu.gpu_id and not db_gpu.available:
                                gpus_to_reset.append(current_gpu.gpu_id)
                                break
                    else:
                        for db_gpu in trainer.gpus:
                            if db_gpu.gpu_id == current_gpu.gpu_id and db_gpu.available:
                                gpus_to_mark_unavailable.append(current_gpu.gpu_id)
                                break

                # Reset GPU availability in database if needed
                if gpus_to_reset:
                    await tournament_sql.update_gpu_availability(trainer.trainer_ip, gpus_to_reset, 0, config.psql_db)
                    logger.info(f"Reset {len(gpus_to_reset)} GPUs for trainer {trainer.trainer_ip}: {gpus_to_reset}")

                if gpus_to_mark_unavailable:
                    await tournament_sql.update_gpu_availability(trainer.trainer_ip, gpus_to_mark_unavailable, 2, config.psql_db)
                    logger.info(
                        f"Marked {len(gpus_to_mark_unavailable)} GPUs as unavailable for trainer "
                        f"{trainer.trainer_ip}: {gpus_to_mark_unavailable}"
                    )

            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as e:
                # Handle both server errors and unreachable trainers
                if isinstance(e, httpx.HTTPStatusError):
                    status_code = e.response.status_code
                    if 500 <= status_code < 600:
                        error_msg = f"returned 5xx error ({status_code})"
                    else:
                        logger.error(f"HTTP error {status_code} from trainer {trainer.trainer_ip}: {str(e)}")
                        continue
                else:
                    # ConnectError or TimeoutException
                    error_msg = f"is unreachable ({type(e).__name__})"

                # Common handling for unreachable trainers - set all GPUs to be available in 2 hours
                all_gpu_ids = [gpu.gpu_id for gpu in trainer.gpus]
                await tournament_sql.update_gpu_availability(trainer.trainer_ip, all_gpu_ids, 2, config.psql_db)
                logger.warning(
                    f"Trainer {trainer.trainer_ip} {error_msg}, setting {len(all_gpu_ids)} GPUs to be available in 2 hours"
                )
                continue
            except Exception as e:
                logger.error(f"Error updating GPU availability for trainer {trainer.trainer_ip}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in _update_all_trainers_gpu_availability: {str(e)}")


async def move_completed_tasks_to_preevaluation(config: Config):
    """
    Find tasks where all training tasks (task_id, hotkey) pairs have completed
    and move those tasks to preevaluation status.
    """
    while True:
        try:
            logger.info("Moving completed tournament tasks to preevaluation")
            await _move_completed_tasks_to_preevaluation(config)
        except Exception as e:
            logger.error(f"Error in move_completed_tasks_to_preevaluation cycle: {str(e)}", exc_info=True)
        finally:
            await asyncio.sleep(cst.MOVE_COMPLETED_TASKS_CYCLE_INTERVAL)


async def _move_completed_tasks_to_preevaluation(config: Config):
    """
    Find tasks where all training tasks (task_id, hotkey) pairs have completed
    and move those tasks to preevaluation status.
    """
    # Get task IDs where all training tasks have completed (only from last month)
    completed_task_ids = await tournament_sql.get_tasks_with_all_training_completed(config.psql_db)
    logger.info(f"Found {len(completed_task_ids)} tasks with all training completed")

    if not completed_task_ids:
        logger.info("No tasks with all training completed, skipping cycle")
        return

    # Get the actual task objects for these IDs
    tasks_to_move = []
    for task_id in completed_task_ids:
        task = await task_sql.get_task(task_id, config.psql_db)
        if task:
            tasks_to_move.append(task)

    logger.info(f"Moving {len(tasks_to_move)} tasks to preevaluation status")

    # Move tasks to preevaluation
    for task in tasks_to_move:
        tournament_id = await get_tournament_id_by_task_id(task.task_id, config.psql_db)
        with LogContext(task_id=str(task.task_id), tournament_id=tournament_id):
            try:
                task.status = TaskStatus.PREEVALUATION
                await task_sql.update_task(task, config.psql_db)
                logger.info(f"Moved task {task.task_id} from training to preevaluation status")
            except Exception as e:
                logger.error(f"Error moving task {task.task_id} to preevaluation: {str(e)}")

    logger.info(f"Successfully moved {len(tasks_to_move)} tasks to preevaluation status")


async def update_all_trainers_gpu_availability_cycle(config: Config):
    """
    Periodically update GPU availability for all trainers.
    """
    while True:
        try:
            logger.info("Periodically updating all trainers' GPU availability")
            await _update_all_trainers_gpu_availability(config)
        except Exception as e:
            logger.error(f"Error in periodic GPU availability update: {str(e)}", exc_info=True)
        finally:
            await asyncio.sleep(cst.PERIODIC_GPU_AVAILABILITY_UPDATE_INTERVAL)


async def run_tournament_orchestrator_cycles():
    config = load_config()
    await try_db_connections(config)

    logger.info("Starting tournament orchestrator cycles")
    await asyncio.gather(
        fetch_tournament_tasks_ready_to_train(config),
        process_pending_tournament_tasks(config),
        monitor_training_tasks(config),
        move_completed_tasks_to_preevaluation(config),
        update_all_trainers_gpu_availability_cycle(config),
    )


if __name__ == "__main__":
    load_dotenv(".vali.env", override=True)
    asyncio.run(run_tournament_orchestrator_cycles())
