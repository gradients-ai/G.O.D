import asyncio
import logging
import math
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from time import monotonic

import httpx
from dotenv import load_dotenv
from huggingface_hub.errors import RepositoryNotFoundError
from pydantic import BaseModel
from pydantic import TypeAdapter
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

import validator.db.constants as db_cst
import validator.infrastructure.service_constants as service_cst
import validator.tasks.datasets.constants as data_cst
import validator.tournament.constants as cst
from core.constants.environments import TrainingStartPoint
from core.logging import REDACTED
from core.logging import LogContext
from core.logging import get_logger
from core.logging import redact_url_queries
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.model_prep_models import BaselineStats
from core.models.payload_models import ModelPrepJob
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestQuant
from core.models.payload_models import TrainRequestText
from core.models.quant_artifact_models import QuantDecodedFile
from core.models.quant_artifact_models import QuantDecodeRequest
from core.models.quant_models import QuantCalibrationReference
from core.models.quant_models import QuantCalibrationRequest
from core.models.quant_models import QuantTarget
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from core.models.trainer_contract_models import GPUInfo
from core.models.trainer_contract_models import GPUInterconnect
from core.models.trainer_contract_models import GPUType
from validator.app.config import Config
from validator.app.config import load_config
from validator.db.sql import gpu_costs as gpu_cost_sql
from validator.db.sql import quant_sources as quant_sources_sql
from validator.db.sql import submissions_and_scoring as submissions_and_scoring_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournament_seeds as seeds_sql
from validator.db.sql import tournaments as tournament_sql
from validator.db.sql.quant_calibration_references import get_quant_calibration_references
from validator.db.sql.quant_references import get_quant_sku_references
from validator.db.sql.tournament_performance import is_final_round_task
from validator.db.sql.tournaments import get_tournament_id_by_task_id
from validator.evaluation.notifications import notify_evaluation_exception
from validator.evaluation.quant_artifacts import decoded_model_prefix
from validator.infrastructure.minio_client import async_minio_client
from validator.infrastructure.service_constants import GET_GPU_AVAILABILITY_ENDPOINT
from validator.infrastructure.service_constants import MODEL_PREP_STATUS_ENDPOINT
from validator.infrastructure.service_constants import PROXY_TRAINING_IMAGE_ENDPOINT
from validator.infrastructure.service_constants import TASK_DETAILS_ENDPOINT
from validator.lifecycle import boss_seed_eval
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import QuantSkuReference
from validator.scoring.quant_ranking import freeform_bar
from validator.scoring.tasks import _get_dataset_type
from validator.tasks.details import try_db_connections
from validator.tasks.models import AnyQuantRawTask
from validator.tasks.models import AnyTypeRawTask
from validator.tasks.models import Backend
from validator.tasks.models import ChatRawTask
from validator.tasks.models import EnvRawTask
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import freeform_penalty_policy
from validator.tasks.prep.model import dispatch_augmentation_and_stats
from validator.tasks.quant_input_repo import delete_merged_repo
from validator.tasks.quant_input_repo import full_model_rejection
from validator.tasks.quant_input_repo import is_adapter_repo
from validator.tasks.synthetics.scheduler import _draw_quant_instruct_dataset
from validator.tasks.synthetics.scheduler import _fetch_continuous_quant_chunk
from validator.tasks.synthetics.scheduler import _get_instruct_text_datasets
from validator.tasks.synthetics.scheduler import compute_hours_from_baseline_stats
from validator.tasks.synthetics.scheduler import quant_hours_after_prep
from validator.tasks.synthetics.scheduler import quant_training_hours
from validator.tournament import chaining
from validator.tournament import quant_sources
from validator.tournament.gpu_requirements import get_task_gpu_requirement
from validator.tournament.models import ANY_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import ANY_QUANT_SEED_ROLES
from validator.tournament.models import ATTACHED_SEED_ROLES
from validator.tournament.models import INPUT_SEED_ROLES
from validator.tournament.models import QUANT_ROUND_SEED_ROLES
from validator.tournament.models import TEXT_ROUND_SEED_ROLES
from validator.tournament.models import TEXT_SEED_ROLES
from validator.tournament.models import GpuRequirement
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TaskTrainingAssignment
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentTaskTraining
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingRepoInfo
from validator.tournament.models import TrainingStatus
from validator.tournament.notifications import notify_model_prep_failure_limit
from validator.tournament.notifications import notify_quant_decode_infra_error
from validator.tournament.quant_reference import is_own_model_start_point
from validator.tournament.quant_reference import reference_hotkey_for
from validator.tournament.quant_reference_prepass import calibration_reference_gave_up
from validator.tournament.quant_reference_prepass import has_calibration_set
from validator.tournament.quant_reference_prepass import measure_quant_references_cycle
from validator.tournament.quant_reference_prepass import reference_gave_up
from validator.tournament.seed_tasks import FIRST_BOSS_ENTRY_TRAINING_RANK
from validator.tournament.seed_tasks import INSTRUCT_INPUT_SEED_ROLES
from validator.tournament.seed_tasks import NON_SEED_TRAINING_RANK
from validator.tournament.seed_tasks import SEED_ADAPTER_MERGE_COOLDOWN_BASE_SECONDS
from validator.tournament.seed_tasks import SEED_ADAPTER_MERGE_COOLDOWN_MAX_SECONDS
from validator.tournament.seed_tasks import SEED_ADAPTER_MERGE_FAILURES_BEFORE_ALERT
from validator.tournament.seed_tasks import SEED_ADAPTER_MERGE_MAX_FAILURES
from validator.tournament.seed_tasks import SEED_ADAPTER_MERGE_UNKNOWN_CYCLES_BEFORE_DISPATCH
from validator.tournament.seed_tasks import alert_seed_issue
from validator.tournament.seed_tasks import boss_output_awaits_merge
from validator.tournament.seed_tasks import is_boss_entry_row
from validator.tournament.seed_tasks import live_seed_tasks
from validator.tournament.seed_tasks import repoint_task_onto_seed
from validator.tournament.seed_tasks import training_ranks


logger = get_logger(__name__)

# The seeds only the boss trains and a chained task is released on: the two round inputs and the boss round's input.
_INPUT_SEED_ROLES = INPUT_SEED_ROLES | frozenset({SeedRole.BOSS_INSTRUCT})

# Persisted as quant_failure for a freeform entrant registered without a decoder snapshot: every
# repo must carry a decoder/, so this is a scored fail and evaluation never spends a GPU on it.
NO_DECODER_SNAPSHOT_FAILURE = "no decoder/ snapshot at registration"
# (task_id, hotkey) pairs already pinged for a held freeform decode. Process-local like
# tournament_manager's alert sets: the hold re-checks every cycle and would otherwise spam the webhook.
_QUANT_DECODE_ALERTED: set[tuple[str, str]] = set()
# A trainer that keeps failing the decode re-poll has likely vanished; after this many consecutive
# HTTP failures for one (task_id, hotkey) the hold is alerted instead of retried silently forever.
QUANT_DECODE_REPOLL_FAILURES_BEFORE_ALERT = 3
_QUANT_DECODE_REPOLL_FAILURES: dict[tuple[str, str], int] = {}
# Quant tasks already pinged for a training row dispatched without calibration rows. Process-local like the sets above: every
# row of the task logs the ERROR, the webhook hears about the task once.
_QUANT_CALIBRATION_MISSING_ALERTED: set[str] = set()
_BASELINE_STATS_ADAPTER: TypeAdapter = TypeAdapter(BaselineStats)


def _training_cost_run_key(task_id: str, hotkey: str) -> str:
    return f"training:{task_id}:{hotkey}"


def _prep_cost_run_key(task_id: str, hotkey: str | None = None) -> str:
    identity = f"miner:{hotkey}" if hotkey is not None else "task"
    return f"prep:{task_id}:{identity}"


async def _max_training_hours_for_task(task, config: Config) -> float | None:
    """Hours above which a round-1 group text task is replaced. None means no extra limit."""
    if task.task_type != TaskType.INSTRUCTTEXTTASK:
        return None
    tourn_task = await tournament_sql.get_tournament_task_by_task_id(str(task.task_id), config.psql_db)
    if tourn_task is None:
        return None
    if (
        tourn_task.group_id is not None
        and tourn_task.pair_id is None
        and tourn_task.round_id.endswith("_round_001")
    ):
        return cst.ROUND_ONE_TEXT_MAX_TRAINING_HOURS
    return None


async def _apply_measured_training_hours(task, config: Config) -> bool:
    """Write post-prep hours from measured stats.

    Returns True when a round-1 text task measures longer than ROUND_ONE_TEXT_MAX_TRAINING_HOURS.
    That task is marked for replacement. Its budget is not shortened to the limit.
    """
    if task.task_type in QUANT_TASK_TYPES:
        new_hours = quant_hours_after_prep(task, task.baseline_stats, boss=await _is_boss_round_task(task, config))
    else:
        new_hours = compute_hours_from_baseline_stats(
            task.hours_to_complete,
            task.baseline_stats,
            task.task_type,
            model_id=task.model_id,
            model_params_count=task.model_params_count,
            training_start_point=task.training_start_point,
            ds=task.ds,
        )
    limit = await _max_training_hours_for_task(task, config)
    if limit is not None and new_hours > limit:
        logger.warning(
            f"Round-1 text task {task.task_id} measured {new_hours}h > {limit}h; "
            f"marking {TaskStatus.PREP_TASK_FAILURE.value} for replacement"
        )
        task.status = TaskStatus.PREP_TASK_FAILURE
        await task_sql.update_task(task, config.psql_db)
        return True

    if task.task_type in NON_BOSS_CAPPED_TASK_TYPES and await _is_non_boss_tournament_task(task, config):
        new_hours = min(new_hours, data_cst.NON_BOSS_INSTRUCT_MAX_TRAINING_HOURS)
    task.hours_to_complete = new_hours
    task.termination_at = datetime.utcnow() + timedelta(hours=new_hours)
    return False


async def _start_prep_cost_run(
    run_key: str,
    task_id: str,
    prep_identity: str,
    trainer_ip: str,
    gpu_ids: list[int],
    config: Config,
    interconnect: str | None = None,
) -> None:
    try:
        await gpu_cost_sql.start_cost_run(
            run_key=run_key,
            task_id=task_id,
            category="prep",
            gpu_type="H100",
            gpu_count=len(gpu_ids),
            interconnect=interconnect,
            psql_db=config.psql_db,
            metadata={
                "prep_identity": prep_identity,
                "trainer_ip": trainer_ip,
                "gpu_ids": gpu_ids,
            },
        )
    except Exception as e:
        logger.error(f"Failed to start GPU cost run {run_key}: {e}", exc_info=True)


async def _finish_training_cost_run(
    training_task: TournamentTaskTraining,
    success: bool,
    config: Config,
    ended_at: datetime | None = None,
) -> None:
    run_key = _training_cost_run_key(str(training_task.task.task_id), training_task.hotkey)
    try:
        await gpu_cost_sql.finish_cost_run(
            run_key=run_key,
            success=success,
            psql_db=config.psql_db,
            ended_at=ended_at,
        )
    except Exception as e:
        logger.error(f"Failed to finalize GPU cost run {run_key}: {e}", exc_info=True)


async def _finish_prep_cost_run(
    run_key: str,
    success: bool,
    config: Config,
    ended_at: datetime | None = None,
) -> None:
    try:
        result = await gpu_cost_sql.finish_cost_run(
            run_key=run_key,
            success=success,
            psql_db=config.psql_db,
            ended_at=ended_at,
        )
    except Exception as e:
        logger.error(f"Failed to finalize GPU cost run {run_key}: {e}", exc_info=True)
        return

    if result is not None and result.get("should_notify_prep_failure", False):
        await notify_model_prep_failure_limit(
            task_id=str(result["task_id"]),
            prep_identity=str(result["metadata"].get("prep_identity", "task")),
            discord_url=config.discord_url,
        )
    if result is not None and not success:
        await _give_up_on_failing_seed_prep(result, config)


async def _give_up_on_failing_seed_prep(run: dict, config: Config) -> None:
    """Stop retrying a seed's task-level model prep after SEED_MAX_PREP_ATTEMPTS failures: PREP_TASK_FAILURE, so the seed path
    redraws it (a dataset rejection, process_tournament_seed_tasks). Prep otherwise retries every cycle with no cap, so a
    deterministic prep error would hold the seed, and the tournament behind it, forever. Ordinary tasks keep that retry.
    """
    if run["category"] != "prep" or run["metadata"].get("prep_identity", "task") != "task":
        return
    if run["prep_failure_count"] < cst.SEED_MAX_PREP_ATTEMPTS:
        return
    try:
        task = await task_sql.get_task(run["task_id"], config.psql_db)
        if task is None or task.status != TaskStatus.AWAITING_MODEL_PREP or await _seed_role(task, config) is None:
            return
        logger.error(
            f"Seed task {task.task_id}: model prep failed {run['prep_failure_count']} times; "
            f"marking {TaskStatus.PREP_TASK_FAILURE.value} so it is redrawn"
        )
        task.status = TaskStatus.PREP_TASK_FAILURE
        await task_sql.update_task(task, config.psql_db)
    except Exception as e:
        logger.error(f"Could not give up on failing seed prep for task {run['task_id']}: {e}", exc_info=True)


simple_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
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


def loggable_training_request(training_request: TrainerProxyRequest) -> str:
    """The request as it may be logged. A quant request is logged without its GitHub token, presigned decoder URL and URL
    queries; any other request as it is."""
    payload = training_request.model_dump()
    if not isinstance(training_request.training_data, TrainRequestQuant):
        return str(payload)
    if payload.get("github_token"):
        payload["github_token"] = REDACTED
    decode = payload["training_data"].get("decode")
    if decode:
        decode["decoder_snapshot_url"] = REDACTED
    return redact_url_queries(str(payload))


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
        logger.error(f"Request payload: {loggable_training_request(training_request)}")
        return False

    async with httpx.AsyncClient(timeout=cst.TRAINER_HTTP_TIMEOUT) as client:
        # Default to port 8001 if no port is specified
        if ":" not in trainer_ip:
            trainer_ip_with_port = f"{trainer_ip}:8001"
        else:
            trainer_ip_with_port = trainer_ip

        url = f"http://{trainer_ip_with_port}{PROXY_TRAINING_IMAGE_ENDPOINT}"
        logger.info(f"Requesting training from trainer at {url} with payload: {loggable_training_request(validated_request)}")

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
    pending_quant_count = 0
    for training_task in pending_training_tasks:
        if training_task.task.task_type == TaskType.IMAGETASK:
            pending_image_count += 1
        elif await _belongs_to_quant_tournament(training_task.task, config):
            pending_quant_count += 1
        else:
            pending_text_count += 1

    logger.info(f"Pending by type - Text: {pending_text_count}, Image: {pending_image_count}")
    if pending_quant_count:
        logger.info(f"Pending quant tournament assignments: {pending_quant_count}")

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
    quant_tasks_to_process = []

    for task in all_tournament_tasks:
        if task.task_type == TaskType.IMAGETASK:
            if pending_image_count < cst.PENDING_QUEUE_THRESHOLD_PER_TYPE:
                image_tasks_to_process.append(task)
        elif await _belongs_to_quant_tournament(task, config):
            if pending_quant_count < cst.PENDING_QUEUE_THRESHOLD_PER_TYPE:
                quant_tasks_to_process.append(task)
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

    if quant_tasks_to_process:
        logger.info(
            f"Pending quant queue below {cst.PENDING_QUEUE_THRESHOLD_PER_TYPE}, "
            f"processing {len(quant_tasks_to_process)} quant tournament tasks"
        )
        await _process_tasks_for_training(quant_tasks_to_process, config, priority=2)

    if pending_count < cst.PENDING_QUEUE_THRESHOLD_FOR_BENCHMARK:
        logger.info(f"Pending queue below {cst.PENDING_QUEUE_THRESHOLD_FOR_BENCHMARK}, fetching benchmark tasks")
        benchmark_tasks = await task_sql.get_tasks_with_status(
            TaskStatus.LOOKING_FOR_NODES, config.psql_db, tournament_filter="exclude", benchmark_filter="only"
        )
        logger.info(f"Found {len(benchmark_tasks)} benchmark tasks looking for nodes")
        await _process_tasks_for_training(benchmark_tasks, config, priority=3)


async def _belongs_to_quant_tournament(task: AnyTypeRawTask, config: Config) -> bool:
    """Whether a task is a round task or a seed task of a quant tournament (the seeds' instruct tasks included)."""
    tournament_type = await tournament_sql.get_tournament_type_for_task(str(task.task_id), config.psql_db)
    return tournament_type == TournamentType.QUANT


def _is_own_model_quant(task: AnyTypeRawTask) -> bool:
    return task.task_type in QUANT_TASK_TYPES and is_own_model_start_point(task.training_start_point)


async def _without_unusable_own_models(task: AnyTypeRawTask, hotkeys: list[str], config: Config) -> list[str]:
    """On an own-model quant task, drop hotkeys with a persisted quant_failure (no usable R1 model).

    That is a scored fail: it gets no training row, so it never takes a GPU and cannot trip the
    majority-failure gate.
    """
    if not _is_own_model_quant(task):
        return hotkeys
    rows = await submissions_and_scoring_sql.get_task_node_losses(task.task_id, config.psql_db)
    failed = {row[db_cst.HOTKEY] for row in rows if row[db_cst.QUANT_FAILURE]}
    for hotkey in failed.intersection(hotkeys):
        logger.info(f"Task {task.task_id}: {hotkey} has a scored quant failure and no usable own model; no training row")
    return [hotkey for hotkey in hotkeys if hotkey not in failed]


async def _awaiting_own_upstream_output(task: AnyTypeRawTask, hotkeys: list[str], config: Config) -> set[str]:
    """Hotkeys of an own-upstream task with no starting model yet: their upstream run is not done, so they get no
    training row from the move to TRAINING. The boss-round seed cycle adds it once the player's own output is checked."""
    if task.training_start_point != TrainingStartPoint.OWN_UPSTREAM_MODEL:
        return set()
    return {hotkey for hotkey in hotkeys if not await task_sql.get_starting_model_repo(str(task.task_id), hotkey, config.psql_db)}


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
        # Environment tournaments are retired; never assign environment tasks for training.
        # This covers leftover pending tournament ENVIRONMENTTASKs and organic environment tasks.
        if task.task_type == TaskType.ENVIRONMENTTASK:
            logger.info(f"Skipping environment task {task.task_id} for training; environment tournaments are retired")
            continue

        # For tournament and benchmark tasks (priority 2 and 3), get actual assigned nodes
        # For organic tasks (priority 1), we'll use EMISSION_BURN_HOTKEY
        if priority in [2, 3]:
            logger.debug(f"Getting nodes for task {task.task_id} (type: {type(task.task_id)})")
            nodes = await task_sql.get_nodes_assigned_to_task(task.task_id, config.psql_db)
            logger.debug(f"Found {len(nodes)} nodes for task {task.task_id}: {[n.hotkey for n in nodes]}")
            hotkeys = [node.hotkey for node in nodes]

            if task.task_type in QUANT_TASK_TYPES:
                assigned_hotkeys = hotkeys
                hotkeys = await _without_unusable_own_models(task, assigned_hotkeys, config)
                awaiting_upstream = await _awaiting_own_upstream_output(task, hotkeys, config)
                hotkeys = [hotkey for hotkey in hotkeys if hotkey not in awaiting_upstream]

                if assigned_hotkeys and not hotkeys and not awaiting_upstream:
                    # Every player lost an own-model task to a scored fail (no usable round-1 model): nothing can ever train,
                    # so fail the task rather than leave it in LOOKING_FOR_NODES. The scored fails are already persisted.
                    logger.error(f"Task {task.task_id}: no player of {assigned_hotkeys} has a usable own model; failing it")
                    task.status = TaskStatus.FAILURE
                    await task_sql.update_task(task, config.psql_db)
                    continue

            if hotkeys:
                # Get tournament_id for this task
                tournament_id = await tournament_sql.get_tournament_id_by_task_id(task.task_id, config.psql_db)

                for hotkey in hotkeys:
                    repo_info = await tournament_sql.get_tournament_training_repo_and_commit(
                        hotkey, tournament_id, config.psql_db
                    ) if tournament_id else TrainingRepoInfo.empty()

                    assignments.append(
                        TaskTrainingAssignment(
                            task_id=str(task.task_id),
                            hotkey=hotkey,
                            created_at=task.created_at,
                            priority=priority,
                            training_repo=repo_info.training_repo,
                            training_commit_hash=repo_info.training_commit_hash,
                            github_token=repo_info.github_token,
                            requested_datasets=repo_info.requested_datasets,
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
            elif task.task_type in QUANT_TASK_TYPES:
                tournament_type = TournamentType.QUANT
            elif task.task_type == TaskType.IMAGETASK:
                tournament_type = TournamentType.IMAGE
            else:
                tournament_type = None

            # Get the last completed tournament winner's repo
            last_tournament = await tournament_sql.get_latest_completed_tournament(config.psql_db, tournament_type)
            if last_tournament and last_tournament.winner_hotkey:
                repo_info = await tournament_sql.get_tournament_training_repo_and_commit(
                    last_tournament.winner_hotkey, last_tournament.tournament_id, config.psql_db
                )
            else:
                repo_info = TrainingRepoInfo.empty()

            assignments.append(
                TaskTrainingAssignment(
                    task_id=str(task.task_id),
                    hotkey=EMISSION_BURN_HOTKEY,
                    created_at=task.created_at,
                    priority=priority,
                    training_repo=repo_info.training_repo,
                    training_commit_hash=repo_info.training_commit_hash,
                    github_token=repo_info.github_token,
                    requested_datasets=repo_info.requested_datasets,
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


def _is_quant_tournament_row(row_rank: int, task: AnyTypeRawTask) -> bool:
    """Whether a training row belongs to a quant tournament: a quant task, or an instruct seed task (only a quant tournament
    has seed tasks, and they are the rows whose rank is not the ordinary one)."""
    return task.task_type in QUANT_TASK_TYPES or row_rank != NON_SEED_TRAINING_RANK


async def schedule_tasks_for_training(pending_training_tasks: list[TournamentTaskTraining], config: Config):
    """
    Process tasks from the list and schedule them for training.
    Only pop tasks when we're 100% sure GPUs are available.
    """
    # Track failed attempts for this scheduling session
    failed_attempts = {}

    tasks_without_gpus = []
    waiting_for_reference = []
    failing_quant_rows = []

    # Rows are taken from the END of the list. Order them small input seed, large input seed, every other row (miner
    # rounds), the boss round's input, then the compression tasks; the rank is the task's, so the boss's and the
    # challenger's rows on a task tie. The query returns newest first and the sort is stable, so popping from the end takes the
    # oldest first within a rank.
    task_ranks = await training_ranks([str(row.task.task_id) for row in pending_training_tasks], config.psql_db)

    def rank_of(row: TournamentTaskTraining) -> int:
        return task_ranks[str(row.task.task_id)]

    pending_training_tasks[:] = sorted(pending_training_tasks, key=lambda row: -rank_of(row))
    miner_row_starved = False
    boss_entries_deferred = []

    while pending_training_tasks:
        oldest_task_training = pending_training_tasks[-1]
        if miner_row_starved and is_boss_entry_row(rank_of(oldest_task_training), oldest_task_training.hotkey):
            # A boss entry (the boss round's input, a compression: they have a day of slack) yields in a cycle where an
            # input seed or a miner row is short of GPUs. Entries do not skip each other, and the challenger's
            # boss-round rows (which share the compressions' rank but are not boss entries) are never skipped.
            boss_entries_deferred.append(pending_training_tasks.pop())
            continue
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

            try:
                reference_ready = await _reference_ready(task, oldest_task_training.hotkey, config)
            except Exception as e:
                if not _is_quant_tournament_row(rank_of(oldest_task_training), task):
                    raise
                # A quant row that cannot be scheduled this cycle must not hold the rows behind it (text, image): it
                # comes back with the next fetch.
                logger.error(f"Exception while checking the reference of task {task.task_id}: {e}, deferring it", exc_info=True)
                failing_quant_rows.append(pending_training_tasks.pop())
                continue
            if not reference_ready:
                waiting_for_reference.append(pending_training_tasks.pop())
                continue

            # Determine required GPUs for this task
            required_gpus = get_task_gpu_requirement(task)
            logger.info(f"Task {task.task_id} requires {required_gpus.value}")
            await _update_all_trainers_gpu_availability(config)
            suitable_gpus_result = await _check_suitable_gpus(config, required_gpus)

            if not suitable_gpus_result:
                logger.info(
                    f"No suitable GPUs found for requirement {required_gpus.value}, skipping this task and continuing with next"
                )

                if rank_of(oldest_task_training) < FIRST_BOSS_ENTRY_TRAINING_RANK:
                    miner_row_starved = True
                tasks_without_gpus.append(pending_training_tasks.pop())
                await asyncio.sleep(1)  # TODO: put in constant or even remove
                continue

            trainer_ip, gpu_ids, interconnect = suitable_gpus_result

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
                    training_task.github_token,
                    training_task.requested_datasets,
                    config,
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
                    attempt = training_task.n_training_attempts + 1
                    run_key = _training_cost_run_key(str(training_task.task.task_id), training_task.hotkey)
                    try:
                        await gpu_cost_sql.start_cost_run(
                            run_key=run_key,
                            task_id=str(training_task.task.task_id),
                            tournament_id=tournament_id,
                            category="training",
                            gpu_type="H100",
                            gpu_count=len(gpu_ids),
                            interconnect=interconnect,
                            psql_db=config.psql_db,
                            metadata={
                                "hotkey": training_task.hotkey,
                                "attempt": attempt,
                                "trainer_ip": trainer_ip,
                                "gpu_ids": gpu_ids,
                            },
                        )
                    except Exception as e:
                        logger.error(f"Failed to start GPU cost run {run_key}: {e}", exc_info=True)
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
            if _is_quant_tournament_row(rank_of(training_task), training_task.task):
                # Do not retry a raising quant row in this session: it would hold the text and image rows behind it. It is
                # fetched again next cycle. (Only when it is still the head: a failure after its pop leaves the queue alone.)
                if pending_training_tasks and pending_training_tasks[-1] is training_task:
                    failing_quant_rows.append(pending_training_tasks.pop())
                continue
            await asyncio.sleep(cst.TRAINING_START_RETRY_INTERVAL)
            continue

    if failing_quant_rows:
        logger.warning(
            f"Deferred {len(failing_quant_rows)} quant training row(s) that raised while being scheduled; they are fetched "
            f"again in the next cycle."
        )

    if boss_entries_deferred:
        logger.info(
            f"A miner-round row is waiting for GPUs; {len(boss_entries_deferred)} boss entry row(s) (boss input / compressions) "
            f"yield this cycle and are retried in the next"
        )

    if tasks_without_gpus:
        logger.info(
            f"Skipped {len(tasks_without_gpus)} tasks due to GPU unavailability. They will get picked up in the next cycle."
        )

    if waiting_for_reference:
        logger.info(
            f"Deferred {len(waiting_for_reference)} own-model quant (task, hotkey) pairs waiting for their bf16 reference "
            f"(quant_reference_prepass). They will get picked up in the next cycle."
        )

    logger.info(f"Completed scheduling cycle, {len(pending_training_tasks)} tasks remaining")


async def _reference_ready(task: AnyTypeRawTask, hotkey: str, config: Config) -> bool:
    """False while a quant miner's bf16 reference is unmeasured: its target cannot be published yet.

    An own-model miner waits for its own row; on any other quant task every miner waits for the task-common row
    (stored under hotkey ''), measured once by the prepass; a non-tournament task never waits, and a task-common reference
    that failed QUANT_TASK_COMMON_REFERENCE_MAX_FAILURES times in a row is given up on.
    """
    if task.task_type not in QUANT_TASK_TYPES:
        return True
    reference_hotkey = reference_hotkey_for(task, hotkey)
    if reference_hotkey == "":
        # The prepass measures tournament tasks only, so an organic task has nothing to wait for.
        if await get_tournament_id_by_task_id(str(task.task_id), config.psql_db) is None:
            return True
    references = await get_quant_sku_references(task.task_id, config.psql_db)
    if not any(stored_hotkey == reference_hotkey for stored_hotkey, _ in references):
        # A task-common reference that keeps failing is given up on (miners train with no target); an own-model one never is.
        return reference_gave_up(task.task_id, reference_hotkey)
    return await _calibration_reference_ready(task, hotkey, config)


async def _calibration_reference_ready(task: AnyTypeRawTask, hotkey: str, config: Config) -> bool:
    """A task with calibration rows also waits for the reference's bf16 loss on them, until the prepass gives up on it."""
    if not has_calibration_set(task):
        return True
    reference_hotkey = reference_hotkey_for(task, hotkey)
    if reference_hotkey in await get_quant_calibration_references(task.task_id, config.psql_db):
        return True
    return calibration_reference_gave_up(task.task_id, reference_hotkey)


async def _check_suitable_gpus(
    config: Config, required_gpus: GpuRequirement
) -> tuple[str, list[int], str] | None:
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
        tuple[str, list[int], str] | None: (trainer_ip, gpu_ids, interconnect) if suitable
        GPUs found, None otherwise
    """
    try:
        trainers = await tournament_sql.get_trainers(config.psql_db)
        required_gpu_count = required_gpus.gpu_count

        best_trainer = None
        best_gpu_ids = None
        best_interconnect = GPUInterconnect.UNKNOWN.value
        best_ratio = -1.0
        for trainer in trainers:
            gpu_ids = _trainer_has_sufficient_gpus(trainer.gpus, required_gpus)
            if gpu_ids:
                free_gpu_count = sum(1 for gpu in trainer.gpus if gpu.available)
                ratio = required_gpu_count / free_gpu_count
                interconnect = _interconnect_for_gpu_ids(trainer.gpus, gpu_ids)

                # >1-GPU jobs only go to SXM / NVLink boxes; PCIe would bottleneck.
                if (
                    required_gpu_count > 1
                    and cst.MULTI_GPU_REQUIRES_HIGH_INTERCONNECT
                    and not _is_high_interconnect(trainer.gpus, gpu_ids)
                ):
                    logger.info(
                        f"Skipping trainer {trainer.trainer_ip} for {required_gpus.value}: "
                        f"interconnect {interconnect} is not SXM/NVLink"
                    )
                    continue

                logger.info(
                    f"Trainer {trainer.trainer_ip}: {len(gpu_ids)} GPUs available for requirement {required_gpus.value}, "
                    f"{free_gpu_count} total free GPUs, ratio: {ratio:.2f}, interconnect: {interconnect}"
                )

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_trainer = trainer.trainer_ip
                    best_gpu_ids = gpu_ids
                    best_interconnect = interconnect

        if best_trainer:
            logger.info(
                f"Selected trainer {best_trainer} with best utilization ratio {best_ratio:.2f} "
                f"for requirement {required_gpus.value} (interconnect={best_interconnect})"
            )
            return best_trainer, best_gpu_ids, best_interconnect

        logger.info(f"No suitable GPUs found for requirement {required_gpus.value}")
        return None

    except Exception as e:
        logger.error(f"Error checking suitable GPUs: {str(e)}")
        return None


def _interconnect_for_gpu_ids(gpus: list[GPUInfo], gpu_ids: list[int]) -> str:
    id_set = set(gpu_ids)
    for gpu in gpus:
        if gpu.gpu_id not in id_set:
            continue
        interconnect = getattr(gpu, "interconnect", None)
        if interconnect is None:
            return GPUInterconnect.UNKNOWN.value
        return interconnect.value if hasattr(interconnect, "value") else str(interconnect)
    return GPUInterconnect.UNKNOWN.value


_HIGH_INTERCONNECTS = {GPUInterconnect.SXM.value, GPUInterconnect.NVL.value}


def _selection_is_whole_nvlink_groups(gpus: list[GPUInfo], gpu_ids: list[int]) -> bool:
    """True if the selection is made of COMPLETE NVLink groups — no group is split.

    Each NVLink group (its connected component over nvlink_peers) that the selection
    touches must be fully inside the selection. So on a box bridged as pairs
    {0-1, 2-3, ...} a 4x job may take two whole pairs (0-1-2-3), an 8x all four, but a
    pair must never be split (e.g. 1+2 straddles two pairs and is rejected). Within a
    group the link is NVLink; across groups a copy still crosses PCIe, which the caller
    accepts for >1-group jobs.
    """
    id_set = set(gpu_ids)
    peers = {gpu.gpu_id: set(gpu.nvlink_peers or []) for gpu in gpus}
    seen: set[int] = set()
    for gid in id_set:
        if gid in seen:
            continue
        component, stack = set(), [gid]  # full NVLink group of gid, across the whole box
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(peers.get(current, ()))
        seen |= component
        if not component <= id_set:  # part of this group sits outside the selection
            return False
    return True


def _is_high_interconnect(gpus: list[GPUInfo], gpu_ids: list[int]) -> bool:
    """True if the selected GPUs are high-interconnect enough to run a >1-GPU job.

    - SXM / NVL form factor (NVSwitch fabric): any subset is fully connected.
    - Otherwise every selected GPU must carry an NVLink bridge (rules out plain PCIe),
      and when the trainer reports topology (nvlink_peers) the selection must consist of
      WHOLE NVLink groups — e.g. one bonded pair for 2x, two pairs for 4x, four for 8x —
      never a split pair. Cross-group traffic falls back to PCIe, accepted for >1 group.
    - When no topology is reported but every selected GPU flags nvlink, accept as a best
      effort (matches the adjacent-pair assumption used when picking the group).
    - Everything else (plain PCIe / unknown) is low-interconnect.
    """
    if _interconnect_for_gpu_ids(gpus, gpu_ids) in _HIGH_INTERCONNECTS:
        return True
    id_set = set(gpu_ids)
    selected = [gpu for gpu in gpus if gpu.gpu_id in id_set]
    if not selected or not all(getattr(gpu, "nvlink", False) for gpu in selected):
        return False
    if any(gpu.nvlink_peers is None for gpu in selected):
        return True  # no topology reported; nvlink flags are the best signal we have
    return _selection_is_whole_nvlink_groups(gpus, gpu_ids)



def _quant_training_data(
    task: AnyQuantRawTask,
    expected_repo_name: str | None,
    baseline_stats: BaselineStats | dict | None,
    decode: QuantDecodeRequest | None = None,
    *,
    model: str,
    quant_target: QuantTarget | None,
    calibration: QuantCalibrationRequest | None = None,
) -> TrainRequestQuant:
    """Build the trainer request for a quant task: the entrant compresses model, its own starting model when set."""
    return TrainRequestQuant(
        model=model,
        task_id=str(task.task_id),
        hours_to_complete=task.hours_to_complete,
        expected_repo_name=expected_repo_name,
        dataset=task.training_data,
        dataset_type=task.dataset_type,
        file_format=FileFormat.S3,  # always an S3 since we task prep
        task_type=task.task_type,
        quant_spec=task.quant_spec if task.task_type == TaskType.QUANTFIXEDTASK else None,
        baseline_stats=baseline_stats,
        decode=decode,
        quant_target=quant_target,
        calibration=calibration,
    )


def quant_target_from_reference(task: AnyQuantRawTask, reference: QuantSkuReference) -> QuantTarget:
    """The target published to a miner from its measured bf16 reference; fixed has no bar, freeform adds the penalty.

    bar_cap / bar point at the knee for a penalised task. A task with no curvature has its knee at its cap and the
    curvature 0.
    """
    if task.task_type != TaskType.QUANTFREEFORMTASK:
        return QuantTarget(reference_model=reference.model, reference_loss=reference.loss, reference_sku=reference.sku)
    policy = freeform_penalty_policy(task)
    knee_excess = task.bar_cap if policy.is_cliff else task.penalty_knee
    knee_loss = freeform_bar(reference.loss, knee_excess)
    return QuantTarget(
        reference_model=reference.model,
        reference_loss=reference.loss,
        reference_sku=reference.sku,
        bar_cap=knee_excess,
        bar=knee_loss,
        knee=policy.knee,
        knee_loss=knee_loss,
        penalty_curvature=policy.curvature,
        cap=policy.cap,
        cap_loss=freeform_bar(reference.loss, task.bar_cap),
    )


async def _participant_for_task(task_id, hotkey: str, config: Config) -> TournamentParticipant | None:
    tournament_id = await get_tournament_id_by_task_id(task_id, config.psql_db)
    if tournament_id is None:
        return None
    return await tournament_sql.get_tournament_participant(tournament_id, hotkey, config.psql_db)


async def _decode_request_for(task: AnyQuantRawTask, hotkey: str, config: Config) -> QuantDecodeRequest | None:
    """The trainer-side decode of a freeform entry, run on the decoder snapshot taken at registration.

    The decode only ever runs that MinIO tar, never a fresh clone, so a miner cannot swap decoders
    after review. Without a snapshot there is nothing to run; the scored fail is persisted when
    training succeeds (_record_quant_decode).
    """
    participant = await _participant_for_task(task.task_id, hotkey, config)
    if participant is None or not participant.has_decoder_snapshot:
        logger.warning(f"No decoder snapshot for {hotkey} on task {task.task_id}; training without a decode")
        return None
    # Presigned for the default 7 days, longer than any training budget. Never log this URL.
    snapshot_url = await async_minio_client.get_presigned_url(service_cst.BUCKET_NAME, participant.decoder_object_name)
    return QuantDecodeRequest(
        decoder_snapshot_url=snapshot_url,
        decoder_sha256=participant.decoder_sha256,
        bucket=service_cst.BUCKET_NAME,
        object_prefix=decoded_model_prefix(task.task_id, hotkey),
    )


async def _quant_target_for(task: AnyQuantRawTask, hotkey: str, config: Config) -> QuantTarget | None:
    """The bar published to a miner from its stored bf16 reference; None when the reference is not measured.

    An own-model miner reads its own rows; any other quant task reads the task-common reference (hotkey ''), which
    the prepass measures before training starts (training waits on it: _reference_ready).
    With several SKUs measured the earliest is the target: the scorer re-normalises per SKU anyway.
    """
    references = await get_quant_sku_references(task.task_id, config.psql_db)
    own = [reference for reference in references.values() if reference.hotkey == reference_hotkey_for(task, hotkey)]
    if not own:
        return None
    first = min(own, key=lambda reference: (reference.measured_at is None, reference.measured_at))
    return quant_target_from_reference(task, first)


async def _quant_calibration_for(task: AnyQuantRawTask, hotkey: str, config: Config) -> QuantCalibrationRequest | None:
    """The calibration rows a quant miner is handed (fixed or freeform), with the reference's bf16 loss on them when measured.

    None for a task with no calibration set. A missing or inconsistent reference (the prepass gave up, or measured
    other examples) publishes the rows with reference None; a presigned-URL failure publishes nothing. Training never
    fails on calibration. The URL is presigned and never logged.
    """
    if not has_calibration_set(task) or not task.calibration_fingerprint:
        return None
    records = await get_quant_calibration_references(task.task_id, config.psql_db)
    record = records.get(reference_hotkey_for(task, hotkey))
    reference = None
    if record is not None and record.eval_set_fingerprint == task.calibration_fingerprint:
        reference = QuantCalibrationReference(
            model=record.model,
            loss=record.loss,
            per_example_losses=[None if math.isnan(loss) else loss for loss in record.per_example_losses],
            sku=record.sku,
        )
    elif record is not None:
        logger.error(f"Task {task.task_id} hotkey {hotkey}: calibration reference is on another example set; publishing none")
    else:
        logger.warning(f"Task {task.task_id} hotkey {hotkey}: no calibration reference; training without calibration losses")
    n_examples = len(reference.per_example_losses) if reference else task.calibration_n_examples
    url = await async_minio_client.get_new_presigned_url(task.calibration_data)
    if n_examples is None or not url:
        logger.warning(f"Task {task.task_id}: calibration set unavailable for {hotkey}; training without it")
        return None
    return QuantCalibrationRequest(
        examples_url=url,
        eval_set_fingerprint=task.calibration_fingerprint,
        n_examples=n_examples,
        reference=reference,
    )


async def _alert_missing_calibration(task: AnyQuantRawTask, hotkey: str, config: Config) -> None:
    """Fail loud, never fail the row: a quant miner dispatched without calibration rows compresses blind.

    Every tournament quant task is meant to carry a calibration set (carved by its upstream seed or from its own chunk),
    so its absence is a validator bug or an infra fault. Logged as an ERROR on every row, alerted once per task.
    """
    if has_calibration_set(task) and task.calibration_fingerprint:
        reason = "its calibration set could not be published (presigned URL or example count unavailable)"
    else:
        reason = "the task has no calibration set (its upstream or its own prep carved none)"
    logger.error(f"Task {task.task_id}: {hotkey} is dispatched WITHOUT calibration rows: {reason}")
    if str(task.task_id) in _QUANT_CALIBRATION_MISSING_ALERTED:
        return
    _QUANT_CALIBRATION_MISSING_ALERTED.add(str(task.task_id))
    await notify_evaluation_exception(
        config,
        task_id=str(task.task_id),
        task_type=task.task_type,
        context="Quant training dispatched without a calibration set",
        error=f"{reason}; the miner trains without calibration rows or their bf16 reference losses",
        hotkeys=[hotkey],
    )


async def _create_training_request(
    task: AnyTypeRawTask,
    hotkey: str,
    available_gpu_ids: list[int],
    training_repo: str,
    training_commit_hash: str,
    github_token: str | None,
    requested_datasets: list[str] | None,
    config: Config,
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

    # Per-miner starting model override (for round continuation)
    starting_model = await task_sql.get_starting_model_repo(str(task.task_id), hotkey, config.psql_db)
    training_model = starting_model or task.augmented_model_id or task.model_id

    # Per-miner baseline_stats (env continuous training), fall back to task-level
    miner_stats = await task_sql.get_miner_baseline_stats(str(task.task_id), hotkey, config.psql_db)
    baseline_stats = miner_stats or task.baseline_stats

    if task.task_type in QUANT_TASK_TYPES:
        decode = await _decode_request_for(task, hotkey, config) if task.task_type == TaskType.QUANTFREEFORMTASK else None
        training_data = _quant_training_data(
            task,
            expected_repo_name,
            baseline_stats,
            decode=decode,
            model=training_model,
            quant_target=await _quant_target_for(task, hotkey, config),
            calibration=await _quant_calibration_for(task, hotkey, config),
        )
        if training_data.calibration is None:
            await _alert_missing_calibration(task, hotkey, config)
    elif task.task_type == TaskType.IMAGETASK:
        training_data = TrainRequestImage(
            model=training_model,
            task_id=str(task.task_id),
            hours_to_complete=task.hours_to_complete,
            expected_repo_name=expected_repo_name,
            dataset_zip=task.training_data,
            model_type=task.model_type,
            trigger_word=task.trigger_word,
            baseline_stats=baseline_stats,
        )
    else:
        dataset_type = _get_dataset_type(task)

        use_kl, kl_coef = (task.use_kl, task.kl_coef) if isinstance(task, InstructTextRawTask) else (False, None)
        training_data = TrainRequestText(
            model=training_model,
            task_id=str(task.task_id),
            hours_to_complete=task.hours_to_complete,
            expected_repo_name=expected_repo_name,
            dataset=task.training_data,
            dataset_type=dataset_type,
            file_format=FileFormat.S3,  # always an S3 since we task prep
            baseline_stats=baseline_stats,
            use_kl=use_kl,
            kl_coef=kl_coef,
        )

    return TrainerProxyRequest(
        training_data=training_data,
        github_repo=training_repo,
        gpu_ids=available_gpu_ids,
        hotkey=hotkey,
        github_commit_hash=training_commit_hash,
        github_token=github_token,
        requested_datasets=requested_datasets,
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
    gpu_type = GPUType.A100 if requirement == GpuRequirement.A100 else GPUType.H100
    typed_gpus = [gpu for gpu in trainer_gpus if gpu.gpu_type == gpu_type]
    available_ids = [gpu.gpu_id for gpu in typed_gpus if gpu.available]
    return _select_grouped_gpu_ids(available_ids, _nvlink_groups(typed_gpus), requirement.gpu_count)


def _nvlink_groups(gpus: list[GPUInfo]) -> list[list[int]]:
    """
    Split a trainer's GPUs into NVLink-connected groups.

    Uses the topology the trainer reports. Trainers on older builds report no
    topology (nvlink_peers is None); for those, assume the adjacent-pair bridging
    (0-1, 2-3, ...) common on PCIe boxes, which is harmless where it is wrong.
    """
    gpu_ids = sorted(gpu.gpu_id for gpu in gpus)
    if any(gpu.nvlink_peers is None for gpu in gpus):
        return [[gpu_id for gpu_id in gpu_ids if gpu_id // 2 == pair] for pair in sorted({gpu_id // 2 for gpu_id in gpu_ids})]

    peers = {gpu.gpu_id: set(gpu.nvlink_peers or []) for gpu in gpus}
    groups: list[list[int]] = []
    seen: set[int] = set()
    for start in gpu_ids:
        if start in seen:
            continue
        seen.add(start)
        stack, group = [start], []
        while stack:
            gpu_id = stack.pop()
            group.append(gpu_id)
            for peer in peers[gpu_id]:
                if peer in peers and peer not in seen:
                    seen.add(peer)
                    stack.append(peer)
        groups.append(sorted(group))
    return groups


def _select_grouped_gpu_ids(available_ids: list[int], groups: list[list[int]], count: int) -> list[int]:
    """
    Pick `count` GPU IDs, keeping NVLink groups together.

    Fully connected (SXM / NVSwitch) and unlinked trainers make every choice
    equivalent, so they get first-fit. Partially linked trainers (PCIe with
    NVLink bridges, where a cross-group copy runs ~7x slower) give multi-GPU jobs
    whole free groups first, and give single-GPU jobs a GPU from an already
    broken group so free groups stay intact.

    Returns:
        list[int]: Sorted GPU IDs, empty list if fewer than `count` are free
    """
    if len(available_ids) < count:
        return []

    free = sorted(available_ids)
    if len(groups) <= 1 or all(len(group) == 1 for group in groups):
        return free[:count]

    free_set = set(free)
    whole_free_groups = [group for group in groups if all(gpu_id in free_set for gpu_id in group)]
    in_whole_group = {gpu_id for group in whole_free_groups for gpu_id in group}
    orphans = [gpu_id for gpu_id in free if gpu_id not in in_whole_group]

    chosen: list[int] = []
    for group in whole_free_groups:
        if len(chosen) + len(group) <= count:
            chosen.extend(group)

    # Fill any remainder from orphans before splitting a free group.
    for gpu_id in orphans + free:
        if len(chosen) == count:
            break
        if gpu_id not in chosen:
            chosen.append(gpu_id)

    return sorted(chosen)


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
                    await _finish_training_cost_run(training_task, False, config)
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
                    successful_log = next(task_log for _, task_log in responses if task_log.status == TaskStatus.SUCCESS)
                    await _finish_training_cost_run(
                        training_task,
                        True,
                        config,
                        ended_at=successful_log.finished_at,
                    )
                    await tournament_sql.update_tournament_task_training_status(
                        training_task.task.task_id, training_task.hotkey, TrainingStatus.SUCCESS, config.psql_db
                    )
                    if training_task.task.task_type == TaskType.QUANTFREEFORMTASK:
                        # Result ignored: an unsettled decode is caught up by the PREEVALUATION hold.
                        await _record_quant_decode(training_task, successful_log, config)
                elif all(s == TaskStatus.FAILURE for s in statuses):
                    any_completed = True
                    logger.info(f"Task {training_task.task.task_id} with hotkey {training_task.hotkey} failed on all trainers")
                    finished_times = [task_log.finished_at for _, task_log in responses if task_log.finished_at is not None]
                    await _finish_training_cost_run(
                        training_task,
                        False,
                        config,
                        ended_at=max(finished_times) if finished_times else None,
                    )
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
                grace_period = timedelta(seconds=cst.GPU_RESERVATION_GRACE_PERIOD_SECONDS)
                now = datetime.now(timezone.utc)

                for current_gpu in current_gpus:
                    if current_gpu.available:
                        # Check if this GPU is marked as used in our database
                        for db_gpu in trainer.gpus:
                            if db_gpu.gpu_id == current_gpu.gpu_id and not db_gpu.available:
                                # Skip reset if reservation was made recently — the task is likely
                                # still in its build phase (clone, docker build) and hasn't started
                                # a container yet, so the trainer incorrectly reports "available".
                                if db_gpu.updated_at and (now - db_gpu.updated_at) < grace_period:
                                    logger.debug(
                                        f"Skipping reset for GPU {db_gpu.gpu_id} on {trainer.trainer_ip}: "
                                        f"reservation is {(now - db_gpu.updated_at).seconds}s old (grace period: {grace_period})"
                                    )
                                    break
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


async def seed_tournament_evaluations_from_training(config: Config):
    """
    Seed evaluation rows for tournament task-hotkey pairs as soon as training succeeds.
    """
    while True:
        try:
            training_tasks = await task_sql.get_tasks_with_status(
                TaskStatus.TRAINING,
                config.psql_db,
                tournament_filter="only",
                benchmark_filter="exclude",
            )
            logger.info(f"Seeding evaluation rows for {len(training_tasks)} in-flight tournament tasks")

            for task in training_tasks:
                try:
                    await task_sql.add_task_evaluation_pairs(task.task_id, config.psql_db)
                except Exception as e:
                    logger.error(f"Error seeding evaluation rows for task {task.task_id}: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Error in seed_tournament_evaluations_from_training cycle: {str(e)}", exc_info=True)
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
    moved = 0
    for task in tasks_to_move:
        tournament_id = await get_tournament_id_by_task_id(task.task_id, config.psql_db)
        with LogContext(task_id=str(task.task_id), tournament_id=tournament_id):
            try:
                if not await seeds_sql.boss_round_seed_ready_to_finalise(task.task_id, config.psql_db):
                    logger.info(f"Holding seed {task.task_id} before PREEVALUATION: unattached or unfinished")
                    continue
                if task.task_type == TaskType.QUANTFREEFORMTASK and not await _freeform_decodes_settled(task, config):
                    continue
                task.status = TaskStatus.PREEVALUATION
                await task_sql.update_task(task, config.psql_db)
                moved += 1
                logger.info(f"Moved task {task.task_id} from training to preevaluation status")
            except Exception as e:
                logger.error(f"Error moving task {task.task_id} to preevaluation: {str(e)}")

    logger.info(f"Successfully moved {moved} tasks to preevaluation status")


async def _hold_quant_decode(task_id: str, hotkey: str, detail: str, config: Config) -> None:
    """Infra trouble on a freeform decode: never scored, so the task is held and a human pinged once."""
    logger.error(f"Holding freeform decode for task {task_id} hotkey {hotkey} (infra, not scored): {detail}")
    if (task_id, hotkey) in _QUANT_DECODE_ALERTED:
        return
    _QUANT_DECODE_ALERTED.add((task_id, hotkey))
    await notify_quant_decode_infra_error(task_id, hotkey, detail, config.discord_url)


async def _persist_quant_decode(
    task_id,
    hotkey: str,
    config: Config,
    *,
    artifact_bytes: int | None = None,
    decoder_bytes: int | None = None,
    decoder_sha256: str | None = None,
    quant_failure: str | None = None,
    build_warning: str | None = None,
    artifact_files: list[QuantDecodedFile] | None = None,
    decoded_files: list[QuantDecodedFile] | None = None,
) -> None:
    if build_warning is not None:
        logger.warning(f"Freeform decode for task {task_id} hotkey {hotkey}: decoder build warning (not scored): {build_warning}")
    await submissions_and_scoring_sql.set_quant_decode_result(
        task_id,
        hotkey,
        artifact_bytes=artifact_bytes,
        decoder_bytes=decoder_bytes,
        decoder_sha256=decoder_sha256,
        quant_failure=quant_failure,
        psql_db=config.psql_db,
        build_warning=build_warning,
        artifact_files=artifact_files,
        decoded_files=decoded_files,
    )


async def _record_quant_decode(training_task: TournamentTaskTraining, task_log: TrainerTaskLog, config: Config) -> bool:
    """Persist a successful freeform training's decode outcome; True once it is settled.

    Only a miner's fault (no decoder at registration, or the trainer's quant_failure) is scored. A
    trainer without a decode result never zeroes a miner: it holds the task and alerts. A sha/byte
    mismatch between the registration snapshot and what the trainer fetched is treated as our fault
    (or tampering) and is never scored either. A task without a tournament (organic) has no
    participant to fault, so nothing is persisted for it.
    """
    task = training_task.task
    hotkey = training_task.hotkey
    if task.task_type != TaskType.QUANTFREEFORMTASK:
        return True

    task_id = str(task.task_id)
    tournament_id = await get_tournament_id_by_task_id(task.task_id, config.psql_db)
    if tournament_id is None:
        logger.info(f"Freeform task {task_id} has no tournament (organic); not recording a decode for {hotkey}")
        return True
    participant = await tournament_sql.get_tournament_participant(tournament_id, hotkey, config.psql_db)
    has_snapshot = participant is not None and participant.has_decoder_snapshot
    decode = task_log.quant_decode

    if decode is None and not has_snapshot:
        await _persist_quant_decode(task.task_id, hotkey, config, quant_failure=NO_DECODER_SNAPSHOT_FAILURE)
        return True
    if decode is None:
        await _hold_quant_decode(task_id, hotkey, "the trainer returned no decode result", config)
        return False
    if decode.infra_error is not None:
        await _hold_quant_decode(task_id, hotkey, decode.infra_error, config)
        return False
    if decode.quant_failure is not None:
        await _persist_quant_decode(
            task.task_id,
            hotkey,
            config,
            decoder_bytes=decode.decoder_bytes,
            decoder_sha256=decode.decoder_sha256,
            quant_failure=decode.quant_failure,
            build_warning=decode.build_warning,
            artifact_files=decode.artifact_files,
        )
        return True
    if (
        not has_snapshot
        or decode.decoder_sha256 != participant.decoder_sha256
        or decode.decoder_bytes != participant.decoder_xz_bytes
    ):
        await _hold_quant_decode(
            task_id,
            hotkey,
            f"decoder snapshot mismatch: trainer ran {decode.decoder_sha256} ({decode.decoder_bytes} bytes)",
            config,
        )
        return False
    await _persist_quant_decode(
        task.task_id,
        hotkey,
        config,
        artifact_bytes=decode.artifact_bytes,
        decoder_bytes=decode.decoder_bytes,
        decoder_sha256=decode.decoder_sha256,
        build_warning=decode.build_warning,
        artifact_files=decode.artifact_files,
        decoded_files=decode.decoded_files,
    )
    return True


async def _poll_quant_decode(training_task: TournamentTaskTraining | None, task_id: str, hotkey: str, config: Config) -> bool:
    """Re-read a successful training's decode result from its trainer and record it; True once settled."""
    if training_task is None or not training_task.trainer_ip:
        await _hold_quant_decode(task_id, hotkey, "no trainer recorded to re-poll the decode result from", config)
        return False
    key = (task_id, hotkey)
    try:
        task_log = await get_training_task_details(training_task.trainer_ip, task_id, hotkey)
    except httpx.HTTPError as e:
        failures = _QUANT_DECODE_REPOLL_FAILURES.get(key, 0) + 1
        _QUANT_DECODE_REPOLL_FAILURES[key] = failures
        detail = f"could not re-poll the decode from {training_task.trainer_ip} ({failures} consecutive failures): {e}"
        if failures >= QUANT_DECODE_REPOLL_FAILURES_BEFORE_ALERT:
            await _hold_quant_decode(task_id, hotkey, detail, config)
        else:
            logger.warning(f"Task {task_id} hotkey {hotkey}: {detail}")
        return False
    _QUANT_DECODE_REPOLL_FAILURES.pop(key, None)
    return await _record_quant_decode(training_task, task_log, config)


async def _freeform_decodes_settled(task: AnyTypeRawTask, config: Config) -> bool:
    """True once every successful entrant of a freeform task has a settled decode.

    Unsettled entrants are re-polled from their trainer each cycle.
    """
    task_id = str(task.task_id)
    statuses = await tournament_sql.get_training_status_for_task(task_id, config.psql_db)
    required = {hotkey for hotkey, status in statuses.items() if status == TrainingStatus.SUCCESS.value}
    rows = await submissions_and_scoring_sql.get_quant_decode_rows(task.task_id, config.psql_db)
    unsettled = sorted(hotkey for hotkey in required if hotkey not in rows or not rows[hotkey].settled)
    if not unsettled:
        return True

    successful = await tournament_sql.get_tournament_training_tasks(config.psql_db, TrainingStatus.SUCCESS)
    trainings = {training.hotkey: training for training in successful if str(training.task.task_id) == task_id}
    held = [hotkey for hotkey in unsettled if not await _poll_quant_decode(trainings.get(hotkey), task_id, hotkey, config)]
    if held:
        logger.info(f"Holding freeform task {task_id} before PREEVALUATION: decode unsettled for {held}")
    return not held


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


_model_prep_in_progress: set[str] = set()


NON_BOSS_CAPPED_TASK_TYPES = frozenset({TaskType.INSTRUCTTEXTTASK, TaskType.CHATTASK, TaskType.DPOTASK})

_MODEL_PREP_STATUS_TIMEOUT = 5.0  # seconds — fast check, don't stall the cycle


async def _hours_after_prep(task, config: Config) -> float:
    """The task's budget once its prep stats are in.

    Quant budgets are resized from the task's own prep stats (flat when they are unusable); the boss
    multiplier follows the round.
    Instruct/chat/DPO tournament tasks outside the boss round are capped at NON_BOSS_INSTRUCT_MAX_TRAINING_HOURS;
    boss-round and organic tasks keep the MAX_TRAINING_HOURS cap of the throughput estimate.
    """
    if task.task_type in QUANT_TASK_TYPES:
        return quant_hours_after_prep(task, task.baseline_stats, boss=await _is_boss_round_task(task, config))
    hours = compute_hours_from_baseline_stats(
        task.hours_to_complete,
        task.baseline_stats,
        task.task_type,
        model_id=task.model_id,
        model_params_count=task.model_params_count,
        training_start_point=task.training_start_point,
        ds=task.ds,
    )
    if task.task_type in NON_BOSS_CAPPED_TASK_TYPES and await _is_non_boss_tournament_task(task, config):
        return min(hours, data_cst.NON_BOSS_INSTRUCT_MAX_TRAINING_HOURS)
    return hours


async def _seed_role(task, config: Config) -> SeedRole | None:
    seed = await seeds_sql.get_seed_task_by_task_id(task.task_id, config.psql_db)
    return seed.role if seed else None


async def _is_non_boss_tournament_task(task, config: Config) -> bool:
    """True for a task of a quant tournament round that is not the final (boss) round; organic tasks and the tasks of
    the text and image tournaments are neither.

    An input seed and a text or quant round seed (rounds 1-3) are non-boss tasks, even before a round attaches them; a
    boss-round seed is a boss task even before the boss round attaches it. The task a round of a seeded text tournament drew
    for itself (its seed role failed) is a non-boss task like the seed it stands in for.
    """
    role = await _seed_role(task, config)
    if role is not None:
        return role in INPUT_SEED_ROLES or role in TEXT_ROUND_SEED_ROLES or role in QUANT_ROUND_SEED_ROLES
    if not await _belongs_to_quant_tournament(task, config):
        tournament_id = await get_tournament_id_by_task_id(task.task_id, config.psql_db)
        if not tournament_id or not await seeds_sql.is_seeded_text_tournament(tournament_id, config.psql_db):
            return False
    return not await is_final_round_task(task.task_id, config.psql_db)


async def _is_boss_round_task(task, config: Config) -> bool:
    """True for a task of a quant tournament's final (boss) round or a boss-round seed; organic tasks are not."""
    role = await _seed_role(task, config)
    if role is not None:
        return role in ANY_BOSS_ROUND_SEED_ROLES
    if not await _belongs_to_quant_tournament(task, config):
        return False
    return await is_final_round_task(task.task_id, config.psql_db)


def _exceeds_near_duplicate_threshold(task, baseline_stats) -> bool:
    """True if the prepped dataset's near-duplicate rate is at/above the cutoff.

    Only text tasks (instruct/dpo/grpo) carry dataset stats; env tasks and any
    stats without a dataset are never rejected. Organic tasks are exempt: we log
    but let them proceed. Instruct tasks are gated too: dropping exact duplicate
    rows before the split does not remove near-duplicates.
    """
    dataset = getattr(baseline_stats, "dataset", None)
    if dataset is None:
        return False
    rate = dataset.near_duplicate_rate
    if rate < cst.MAX_NEAR_DUPLICATE_RATE:
        return False

    # Continuous-SFT reuses a fixed curated chunk, so its near-duplicate rate is expected — never reject.
    if cst.is_continuous_sft_task(task):
        logger.warning(
            f"Task {task.task_id} has near_duplicate_rate={rate:.3f} "
            f">= {cst.MAX_NEAR_DUPLICATE_RATE} but is the continuous-SFT curated chunk — allowing through"
        )
        return False

    if task.is_organic:
        logger.warning(
            f"Task {task.task_id} has near_duplicate_rate={rate:.3f} "
            f">= {cst.MAX_NEAR_DUPLICATE_RATE} but is organic — allowing through"
        )
        return False

    logger.warning(
        f"Task {task.task_id} rejected: near_duplicate_rate={rate:.3f} "
        f">= {cst.MAX_NEAR_DUPLICATE_RATE}, marking {TaskStatus.PREP_TASK_FAILURE.value} "
        f"for replacement"
    )
    return True


async def _recover_model_prep_from_trainer(task, config: Config) -> bool:
    """Check all trainers for an existing model prep job for this task.

    Returns True if the task was handled (completed result recovered, or
    still in progress on a trainer), meaning the caller should skip dispatch.
    Returns False if no job was found on any trainer.
    """
    task_id_str = str(task.task_id)
    trainers = await tournament_sql.get_trainers(config.psql_db)

    for trainer in trainers:
        trainer_ip = trainer.trainer_ip
        if ":" not in trainer_ip:
            trainer_ip_with_port = f"{trainer_ip}:8001"
        else:
            trainer_ip_with_port = trainer_ip

        try:
            url = f"http://{trainer_ip_with_port}{MODEL_PREP_STATUS_ENDPOINT.format(task_id=task_id_str)}"
            async with httpx.AsyncClient(timeout=_MODEL_PREP_STATUS_TIMEOUT) as client:
                response = await client.get(url)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                job = ModelPrepJob.model_validate(response.json())
        except Exception:
            continue

        if job.status == TaskStatus.SUCCESS and job.result is not None:
            await _finish_prep_cost_run(
                _prep_cost_run_key(task_id_str),
                True,
                config,
                ended_at=job.finished_at,
            )
            if job.result.augmented_model_id:
                task.augmented_model_id = job.result.augmented_model_id
            if job.result.baseline_stats:
                task.baseline_stats = job.result.baseline_stats

            if job.result.baseline_stats and _exceeds_near_duplicate_threshold(
                task, job.result.baseline_stats
            ):
                task.status = TaskStatus.PREP_TASK_FAILURE
                await task_sql.update_task(task, config.psql_db)
                return True

            if job.result.baseline_stats and await _apply_measured_training_hours(task, config):
                return True

            task.status = TaskStatus.LOOKING_FOR_NODES
            await task_sql.update_task(task, config.psql_db)
            logger.info(
                f"Recovered model prep result for task {task.task_id} from trainer "
                f"{trainer_ip}, moved to {TaskStatus.LOOKING_FOR_NODES.value}"
            )
            return True

        if job.status == TaskStatus.TRAINING:
            logger.info(
                f"Model prep for task {task.task_id} still in progress on trainer "
                f"{trainer_ip}, skipping dispatch"
            )
            return True

        if job.status == TaskStatus.FAILURE:
            await _finish_prep_cost_run(
                _prep_cost_run_key(task_id_str),
                False,
                config,
                ended_at=job.finished_at,
            )

    return False


async def _recover_miner_preps_from_trainers(task, miners_needing: list[tuple[str, str]], config: Config) -> set[str]:
    """Check trainers for completed per-miner model prep results after a restart.

    Polls each trainer for each miner's prep job (keyed by task_id + hotkey).
    Stores recovered results and returns the set of hotkeys that are already
    handled this cycle (baseline_stats recovered, or a prep still running on a
    trainer). Those hotkeys must be skipped for dispatch, but any remaining
    miners should still be scheduled.
    """
    task_id_str = str(task.task_id)
    trainers = await tournament_sql.get_trainers(config.psql_db)
    handled: set[str] = set()

    for trainer in trainers:
        trainer_ip = trainer.trainer_ip
        trainer_ip_with_port = f"{trainer_ip}:8001" if ":" not in trainer_ip else trainer_ip

        for hotkey, _starting_model in miners_needing:
            if hotkey in handled:
                continue
            try:
                url = f"http://{trainer_ip_with_port}{MODEL_PREP_STATUS_ENDPOINT.format(task_id=task_id_str)}"
                async with httpx.AsyncClient(timeout=_MODEL_PREP_STATUS_TIMEOUT) as client:
                    response = await client.get(url, params={"hotkey": hotkey})
                    if response.status_code == 404:
                        continue
                    response.raise_for_status()
                    job = ModelPrepJob.model_validate(response.json())
            except Exception:
                continue

            if job.status == TaskStatus.TRAINING:
                handled.add(hotkey)
                continue

            if job.status == TaskStatus.SUCCESS and job.result is not None and job.result.baseline_stats:
                await _finish_prep_cost_run(
                    _prep_cost_run_key(task_id_str, hotkey),
                    True,
                    config,
                    ended_at=job.finished_at,
                )
                await task_sql.set_miner_baseline_stats(
                    task_id_str, hotkey, job.result.baseline_stats, config.psql_db,
                )
                logger.info(
                    f"Recovered per-miner baseline_stats for task {task.task_id} "
                    f"hotkey={hotkey[:8]}... from trainer {trainer_ip}"
                )
                handled.add(hotkey)

            elif job.status == TaskStatus.FAILURE:
                await _finish_prep_cost_run(
                    _prep_cost_run_key(task_id_str, hotkey),
                    False,
                    config,
                    ended_at=job.finished_at,
                )

    return handled


async def _copy_quant_sibling_prep(task, config: Config) -> bool:
    """Quant half of the sibling reuse: a freeform and a fixed task on one model and dataset share one prep.

    Copies the newest sibling's baseline_stats and sizes this task's own hours from them; waits while a sibling's prep is
    running. Returns True if handled (copied or waiting), False to dispatch a prep.
    """
    task_id_str = str(task.task_id)
    sibling_stats = await tournament_sql.get_quant_sibling_baseline_stats(task_id_str, config.psql_db)
    if sibling_stats is not None:
        task.baseline_stats = _BASELINE_STATS_ADAPTER.validate_python(sibling_stats)
        if _exceeds_near_duplicate_threshold(task, task.baseline_stats):
            task.status = TaskStatus.PREP_TASK_FAILURE
            await task_sql.update_task(task, config.psql_db)
            return True
        new_hours = await _hours_after_prep(task, config)
        task.hours_to_complete = new_hours
        task.termination_at = datetime.utcnow() + timedelta(hours=new_hours)
        task.status = TaskStatus.LOOKING_FOR_NODES
        await task_sql.update_task(task, config.psql_db)
        logger.info(
            f"Copied baseline_stats from a sibling quant prep of the same model and dataset for task {task.task_id}, "
            f"skipping model prep; hours {new_hours}, moved to {TaskStatus.LOOKING_FOR_NODES.value}"
        )
        return True

    sibling_ids = await tournament_sql.get_quant_prep_sibling_task_ids(task_id_str, config.psql_db)
    return any(sid in _model_prep_in_progress for sid in sibling_ids)


async def _try_reuse_sibling_quant_prep(task, config: Config) -> bool:
    """A quant task with no augmentation reuses another quant task's prep: same tournament, model and training data.

    Returns True if the task was handled (copied or should wait), False to dispatch its own prep.
    """
    if task.augmentation_config is not None:
        return False
    try:
        return await _copy_quant_sibling_prep(task, config)
    except Exception as e:
        logger.warning(f"Quant sibling prep lookup failed for task {task.task_id}, dispatching its own prep: {e}")
        return False


async def _try_reuse_sibling_model_prep(task, config: Config) -> bool:
    """For env tasks with no augmentation and no per-miner starting models,
    try to copy baseline_stats from a sibling in the same round, or skip if
    a sibling is already running prep.

    Skipped entirely when miners have per-miner starting models (continuous
    training), since each miner needs their own baseline.

    Quant tasks take the quant path instead (_copy_quant_sibling_prep).

    Returns True if the task was handled (copied or should wait), False to proceed normally.
    """
    if task.task_type in QUANT_TASK_TYPES:
        return await _try_reuse_sibling_quant_prep(task, config)
    if task.task_type != TaskType.ENVIRONMENTTASK or task.augmentation_config is not None:
        return False

    task_id_str = str(task.task_id)

    # Per-miner starting models → each miner needs their own baseline, no reuse
    miners_with_starting = await task_sql.get_miners_needing_baseline_stats(
        task_id_str, config.psql_db,
    )
    if miners_with_starting:
        return False

    sibling_stats = await tournament_sql.get_sibling_env_baseline_stats(
        task_id_str, task.model_id, config.psql_db,
    )
    if sibling_stats is not None:
        task.baseline_stats = sibling_stats
        task.status = TaskStatus.LOOKING_FOR_NODES
        await task_sql.update_task(task, config.psql_db)
        logger.info(f"Copied baseline_stats from sibling for env task {task.task_id}, skipping model prep")
        return True

    sibling_ids = await tournament_sql.get_matching_sibling_task_ids(task_id_str, task.model_id, config.psql_db)
    if any(sid in _model_prep_in_progress for sid in sibling_ids):
        return True

    return False


async def process_awaiting_model_prep_tasks(config: Config):
    """Poll for tasks awaiting model prep and dispatch to a trainer with GPU.

    Dispatches concurrently so a slow large-model prep doesn't block smaller
    models that could run on other GPUs.  Tasks FIFO (oldest first) for the
    GPU availability check, but once dispatched they run in parallel.
    """
    # Track per-miner preps independently: "task_id:hotkey"
    _miner_prep_in_progress: set[str] = set()

    async def _run_miner_prep(task, hotkey, starting_model, trainer_ip, gpu_ids, interconnect=None):
        """Run model prep for a single miner's starting model. Releases GPUs when done."""
        task_id_str = str(task.task_id)
        prep_key = f"{task_id_str}:{hotkey}"
        cost_run_key = _prep_cost_run_key(task_id_str, hotkey)
        prep_success = False
        try:
            logger.info(
                f"Running per-miner model prep for task {task.task_id} "
                f"hotkey={hotkey[:8]}... model={starting_model}"
            )
            await _start_prep_cost_run(
                cost_run_key,
                task_id_str,
                f"miner:{hotkey}",
                trainer_ip,
                gpu_ids,
                config,
                interconnect=interconnect,
            )
            reward_fns = getattr(task, "reward_functions", None)
            prep_result = await dispatch_augmentation_and_stats(
                task_id=task_id_str,
                model_id=starting_model,
                training_data_url=task.training_data,
                augmentation_config=task.augmentation_config,
                task_type=task.task_type,
                trainer_ip=trainer_ip,
                gpu_ids=gpu_ids,
                reward_functions=reward_fns,
                is_env_task=True,
                hotkey=hotkey,
                environment_names=task.environment_names if isinstance(task, EnvRawTask) else None,
            )
            if prep_result is not None and prep_result.baseline_stats:
                prep_success = True
                await task_sql.set_miner_baseline_stats(
                    task_id_str, hotkey, prep_result.baseline_stats, config.psql_db,
                )
                logger.info(f"Stored baseline_stats for task {task.task_id} hotkey={hotkey[:8]}...")
            else:
                logger.warning(
                    f"Model prep returned no baseline_stats for task {task.task_id} "
                    f"hotkey={hotkey[:8]}..., will retry next cycle"
                )
        except Exception as e:
            logger.error(
                f"Per-miner model prep failed for task {task.task_id} hotkey={hotkey[:8]}...: {e}",
                exc_info=True,
            )
        finally:
            await _finish_prep_cost_run(cost_run_key, prep_success, config)
            _miner_prep_in_progress.discard(prep_key)
            try:
                await tournament_sql.update_gpu_availability(
                    trainer_ip, gpu_ids, 0, config.psql_db
                )
            except Exception:
                pass

    async def _run_single_prep(task_id_str, model_id, task, trainer_ip, gpu_ids):
        """Run one model prep for a given model_id and return the result."""
        reward_fns = getattr(task, "reward_functions", None)
        is_env_task = task.task_type == TaskType.ENVIRONMENTTASK
        return await dispatch_augmentation_and_stats(
            task_id=task_id_str,
            model_id=model_id,
            training_data_url=task.training_data,
            augmentation_config=task.augmentation_config,
            task_type=task.task_type,
            trainer_ip=trainer_ip,
            gpu_ids=gpu_ids,
            reward_functions=reward_fns,
            is_env_task=is_env_task,
            environment_names=task.environment_names if isinstance(task, EnvRawTask) else None,
        )

    async def _run_task_prep(task, trainer_ip, gpu_ids, interconnect=None):
        """Standard task-level model prep (text tasks, env round-1)."""
        task_id_str = str(task.task_id)
        cost_run_key = _prep_cost_run_key(task_id_str)
        prep_success = False
        try:
            await _start_prep_cost_run(
                cost_run_key,
                task_id_str,
                "task",
                trainer_ip,
                gpu_ids,
                config,
                interconnect=interconnect,
            )
            prep_result = await _run_single_prep(
                task_id_str, task.model_id, task, trainer_ip, gpu_ids,
            )
            if prep_result is not None:
                prep_success = True
                if prep_result.augmented_model_id:
                    task.augmented_model_id = prep_result.augmented_model_id
                if prep_result.baseline_stats:
                    task.baseline_stats = prep_result.baseline_stats

                if prep_result.baseline_stats and _exceeds_near_duplicate_threshold(
                    task, prep_result.baseline_stats
                ):
                    task.status = TaskStatus.PREP_TASK_FAILURE
                    await task_sql.update_task(task, config.psql_db)
                    return

                if prep_result.baseline_stats and await _apply_measured_training_hours(task, config):
                    return

                task.status = TaskStatus.LOOKING_FOR_NODES
                await task_sql.update_task(task, config.psql_db)
                logger.info(
                    f"Model prep complete for task {task.task_id}, "
                    f"moved to {TaskStatus.LOOKING_FOR_NODES.value}"
                )
            else:
                logger.warning(
                    f"Model prep dispatch returned None for task {task.task_id}, "
                    f"will retry next cycle"
                )
        except Exception as e:
            logger.error(
                f"Model prep failed for task {task.task_id}: {e}",
                exc_info=True,
            )
        finally:
            await _finish_prep_cost_run(cost_run_key, prep_success, config)
            _model_prep_in_progress.discard(task_id_str)
            try:
                await tournament_sql.update_gpu_availability(
                    trainer_ip, gpu_ids, 0, config.psql_db
                )
            except Exception:
                pass

    while True:
        try:
            tasks = await task_sql.get_tasks_with_status(
                TaskStatus.AWAITING_MODEL_PREP, config.psql_db
            )
            if not tasks:
                logger.debug("No tasks awaiting model prep")
                await asyncio.sleep(cst.MODEL_PREP_CYCLE_INTERVAL)
                continue

            tasks.sort(key=lambda t: t.created_at)
            logger.info(f"Found {len(tasks)} tasks awaiting model prep")

            for task in tasks:
                task_id_str = str(task.task_id)

                with LogContext(task_id=task_id_str):
                    is_env_task = task.task_type == TaskType.ENVIRONMENTTASK

                    # --- Per-miner prep path (continuous training) ---
                    if is_env_task:
                        miners_needing = await task_sql.get_miners_needing_baseline_stats(
                            task_id_str, config.psql_db,
                        )
                        if miners_needing:
                            # Recover completed/in-flight per-miner preps from trainers (e.g. after
                            # restart). Returns the hotkeys already handled this cycle; the remaining
                            # miners must still be dispatched so one in-progress miner doesn't block
                            # the others.
                            handled_hotkeys = await _recover_miner_preps_from_trainers(
                                task, miners_needing, config
                            )

                            gpu_req = get_task_gpu_requirement(task)
                            for hotkey, starting_model in miners_needing:
                                prep_key = f"{task_id_str}:{hotkey}"
                                if hotkey in handled_hotkeys or prep_key in _miner_prep_in_progress:
                                    continue
                                suitable = await _check_suitable_gpus(config, gpu_req)
                                if suitable is None:
                                    continue
                                trainer_ip, gpu_ids, interconnect = suitable
                                await tournament_sql.update_gpu_availability(
                                    trainer_ip, gpu_ids, cst.MODEL_PREP_GPU_RESERVE_HOURS, config.psql_db
                                )
                                _miner_prep_in_progress.add(prep_key)
                                asyncio.create_task(
                                    _run_miner_prep(
                                        task, hotkey, starting_model, trainer_ip, gpu_ids, interconnect
                                    )
                                )
                            continue

                        # No miners need stats — check if per-miner task is fully done
                        if await task_sql.has_miners_with_starting_model(task_id_str, config.psql_db):
                            task.status = TaskStatus.LOOKING_FOR_NODES
                            await task_sql.update_task(task, config.psql_db)
                            logger.info(
                                f"All per-miner baseline_stats complete for task {task.task_id}, "
                                f"moved to {TaskStatus.LOOKING_FOR_NODES.value}"
                            )
                            continue

                    # --- Standard task-level prep path ---
                    if task_id_str in _model_prep_in_progress:
                        continue

                    recovered = await _recover_model_prep_from_trainer(task, config)
                    if recovered:
                        continue

                    if await _try_reuse_sibling_model_prep(task, config):
                        continue

                    gpu_req = get_task_gpu_requirement(task)
                    suitable = await _check_suitable_gpus(config, gpu_req)
                    if suitable is None:
                        logger.info(
                            f"No GPUs available for model prep of task {task.task_id} "
                            f"(model={task.model_id}, req={gpu_req.value}), will retry next cycle"
                        )
                        continue

                    trainer_ip, gpu_ids, interconnect = suitable
                    await tournament_sql.update_gpu_availability(
                        trainer_ip, gpu_ids, cst.MODEL_PREP_GPU_RESERVE_HOURS, config.psql_db
                    )

                    _model_prep_in_progress.add(task_id_str)
                    try:
                        asyncio.create_task(_run_task_prep(task, trainer_ip, gpu_ids, interconnect))
                    except Exception:
                        _model_prep_in_progress.discard(task_id_str)
                        await tournament_sql.update_gpu_availability(
                            trainer_ip, gpu_ids, 0, config.psql_db
                        )
                        raise
        except Exception as e:
            logger.error(f"Error in process_awaiting_model_prep_tasks cycle: {e}", exc_info=True)

        await asyncio.sleep(cst.MODEL_PREP_CYCLE_INTERVAL)


async def _is_full_model_repo(repo_id: str) -> bool:
    """Whether a hub repo exists and holds a loadable full model (quant_input_repo.full_model_rejection).

    Any hub error but a missing repo raises, so the task is held and retried rather than handed to the
    other player on a transient failure.
    """
    try:
        rejection = await full_model_rejection(repo_id)
    except RepositoryNotFoundError:
        logger.warning(f"{repo_id} is not a full model: repo not found")
        return False
    if rejection is not None:
        logger.warning(f"{repo_id} is not a full model: {rejection}")
        return False
    return True


async def _shared_chained_input(upstream: InstructTextRawTask, hotkeys: list[str], config: Config) -> str | None:
    """The model both players compress: the best usable upstream output that is a full model, or None.

    The repo is the one our trainer uploaded for the upstream task, never a miner-supplied id. Players are
    tried best upstream loss first, so a non-full-model repo falls back to the other player's.
    """
    for hotkey in await chaining.shared_input_candidates(upstream.task_id, hotkeys, config.psql_db):
        repo = await chaining.chained_input_repo(upstream.task_id, hotkey, config.psql_db)
        if repo is not None and await _is_full_model_repo(repo):
            logger.info(f"Chained input from upstream {upstream.task_id}: {hotkey}'s output {repo}")
            return repo
        logger.warning(f"Upstream {upstream.task_id}: {hotkey}'s output {repo} is not a full model, trying the next player")
    return None


async def _fail_chained_task(task: AnyQuantRawTask, hotkeys: list[str], config: Config) -> None:
    """No usable upstream output: every player loses the task as a scored fail, without a GPU."""
    logger.error(f"Chained task {task.task_id}: no player of {hotkeys} has a usable upstream output; failing it")
    for hotkey in hotkeys:
        await _persist_quant_decode(task.task_id, hotkey, config, quant_failure=cst.UPSTREAM_FAILURE_REASON)
    task.status = TaskStatus.FAILURE
    await task_sql.update_task(task, config.psql_db)


def _copy_upstream_data(task: AnyQuantRawTask, upstream: InstructTextRawTask, boss: bool = True) -> None:
    """A pinned chained task trains and is scored on the upstream's own prepped split.

    test_data is the raw test split the instruct task was scored on (quant prep tokenises it), and the
    dataset type carries the standard column names the upstream's prep wrote. calibration_data is the upstream's raw
    calibration JSON (None when it carved none), which quant prep tokenises and publishes. The budget is recomputed
    here from the upstream's model and its prep stats (flat when it has none): the boss budget for the type with boss,
    else the non-boss one (the miner rounds R1-R3 on an input seed).
    """
    task.training_data = upstream.training_data
    task.test_data = upstream.test_data
    task.calibration_data = upstream.calibration_data
    task.file_format = FileFormat.S3
    task.dataset_type = InstructTextDatasetType(
        field_instruction=upstream.field_instruction,
        field_input=upstream.field_input,
        field_output=upstream.field_output,
        field_system=upstream.field_system,
        format=upstream.format,
        no_input_format=upstream.no_input_format,
    )
    task.hours_to_complete = quant_training_hours(
        task.task_type, upstream.model_params_count, boss=boss, baseline_stats=upstream.baseline_stats
    )
    task.termination_at = datetime.utcnow() + timedelta(hours=task.hours_to_complete)


class _TextBossModel(BaseModel):
    """The model a quant task chained on a text seed task is released on: the repo, and whether it is the fallback (the text
    task's own base model, used because the text boss produced no usable model) and why."""

    repo: str
    is_fallback: bool
    reason: str = ""


# A text seed row in one of these states will not be replaced by the seed loop: its boss run is the boss's final entry.
_FINAL_TEXT_SEED_STATUSES = frozenset({SeedStatus.TRAINED, SeedStatus.READY, SeedStatus.FAILED})


async def _text_fallback_model(
    task: AnyQuantRawTask, upstream, seed: TournamentSeedTask, reason: str, config: Config
) -> _TextBossModel | None:
    """The fallback for a text boss that never produced a usable model: the text task's own base model when it is a full model;
    for the continuous task, whose base is the carried previous winner and may be an adapter, the lineage's seed model after it.
    None (held, alerted once) when neither is a full model: nothing is failed and nobody is halted."""
    candidates = [upstream.model_id]
    if seed.role == SeedRole.TEXT_CONTINUOUS:
        candidates.append(cst.continuous_sft_seed_repo_for_ds(upstream.ds))
    for repo in dict.fromkeys(candidate for candidate in candidates if candidate):
        if await _is_full_model_repo(repo):
            return _TextBossModel(repo=repo, is_fallback=True, reason=reason)
    await alert_seed_issue(
        f"{task.task_id}:quant-source-no-fallback",
        f"Quant task {task.task_id}: the text boss produced no usable model on text seed {seed.role.value} task "
        f"{upstream.task_id} ({reason}) and no fallback model is a full model ({candidates}); the task stays held",
        config,
    )
    return None


def _seed_row_settled(seed: TournamentSeedTask, seconds: int) -> bool:
    """Whether the seed loop has seen the boss's finished run and the row has not changed for `seconds`.

    The row's updated_at is written only when the seed loop moves it, so it dates the run's end only once the row is trained
    or ready; a row still `training` has not been looked at since the run finished and never counts, whatever its age. A
    trained or ready row with no timestamp counts as old enough.
    """
    if seed.status not in (SeedStatus.TRAINED, SeedStatus.READY):
        return False
    if seed.updated_at is None:
        return True
    changed = seed.updated_at if seed.updated_at.tzinfo is not None else seed.updated_at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - changed).total_seconds() >= seconds


async def _text_tournament_is_over(seed: TournamentSeedTask, config: Config) -> bool:
    """Whether the text tournament of a seed is neither active nor pending: nothing will merge, replace or train on it."""
    text_tournament = await tournament_sql.get_tournament(seed.tournament_id, config.psql_db)
    return text_tournament is not None and text_tournament.status not in (TournamentStatus.ACTIVE, TournamentStatus.PENDING)


async def _text_boss_model(task: AnyQuantRawTask, upstream, seed: TournamentSeedTask, config: Config) -> _TextBossModel | None:
    """The model a quant task chained on a text seed task compresses, or None to keep holding. Decided from persisted state
    only (the seed row, the seed's tournament, the boss's training row, the hub), so a restart at any point reaches the same
    answer.

    Landed: the boss's training run on the text task succeeded and its output is a full model (the merged repo when an
    adapter was merged). A seed row that is `failed` (out of replacements) never waits for anything: that landed model is
    used, and otherwise the fallback is taken at once, whatever the boss's training row says. While the text tournament is
    ACTIVE (or PENDING) and the row is not failed:
      * an adapter whose merge is not stored yet holds;
      * a run that failed takes the fallback once the seed row is final (it will not be replaced), else the task holds: the
        re-point moves it to the replacement;
      * a run that succeeded with an output that is missing or not a full model takes the fallback only when the seed loop has
        already seen the finished run (the row is trained or ready) and the row has not changed for
        QUANT_SOURCE_FALLBACK_GRACE_SECONDS (a hub lookup right after the run must not decide it); a row still `training`
        holds;
      * a run that has not finished holds.
    Once the text tournament is over nothing will merge, replace or train anything on the seed, so its state is final whatever
    its row says: a succeeded run with a full-model output is used, every other case takes the fallback at once.
    """
    boss_status = await _training_status_of(upstream.task_id, EMISSION_BURN_HOTKEY, config)
    text_tournament_over = await _text_tournament_is_over(seed, config)
    seed_failed = seed.status == SeedStatus.FAILED
    settled = text_tournament_over or seed_failed
    if boss_status == TrainingStatus.SUCCESS.value:
        if not settled and await boss_output_awaits_merge(seed, config):
            logger.debug(f"Quant task {task.task_id}: the boss's output on text seed {upstream.task_id} awaits its merge")
            return None
        repo = await chaining.chained_input_repo(upstream.task_id, EMISSION_BURN_HOTKEY, config.psql_db)
        if repo is not None and await _is_full_model_repo(repo):
            return _TextBossModel(repo=repo, is_fallback=False)
        if not settled and not _seed_row_settled(seed, cst.QUANT_SOURCE_FALLBACK_GRACE_SECONDS):
            logger.debug(f"Quant task {task.task_id}: the boss's output {repo} is not usable yet; holding for the grace period")
            return None
        return await _text_fallback_model(task, upstream, seed, f"its output {repo} is not a full model", config)
    if text_tournament_over:
        text_tournament = await tournament_sql.get_tournament(seed.tournament_id, config.psql_db)
        over_status = TournamentStatus(text_tournament.status).value
        return await _text_fallback_model(
            task, upstream, seed, f"its text tournament is {over_status} and its run is {boss_status}", config
        )
    if seed_failed:
        return await _text_fallback_model(
            task, upstream, seed, f"its text seed is out of replacements and its run is {boss_status}", config
        )
    if boss_status == TrainingStatus.FAILURE.value and seed.status in _FINAL_TEXT_SEED_STATUSES:
        return await _text_fallback_model(task, upstream, seed, "its training run failed", config)
    logger.debug(f"Quant task {task.task_id}: the text boss has not produced its model on {upstream.task_id} yet")
    return None


async def _release_on_text_boss_model(task: AnyQuantRawTask, upstream, seed: TournamentSeedTask, config: Config) -> None:
    """Release a held quant task chained on a text seed task once the text boss's model has landed (see _text_boss_model).

    The quant task is its own task: only its own row, its source row and its own nodes are read or written here, never the
    text task, its seed row or its status. model_id becomes the model repo; the upstream's prepped data is copied for a
    pinned-data task exactly as for any chained task, a task on its own continuous chunk keeps its data and gets freshly
    signed urls for it. The source row is written before the task moves, so a crash in between is repaired by the next pass.
    Held while no node is assigned to the quant task.

    A task still on a replaced seed follows the role's successor (its live row, or the failed row of a role out of
    replacements) when there is one and the re-point succeeds; the next pass releases it. With no successor to follow, or a
    refused re-point, it holds while the text tournament runs and, once that is over, is released like any other from the seed
    it is on.

    A task on a `failed` seed (or a replaced one it could not leave) whose upstream was never prepped has no data to copy, and
    nothing will ever prep the upstream:
    it is released on a fresh instruct dataset of its own, which its own prep splits (carving the calibration set) like any
    quant task with its own data; a recreated copy of it is released on another fresh dataset the same way.
    """
    if seed.status == SeedStatus.REPLACED:
        successor = await quant_sources.final_text_seed(seed.tournament_id, seed.role, config)
        new_upstream = (
            await task_sql.get_task(successor.task_id, config.psql_db)
            if successor is not None and successor.task_id != upstream.task_id
            else None
        )
        if new_upstream is not None and await repoint_task_onto_seed(task, upstream, new_upstream, config):
            return
        if not await _text_tournament_is_over(seed, config):
            return
    if not await task_sql.get_nodes_assigned_to_task(task.task_id, config.psql_db):
        logger.debug(f"Chained task {task.task_id}: no nodes assigned yet, holding")
        return
    model = await _text_boss_model(task, upstream, seed, config)
    if model is None:
        return
    unprepped = upstream.training_data is None or upstream.test_data is None
    # A failed seed, or a replaced one that no successor will replace (the text tournament is over), is never prepped.
    draws_fresh_data = (
        seed.status in (SeedStatus.FAILED, SeedStatus.REPLACED)
        and unprepped
        and task.training_data is None
        and not cst.is_continuous_quant_task(task)
    )
    copies_data = task.training_data is None and task.ds == upstream.ds and not draws_fresh_data
    if copies_data and unprepped:
        logger.debug(f"Chained task {task.task_id}: text seed {upstream.task_id} is not prepped yet, holding")
        return
    if draws_fresh_data:
        try:
            ds, dataset_type, _ = await _draw_quant_instruct_dataset(config, _get_instruct_text_datasets(config.keypair))
        except Exception as e:
            logger.warning(redact_url_queries(f"Chained task {task.task_id}: could not draw a fresh dataset ({e}); holding"))
            return
    if cst.is_continuous_quant_task(task):
        try:
            chunk = await _fetch_continuous_quant_chunk(config, cst.continuous_quant_train_index(task.ds))
        except Exception as e:
            logger.warning(redact_url_queries(f"Chained task {task.task_id}: could not refresh the chunk ({e}); holding"))
            return
        task.training_data, task.test_data = chunk.train_url, chunk.test_url
    task.model_id = model.repo
    if upstream.model_params_count:
        # The task was created when the text task might not have been prepped yet (no size then): the GPU tier and the hours
        # follow the size the text task's prep recorded.
        task.model_params_count = upstream.model_params_count
    if copies_data:
        _copy_upstream_data(task, upstream, boss=not await _is_non_boss_tournament_task(task, config))
    elif draws_fresh_data:
        task.ds, task.dataset_type, task.file_format = ds, dataset_type, FileFormat.HF
        task.carve_calibration = True
        task.training_data = task.test_data = task.calibration_data = None
        task.termination_at = datetime.utcnow() + timedelta(hours=task.hours_to_complete)
    await quant_sources_sql.set_quant_task_source_model(task.task_id, model.repo, model.is_fallback, config.psql_db)
    task.status = TaskStatus.PENDING
    await task_sql.update_task(task, config.psql_db)
    logger.info(
        f"Released chained task {task.task_id} on {model.repo} from text seed {upstream.task_id} to PENDING"
        f"{f' (fallback: {model.reason})' if model.is_fallback else ''}{' on a fresh dataset' if draws_fresh_data else ''}"
    )
    if model.is_fallback:
        message = (
            f"Quant task {task.task_id} was released on the fallback model {model.repo}: no usable text boss model on text "
            f"seed {seed.role.value} task {upstream.task_id} ({model.reason})"
            f"{'; it trains on a fresh dataset because the text task was never prepped' if draws_fresh_data else ''}"
        )
        logger.warning(message)
        await alert_seed_issue(f"{task.task_id}:quant-source-fallback", message, config)


async def _release_chained_task(task: AnyQuantRawTask, config: Config) -> None:
    """Release a held chained task to PENDING, on its shared input, once its upstream instruct task is SUCCESS.

    Held while the upstream is unscored or nodes are unassigned. From here it is an ordinary quant task
    whose model_id is the shared input. A task with its own ds keeps its data and takes the
    normal split path in prep.

    On an input seed (the two round inputs R1-R3 compress, or the boss round's instruct input) the players have no run on
    the upstream: the only candidate is the boss's (EMISSION_BURN_HOTKEY) output, checked as a full model.
    A seed is SUCCESS only with a usable boss output, so when that output is missing the task is held for the seed's
    replacement rather than failed.

    A task chained on a TEXT seed task is released by _release_on_text_boss_model instead: a text seed is not SUCCESS until
    its text round is over, and the quant task starts as soon as the text boss's model has landed.
    """
    upstream = await task_sql.get_task(task.upstream_task_id, config.psql_db)
    if upstream is None:
        logger.error(f"Chained task {task.task_id}: upstream task {task.upstream_task_id} not found, holding")
        return
    upstream_seed = await seeds_sql.get_seed_task_by_task_id(upstream.task_id, config.psql_db)
    if upstream_seed is not None and upstream_seed.role in TEXT_SEED_ROLES:
        await _release_on_text_boss_model(task, upstream, upstream_seed, config)
        return
    if upstream.status != TaskStatus.SUCCESS:
        return

    hotkeys = [node.hotkey for node in await task_sql.get_nodes_assigned_to_task(task.task_id, config.psql_db)]
    if not hotkeys:
        logger.info(f"Chained task {task.task_id}: no nodes assigned yet, holding")
        return

    upstream_is_input_seed = await _is_input_seed(upstream, config)
    candidates = [EMISSION_BURN_HOTKEY] if upstream_is_input_seed else hotkeys
    shared_input = await _shared_chained_input(upstream, candidates, config)
    if shared_input is None:
        if upstream_is_input_seed:
            logger.error(f"Chained task {task.task_id}: input seed {upstream.task_id} has no usable boss output, holding")
            return
        await _fail_chained_task(task, hotkeys, config)
        return

    task.model_id = shared_input
    if task.training_data is None and task.ds == upstream.ds:
        _copy_upstream_data(task, upstream, boss=not await _is_non_boss_tournament_task(task, config))
    task.status = TaskStatus.PENDING
    await task_sql.update_task(task, config.psql_db)
    logger.info(f"Released chained task {task.task_id} on {shared_input} from upstream {upstream.task_id} to PENDING")


async def _is_input_seed(task, config: Config) -> bool:
    """Whether a task is an input seed (the upstreams of the miner rounds) or the boss round's instruct input."""
    role = await _seed_role(task, config)
    return role in _INPUT_SEED_ROLES


async def _training_status_of(task_id, hotkey: str, config: Config) -> str | None:
    return (await tournament_sql.get_training_status_for_task_and_hotkeys(str(task_id), [hotkey], config.psql_db)).get(hotkey)


async def _own_upstream_output_rejection(upstream_id, hotkey: str, config: Config) -> str | None:
    """Why a player's output on the upstream task cannot be compressed (None when it is a full model)."""
    repo = await chaining.chained_input_repo(upstream_id, hotkey, config.psql_db)
    if repo is None:
        return "no validator-assigned repo"
    if not await _is_full_model_repo(repo):
        return f"{repo} is not a full model"
    return None


async def _release_own_upstream_task(task: AnyQuantRawTask, config: Config) -> None:
    """Release the boss's side of an own-upstream boss-round seed as soon as the boss's upstream run is SUCCESS.

    Does not wait for the upstream task itself: it is held until the challenger has trained. The boss compresses
    its own upstream output (starting_model_repo); model_id stays the upstream's base (both players fine-tune the same
    base, so prep's tokenisation is right). The compression tasks take the upstream's prepped split and calibration set with
    boss hours. A compression task on a continuous chunk (cst.is_continuous_quant_task) carries its own chunk from creation:
    nothing is copied from the upstream, and its hours are its own. Its presigned urls were signed at creation, hours or days
    before prep, so the release re-fetches the same chunk (same index and ds) and overwrites both urls; when the content service
    fails the task stays held, never released with stale urls.
    Held while the upstream is unprepped or the boss's output is not a full model (seed replacement handles that).
    """
    upstream = await task_sql.get_task(task.upstream_task_id, config.psql_db)
    if upstream is None:
        logger.error(f"Own-upstream task {task.task_id}: upstream task {task.upstream_task_id} not found, holding")
        return
    if await _training_status_of(upstream.task_id, EMISSION_BURN_HOTKEY, config) != TrainingStatus.SUCCESS.value:
        return
    if upstream.training_data is None or upstream.test_data is None:
        logger.info(f"Own-upstream task {task.task_id}: upstream {upstream.task_id} is not prepped yet, holding")
        return
    rejection = await _own_upstream_output_rejection(upstream.task_id, EMISSION_BURN_HOTKEY, config)
    if rejection:
        logger.warning(f"Own-upstream task {task.task_id}: the boss's output on {upstream.task_id}: {rejection}; holding")
        return
    repo = await chaining.chained_input_repo(upstream.task_id, EMISSION_BURN_HOTKEY, config.psql_db)
    if cst.is_continuous_quant_task(task):
        try:
            chunk = await _fetch_continuous_quant_chunk(config, cst.continuous_quant_train_index(task.ds))
        except Exception as e:
            logger.warning(redact_url_queries(f"Own-upstream task {task.task_id}: could not refresh the chunk ({e}); holding"))
            return
        task.training_data, task.test_data = chunk.train_url, chunk.test_url
    elif task.training_data is None:
        _copy_upstream_data(task, upstream, boss=True)
    await task_sql.set_starting_model_repo(str(task.task_id), EMISSION_BURN_HOTKEY, repo, config.psql_db)
    task.status = TaskStatus.PENDING
    await task_sql.update_task(task, config.psql_db)
    logger.info(f"Released own-upstream task {task.task_id}: the boss starts from {repo} (upstream {upstream.task_id})")


async def process_awaiting_upstream_tasks(config: Config):
    """Release held chained boss tasks: shared-input ones when their upstream is scored, own-upstream ones per player."""
    while True:
        try:
            tasks = await task_sql.get_tasks_with_status(TaskStatus.AWAITING_UPSTREAM, config.psql_db, tournament_filter="only")
            for task in tasks:
                tournament_id = await get_tournament_id_by_task_id(task.task_id, config.psql_db)
                with LogContext(task_id=str(task.task_id), tournament_id=tournament_id):
                    try:
                        if task.training_start_point == TrainingStartPoint.OWN_UPSTREAM_MODEL:
                            # Own-model slots never share an input: the boss side releases here, the challenger's row is
                            # added by process_boss_round_seed_rows once its own upstream run is done.
                            await _release_own_upstream_task(task, config)
                        else:
                            await _release_chained_task(task, config)
                    except Exception as e:
                        logger.error(f"Error releasing chained task {task.task_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error in process_awaiting_upstream_tasks cycle: {e}", exc_info=True)
        await asyncio.sleep(cst.PROCESS_PENDING_TASKS_CYCLE_INTERVAL)


async def _prepare_challenger_start(task: AnyTypeRawTask, hotkey: str, config: Config) -> bool:
    """Whether a challenger may train an own-upstream task: its starting model is its OWN upstream output.

    Guarded property: on a chained slot the challenger's starting model is never anything but its own output on the
    upstream task. While its upstream run is unfinished the answer is False (wait). A failed run or an output that
    is not a full model persists the scored fail UPSTREAM_FAILURE_REASON and the player gets no row.
    """
    task_id = str(task.task_id)
    if await task_sql.get_starting_model_repo(task_id, hotkey, config.psql_db):
        return True
    rows = await submissions_and_scoring_sql.get_task_node_losses(task.task_id, config.psql_db)
    if any(row[db_cst.HOTKEY] == hotkey and row[db_cst.QUANT_FAILURE] for row in rows):
        return False
    status = await _training_status_of(task.upstream_task_id, hotkey, config)
    if status not in (TrainingStatus.SUCCESS.value, TrainingStatus.FAILURE.value):
        return False
    rejection = (
        f"its upstream run ended {status}"
        if status == TrainingStatus.FAILURE.value
        else await _own_upstream_output_rejection(task.upstream_task_id, hotkey, config)
    )
    if rejection:
        logger.warning(f"Own-upstream task {task_id}: {hotkey[:8]} is unusable ({rejection}); scored fail, no training row")
        await submissions_and_scoring_sql.set_task_node_quant_failure(
            task.task_id, hotkey, cst.UPSTREAM_FAILURE_REASON, config.psql_db
        )
        return False
    repo = await chaining.chained_input_repo(task.upstream_task_id, hotkey, config.psql_db)
    await task_sql.set_starting_model_repo(task_id, hotkey, repo, config.psql_db)
    logger.info(f"Own-upstream task {task_id}: {hotkey[:8]} starts from its own upstream output {repo}")
    return True


async def _add_challenger_rows(seed_role: SeedRole, task: AnyTypeRawTask, config: Config) -> None:
    """Training rows for every assigned non-boss player of an attached boss-round seed that has none yet.

    The boss (EMISSION_BURN_HOTKEY) never gets a row here: its upfront run was scheduled when the first round was created. A task
    still before TRAINING gets its rows when it moves there (_process_tasks_for_training), so this only adds rows to
    a task already past that; for an own-upstream task it also settles each player's starting model first.
    """
    challengers = [
        node.hotkey
        for node in await task_sql.get_nodes_assigned_to_task(str(task.task_id), config.psql_db)
        if node.hotkey != EMISSION_BURN_HOTKEY
    ]
    own_upstream = task.training_start_point == TrainingStartPoint.OWN_UPSTREAM_MODEL
    if own_upstream:
        challengers = [hotkey for hotkey in challengers if await _prepare_challenger_start(task, hotkey, config)]
    if task.status != TaskStatus.TRAINING or not challengers:
        return
    existing = await tournament_sql.get_training_status_for_task(str(task.task_id), config.psql_db)
    tournament_id = await tournament_sql.get_tournament_id_by_task_id(task.task_id, config.psql_db)
    assignments = []
    for hotkey in challengers:
        if hotkey in existing:
            continue
        repo_info = (
            await tournament_sql.get_tournament_training_repo_and_commit(hotkey, tournament_id, config.psql_db)
            if tournament_id
            else TrainingRepoInfo.empty()
        )
        assignments.append(
            TaskTrainingAssignment(
                task_id=str(task.task_id),
                hotkey=hotkey,
                created_at=task.created_at,
                priority=2,
                training_repo=repo_info.training_repo,
                training_commit_hash=repo_info.training_commit_hash,
                github_token=repo_info.github_token,
                requested_datasets=repo_info.requested_datasets,
            )
        )
    if assignments:
        added = [assignment.hotkey for assignment in assignments]
        logger.info(f"Boss-round seed {seed_role.value} task {task.task_id}: adding challenger rows {added}")
        await tournament_sql.add_tournament_task_hotkey_pairs_for_training(assignments, config.psql_db)


async def _boss_round_seed_cycle(config: Config) -> None:
    """One pass over the attached seeds (boss-round compressions and text seeds) of every active quant and text tournament.

    Adds the miners' training rows (never the boss's) and, while a seed's boss row is not terminal, logs
    `boss entry <role> <task_id> still training` at WARNING; the round never waits on it.
    """
    for tournament in await tournament_sql.get_tournaments_with_status(TournamentStatus.ACTIVE, config.psql_db):
        if tournament.tournament_type not in (TournamentType.QUANT, TournamentType.TEXT):
            continue
        for role, seed in (await live_seed_tasks(tournament.tournament_id, config.psql_db)).items():
            if role not in ATTACHED_SEED_ROLES or not await seeds_sql.is_task_attached_to_round(seed.task_id, config.psql_db):
                continue
            with LogContext(task_id=str(seed.task_id), tournament_id=tournament.tournament_id):
                try:
                    task = await task_sql.get_task(seed.task_id, config.psql_db)
                    if task is None or task.status in (TaskStatus.FAILURE, TaskStatus.PREP_TASK_FAILURE, TaskStatus.SUCCESS):
                        continue
                    boss_status = await _training_status_of(seed.task_id, EMISSION_BURN_HOTKEY, config)
                    if boss_status not in (TrainingStatus.SUCCESS.value, TrainingStatus.FAILURE.value):
                        logger.warning(f"boss entry {role.label} {seed.task_id} still training")
                    await _add_challenger_rows(role, task, config)
                    if boss_status == TrainingStatus.SUCCESS.value and role in ANY_QUANT_SEED_ROLES:
                        # Trained at seed time but its seed-time evaluation is not settled: alert; it is evaluated now.
                        await boss_seed_eval.alert_if_boss_evaluation_missing(task, config)
                except Exception as e:
                    logger.error(f"Error processing boss-round seed {role.value} task {seed.task_id}: {e}", exc_info=True)


async def process_boss_round_seed_rows(config: Config):
    while True:
        try:
            await _boss_round_seed_cycle(config)
        except Exception as e:
            logger.error(f"Error in process_boss_round_seed_rows cycle: {e}", exc_info=True)
        await asyncio.sleep(cst.PROCESS_PENDING_TASKS_CYCLE_INTERVAL)


_seed_merge_in_progress: set[str] = set()
_seed_merge_failures: dict[str, int] = {}
# Per seed task id, all in memory by design (a restart of this process resets them and resumes a parked merge):
_seed_merge_retry_after: dict[str, float] = {}  # monotonic time before which the seed is not merged again
_seed_merge_parked: set[str] = set()  # seeds whose merge hit SEED_ADAPTER_MERGE_MAX_FAILURES: no more attempts
_seed_merge_unknown_cycles: dict[str, int] = {}  # cycles in a row a trainer could not be asked and none had the job
_seed_output_known_full: set[str] = set()  # seeds whose boss output is verified to be a full model: not polled again


class SeedMergeLookup(BaseModel):
    """What the trainers said about a seed's merge job: the job if one has it, and how many trainers could not be asked."""

    job: ModelPrepJob | None = None
    unreachable_trainers: int = 0


def _seed_merge_job_id(seed: TournamentSeedTask) -> str:
    """The trainer job id of a seed's adapter merge: distinct from the seed task's own model-prep job."""
    return f"{seed.task_id}-merge"


async def _find_seed_merge_job(job_id: str, config: Config) -> SeedMergeLookup:
    """A finished (with its result) or running merge job of the given id on any trainer. Finished wins. A trainer that
    errors (anything but 404) is counted: with no job found, the job may be running there."""
    running: ModelPrepJob | None = None
    unreachable = 0
    for trainer in await tournament_sql.get_trainers(config.psql_db):
        trainer_ip = trainer.trainer_ip if ":" in trainer.trainer_ip else f"{trainer.trainer_ip}:8001"
        try:
            url = f"http://{trainer_ip}{MODEL_PREP_STATUS_ENDPOINT.format(task_id=job_id)}"
            async with httpx.AsyncClient(timeout=_MODEL_PREP_STATUS_TIMEOUT) as client:
                response = await client.get(url)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                job = ModelPrepJob.model_validate(response.json())
        except Exception as e:
            unreachable += 1
            logger.warning(f"Could not ask trainer {trainer_ip} about merge job {job_id}: {e}")
            continue
        if job.status == TaskStatus.SUCCESS and job.result is not None and job.result.augmented_model_id:
            return SeedMergeLookup(job=job, unreachable_trainers=unreachable)
        if job.status == TaskStatus.TRAINING:
            running = job
    return SeedMergeLookup(job=running, unreachable_trainers=unreachable)


async def _seed_merge_failed(seed: TournamentSeedTask, reason: str, config: Config) -> None:
    """Count a failed merge attempt; the seed is never replaced or failed for it, the merge retries on later cycles."""
    key = str(seed.task_id)
    failures = _seed_merge_failures.get(key, 0) + 1
    _seed_merge_failures[key] = failures
    cooldown = min(SEED_ADAPTER_MERGE_COOLDOWN_BASE_SECONDS * 2 ** (failures - 1), SEED_ADAPTER_MERGE_COOLDOWN_MAX_SECONDS)
    _seed_merge_retry_after[key] = monotonic() + cooldown
    logger.warning(
        f"Seed {seed.role.value} task {seed.task_id}: adapter merge failed ({failures} in a row), next attempt in "
        f"{cooldown // 60} min: {reason}"
    )
    if failures >= SEED_ADAPTER_MERGE_MAX_FAILURES:
        _seed_merge_parked.add(key)
        await alert_seed_issue(
            f"{seed.tournament_id}:{seed.role.value}:adapter-merge-parked",
            f"Tournament {seed.tournament_id}: seed {seed.role.value} task {seed.task_id}: the adapter merge failed {failures} "
            f"times in a row and is PARKED, no further merge is attempted (last: {reason}). The seed is not replaced or "
            f"charged. Fix the cause and check the merged repo on the hub, then restart the validator's tournament orchestrator "
            f"process to resume (the counters are in memory).",
            config,
        )
        return
    if failures >= SEED_ADAPTER_MERGE_FAILURES_BEFORE_ALERT:
        await alert_seed_issue(
            f"{seed.tournament_id}:{seed.role.value}:adapter-merge",
            f"Tournament {seed.tournament_id}: seed {seed.role.value} task {seed.task_id}: the boss uploaded a LoRA adapter and "
            f"merging it into a full model failed {failures} times in a row (last: {reason}). The seed is not replaced and the "
            f"merge keeps retrying with a growing cooldown; check model-prep job {_seed_merge_job_id(seed)} and the merged "
            f"repo on the hub.",
            config,
        )


async def _persist_seed_merge(
    seed: TournamentSeedTask, merged_repo: str | None, raw_repo: str, base_model_id: str, config: Config
) -> bool:
    """Record the merged repo as the seed's output once it verifiably holds a full model (the prep container swallows merge
    errors and does not check shard indices). False, with the failure counted, otherwise. A merged repo that exists but is
    not a full model is deleted: the trainer skips the upload of a repo that already has a config and a weight file, so a
    half-uploaded one would otherwise stay broken for every later attempt."""
    if not merged_repo:
        await _seed_merge_failed(seed, "the merge job returned no merged repo", config)
        return False
    rejected_existing_repo = False
    try:
        rejection = await full_model_rejection(merged_repo)
        rejected_existing_repo = rejection is not None
    except RepositoryNotFoundError:
        rejection = "repo not found"
    except Exception as e:
        rejection = f"could not verify it ({e})"
    if rejection is not None:
        reason = f"{merged_repo} is not a full model: {rejection}"
        if rejected_existing_repo:
            try:
                await delete_merged_repo(merged_repo, raw_repo, base_model_id)
                reason += " (deleted it so the next attempt re-uploads it)"
            except Exception as e:
                logger.error(f"Seed {seed.role.value} task {seed.task_id}: could not delete {merged_repo}: {e}")
                reason += f" (deleting it failed: {e})"
        await _seed_merge_failed(seed, reason, config)
        return False
    await seeds_sql.set_seed_merged_output_repo(seed.task_id, merged_repo, config.psql_db)
    for state in (_seed_merge_failures, _seed_merge_retry_after, _seed_merge_unknown_cycles):
        state.pop(str(seed.task_id), None)
    logger.info(f"Seed {seed.role.value} task {seed.task_id}: the boss's adapter output is merged into {merged_repo}")
    return True


async def _run_seed_merge(
    seed: TournamentSeedTask,
    task: InstructTextRawTask | ChatRawTask,
    raw_repo: str,
    trainer_ip: str,
    gpu_ids: list[int],
    interconnect: str | None,
    config: Config,
) -> None:
    """Merge the boss's adapter output into its base on a trainer (a model-prep job without augmentation) and record the
    result. Releases the GPUs on every exit path."""
    job_id = _seed_merge_job_id(seed)
    cost_run_key = _prep_cost_run_key(job_id)
    success = False
    try:
        await _start_prep_cost_run(
            cost_run_key, str(seed.task_id), "seed-merge", trainer_ip, gpu_ids, config, interconnect=interconnect
        )
        result = await dispatch_augmentation_and_stats(
            task_id=job_id,
            model_id=raw_repo,
            training_data_url=task.training_data,
            augmentation_config=None,
            task_type=task.task_type,
            trainer_ip=trainer_ip,
            gpu_ids=gpu_ids,
        )
        if result is None:
            await _seed_merge_failed(seed, "the merge job failed on the trainer", config)
        else:
            success = await _persist_seed_merge(seed, result.augmented_model_id, raw_repo, task.model_id, config)
    except Exception as e:
        logger.error(f"Seed {seed.role.value} task {seed.task_id}: adapter merge errored: {e}", exc_info=True)
        await _seed_merge_failed(seed, f"error: {e}", config)
    finally:
        await _finish_prep_cost_run(cost_run_key, success, config)
        _seed_merge_in_progress.discard(str(seed.task_id))
        try:
            await tournament_sql.update_gpu_availability(trainer_ip, gpu_ids, 0, config.psql_db)
        except Exception:
            pass


async def _remember_if_full_model(seed: TournamentSeedTask, config: Config) -> None:
    """Stop polling the hub for a seed whose boss output is positively a full model. boss_output_awaits_merge also says False
    for a missing repo or no repo, which are not 'known full', so the repo is checked once more here."""
    repo = await chaining.chained_input_repo(seed.task_id, EMISSION_BURN_HOTKEY, config.psql_db)
    if repo is None:
        return
    try:
        if not await is_adapter_repo(repo):
            _seed_output_known_full.add(str(seed.task_id))
    except RepositoryNotFoundError:
        return


async def _merge_seed_adapter(seed: TournamentSeedTask, config: Config) -> None:
    """Start, recover or finish the merge of one seed's adapter output: an instruct seed or the continuous-SFT chat seed
    (see process_seed_adapter_merges)."""
    if await _training_status_of(seed.task_id, EMISSION_BURN_HOTKEY, config) != TrainingStatus.SUCCESS.value:
        return
    if not await boss_output_awaits_merge(seed, config):
        await _remember_if_full_model(seed, config)
        return
    task = await task_sql.get_task(seed.task_id, config.psql_db)
    raw_repo = await chaining.chained_input_repo(seed.task_id, EMISSION_BURN_HOTKEY, config.psql_db)
    if not isinstance(task, (InstructTextRawTask, ChatRawTask)) or raw_repo is None or task.training_data is None:
        return

    key = str(seed.task_id)
    lookup = await _find_seed_merge_job(_seed_merge_job_id(seed), config)
    job = lookup.job
    if job is not None:
        _seed_merge_unknown_cycles.pop(key, None)
    if job is not None and job.status == TaskStatus.TRAINING:
        logger.info(f"Seed {seed.role.value} task {seed.task_id}: the adapter merge is still running on a trainer")
        return
    if job is not None and job.result is not None:
        # Verified (and counted as a failure, with its cooldown, when it is not usable): either way no dispatch this cycle.
        await _persist_seed_merge(seed, job.result.augmented_model_id, raw_repo, task.model_id, config)
        return
    if lookup.unreachable_trainers:
        # The job may be running on a trainer that could not be asked; a second merge would push to the same repo.
        unknown_cycles = _seed_merge_unknown_cycles.get(key, 0) + 1
        _seed_merge_unknown_cycles[key] = unknown_cycles
        if unknown_cycles < SEED_ADAPTER_MERGE_UNKNOWN_CYCLES_BEFORE_DISPATCH:
            logger.warning(
                f"Seed {seed.role.value} task {seed.task_id}: {lookup.unreachable_trainers} trainer(s) could not be asked "
                f"about the merge job and none had it; not dispatching a merge this cycle ({unknown_cycles} in a row)"
            )
            return
        logger.warning(
            f"Seed {seed.role.value} task {seed.task_id}: trainers have been unreachable for {unknown_cycles} cycles; "
            f"dispatching the merge anyway"
        )
    _seed_merge_unknown_cycles.pop(key, None)

    gpu_req = get_task_gpu_requirement(task)
    suitable = await _check_suitable_gpus(config, gpu_req)
    if suitable is None:
        logger.info(f"No GPUs available to merge the adapter output of seed task {seed.task_id}, will retry next cycle")
        return
    trainer_ip, gpu_ids, interconnect = suitable
    await tournament_sql.update_gpu_availability(trainer_ip, gpu_ids, cst.MODEL_PREP_GPU_RESERVE_HOURS, config.psql_db)
    _seed_merge_in_progress.add(str(seed.task_id))
    try:
        asyncio.create_task(_run_seed_merge(seed, task, raw_repo, trainer_ip, gpu_ids, interconnect, config))
    except Exception:
        _seed_merge_in_progress.discard(str(seed.task_id))
        await tournament_sql.update_gpu_availability(trainer_ip, gpu_ids, 0, config.psql_db)
        raise


async def _seed_adapter_merge_cycle(config: Config) -> None:
    """One pass over the instruct seeds of every active quant and text tournament whose boss output is an adapter without a merge.

    A boss training that uploads a LoRA adapter instead of full weights is not a seed defect: the adapter is merged into
    its base by a trainer model-prep job (the one that flattens a carried continuous-SFT adapter) and the merged full
    model becomes the seed's output (chaining.chained_input_repo). Until then the seed waits (seed_tasks._evaluate_seed);
    it is never replaced, charged or halted for it.
    """
    for tournament in await tournament_sql.get_tournaments_with_status(TournamentStatus.ACTIVE, config.psql_db):
        if tournament.tournament_type == TournamentType.QUANT:
            merge_roles = INSTRUCT_INPUT_SEED_ROLES
        elif tournament.tournament_type == TournamentType.TEXT:
            merge_roles = TEXT_SEED_ROLES
        else:
            continue
        for role, seed in (await live_seed_tasks(tournament.tournament_id, config.psql_db)).items():
            key = str(seed.task_id)
            if role not in merge_roles or seed.merged_output_repo or key in _seed_merge_in_progress:
                continue
            if key in _seed_output_known_full or key in _seed_merge_parked or _seed_merge_retry_after.get(key, 0.0) > monotonic():
                continue
            with LogContext(task_id=str(seed.task_id), tournament_id=tournament.tournament_id):
                try:
                    await _merge_seed_adapter(seed, config)
                except Exception as e:
                    logger.error(f"Error merging the adapter output of seed {role.value} task {seed.task_id}: {e}", exc_info=True)


async def process_seed_adapter_merges(config: Config):
    while True:
        try:
            await _seed_adapter_merge_cycle(config)
        except Exception as e:
            logger.error(f"Error in process_seed_adapter_merges cycle: {e}", exc_info=True)
        await asyncio.sleep(cst.PROCESS_PENDING_TASKS_CYCLE_INTERVAL)


async def run_tournament_orchestrator_cycles():
    config = load_config()
    await try_db_connections(config)

    logger.info("Starting tournament orchestrator cycles")
    await asyncio.gather(
        fetch_tournament_tasks_ready_to_train(config),
        process_pending_tournament_tasks(config),
        monitor_training_tasks(config),
        seed_tournament_evaluations_from_training(config),
        update_all_trainers_gpu_availability_cycle(config),
        process_awaiting_model_prep_tasks(config),
        process_awaiting_upstream_tasks(config),
        process_boss_round_seed_rows(config),
        process_seed_adapter_merges(config),
        measure_quant_references_cycle(config),
    )


if __name__ == "__main__":
    load_dotenv(".vali.env", override=True)
    asyncio.run(run_tournament_orchestrator_cycles())
