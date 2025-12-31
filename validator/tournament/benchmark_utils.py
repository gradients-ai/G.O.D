from typing import List

from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentType
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.config import Config
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournaments as tournament_sql
from validator.db.sql.nodes import get_node_by_hotkey
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def create_benchmark_training_pairs_for_participant(
    participant: TournamentParticipant, task_type: TaskType, config: Config
) -> List[str]:
    """
    Create benchmark task copies for a tournament participant.

    This function:
    1. Gets benchmark root tasks of the specified task type
    2. Creates copies of these tasks for the participant
    3. Assigns the participant's node to each copied task (for evaluation)
    4. Sets expected repo names for the tasks
    5. Returns the list of task IDs that were created

    The orchestrator will handle:
    - Creating training pairs from assigned nodes
    - Moving tasks to TRAINING status
    - Scheduling and monitoring training
    """
    try:
        logger.info(f"Creating benchmark training pairs for participant {participant.hotkey} with task type {task_type.value}")

        root_task_ids = await tournament_sql.get_benchmark_root_tasks(task_type, config.psql_db)

        if not root_task_ids:
            logger.info(f"No benchmark root tasks found for task type {task_type.value}")
            return []

        logger.info(f"Found {len(root_task_ids)} benchmark root tasks of type {task_type.value}")

        copied_task_ids = []
        for root_task_id in root_task_ids:
            original_task = await task_sql.get_task(root_task_id, config.psql_db)
            if not original_task:
                logger.warning(f"Could not find original task {root_task_id}")
                continue

            copied_task = await task_sql.copy_task_for_benchmark(original_task, config.psql_db)

            participant_node = await get_node_by_hotkey(participant.hotkey, config.psql_db)
            if participant_node:
                await task_sql.assign_node_to_task(str(copied_task.task_id), participant_node, config.psql_db)
                logger.info(f"Assigned node {participant.hotkey} to benchmark task copy {copied_task.task_id}")

                expected_repo_name = f"benchmark-{original_task.task_id}-{participant.tournament_id}-{participant.hotkey[:8]}"
                await task_sql.set_expected_repo_name(
                    str(copied_task.task_id), participant_node, config.psql_db, expected_repo_name
                )
                logger.info(f"Set expected repo name for benchmark task {copied_task.task_id}: {expected_repo_name}")
            else:
                logger.warning(f"Could not find node for hotkey {participant.hotkey}, skipping node assignment")
                raise Exception(f"Could not find node for hotkey {participant.hotkey}")

            await tournament_sql.add_benchmark_task_copy(
                copy_task_id=copied_task.task_id,
                root_task_id=root_task_id,
                participant_hotkey=participant.hotkey,
                tournament_id=participant.tournament_id,
                psql_db=config.psql_db,
            )

            copied_task.status = TaskStatus.LOOKING_FOR_NODES  # the orchestrator will take it from here
            await task_sql.update_task(copied_task, config.psql_db)

            copied_task_ids.append(copied_task.task_id)
            logger.info(f"Created benchmark task copy {copied_task.task_id} for participant {participant.hotkey}")

        if not copied_task_ids:
            logger.warning("No benchmark task copies were created")
            return []

        logger.info(f"Successfully created benchmark task copies for participant {participant.hotkey}: {copied_task_ids}")
        return copied_task_ids

    except Exception as e:
        logger.error(f"Error creating benchmark training pairs for participant {participant.hotkey}: {str(e)}", exc_info=True)
        raise


async def create_benchmark_tasks_for_tournament_winner(tournament_id: str, winner_hotkey: str, config: Config) -> List[str]:
    try:
        logger.info(f"Creating benchmark tasks for tournament winner {winner_hotkey} from tournament {tournament_id}")

        tournament = await tournament_sql.get_tournament(tournament_id, config.psql_db)
        if not tournament:
            logger.error(f"Could not find tournament {tournament_id}")
            return []

        winner_participant = await tournament_sql.get_tournament_participant(tournament_id, winner_hotkey, config.psql_db)
        if not winner_participant:
            logger.error(f"Could not find tournament participant data for winner {winner_hotkey} in tournament {tournament_id}")
            return []

        logger.info(
            f"Found winner participant: {winner_participant.hotkey} with training repo: {winner_participant.training_repo}"
        )

        all_created_task_ids = []

        if tournament.tournament_type == TournamentType.TEXT:
            task_types = [
                TaskType.INSTRUCTTEXTTASK,
                TaskType.CHATTASK,
                TaskType.DPOTASK,
                TaskType.GRPOTASK,
            ]
            logger.info(f"Creating text benchmark tasks for {tournament.tournament_type.value} tournament")
        elif tournament.tournament_type == TournamentType.IMAGE:
            task_types = [
                TaskType.IMAGETASK,
            ]
            logger.info(f"Creating image benchmark tasks for {tournament.tournament_type.value} tournament")
        elif tournament.tournament_type == TournamentType.ENVIRONMENT:
            task_types = [
                TaskType.ENVIRONMENTTASK,
            ]
            logger.info(f"Creating environment benchmark tasks for {tournament.tournament_type.value} tournament")
        else:
            logger.error(f"Unknown tournament type: {tournament.tournament_type.value}")
            return []

        for task_type in task_types:
            try:
                logger.info(f"Creating benchmark tasks for task type: {task_type.value}")
                created_task_ids = await create_benchmark_training_pairs_for_participant(winner_participant, task_type, config)
                all_created_task_ids.extend(created_task_ids)
                logger.info(f"Created {len(created_task_ids)} benchmark tasks for task type {task_type.value}")
            except Exception as e:
                logger.error(f"Error creating benchmark tasks for task type {task_type.value}: {str(e)}")
                continue

        logger.info(
            f"Successfully created {len(all_created_task_ids)} total benchmark tasks for tournament winner {winner_hotkey}"
        )
        return all_created_task_ids

    except Exception as e:
        logger.error(f"Error creating benchmark tasks for tournament winner {winner_hotkey}: {str(e)}", exc_info=True)
        raise
