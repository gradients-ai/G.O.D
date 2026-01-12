from datetime import datetime
from datetime import timedelta
from datetime import timezone

import validator.db.constants as cst
from core.models.tournament_models import GroupRound
from core.models.tournament_models import HotkeyTaskParticipation
from core.models.tournament_models import TaskTrainingAssignment
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentGroupData
from core.models.tournament_models import TournamentPairData
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentResults
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentRoundResult
from core.models.tournament_models import TournamentStatus
from core.models.tournament_models import TournamentTask
from core.models.tournament_models import TournamentTaskScore
from core.models.tournament_models import TournamentTaskTraining
from core.models.tournament_models import TournamentType
from core.models.utility_models import GPUInfo
from core.models.utility_models import TaskType
from core.models.utility_models import TrainerInfo
from core.models.utility_models import TrainingStatus
from validator.db.database import PSQLDB
from validator.db.sql import tasks as task_sql
from validator.db.sql.submissions_and_scoring import get_all_scores_and_losses_for_task
from validator.db.sql.submissions_and_scoring import get_task_winners
from validator.utils.logging import get_logger
from validator.utils.util import normalise_float


logger = get_logger(__name__)


def is_champion_winner(winner_hotkey: str | None, base_winner_hotkey: str | None, champion_hotkey: str) -> bool:
    """
    Check if champion_hotkey won the tournament.

    Handles the case where the defending champion defends successfully
    (winner_hotkey == EMISSION_BURN_HOTKEY but base_winner_hotkey == champion_hotkey).

    Args:
        winner_hotkey: The tournament's winner_hotkey field
        base_winner_hotkey: The tournament's base_winner_hotkey field
        champion_hotkey: The hotkey to check

    Returns:
        True if champion_hotkey won the tournament
    """
    from validator.core.constants import EMISSION_BURN_HOTKEY

    if not winner_hotkey:
        return False

    return winner_hotkey == champion_hotkey or (winner_hotkey == EMISSION_BURN_HOTKEY and base_winner_hotkey == champion_hotkey)


async def create_tournament(tournament: TournamentData, psql_db: PSQLDB) -> str:
    async with await psql_db.connection() as connection:
        query = f"""
            INSERT INTO {cst.TOURNAMENTS_TABLE}
            ({cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS},
             {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}, {cst.CREATED_AT}, {cst.UPDATED_AT})
            VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING {cst.TOURNAMENT_ID}
        """
        result = await connection.fetchrow(
            query,
            tournament.tournament_id,
            tournament.tournament_type,
            tournament.status,
            tournament.base_winner_hotkey,
            tournament.winner_hotkey,
        )
        logger.info(f"Created tournament: {tournament.tournament_id}")
        return result[cst.TOURNAMENT_ID]


async def insert_tournament_round(round_data: TournamentRoundData, psql_db: PSQLDB) -> str:
    async with await psql_db.connection() as connection:
        query = f"""
            INSERT INTO {cst.TOURNAMENT_ROUNDS_TABLE}
            ({cst.ROUND_ID}, {cst.TOURNAMENT_ID}, {cst.ROUND_NUMBER}, {cst.ROUND_TYPE},
             {cst.IS_FINAL_ROUND}, {cst.ROUND_STATUS}, {cst.CREATED_AT})
            VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
            RETURNING {cst.ROUND_ID}
        """
        result = await connection.fetchrow(
            query,
            round_data.round_id,
            round_data.tournament_id,
            round_data.round_number,
            round_data.round_type,
            round_data.is_final_round,
            round_data.status,
        )
        logger.info(f"Created tournament round: {round_data.round_id}")
        return result[cst.ROUND_ID]


async def insert_tournament_groups_with_members(round_id: str, round_structure: GroupRound, psql_db: PSQLDB) -> list[str]:
    group_ids = []
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            for i, group in enumerate(round_structure.groups):
                group_id = f"{round_id}_group_{i + 1:03d}"

                group_query = f"""
                    INSERT INTO {cst.TOURNAMENT_GROUPS_TABLE}
                    ({cst.GROUP_ID}, {cst.ROUND_ID}, {cst.CREATED_AT})
                    VALUES ($1, $2, CURRENT_TIMESTAMP)
                    RETURNING {cst.GROUP_ID}
                """
                await connection.execute(group_query, group_id, round_id)

                member_query = f"""
                    INSERT INTO {cst.TOURNAMENT_GROUP_MEMBERS_TABLE}
                    ({cst.GROUP_ID}, {cst.HOTKEY}, {cst.CREATED_AT})
                    VALUES ($1, $2, CURRENT_TIMESTAMP)
                """
                for hotkey in group.member_ids:
                    await connection.execute(member_query, group_id, hotkey)

                group_ids.append(group_id)

            logger.info(f"Created {len(group_ids)} groups for round {round_id}")
    return group_ids


async def insert_tournament_pairs(round_id: str, hotkey_pairs: list[tuple[str, str]], psql_db: PSQLDB) -> list[str]:
    pair_ids = []
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            query = f"""
                INSERT INTO {cst.TOURNAMENT_PAIRS_TABLE}
                ({cst.PAIR_ID}, {cst.ROUND_ID}, {cst.HOTKEY1}, {cst.HOTKEY2}, {cst.CREATED_AT})
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                RETURNING {cst.PAIR_ID}
            """
            for i, (hotkey1, hotkey2) in enumerate(hotkey_pairs):
                pair_id = f"{round_id}_pair_{i + 1:03d}"
                await connection.execute(query, pair_id, round_id, hotkey1, hotkey2)
                pair_ids.append(pair_id)

            logger.info(f"Created {len(pair_ids)} pairs for round {round_id}")
    return pair_ids


async def add_tournament_participants(participants: list[TournamentParticipant], psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            query = f"""
                INSERT INTO {cst.TOURNAMENT_PARTICIPANTS_TABLE}
                ({cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.TRAINING_REPO}, {cst.TRAINING_COMMIT_HASH}, {cst.CREATED_AT})
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                ON CONFLICT ({cst.TOURNAMENT_ID}, {cst.HOTKEY}) DO NOTHING
            """
            for participant in participants:
                await connection.execute(
                    query,
                    participant.tournament_id,
                    participant.hotkey,
                    participant.training_repo,
                    participant.training_commit_hash,
                )
            logger.info(f"Added {len(participants)} participants to tournament")


async def add_tournament_tasks(tasks: list[TournamentTask], psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            query = f"""
                INSERT INTO {cst.TOURNAMENT_TASKS_TABLE}
                ({cst.TOURNAMENT_ID}, {cst.ROUND_ID}, {cst.TASK_ID}, {cst.GROUP_ID}, {cst.PAIR_ID}, {cst.CREATED_AT})
                VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
            """
            for task in tasks:
                await connection.execute(query, task.tournament_id, task.round_id, task.task_id, task.group_id, task.pair_id)
            logger.info(f"Added {len(tasks)} tasks to tournament")


async def get_tournament(tournament_id: str, psql_db: PSQLDB) -> TournamentData | None:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS},
                   {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}, {cst.WINNING_PERFORMANCE_DIFFERENCE}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        result = await connection.fetchrow(query, tournament_id)
        if result:
            return TournamentData(
                tournament_id=result[cst.TOURNAMENT_ID],
                tournament_type=result[cst.TOURNAMENT_TYPE],
                status=result[cst.TOURNAMENT_STATUS],
                base_winner_hotkey=result[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=result[cst.WINNER_HOTKEY],
                winning_performance_difference=result[cst.WINNING_PERFORMANCE_DIFFERENCE],
            )
        return None


async def get_tournament_where_champion_first_won(
    psql_db: PSQLDB, tournament_type: TournamentType, champion_hotkey: str
) -> TournamentData | None:
    """
    Get the LATEST tournament where the champion's actual hotkey was in the winner_hotkey field.
    This finds when they actually won (not defended with a burn placeholder).
    """
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS}, {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}, {cst.WINNING_PERFORMANCE_DIFFERENCE}, {cst.UPDATED_AT}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_TYPE} = $1
              AND {cst.TOURNAMENT_STATUS} = 'completed'
              AND {cst.WINNER_HOTKEY} = $2
            ORDER BY {cst.CREATED_AT} DESC
            LIMIT 1
        """
        result = await connection.fetchrow(query, tournament_type.value, champion_hotkey)
        if result:
            return TournamentData(
                tournament_id=result[cst.TOURNAMENT_ID],
                tournament_type=result[cst.TOURNAMENT_TYPE],
                status=result[cst.TOURNAMENT_STATUS],
                base_winner_hotkey=result[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=result[cst.WINNER_HOTKEY],
                winning_performance_difference=result[cst.WINNING_PERFORMANCE_DIFFERENCE],
                updated_at=result[cst.UPDATED_AT],
            )
        return None


async def get_last_tournament_before_current_champion(
    psql_db: PSQLDB, tournament_type: TournamentType, current_champion_hotkey: str
) -> TournamentData | None:
    """
    Get the last tournament completed before the current champion started their reign.
    This represents the previous champion's final tournament.

    Returns None if:
    - Current champion never won
    - No tournament exists before current champion's first win
    """
    first_win_tournament = await get_tournament_where_champion_first_won(psql_db, tournament_type, current_champion_hotkey)

    if not first_win_tournament or not first_win_tournament.updated_at:
        return None

    # Get the tournament completed immediately before the current champion's first win
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS},
                   {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}, {cst.WINNING_PERFORMANCE_DIFFERENCE},
                   {cst.UPDATED_AT}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_TYPE} = $1
              AND {cst.TOURNAMENT_STATUS} = 'completed'
              AND {cst.UPDATED_AT} < $2
            ORDER BY {cst.UPDATED_AT} DESC
            LIMIT 1
        """
        result = await connection.fetchrow(query, tournament_type.value, first_win_tournament.updated_at)

        if result:
            return TournamentData(
                tournament_id=result[cst.TOURNAMENT_ID],
                tournament_type=result[cst.TOURNAMENT_TYPE],
                status=result[cst.TOURNAMENT_STATUS],
                base_winner_hotkey=result[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=result[cst.WINNER_HOTKEY],
                winning_performance_difference=result[cst.WINNING_PERFORMANCE_DIFFERENCE],
                updated_at=result[cst.UPDATED_AT],
            )
        else:
            return None


async def get_latest_completed_tournament(
    psql_db: PSQLDB, tournament_type: TournamentType, exclude_tournament_id: str | None = None
) -> TournamentData | None:
    async with await psql_db.connection() as connection:
        exclude_clause = f"AND {cst.TOURNAMENT_ID} != $2" if exclude_tournament_id else ""
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS},
                   {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}, {cst.WINNING_PERFORMANCE_DIFFERENCE},
                   {cst.UPDATED_AT}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_TYPE} = $1 AND {cst.TOURNAMENT_STATUS} = 'completed'
            {exclude_clause}
            ORDER BY {cst.CREATED_AT} DESC
            LIMIT 1
        """
        params = [tournament_type.value]
        if exclude_tournament_id:
            params.append(exclude_tournament_id)

        result = await connection.fetchrow(query, *params)
        if result:
            return TournamentData(
                tournament_id=result[cst.TOURNAMENT_ID],
                tournament_type=result[cst.TOURNAMENT_TYPE],
                status=result[cst.TOURNAMENT_STATUS],
                base_winner_hotkey=result[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=result[cst.WINNER_HOTKEY],
                winning_performance_difference=result[cst.WINNING_PERFORMANCE_DIFFERENCE],
                updated_at=result[cst.UPDATED_AT],
            )
        return None


async def get_tournament_rounds(tournament_id: str, psql_db: PSQLDB) -> list[TournamentRoundData]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.ROUND_ID}, {cst.TOURNAMENT_ID}, {cst.ROUND_NUMBER}, {cst.ROUND_TYPE},
                   {cst.IS_FINAL_ROUND}, {cst.ROUND_STATUS}
            FROM {cst.TOURNAMENT_ROUNDS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1
            ORDER BY {cst.ROUND_NUMBER} ASC
        """
        results = await connection.fetch(query, tournament_id)
        return [
            TournamentRoundData(
                round_id=row[cst.ROUND_ID],
                tournament_id=row[cst.TOURNAMENT_ID],
                round_number=row[cst.ROUND_NUMBER],
                round_type=row[cst.ROUND_TYPE],
                is_final_round=row[cst.IS_FINAL_ROUND],
                status=row[cst.ROUND_STATUS],
            )
            for row in results
        ]


async def get_tournament_rounds_with_status(status: str, psql_db: PSQLDB) -> list[TournamentRoundData]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.ROUND_ID}, {cst.TOURNAMENT_ID}, {cst.ROUND_NUMBER}, {cst.ROUND_TYPE},
                   {cst.IS_FINAL_ROUND}, {cst.ROUND_STATUS}
            FROM {cst.TOURNAMENT_ROUNDS_TABLE}
            WHERE {cst.ROUND_STATUS} = $1
            ORDER BY {cst.TOURNAMENT_ID}, {cst.ROUND_NUMBER} ASC
        """
        results = await connection.fetch(query, status)
        return [
            TournamentRoundData(
                round_id=row[cst.ROUND_ID],
                tournament_id=row[cst.TOURNAMENT_ID],
                round_number=row[cst.ROUND_NUMBER],
                round_type=row[cst.ROUND_TYPE],
                is_final_round=row[cst.IS_FINAL_ROUND],
                status=row[cst.ROUND_STATUS],
            )
            for row in results
        ]


async def get_final_round_id(tournament_id: str, psql_db: PSQLDB) -> str | None:
    """Get the final round ID for a tournament."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.ROUND_ID}
            FROM {cst.TOURNAMENT_ROUNDS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.IS_FINAL_ROUND} = true
            LIMIT 1
        """
        result = await connection.fetchrow(query, tournament_id)
        return result[cst.ROUND_ID] if result else None


async def get_tournament_tasks(round_id: str, psql_db: PSQLDB) -> list[TournamentTask]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.ROUND_ID}, {cst.TASK_ID}, {cst.GROUP_ID}, {cst.PAIR_ID}
            FROM {cst.TOURNAMENT_TASKS_TABLE}
            WHERE {cst.ROUND_ID} = $1
        """
        results = await connection.fetch(query, round_id)
        return [
            TournamentTask(
                tournament_id=row[cst.TOURNAMENT_ID],
                round_id=row[cst.ROUND_ID],
                task_id=row[cst.TASK_ID],
                group_id=row[cst.GROUP_ID],
                pair_id=row[cst.PAIR_ID],
            )
            for row in results
        ]


async def get_tournament_pairs(round_id: str, psql_db: PSQLDB) -> list[TournamentPairData]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.PAIR_ID}, {cst.ROUND_ID}, {cst.HOTKEY1}, {cst.HOTKEY2}
            FROM {cst.TOURNAMENT_PAIRS_TABLE}
            WHERE {cst.ROUND_ID} = $1
        """
        results = await connection.fetch(query, round_id)
        return [
            TournamentPairData(
                pair_id=row[cst.PAIR_ID],
                round_id=row[cst.ROUND_ID],
                hotkey1=row[cst.HOTKEY1],
                hotkey2=row[cst.HOTKEY2],
            )
            for row in results
        ]


async def get_tournament_groups(round_id: str, psql_db: PSQLDB) -> list[TournamentGroupData]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.GROUP_ID}, {cst.ROUND_ID}
            FROM {cst.TOURNAMENT_GROUPS_TABLE}
            WHERE {cst.ROUND_ID} = $1
        """
        results = await connection.fetch(query, round_id)
        return [TournamentGroupData(group_id=row[cst.GROUP_ID], round_id=row[cst.ROUND_ID]) for row in results]


async def get_tournament_group_members(group_id: str, psql_db: PSQLDB) -> list[TournamentParticipant]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.GROUP_ID}, {cst.HOTKEY}
            FROM {cst.TOURNAMENT_GROUP_MEMBERS_TABLE}
            WHERE {cst.GROUP_ID} = $1
        """
        results = await connection.fetch(query, group_id)
        # TODO: join with full participant table
        return [
            TournamentParticipant(
                tournament_id="",
                hotkey=row[cst.HOTKEY],
            )
            for row in results
        ]


async def update_round_status(round_id: str, status: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENT_ROUNDS_TABLE}
            SET {cst.ROUND_STATUS} = $2
            WHERE {cst.ROUND_ID} = $1
        """
        await connection.execute(query, round_id, status)
        logger.info(f"Updated round {round_id} status to {status}")


async def update_tournament_status(tournament_id: str, status: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENTS_TABLE}
            SET {cst.TOURNAMENT_STATUS} = $2, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        await connection.execute(query, tournament_id, status)
        logger.info(f"Updated tournament {tournament_id} status to {status}")


async def update_tournament_winner_hotkey(tournament_id: str, winner_hotkey: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENTS_TABLE}
            SET {cst.WINNER_HOTKEY} = $2, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        await connection.execute(query, tournament_id, winner_hotkey)
        logger.info(f"Updated tournament {tournament_id} winner hotkey to {winner_hotkey}")


async def get_tournaments_with_status(status: TournamentStatus, psql_db: PSQLDB) -> list[TournamentData]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS},
                   {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_STATUS} = $1
            ORDER BY {cst.CREATED_AT} DESC
        """
        results = await connection.fetch(query, status.value)
        return [
            TournamentData(
                tournament_id=row[cst.TOURNAMENT_ID],
                tournament_type=row[cst.TOURNAMENT_TYPE],
                status=row[cst.TOURNAMENT_STATUS],
                base_winner_hotkey=row[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=row[cst.WINNER_HOTKEY],
            )
            for row in results
        ]


async def update_tournament_participant_training_repo(
    tournament_id: str, hotkey: str, training_repo: str, training_commit_hash: str, psql_db: PSQLDB
):
    """Update the training repo information for a tournament participant."""
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENT_PARTICIPANTS_TABLE}
            SET {cst.TRAINING_REPO} = $1, {cst.TRAINING_COMMIT_HASH} = $2
            WHERE {cst.TOURNAMENT_ID} = $3 AND {cst.HOTKEY} = $4
        """
        await connection.execute(query, training_repo, training_commit_hash, tournament_id, hotkey)
        logger.info(f"Updated training repo for participant {hotkey} in tournament {tournament_id}")


async def update_tournament_participant_backup_repo(tournament_id: str, hotkey: str, backup_repo: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENT_PARTICIPANTS_TABLE}
            SET {cst.BACKUP_REPO} = $1
            WHERE {cst.TOURNAMENT_ID} = $2 AND {cst.HOTKEY} = $3
        """
        await connection.execute(query, backup_repo, tournament_id, hotkey)
        logger.info(f"Updated backup repo for participant {hotkey} in tournament {tournament_id}")


async def get_tournament_participant(tournament_id: str, hotkey: str, psql_db: PSQLDB) -> TournamentParticipant | None:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.ELIMINATED_IN_ROUND_ID},
                   {cst.FINAL_POSITION}, {cst.TRAINING_REPO}, {cst.TRAINING_COMMIT_HASH}, {cst.BACKUP_REPO}
            FROM {cst.TOURNAMENT_PARTICIPANTS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.HOTKEY} = $2
        """
        result = await connection.fetchrow(query, tournament_id, hotkey)
        if result:
            return TournamentParticipant(
                tournament_id=result[cst.TOURNAMENT_ID],
                hotkey=result[cst.HOTKEY],
                eliminated_in_round_id=result[cst.ELIMINATED_IN_ROUND_ID],
                final_position=result[cst.FINAL_POSITION],
                training_repo=result[cst.TRAINING_REPO],
                training_commit_hash=result[cst.TRAINING_COMMIT_HASH],
                backup_repo=result[cst.BACKUP_REPO],
            )
        return None


async def get_tournament_participants(tournament_id: str, psql_db: PSQLDB) -> list[TournamentParticipant]:
    """Get all participants for a tournament with their training repo information."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.HOTKEY}, {cst.ELIMINATED_IN_ROUND_ID},
                   {cst.FINAL_POSITION}, {cst.TRAINING_REPO}, {cst.TRAINING_COMMIT_HASH}, {cst.BACKUP_REPO}
            FROM {cst.TOURNAMENT_PARTICIPANTS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        results = await connection.fetch(query, tournament_id)
        return [
            TournamentParticipant(
                tournament_id=row[cst.TOURNAMENT_ID],
                hotkey=row[cst.HOTKEY],
                eliminated_in_round_id=row[cst.ELIMINATED_IN_ROUND_ID],
                final_position=row[cst.FINAL_POSITION],
                training_repo=row[cst.TRAINING_REPO],
                training_commit_hash=row[cst.TRAINING_COMMIT_HASH],
                backup_repo=row[cst.BACKUP_REPO],
            )
            for row in results
        ]


async def add_trainer_gpus(trainer_ip: str, gpu_infos: list[GPUInfo], psql_db: PSQLDB):
    """Add or update GPU information for a trainer"""
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            # First, remove existing entries for this trainer
            delete_query = f"""
                DELETE FROM {cst.TRAINERS_GPUS_TABLE}
                WHERE {cst.TRAINER_IP} = $1
            """
            await connection.execute(delete_query, trainer_ip)

            # Then insert new GPU information
            insert_query = f"""
                INSERT INTO {cst.TRAINERS_GPUS_TABLE}
                ({cst.TRAINER_IP}, {cst.GPU_ID}, {cst.GPU_TYPE}, {cst.VRAM_GB}, {cst.USED_UNTIL})
                VALUES ($1, $2, $3, $4, $5)
            """

            for gpu_info in gpu_infos:
                used_until = None
                if not gpu_info.available:
                    used_until = datetime.now(timezone.utc) + timedelta(hours=48)

                await connection.execute(
                    insert_query, trainer_ip, gpu_info.gpu_id, gpu_info.gpu_type, gpu_info.vram_gb, used_until
                )

            logger.info(f"Added {len(gpu_infos)} GPUs for trainer {trainer_ip}")


async def remove_trainer(trainer_ip: str, psql_db: PSQLDB):
    """Remove a trainer and all its GPUs from the database"""
    async with await psql_db.connection() as connection:
        query = f"""
            DELETE FROM {cst.TRAINERS_GPUS_TABLE}
            WHERE {cst.TRAINER_IP} = $1
        """
        await connection.execute(query, trainer_ip)
        logger.info(f"Removed trainer {trainer_ip} from the database")


async def get_trainers(psql_db: PSQLDB) -> list[TrainerInfo]:
    """Get all trainers and their GPU information from the database"""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TRAINER_IP}, {cst.GPU_ID}, {cst.GPU_TYPE}, {cst.VRAM_GB}, {cst.USED_UNTIL}
            FROM {cst.TRAINERS_GPUS_TABLE}
            ORDER BY {cst.TRAINER_IP}, {cst.GPU_ID}
        """
        results = await connection.fetch(query)

        # Group by trainer IP
        trainers = {}
        for row in results:
            trainer_ip = row[cst.TRAINER_IP]
            if trainer_ip not in trainers:
                trainers[trainer_ip] = TrainerInfo(trainer_ip=trainer_ip, gpus=[])

            # Determine availability based on used_until
            used_until = row[cst.USED_UNTIL]
            available = used_until is None or used_until < datetime.now(timezone.utc)

            trainers[trainer_ip].gpus.append(
                GPUInfo(
                    gpu_id=row[cst.GPU_ID],
                    gpu_type=row[cst.GPU_TYPE],
                    vram_gb=row[cst.VRAM_GB],
                    available=available,
                    used_until=used_until,
                )
            )

        return list(trainers.values())


async def add_tournament_task_hotkey_pairs_for_training(assignments: list[TaskTrainingAssignment], psql_db: PSQLDB):
    """
    Add task-hotkey pairs to the tournament_task_hotkey_trainings table using batch insert.
    Each task-hotkey pair defines a training task that we'll send to a trainer later.

    Args:
        assignments: List of TaskTrainingAssignment objects containing task info, repo details, and priority
        psql_db: Database connection
    """
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            if not assignments:
                logger.info("No task training assignments to insert")
                return

            query = f"""
                INSERT INTO {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
                ({cst.TASK_ID}, {cst.HOTKEY}, {cst.CREATED_AT}, {cst.PRIORITY}, {cst.TRAINING_REPO}, {cst.TRAINING_COMMIT_HASH})
                SELECT * FROM unnest($1::uuid[], $2::text[], $3::timestamptz[], $4::integer[], $5::text[], $6::text[])
                ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}) DO NOTHING
            """

            task_ids = [assignment.task_id for assignment in assignments]
            hotkeys = [assignment.hotkey for assignment in assignments]
            timestamps = [assignment.created_at for assignment in assignments]
            priorities = [assignment.priority for assignment in assignments]
            training_repos = [assignment.training_repo for assignment in assignments]
            training_commit_hashes = [assignment.training_commit_hash for assignment in assignments]

            await connection.execute(query, task_ids, hotkeys, timestamps, priorities, training_repos, training_commit_hashes)

            priority_counts = {}
            for assignment in assignments:
                priority_counts[assignment.priority] = priority_counts.get(assignment.priority, 0) + 1

            logger.info(f"Added {len(assignments)} task training assignments - priorities: {priority_counts}")


async def get_tournament_training_tasks(psql_db: PSQLDB, status: TrainingStatus) -> list[TournamentTaskTraining]:
    """
    Fetch tournament tasks with specific training status ordered by priority and creation time.
    Priority 1 tasks (regular tournament tasks) are processed before Priority 2 tasks (benchmark tasks).

    Args:
        psql_db: Database connection
        status: Training status to filter by

    Returns:
        List of TournamentTaskTraining objects ordered by priority (1 first) then creation time (newest first)
    """
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TASK_ID}, {cst.HOTKEY}, {cst.TRAINING_STATUS}, {cst.N_TRAINING_ATTEMPTS},
                   {cst.CREATED_AT}, {cst.UPDATED_AT}, {cst.PRIORITY}, {cst.TRAINING_REPO}, {cst.TRAINING_COMMIT_HASH},
                   {cst.TRAINER_IP}
            FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
            WHERE {cst.TRAINING_STATUS} = $1
            ORDER BY {cst.PRIORITY} ASC, {cst.CREATED_AT} DESC
        """
        results = await connection.fetch(query, status)

        if not results:
            return []

        unique_task_ids = list({row[cst.TASK_ID] for row in results})
        tasks = await task_sql.get_tasks_by_ids(unique_task_ids, psql_db, connection)

        # Create a mapping for quick lookup
        tasks_dict = {task.task_id: task for task in tasks}

        tournament_tasks = []
        missing_tasks = []

        for row in results:
            task = tasks_dict.get(row[cst.TASK_ID])
            if task:
                tournament_tasks.append(
                    TournamentTaskTraining(
                        task=task,
                        hotkey=row[cst.HOTKEY],
                        training_status=row[cst.TRAINING_STATUS],
                        n_training_attempts=row[cst.N_TRAINING_ATTEMPTS],
                        created_at=row[cst.CREATED_AT],
                        updated_at=row[cst.UPDATED_AT],
                        training_repo=row[cst.TRAINING_REPO],
                        training_commit_hash=row[cst.TRAINING_COMMIT_HASH],
                        priority=row[cst.PRIORITY],
                        trainer_ip=row[cst.TRAINER_IP],
                    )
                )

            else:
                missing_tasks.append(row[cst.TASK_ID])

        if missing_tasks:
            logger.warning(f"Tasks not found in batch load: {missing_tasks}")

        return tournament_tasks


async def update_tournament_task_training_status(
    task_id: str, hotkey: str, status: TrainingStatus, psql_db: PSQLDB, trainer_ip: str | None = None
):
    """Update the training status of a specific task-hotkey pair"""
    async with await psql_db.connection() as connection:
        increment_clause = (
            f", {cst.N_TRAINING_ATTEMPTS} = {cst.N_TRAINING_ATTEMPTS} + 1" if status == TrainingStatus.TRAINING else ""
        )
        trainer_ip_clause = f", {cst.TRAINER_IP} = $4" if trainer_ip else ""

        query = f"""
            UPDATE {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
            SET {cst.TRAINING_STATUS} = $3{increment_clause}{trainer_ip_clause}, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2
        """

        params = [task_id, hotkey, status]
        if trainer_ip:
            params.append(trainer_ip)
        await connection.execute(query, *params)
        logger.info(f"Marked task-hotkey pair ({task_id}, {hotkey}) as {status}")


async def update_dstack_runname(task_id: str, hotkey: str, runname: str, psql_db: PSQLDB):
    """Update the dstack runname for a specific task-hotkey pair"""
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
            SET {cst.DSTACK_RUNNAME} = $3, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2
        """
        await connection.execute(query, task_id, hotkey, runname)
        logger.info(f"Updated dstack runname for task {task_id}, hotkey {hotkey} to {runname}")


async def get_dstack_runname(task_id: str, hotkey: str, psql_db: PSQLDB) -> str | None:
    """Get the dstack runname for a specific task-hotkey pair"""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.DSTACK_RUNNAME}
            FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2
        """
        result = await connection.fetchval(query, task_id, hotkey)
        return result


async def get_training_status_for_task_and_hotkeys(task_id: str, hotkeys: list[str], psql_db: PSQLDB) -> dict[str, str]:
    """Get the training status for a task and list of hotkeys"""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.HOTKEY}, {cst.TRAINING_STATUS}
            FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = ANY($2)
        """
        results = await connection.fetch(query, task_id, hotkeys)
        return {row[cst.HOTKEY]: row[cst.TRAINING_STATUS] for row in results}


async def get_training_status_for_task(task_id: str, psql_db: PSQLDB) -> dict[str, str]:
    """Get the training status for all hotkeys in a given task_id"""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.HOTKEY}, {cst.TRAINING_STATUS}
            FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
            WHERE {cst.TASK_ID} = $1
        """
        results = await connection.fetch(query, task_id)
        return {row[cst.HOTKEY]: row[cst.TRAINING_STATUS] for row in results}


async def get_tournament_training_repo_and_commit(
    hotkey: str, tournament_id: str, psql_db: PSQLDB
) -> tuple[str | None, str | None]:
    """Get the training_repo and training_commit_hash for a hotkey from tournament_participants table for a specific tournament.
    If backup_repo is present, it will be used instead of training_repo."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TRAINING_REPO}, {cst.TRAINING_COMMIT_HASH}, {cst.BACKUP_REPO}
            FROM {cst.TOURNAMENT_PARTICIPANTS_TABLE}
            WHERE {cst.HOTKEY} = $1 AND {cst.TOURNAMENT_ID} = $2
            ORDER BY {cst.CREATED_AT} DESC
            LIMIT 1
        """
        result = await connection.fetchrow(query, hotkey, tournament_id)
        if result:
            if result[cst.BACKUP_REPO]:
                logger.info(f"Using backup repo for hotkey {hotkey} in tournament {tournament_id}: {result[cst.BACKUP_REPO]}")
                repo = result[cst.BACKUP_REPO]
                return repo, None
            else:
                repo = result[cst.TRAINING_REPO]
                logger.info(f"Using training repo for hotkey {hotkey} in tournament {tournament_id}: {repo}")
                return repo, result[cst.TRAINING_COMMIT_HASH]
        logger.warning(f"No training repository found for hotkey {hotkey} in tournament {tournament_id}")
        return None, None


async def get_tournament_training_stats(psql_db: PSQLDB) -> dict:
    """Get statistics about tournament training status"""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT
                {cst.TRAINING_STATUS},
                COUNT(*) as count,
                AVG({cst.N_TRAINING_ATTEMPTS}) as avg_attempts,
                MAX({cst.N_TRAINING_ATTEMPTS}) as max_attempts
            FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
            GROUP BY {cst.TRAINING_STATUS}
        """
        results = await connection.fetch(query)

        stats = {"total_pairs": 0, "pending": 0, "success": 0, "failure": 0, "avg_attempts": 0, "max_attempts": 0}

        for row in results:
            status = row[cst.TRAINING_STATUS]
            count = row["count"]
            avg_attempts = row["avg_attempts"] or 0
            max_attempts = row["max_attempts"] or 0

            stats["total_pairs"] += count
            stats[status] = count
            stats["avg_attempts"] = max(stats["avg_attempts"], avg_attempts)
            stats["max_attempts"] = max(stats["max_attempts"], max_attempts)

        return stats


async def update_gpu_availability(trainer_ip: str, gpu_ids: list[int], hours_to_complete: int, psql_db: PSQLDB):
    """Update GPU availability by setting used_until based on hours_to_complete"""
    async with await psql_db.connection() as connection:
        if hours_to_complete == 0:
            # Reset GPU availability by setting used_until to NULL
            query = f"""
                UPDATE {cst.TRAINERS_GPUS_TABLE}
                SET {cst.USED_UNTIL} = NULL, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                WHERE {cst.TRAINER_IP} = $1 AND {cst.GPU_ID} = ANY($2)
            """
        else:
            # Set GPU as used for specified hours
            used_until = f"CURRENT_TIMESTAMP + INTERVAL '{hours_to_complete} hours'"
            query = f"""
                UPDATE {cst.TRAINERS_GPUS_TABLE}
                SET {cst.USED_UNTIL} = {used_until}, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                WHERE {cst.TRAINER_IP} = $1 AND {cst.GPU_ID} = ANY($2)
            """

        await connection.execute(query, trainer_ip, gpu_ids)
        action = "reset" if hours_to_complete == 0 else f"used for {hours_to_complete} hours"
        logger.info(f"Updated GPU availability for trainer {trainer_ip}, GPUs {gpu_ids} - {action}")


async def get_tasks_with_all_training_completed(psql_db: PSQLDB) -> list[str]:
    """
    Get task IDs where all training tasks (task_id, hotkey) pairs have completed.
    Only returns tasks from the last month and includes benchmark tasks.
    """
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT DISTINCT t1.{cst.TASK_ID}
            FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE} t1
            JOIN {cst.TASKS_TABLE} {cst.TASKS_TABLE} ON t1.{cst.TASK_ID} = {cst.TASKS_TABLE}.{cst.TASK_ID}
            WHERE {cst.TASKS_TABLE}.{cst.STATUS} = 'training'
            AND {cst.TASKS_TABLE}.{cst.CREATED_AT} >= NOW() - INTERVAL '1 month'
            AND NOT EXISTS (
                SELECT 1
                FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE} t2
                WHERE t2.{cst.TASK_ID} = t1.{cst.TASK_ID}
                AND t2.{cst.TRAINING_STATUS} NOT IN ('success', 'failure')
            )
        """
        results = await connection.fetch(query)
        return [row[cst.TASK_ID] for row in results]


async def eliminate_tournament_participants(tournament_id: str, round_id: str, hotkeys: list[str], psql_db: PSQLDB):
    """Mark tournament participants as eliminated in the specified round."""
    if not hotkeys:
        return

    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENT_PARTICIPANTS_TABLE}
            SET {cst.ELIMINATED_IN_ROUND_ID} = $2
            WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.HOTKEY} = ANY($3)
            AND {cst.ELIMINATED_IN_ROUND_ID} IS NULL
        """
        await connection.execute(query, tournament_id, round_id, hotkeys)
        logger.info(f"Eliminated {len(hotkeys)} participants from tournament {tournament_id} in round {round_id}")


async def get_active_tournament_participants(psql_db: PSQLDB) -> list[str]:
    """Get hotkeys of all active tournament participants for participation weights."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT DISTINCT tp.{cst.HOTKEY}
            FROM {cst.TOURNAMENT_PARTICIPANTS_TABLE} tp
            JOIN {cst.TOURNAMENTS_TABLE} t ON tp.{cst.TOURNAMENT_ID} = t.{cst.TOURNAMENT_ID}
            WHERE t.{cst.TOURNAMENT_STATUS} = 'active'
            AND tp.{cst.ELIMINATED_IN_ROUND_ID} IS NULL
        """
        results = await connection.fetch(query)
        hotkeys = [row[cst.HOTKEY] for row in results]
        logger.info(f"Found {len(hotkeys)} active tournament participants for participation weights")
        return hotkeys


async def get_tournament_full_results(tournament_id: str, psql_db: PSQLDB) -> TournamentResults:
    rounds = await get_tournament_rounds(tournament_id, psql_db)

    round_results = []

    for round_data in rounds:
        tasks = await get_tournament_tasks(round_data.round_id, psql_db)

        task_scores = []
        task_ids = [task.task_id for task in tasks]
        if task_ids:
            task_winners = await get_task_winners(task_ids, psql_db)

            for task in tasks:
                participant_scores = await get_all_scores_and_losses_for_task(task.task_id, psql_db)
                task_score = TournamentTaskScore(
                    task_id=str(task.task_id),
                    group_id=task.group_id,
                    pair_id=task.pair_id,
                    winner=task_winners.get(str(task.task_id)),
                    participant_scores=participant_scores,
                )
                task_scores.append(task_score)

        round_result = TournamentRoundResult(
            round_id=round_data.round_id,
            round_number=round_data.round_number,
            round_type=round_data.round_type,
            is_final_round=round_data.is_final_round,
            tasks=task_scores,
        )
        round_results.append(round_result)

    return TournamentResults(tournament_id=tournament_id, rounds=round_results)


async def get_tournament_with_created_at(tournament_id: str, psql_db: PSQLDB) -> tuple[TournamentData | None, datetime | None]:
    """Get a tournament with its created_at timestamp."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS},
                   {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}, {cst.CREATED_AT}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        result = await connection.fetchrow(query, tournament_id)
        if result:
            tournament = TournamentData(
                tournament_id=result[cst.TOURNAMENT_ID],
                tournament_type=result[cst.TOURNAMENT_TYPE],
                status=result[cst.TOURNAMENT_STATUS],
                base_winner_hotkey=result[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=result[cst.WINNER_HOTKEY],
            )
            created_at = result[cst.CREATED_AT]
            return tournament, created_at
        return None, None


async def get_active_tournament(psql_db: PSQLDB, tournament_type: TournamentType) -> TournamentData | None:
    """Get the active tournament for a given type (if any)."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS},
                   {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_TYPE} = $1 AND {cst.TOURNAMENT_STATUS} = 'active'
            ORDER BY {cst.CREATED_AT} DESC
            LIMIT 1
        """
        result = await connection.fetchrow(query, tournament_type.value)
        if result:
            return TournamentData(
                tournament_id=result[cst.TOURNAMENT_ID],
                tournament_type=result[cst.TOURNAMENT_TYPE],
                status=result[cst.TOURNAMENT_STATUS],
                base_winner_hotkey=result[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=result[cst.WINNER_HOTKEY],
            )
        return None


async def get_latest_tournament_with_created_at(
    psql_db: PSQLDB, tournament_type: TournamentType
) -> tuple[TournamentData | None, datetime | None]:
    """Get the latest tournament (active or completed) with its created_at timestamp."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}, {cst.TOURNAMENT_TYPE}, {cst.TOURNAMENT_STATUS},
                   {cst.BASE_WINNER_HOTKEY}, {cst.WINNER_HOTKEY}, {cst.CREATED_AT}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_TYPE} = $1
              AND {cst.TOURNAMENT_STATUS} != 'cancelled'
            ORDER BY {cst.CREATED_AT} DESC
            LIMIT 1
        """
        result = await connection.fetchrow(query, tournament_type.value)
        if result:
            tournament = TournamentData(
                tournament_id=result[cst.TOURNAMENT_ID],
                tournament_type=result[cst.TOURNAMENT_TYPE],
                status=result[cst.TOURNAMENT_STATUS],
                base_winner_hotkey=result[cst.BASE_WINNER_HOTKEY],
                winner_hotkey=result[cst.WINNER_HOTKEY],
            )
            created_at = result[cst.CREATED_AT]
            return tournament, created_at
        return None, None


async def count_champion_consecutive_wins(psql_db: PSQLDB, tournament_type: TournamentType, champion_hotkey: str) -> int:
    """Count consecutive tournament wins for the current champion (their current winning streak)."""
    async with await psql_db.connection() as connection:
        # Get all completed tournaments ordered by date descending
        # Include base_winner_hotkey to handle EMISSION_BURN_HOTKEY wins correctly
        query = f"""
            SELECT {cst.WINNER_HOTKEY}, {cst.BASE_WINNER_HOTKEY}, {cst.CREATED_AT}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_TYPE} = $1
              AND {cst.TOURNAMENT_STATUS} = 'completed'
            ORDER BY {cst.CREATED_AT} DESC
        """
        results = await connection.fetch(query, tournament_type.value)

        if not results:
            return 0

        consecutive_wins = 0
        for row in results:
            winner = row[cst.WINNER_HOTKEY]
            base_winner = row[cst.BASE_WINNER_HOTKEY]

            if is_champion_winner(winner, base_winner, champion_hotkey):
                consecutive_wins += 1
            else:
                # Stop counting when we hit a tournament won by someone else
                break

        return consecutive_wins


async def count_champion_consecutive_wins_at_tournament(
    psql_db: PSQLDB, tournament_type: TournamentType, champion_hotkey: str, tournament_id: str
) -> int:
    """Count consecutive tournament wins for a champion at the time a specific tournament started."""
    async with await psql_db.connection() as connection:
        # First get the created_at time of the target tournament
        target_query = f"""
            SELECT {cst.CREATED_AT}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        target_result = await connection.fetchval(target_query, tournament_id)

        if not target_result:
            return 0

        # Get all completed tournaments of the same type that finished before this tournament started
        # Include base_winner_hotkey to handle EMISSION_BURN_HOTKEY wins correctly
        query = f"""
            SELECT {cst.WINNER_HOTKEY}, {cst.BASE_WINNER_HOTKEY}, {cst.CREATED_AT}
            FROM {cst.TOURNAMENTS_TABLE}
            WHERE {cst.TOURNAMENT_TYPE} = $1
              AND {cst.TOURNAMENT_STATUS} = 'completed'
              AND {cst.CREATED_AT} < $2
            ORDER BY {cst.CREATED_AT} DESC
        """
        results = await connection.fetch(query, tournament_type.value, target_result)

        if not results:
            return 0

        consecutive_wins = 0
        for row in results:
            winner = row[cst.WINNER_HOTKEY]
            base_winner = row[cst.BASE_WINNER_HOTKEY]

            if is_champion_winner(winner, base_winner, champion_hotkey):
                consecutive_wins += 1
            else:
                # Stop counting when we hit a tournament won by someone else
                break

        return consecutive_wins


async def get_tournament_id_by_task_id(task_id: str, psql_db: PSQLDB) -> str | None:
    """Get the tournament ID for a given task ID."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TOURNAMENT_ID}
            FROM {cst.TOURNAMENT_TASKS_TABLE}
            WHERE {cst.TASK_ID} = $1
        """
        result = await connection.fetchrow(query, task_id)
        if result:
            return result[cst.TOURNAMENT_ID]

        benchmark_query = f"""
            SELECT {cst.TOURNAMENT_ID}
            FROM {cst.BENCHMARK_TASK_COPIES_TABLE}
            WHERE {cst.COPY_TASK_ID} = $1
        """
        benchmark_result = await connection.fetchrow(benchmark_query, task_id)
        if benchmark_result:
            return benchmark_result[cst.TOURNAMENT_ID]

        return None


async def add_benchmark_root_task(task_id: str, task_type: TaskType, psql_db: PSQLDB):
    """Add a task as a benchmark root task (template for benchmarking)."""
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            query = f"""
                INSERT INTO {cst.BENCHMARK_ROOT_TASKS_TABLE}
                ({cst.TASK_ID}, {cst.TASK_TYPE}, {cst.CREATED_AT})
                VALUES ($1, $2, CURRENT_TIMESTAMP)
                ON CONFLICT ({cst.TASK_ID}) DO NOTHING
            """
            await connection.execute(query, task_id, task_type.value)
            logger.info(f"Added benchmark root task {task_id} with type {task_type.value}")


async def get_benchmark_root_tasks(task_type: TaskType, psql_db: PSQLDB) -> list[str]:
    """Get all benchmark root tasks of a specific type."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT {cst.TASK_ID}
            FROM {cst.BENCHMARK_ROOT_TASKS_TABLE}
            WHERE {cst.TASK_TYPE} = $1
        """
        results = await connection.fetch(query, task_type.value)
        return [row[cst.TASK_ID] for row in results]


async def add_benchmark_task_copy(
    copy_task_id: str, root_task_id: str, participant_hotkey: str, tournament_id: str, psql_db: PSQLDB
):
    """Add a benchmark task copy for a specific participant."""
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            query = f"""
                INSERT INTO {cst.BENCHMARK_TASK_COPIES_TABLE}
                ({cst.COPY_TASK_ID}, {cst.ROOT_TASK_ID}, {cst.PARTICIPANT_HOTKEY}, {cst.TOURNAMENT_ID}, {cst.CREATED_AT})
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                ON CONFLICT ({cst.ROOT_TASK_ID}, {cst.PARTICIPANT_HOTKEY}, {cst.TOURNAMENT_ID}) DO NOTHING
            """
            await connection.execute(query, copy_task_id, root_task_id, participant_hotkey, tournament_id)
            logger.info(
                f"Added benchmark task copy {copy_task_id} for participant {participant_hotkey} from tournament {tournament_id}"
            )


async def get_all_benchmark_results(psql_db: PSQLDB) -> list[dict]:
    """
    Get all benchmark results across all root tasks.

    Args:
        psql_db: Database connection

    Returns:
        List of all benchmark results grouped by root task
    """
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT
                brt.{cst.TASK_ID} as root_task_id,
                btc.{cst.COPY_TASK_ID},
                btc.{cst.PARTICIPANT_HOTKEY},
                btc.{cst.TOURNAMENT_ID},
                btc.{cst.CREATED_AT},
                t.{cst.MODEL_ID},
                t.{cst.DS},
                t.{cst.TASK_TYPE},
                tn.{cst.TASK_NODE_QUALITY_SCORE} as quality_score,
                tn.{cst.TEST_LOSS},
                tn.{cst.SYNTH_LOSS},
                s.{cst.REPO},
                t.{cst.COMPLETED_AT}
            FROM {cst.BENCHMARK_ROOT_TASKS_TABLE} brt
            JOIN {cst.BENCHMARK_TASK_COPIES_TABLE} btc ON brt.{cst.TASK_ID} = btc.{cst.ROOT_TASK_ID}
            JOIN {cst.TASKS_TABLE} t ON btc.{cst.COPY_TASK_ID} = t.{cst.TASK_ID}
            LEFT JOIN {cst.TASK_NODES_TABLE} tn ON btc.{cst.COPY_TASK_ID} = tn.{cst.TASK_ID}
                AND btc.{cst.PARTICIPANT_HOTKEY} = tn.{cst.HOTKEY}
            LEFT JOIN {cst.SUBMISSIONS_TABLE} s ON btc.{cst.COPY_TASK_ID} = s.{cst.TASK_ID}
                AND btc.{cst.PARTICIPANT_HOTKEY} = s.{cst.HOTKEY}
            WHERE tn.{cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL
            ORDER BY brt.{cst.TASK_ID}, tn.{cst.TASK_NODE_QUALITY_SCORE} DESC
        """
        rows = await connection.fetch(query)

        # Group results by root task
        results_by_root = {}
        for row in rows:
            root_task_id = str(row["root_task_id"])
            if root_task_id not in results_by_root:
                results_by_root[root_task_id] = []

            results_by_root[root_task_id].append(
                {
                    "copy_task_id": str(row[cst.COPY_TASK_ID]),
                    "participant_hotkey": row[cst.PARTICIPANT_HOTKEY],
                    "tournament_id": row[cst.TOURNAMENT_ID],
                    "quality_score": normalise_float(row["quality_score"]),
                    "test_loss": normalise_float(row[cst.TEST_LOSS]),
                    "synth_loss": normalise_float(row[cst.SYNTH_LOSS]),
                    "repo": row[cst.REPO],
                    "completed_at": row[cst.COMPLETED_AT],
                    "created_at": row[cst.CREATED_AT],
                    "model_id": row[cst.MODEL_ID],
                    "dataset": row[cst.DS],
                    "task_type": row[cst.TASK_TYPE],
                }
            )

        logger.info(f"Retrieved benchmark results for {len(results_by_root)} root tasks")
        return results_by_root


async def get_weekly_task_participation_data(psql_db: PSQLDB) -> list[HotkeyTaskParticipation]:
    """
    Get weekly task participation proportions by type.
    Returns list of HotkeyTaskParticipation objects.
    """
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT
                tn.{cst.HOTKEY},
                t.{cst.TASK_TYPE},
                COUNT(*) as task_count
            FROM {cst.TASK_NODES_TABLE} tn
            JOIN {cst.TASKS_TABLE} t ON tn.{cst.TASK_ID} = t.{cst.TASK_ID}
            WHERE t.{cst.CREATED_AT} > NOW() - INTERVAL '7 days'
            AND tn.{cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL
            GROUP BY tn.{cst.HOTKEY}, t.{cst.TASK_TYPE}
        """
        results = await connection.fetch(query)

        # Aggregate task counts by hotkey and category
        hotkey_counts = {}
        for row in results:
            hotkey = row[cst.HOTKEY]
            task_type = row[cst.TASK_TYPE]
            count = row["task_count"]

            if hotkey not in hotkey_counts:
                hotkey_counts[hotkey] = {"text": 0, "image": 0}

            # Categorize task types (TEXT = DPO, GRPO, INSTRUCT; IMAGE = IMAGE)
            if task_type in [
                TaskType.DPOTASK.value,
                TaskType.GRPOTASK.value,
                TaskType.INSTRUCTTEXTTASK.value,
                TaskType.CHATTASK.value,
            ]:
                hotkey_counts[hotkey]["text"] += count
            elif task_type == TaskType.IMAGETASK.value:
                hotkey_counts[hotkey]["image"] += count

        # Convert to proper objects with proportions
        result = []
        for hotkey, counts in hotkey_counts.items():
            total_tasks = counts["text"] + counts["image"]
            if total_tasks > 0:
                result.append(
                    HotkeyTaskParticipation(
                        hotkey=hotkey,
                        text_task_proportion=counts["text"] / total_tasks,
                        image_task_proportion=counts["image"] / total_tasks,
                        total_tasks=total_tasks,
                    )
                )

        logger.info(f"Found weekly task participation for {len(result)} hotkeys over 7 days")
        return result


