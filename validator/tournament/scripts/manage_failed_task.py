import asyncio
import sys
from uuid import UUID

from validator.core.config import load_config
from validator.db import constants as cst
from validator.db.database import PSQLDB
from validator.utils.logging import get_logger

logger = get_logger(__name__)


async def replace_task(task_id: str, psql_db: PSQLDB) -> None:
    """Set task status to PREP_TASK_FAILURE to trigger automatic replacement."""
    try:
        # Validate task_id is a valid UUID
        UUID(task_id)
    except ValueError:
        logger.error(f"Invalid task_id format: {task_id}. Must be a valid UUID.")
        return

    async with await psql_db.connection() as connection:
        # Update task status to PREP_TASK_FAILURE
        update_query = f"""
            UPDATE {cst.TASKS_TABLE}
            SET {cst.STATUS} = $1, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TASK_ID} = $2
        """
        result = await connection.execute(update_query, "prep_task_failure", task_id)
        
        # asyncpg returns string like "UPDATE 1" or "UPDATE 0" - check if any rows were affected
        rows_affected = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
        if rows_affected == 0:
            logger.error(f"Task {task_id} not found in database.")
            return
        
        logger.info(f"Successfully set task {task_id} status to prep_task_failure. Replacement will be created automatically.")


async def retry_task(task_id: str, psql_db: PSQLDB) -> None:
    """Reset task to training status and reset all training attempts."""
    try:
        # Validate task_id is a valid UUID
        UUID(task_id)
    except ValueError:
        logger.error(f"Invalid task_id format: {task_id}. Must be a valid UUID.")
        return

    async with await psql_db.connection() as connection:
        # Update task status to TRAINING
        update_task_query = f"""
            UPDATE {cst.TASKS_TABLE}
            SET {cst.STATUS} = $1, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TASK_ID} = $2
        """
        result = await connection.execute(update_task_query, "training", task_id)
        
        # asyncpg returns string like "UPDATE 1" or "UPDATE 0" - check if any rows were affected
        rows_affected = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
        if rows_affected == 0:
            logger.error(f"Task {task_id} not found in database.")
            return
        
        logger.info(f"Successfully set task {task_id} status to training.")

        # Reset all training attempts in tournament_task_hotkey_trainings table
        update_trainings_query = f"""
            UPDATE {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
            SET {cst.TRAINING_STATUS} = $1,
                {cst.N_TRAINING_ATTEMPTS} = $2,
                {cst.UPDATED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.TASK_ID} = $3
        """
        await connection.execute(update_trainings_query, "pending", 0, task_id)
        logger.info(f"Successfully reset training attempts for task {task_id}.")


async def allow_task(task_id: str, psql_db: PSQLDB) -> None:
    """Set round status to COMPLETED to allow tournament to proceed despite task failure."""
    try:
        # Validate task_id is a valid UUID
        UUID(task_id)
    except ValueError:
        logger.error(f"Invalid task_id format: {task_id}. Must be a valid UUID.")
        return

    async with await psql_db.connection() as connection:
        # First, get the round_id for this task
        get_round_query = f"""
            SELECT {cst.ROUND_ID}
            FROM {cst.TOURNAMENT_TASKS_TABLE}
            WHERE {cst.TASK_ID} = $1
            LIMIT 1
        """
        round_id = await connection.fetchval(get_round_query, task_id)
        
        if not round_id:
            logger.error(f"Task {task_id} not found in tournament_tasks table.")
            return
        
        # Update round status to COMPLETED (allows tournament to proceed)
        # Winner determination will handle missing results gracefully
        update_round_query = f"""
            UPDATE {cst.TOURNAMENT_ROUNDS_TABLE}
            SET {cst.ROUND_STATUS} = $1, {cst.COMPLETED_AT} = CURRENT_TIMESTAMP
            WHERE {cst.ROUND_ID} = $2
        """
        result = await connection.execute(update_round_query, "completed", round_id)
        
        # asyncpg returns string like "UPDATE 1" or "UPDATE 0" - check if any rows were affected
        rows_affected = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
        if rows_affected == 0:
            logger.error(f"Round {round_id} not found in database.")
            return
        
        logger.info(
            f"Successfully set round {round_id} status to completed. "
            f"Tournament will proceed - winner determination will handle missing results."
        )


async def main():
    if len(sys.argv) != 3:
        print("Usage: python manage_tournament_task.py <task_id> <action>")
        print("Actions: replace, retry, allow")
        print("  replace - Set task status to prep_task_failure (triggers automatic replacement)")
        print("  retry   - Reset task to training status and reset all training attempts")
        print("  allow   - Set round status to completed (allows tournament to proceed despite task failure)")
        sys.exit(1)

    task_id = sys.argv[1]
    action = sys.argv[2].lower()

    if action not in ["replace", "retry", "allow"]:
        logger.error(f"Invalid action: {action}. Must be 'replace', 'retry', or 'allow'.")
        sys.exit(1)

    # Initialize database connection
    config = load_config()
    psql_db = config.psql_db

    try:
        # Connect to database
        await psql_db.connect()
        
        if action == "replace":
            await replace_task(task_id, psql_db)
        elif action == "retry":
            await retry_task(task_id, psql_db)
        elif action == "allow":
            await allow_task(task_id, psql_db)
    except Exception as e:
        logger.error(f"Error managing task {task_id}: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await psql_db.close()


if __name__ == "__main__":
    asyncio.run(main())

