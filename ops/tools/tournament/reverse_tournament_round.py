#!/usr/bin/env python3
"""
Script to reverse a tournament round by deleting it and setting the previous round back to active.

Usage:
    python reverse_tournament_round.py <tournament_id> <round_id_to_delete>

This script will:
1. Delete the specified round and all its associated data
2. Set the previous round to 'completed' status
3. Un-eliminate participants who were eliminated in the deleted round
4. Set tournament status back to 'active' if it was completed
5. Allow the tournament cycle to re-run the advancement logic

Example:
    python reverse_tournament_round.py tourn_f4eb788ad3d66f60_20250723 tourn_f4eb788ad3d66f60_20250723_round_002
"""

import asyncio
import os
import sys
from pathlib import Path

from validator.tournament.models import RoundStatus
from validator.tournament.models import TournamentStatus
from validator.db.database import PSQLDB
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_tasks
from validator.db.sql.tournaments import update_round_status
from validator.db.sql.tournaments import update_tournament_status
from core.logging import get_logger


logger = get_logger(__name__)


def load_database_url_from_env_file() -> str:
    env_file = Path(".vali.env")
    if not env_file.exists():
        raise FileNotFoundError(".vali.env file not found")

    database_url = None
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("DATABASE_URL="):
                database_url = line.split("=", 1)[1].strip("\"'")
                break

    if not database_url:
        raise ValueError("DATABASE_URL not found in .vali.env file")

    return database_url


async def reverse_tournament_round(tournament_id: str, round_id_to_delete: str, psql_db: PSQLDB) -> bool:
    """
    Reverse a tournament round by deleting it and resetting the previous round.

    Args:
        tournament_id: The ID of the tournament
        round_id_to_delete: The round ID to delete
        psql_db: Database connection

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get tournament
        tournament = await get_tournament(tournament_id, psql_db)
        if not tournament:
            logger.error(f"Tournament {tournament_id} not found")
            return False

        logger.info(f"Found tournament: {tournament_id} (status: {tournament.status})")

        # Get all rounds
        rounds = await get_tournament_rounds(tournament_id, psql_db)
        logger.info(f"Found {len(rounds)} rounds for tournament {tournament_id}")

        # Find the round to delete
        round_to_delete = None
        previous_round = None
        for i, round_data in enumerate(rounds):
            if round_data.round_id == round_id_to_delete:
                round_to_delete = round_data
                if i > 0:
                    previous_round = rounds[i - 1]
                break

        if not round_to_delete:
            logger.error(f"Round {round_id_to_delete} not found in tournament {tournament_id}")
            return False

        logger.info(f"Deleting round: {round_id_to_delete} (round number: {round_to_delete.round_number})")

        # Start transaction
        async with await psql_db.connection() as connection:
            async with connection.transaction():
                # 1. Delete all tasks for the round to be deleted
                round_tasks = await get_tournament_tasks(round_id_to_delete, psql_db)
                logger.info(f"Found {len(round_tasks)} tasks to delete")

                # Delete the actual tasks from the tasks table
                # This will automatically cascade to delete from tournament_task_hotkey_trainings
                # and tournament_tasks due to ON DELETE CASCADE constraints
                for task in round_tasks:
                    query = """
                        DELETE FROM tasks
                        WHERE task_id = $1
                    """
                    await connection.execute(query, task.task_id)
                    logger.info(f"Deleted task {task.task_id} from tasks table (cascaded to related tables)")

                # 2. Delete pairs if it's a knockout round
                query = """
                    DELETE FROM tournament_pairs
                    WHERE round_id = $1
                """
                await connection.execute(query, round_id_to_delete)

                # 3. Delete group members and groups if it's a group round
                query = """
                    DELETE FROM tournament_group_members
                    WHERE group_id IN (
                        SELECT group_id FROM tournament_groups WHERE round_id = $1
                    )
                """
                await connection.execute(query, round_id_to_delete)

                query = """
                    DELETE FROM tournament_groups
                    WHERE round_id = $1
                """
                await connection.execute(query, round_id_to_delete)

                # 4. Un-eliminate participants who were eliminated in this round
                query = """
                    UPDATE tournament_participants
                    SET eliminated_in_round_id = NULL
                    WHERE tournament_id = $1 AND eliminated_in_round_id = $2
                """
                await connection.execute(query, tournament_id, round_id_to_delete)

                # 5. Delete the round
                query = """
                    DELETE FROM tournament_rounds
                    WHERE round_id = $1
                """
                await connection.execute(query, round_id_to_delete)

                # 6. Set previous round to active (so cycle can check completion and advance)
                if previous_round:
                    await update_round_status(previous_round.round_id, RoundStatus.ACTIVE, psql_db)
                    logger.info(f"Set previous round {previous_round.round_id} to active status")

                # 7. Set tournament back to active if it was completed
                if tournament.status == TournamentStatus.COMPLETED:
                    await update_tournament_status(tournament_id, TournamentStatus.ACTIVE, psql_db)

                    # Clear winner if set
                    query = """
                        UPDATE tournaments
                        SET winner_hotkey = NULL
                        WHERE tournament_id = $1
                    """
                    await connection.execute(query, tournament_id)

        logger.info(f"Successfully reversed round {round_id_to_delete}")
        return True

    except Exception as e:
        logger.error(f"Error reversing tournament round: {str(e)}")
        return False


async def main():
    """Main function to handle command line arguments and execute reversal."""
    if len(sys.argv) != 3:
        print("Usage: python reverse_tournament_round.py <tournament_id> <round_id_to_delete>")
        print(
            "Example: python reverse_tournament_round.py tourn_f4eb788ad3d66f60_20250723 "
            "tourn_f4eb788ad3d66f60_20250723_round_002"
        )
        sys.exit(1)

    tournament_id = sys.argv[1]
    round_id_to_delete = sys.argv[2]

    try:
        # Load database URL from .vali.env
        database_url = load_database_url_from_env_file()
        print("Using database URL from .vali.env")

        # Set environment variable for PSQLDB
        os.environ["DATABASE_URL"] = database_url

        # Initialize database connection
        psql_db = PSQLDB()

        # Establish database connection
        await psql_db.connect()

        success = await reverse_tournament_round(tournament_id, round_id_to_delete, psql_db)
        if success:
            print(f"✅ Successfully reversed round {round_id_to_delete} in tournament {tournament_id}")
            print("🔄 The tournament cycle will now re-run the advancement logic")
            sys.exit(0)
        else:
            print("❌ Failed to reverse tournament round")
            sys.exit(1)
    except FileNotFoundError:
        print("❌ .vali.env file not found")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        if "psql_db" in locals():
            await psql_db.close()


if __name__ == "__main__":
    asyncio.run(main())
