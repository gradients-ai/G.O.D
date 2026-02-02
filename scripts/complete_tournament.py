#!/usr/bin/env python3
"""
Script to manually complete a tournament after it has finished.

Usage:
    python complete_tournament.py <tournament_id>

This script will:
1. Verify the tournament exists and has a winner_hotkey
2. Check that all rounds are completed (optional warning)
3. Set tournament status to 'completed'

Note: This script should be used when a tournament has finished but the status
hasn't been automatically updated to 'completed' in the database.
"""

import asyncio
import os
import sys
from pathlib import Path

from core.models.tournament_models import RoundStatus
from core.models.tournament_models import TournamentStatus
from validator.db.database import PSQLDB
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import update_tournament_status
from validator.utils.logging import get_logger


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


async def complete_tournament(tournament_id: str, psql_db: PSQLDB) -> bool:
    """
    Manually complete a tournament by setting its status to 'completed'.

    Args:
        tournament_id: The ID of the tournament to complete
        psql_db: Database connection

    Returns:
        True if successful, False otherwise
    """
    try:
        tournament = await get_tournament(tournament_id, psql_db)
        if not tournament:
            logger.error(f"Tournament {tournament_id} not found")
            return False

        logger.info(f"Found tournament: {tournament_id}")
        logger.info(f"Current status: {tournament.status}")
        logger.info(f"Tournament type: {tournament.tournament_type.value}")
        logger.info(f"Winner hotkey: {tournament.winner_hotkey}")
        logger.info(f"Base winner hotkey: {tournament.base_winner_hotkey}")

        # Check if tournament already completed
        if tournament.status == TournamentStatus.COMPLETED:
            logger.warning(f"Tournament {tournament_id} is already marked as completed")
            return True

        # Verify that tournament has a winner
        if not tournament.winner_hotkey:
            logger.error(
                f"Tournament {tournament_id} does not have a winner_hotkey. "
                f"Cannot complete tournament without a winner. "
                f"Please ensure the tournament has finished and a winner has been determined."
            )
            return False

        # Check round statuses (informational)
        rounds = await get_tournament_rounds(tournament_id, psql_db)
        logger.info(f"Found {len(rounds)} rounds for tournament {tournament_id}")

        incomplete_rounds = [r for r in rounds if r.status != RoundStatus.COMPLETED]
        if incomplete_rounds:
            logger.warning(
                f"Found {len(incomplete_rounds)} rounds that are not completed: "
                f"{[r.round_id for r in incomplete_rounds]}"
            )
            logger.warning("Proceeding with tournament completion anyway...")

        # Update tournament status to completed
        await update_tournament_status(tournament_id, TournamentStatus.COMPLETED, psql_db)
        logger.info(f"Set tournament {tournament_id} status to completed")

        logger.info(f"Successfully completed tournament {tournament_id}")
        return True

    except Exception as e:
        logger.error(f"Error completing tournament {tournament_id}: {str(e)}")
        return False


async def main():
    """Main function to handle command line arguments and execute completion."""
    if len(sys.argv) != 2:
        print("Usage: python complete_tournament.py <tournament_id>")
        print("Example: python complete_tournament.py tourn_abc123_20250101")
        sys.exit(1)

    tournament_id = sys.argv[1]

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

        success = await complete_tournament(tournament_id, psql_db)
        if success:
            print(f"✅ Tournament {tournament_id} completed successfully")
            sys.exit(0)
        else:
            print(f"❌ Failed to complete tournament {tournament_id}")
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







































