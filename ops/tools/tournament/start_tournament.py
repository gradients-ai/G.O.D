"""Manually create a basic tournament of the given type.

Usage:
    python -m ops.tools.tournament.start_tournament <text|image|quant>
"""

import argparse
import asyncio

from core.logging import get_logger
from validator.app.config import load_config
from validator.tournament.models import TournamentType
from validator.tournament.tournament_manager import create_basic_tournament


logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a basic tournament (no participants or rounds yet).")
    choices = [t.value for t in TournamentType if t != TournamentType.ENVIRONMENT]
    parser.add_argument("tournament_type", choices=choices, help="Tournament type")
    args = parser.parse_args(argv)
    args.tournament_type = TournamentType(args.tournament_type)
    return args


async def main(tournament_type: TournamentType):
    logger.info(f"Starting {tournament_type.value} tournament...")

    config = load_config()

    await config.psql_db.connect()
    logger.info("Connected to database")

    try:
        logger.info("Creating basic tournament...")
        tournament_id = await create_basic_tournament(tournament_type, config.psql_db, config)
        logger.info(f"Created basic {tournament_type.value} tournament: {tournament_id}")

    except Exception as e:
        logger.error(f"Error starting tournament: {e}")
        raise
    finally:
        await config.psql_db.close()


if __name__ == "__main__":
    asyncio.run(main(parse_args().tournament_type))
