import asyncio

from validator.tournament.models import TournamentType
from validator.app.config import load_config
from validator.tournament.tournament_manager import create_basic_tournament
from core.logging import get_logger


logger = get_logger(__name__)


async def main():
    logger.info("Starting tournament...")

    config = load_config()

    await config.psql_db.connect()
    logger.info("Connected to database")

    try:
        logger.info("Creating basic tournament...")
        tournament_id = await create_basic_tournament(TournamentType.IMAGE, config.psql_db, config)
        logger.info(f"Created basic tournament: {tournament_id}")

    except Exception as e:
        logger.error(f"Error starting tournament: {e}")
        raise
    finally:
        await config.psql_db.close()


if __name__ == "__main__":
    asyncio.run(main())
