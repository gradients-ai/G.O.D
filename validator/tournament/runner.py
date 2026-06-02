import asyncio

from validator.app.config import load_config
from validator.tournament.tournament_manager import process_active_tournaments
from validator.tournament.tournament_manager import process_pending_rounds
from validator.tournament.tournament_manager import process_pending_tournaments
from validator.tournament.tournament_manager import process_tournament_scheduling
from validator.tournament.transfer_monitoring import transfer_monitoring_cycle
from core.logging import get_logger
from validator.tasks.details import try_db_connections


logger = get_logger(__name__)


async def cycle():
    config = load_config()

    await try_db_connections(config)

    await asyncio.gather(
        # this monitors TAO transfers and updates coldkey balances
        transfer_monitoring_cycle(config),
        process_pending_tournaments(config),
        # this processes pending rounds by creating tasks and assigning nodes
        process_pending_rounds(config),
        # this advances the tournament till completion
        process_active_tournaments(config),
        # this auto-creates tournaments on schedule
        process_tournament_scheduling(config),
    )


if __name__ == "__main__":
    asyncio.run(cycle())
