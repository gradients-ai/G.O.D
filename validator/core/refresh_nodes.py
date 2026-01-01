"""
Gets the latest nodes from the network and stores them in the database,
migrating the old nodes to history in the process
"""

import asyncio
from datetime import datetime
from datetime import timedelta

from fiber.chain import fetch_nodes
from fiber.chain.models import Node

from validator.core.config import Config
from validator.db import constants as cst
from validator.db.sql.nodes import get_all_nodes
from validator.db.sql.nodes import get_last_updated_time_for_nodes
from validator.db.sql.nodes import insert_nodes
from validator.db.sql.nodes import migrate_nodes_to_history
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def _fetch_nodes_from_substrate(config: Config) -> list[Node]:
    # It won't cause issues with substrate, as it creates a new connection.
    return await asyncio.to_thread(fetch_nodes.get_nodes_for_netuid, config.substrate, config.netuid)


async def _is_recent_update(config: Config) -> bool:
    async with await config.psql_db.connection() as connection:
        last_updated_time = await get_last_updated_time_for_nodes(connection)
        if last_updated_time is not None and datetime.now() - last_updated_time.replace(tzinfo=None) < timedelta(minutes=30):
            logger.info(
                f"Last update for nodes table was at {last_updated_time}, which is less than 30 minutes ago - skipping refresh"
            )
            return True
        return False


async def _get_and_store_nodes(config: Config) -> list[Node]:
    try:
        async with config.psql_db.pool.acquire(timeout=cst.TIMEOUT) as conn:
            await conn.execute("SELECT 1")
    except Exception as e:
        logger.warning(f"DB pool not ready, reconnecting... {e}")
    if await _is_recent_update(config):
        nodes = await get_all_nodes(config.psql_db)

    logger.info("At fetch")
    raw_nodes = await _fetch_nodes_from_substrate(config)
    logger.info("afer  fetch")
    nodes = [Node(**node.model_dump(mode="json")) for node in raw_nodes]
    logger.info("afer  nodes")

    async with await config.psql_db.connection() as connection:
        logger.info("connection made")
        await migrate_nodes_to_history(connection)
        logger.info("after migrate")
        await insert_nodes(connection, nodes)
        logger.info("after insret")

    logger.info(f"Stored {len(nodes)} nodes.")
    return nodes


async def refresh_nodes_periodically(config: Config) -> None:
    while True:
        try:
            logger.info("Attempting to refresh nodes with the metagraph")
            # 1 minute timeout
            await asyncio.wait_for(_get_and_store_nodes(config), timeout=5 * 60)
            logger.info("Node refresh cycle complete! Waiting 15 minutes before next refresh...")
            await asyncio.sleep(60 * 15)  # 15 minutes
        except asyncio.TimeoutError:
            logger.error("Node refresh timed out after 5 minutes... :( Please look into this!!")
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error refreshing nodes: {e}", exc_info=True)
            await asyncio.sleep(60)  # Wait 1 minute before retrying on error
