"""
Gets the latest nodes from the network and stores them in the database,
migrating the old nodes to history in the process
"""

import asyncio
import multiprocessing as mp
import queue as queue_lib
from datetime import datetime
from datetime import timedelta

from fiber.chain.models import Node

from validator.core.config import Config
from validator.db import constants as cst
from validator.db.sql.nodes import get_all_nodes
from validator.db.sql.nodes import get_last_updated_time_for_nodes
from validator.db.sql.nodes import insert_nodes
from validator.db.sql.nodes import migrate_nodes_to_history
from validator.utils.logging import get_logger


logger = get_logger(__name__)

# Hard timeout for the isolated fetch subprocess. Kept below the outer
# asyncio.wait_for backstop in refresh_nodes_periodically so it always fires first.
SUBSTRATE_FETCH_TIMEOUT_SECS = 120


def _fetch_nodes_subprocess(url: str, netuid: int, result_queue: "mp.Queue") -> None:
    """Fetch nodes inside a short-lived child process.

    Substrate-interface blocking I/O cannot be cancelled by asyncio, and a hung
    connection holds process-global locks that poison every later attempt in the
    long-lived validator_cycle process. Running it in an isolated child process
    means a hang can be killed cleanly via terminate()/kill() without affecting
    the parent. Imports happen here so the spawned interpreter stays minimal.
    """
    try:
        from fiber.chain import fetch_nodes
        from fiber.chain import interface

        substrate = interface.get_substrate(subtensor_address=url)
        try:
            nodes = fetch_nodes._get_nodes_for_uid(substrate, netuid)
            result_queue.put(("ok", [node.model_dump(mode="json") for node in nodes]))
        finally:
            try:
                substrate.close()
            except Exception:
                pass
    except Exception as e:
        result_queue.put(("error", f"{type(e).__name__}: {e}"))


def _terminate_process(proc: "mp.Process") -> None:
    """Ensure the child process is fully gone (escalating terminate -> kill)."""
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            logger.error(f"Node fetch subprocess (pid={proc.pid}) did not terminate; killing")
            proc.kill()
            proc.join(5)
    else:
        proc.join(5)


def _run_fetch_in_subprocess(url: str, netuid: int) -> list[Node]:
    # spawn (not fork) gives a fresh interpreter with no inherited locks,
    # event loop, or DB connections from the parent.
    ctx = mp.get_context("spawn")
    result_queue: "mp.Queue" = ctx.Queue()
    proc = ctx.Process(
        target=_fetch_nodes_subprocess,
        args=(url, netuid, result_queue),
        daemon=True,
    )
    proc.start()
    try:
        try:
            status, payload = result_queue.get(timeout=SUBSTRATE_FETCH_TIMEOUT_SECS)
        except queue_lib.Empty:
            raise TimeoutError(f"Node fetch subprocess timed out after {SUBSTRATE_FETCH_TIMEOUT_SECS}s")
    finally:
        _terminate_process(proc)

    if status == "error":
        raise RuntimeError(f"Node fetch subprocess failed: {payload}")

    return [Node(**node) for node in payload]


async def _fetch_nodes_from_substrate(config: Config) -> list[Node]:
    loop = asyncio.get_running_loop()
    # The blocking helper has its own internal timeout and always returns,
    # so the worker thread is never leaked.
    return await loop.run_in_executor(None, _run_fetch_in_subprocess, config.substrate.url, config.netuid)


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
        return await get_all_nodes(config.psql_db)

    logger.info("Fetching nodes from substrate")
    raw_nodes = await _fetch_nodes_from_substrate(config)
    nodes = [Node(**node.model_dump(mode="json")) for node in raw_nodes]

    async with await config.psql_db.connection() as connection:
        async with connection.transaction():
            await migrate_nodes_to_history(connection)
            await insert_nodes(connection, nodes)

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
