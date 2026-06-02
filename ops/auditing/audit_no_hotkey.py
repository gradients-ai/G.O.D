from dotenv import load_dotenv


load_dotenv(".vali.env")

import asyncio
import os
import time
from dataclasses import dataclass

import httpx
import websocket
from fiber.chain import interface
from substrateinterface import SubstrateInterface

from ops.auditing.audit import audit_weights
from validator.db.database import PSQLDB
from core.logging import get_logger


logger = get_logger(__name__)

# Similar config, but stripped of any hotkey stuff


@dataclass
class Config:
    substrate: SubstrateInterface
    psql_db: PSQLDB
    subtensor_network: str | None
    subtensor_address: str | None
    netuid: int
    httpx_client: httpx.AsyncClient
    debug: bool = os.getenv("ENV", "prod").lower() != "prod"


def load_mini_config() -> Config:
    subtensor_network = os.getenv("SUBTENSOR_NETWORK")
    subtensor_address = os.getenv("SUBTENSOR_ADDRESS") or None
    dev_env = os.getenv("ENV", "prod").lower() != "prod"
    netuid = os.getenv("NETUID")
    if netuid is None:
        netuid = 201 if subtensor_network == "test" else 69420
        logger.warning(f"NETUID not set, using {netuid}")
    else:
        netuid = int(netuid)

    try:
        substrate = interface.get_substrate(subtensor_network=subtensor_network, subtensor_address=subtensor_address)
    except websocket._exceptions.WebSocketBadStatusException as e:
        logger.error(f"Failed to get substrate: {e}. Sleeping for 20 seconds and then trying again...")
        time.sleep(20)
        substrate = interface.get_substrate(subtensor_network=subtensor_network, subtensor_address=subtensor_address)

    httpx_limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
    httpx_client = httpx.AsyncClient(limits=httpx_limits)

    _config = Config(
        substrate=substrate,
        psql_db=PSQLDB(),
        subtensor_network=subtensor_network,
        subtensor_address=subtensor_address,
        netuid=netuid,
        httpx_client=httpx_client,
        debug=dev_env,
    )
    return _config


async def main():
    config = load_mini_config()

    success = await audit_weights(config, set_weights_on_chain=False)

    logger.info(f"Audit weights success: {success}")


if __name__ == "__main__":
    asyncio.run(main())
