from dotenv import load_dotenv


load_dotenv(".vali.env")

import os
import time
from dataclasses import dataclass

import httpx
import websocket
from fiber.chain import chain_utils
from fiber.chain import interface
from redis.asyncio import Redis
from substrateinterface import Keypair
from substrateinterface import SubstrateInterface

from validator.db.database import PSQLDB
from core.logging import get_logger


logger = get_logger(__name__)


@dataclass
class Config:
    substrate: SubstrateInterface
    keypair: Keypair
    psql_db: PSQLDB
    redis_db: Redis
    subtensor_network: str | None
    subtensor_address: str | None
    netuid: int
    refresh_nodes: bool
    httpx_client: httpx.AsyncClient
    set_metagraph_weights_with_high_updated_to_not_dereg: bool
    github_token: str | None = None
    github_username: str | None = None
    discord_url: str | None = None
    testnet: bool = os.getenv("SUBTENSOR_NETWORK", "").lower() == "test"
    debug: bool = os.getenv("ENV", "prod").lower() != "prod"
    taostats_api_key: str | None = None
    transfer_target_address: str | None = None
    transfer_network: str | None = None
    transfer_processing_interval_hours: int = 24


_config = None


def load_config() -> Config:
    global _config
    if _config is None:
        subtensor_network = os.getenv("SUBTENSOR_NETWORK")
        subtensor_address = os.getenv("SUBTENSOR_ADDRESS") or None
        dev_env = os.getenv("ENV", "prod").lower() != "prod"
        wallet_name = os.getenv("WALLET_NAME", "default")
        hotkey_name = os.getenv("HOTKEY_NAME", "default")
        netuid = os.getenv("NETUID")
        if netuid is None:
            netuid = 201 if subtensor_network == "test" else 69420
            logger.warning(f"NETUID not set, using {netuid}")
        else:
            netuid = int(netuid)

        redis_host = "localhost"

        refresh_nodes: bool = os.getenv("REFRESH_NODES", "true").lower() == "true"
        if refresh_nodes:
            try:
                substrate = interface.get_substrate(subtensor_network=subtensor_network, subtensor_address=subtensor_address)
            except websocket._exceptions.WebSocketBadStatusException as e:
                logger.error(f"Failed to get substrate: {e}. Sleeping for 20 seconds and then trying again...")
                time.sleep(20)
                substrate = interface.get_substrate(subtensor_network=subtensor_network, subtensor_address=subtensor_address)
        else:
            # this is only used for testing
            substrate = None
        keypair = chain_utils.load_hotkey_keypair(wallet_name=wallet_name, hotkey_name=hotkey_name)
        logger.info(f"This is my own keypair {keypair}")

        # Debug logging for keypair
        logger.info(f"🔍 CONFIG DEBUG: keypair type: {type(keypair)}")
        logger.info(f"🔍 CONFIG DEBUG: keypair module: {keypair.__class__.__module__}")
        logger.info(f"🔍 CONFIG DEBUG: keypair class: {keypair.__class__.__name__}")
        if hasattr(keypair, "_mock_name"):
            logger.error(f"🚨 CONFIG: Keypair is a Mock object: {keypair}")
        if hasattr(keypair, "ss58_address"):
            logger.info(f"🔍 CONFIG DEBUG: ss58_address type: {type(keypair.ss58_address)}")
            if hasattr(keypair.ss58_address, "_mock_name"):
                logger.error(f"🚨 CONFIG: ss58_address is a Mock object: {keypair.ss58_address}")

        httpx_limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
        httpx_client = httpx.AsyncClient(limits=httpx_limits)

        set_metagraph_weights_with_high_updated_to_not_dereg = bool(
            os.getenv("SET_METAGRAPH_WEIGHTS_WITH_HIGH_UPDATED_TO_NOT_DEREG", "false").lower() == "true"
        )

        github_token = os.getenv("GITHUB_TOKEN")
        github_username = os.getenv("GITHUB_USERNAME")
        discord_url = os.getenv("DISCORD_WEBHOOK", None)

        taostats_api_key = os.getenv("TAOSTATS_API_KEY")
        transfer_target_address = os.getenv("TRANSFER_TARGET_ADDRESS")
        transfer_network = os.getenv("TRANSFER_NETWORK", "finney")
        transfer_processing_interval_hours = int(os.getenv("TRANSFER_PROCESSING_INTERVAL_HOURS", "24"))

        _config = Config(
            substrate=substrate,
            keypair=keypair,
            psql_db=PSQLDB(),
            redis_db=Redis(host=redis_host),
            subtensor_network=subtensor_network,
            subtensor_address=subtensor_address,
            netuid=netuid,
            refresh_nodes=refresh_nodes,
            httpx_client=httpx_client,
            debug=dev_env,
            set_metagraph_weights_with_high_updated_to_not_dereg=set_metagraph_weights_with_high_updated_to_not_dereg,
            github_token=github_token,
            github_username=github_username,
            discord_url=discord_url,
            taostats_api_key=taostats_api_key,
            transfer_target_address=transfer_target_address,
            transfer_network=transfer_network,
            transfer_processing_interval_hours=transfer_processing_interval_hours,
        )
    return _config
