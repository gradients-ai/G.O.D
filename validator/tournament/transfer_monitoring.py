"""
Transfer monitoring for TaoStats API integration
"""

import asyncio
from datetime import datetime
from typing import Optional

import requests

from validator.app.config import Config
from validator.transfers.models import TaoStatsTransferData
from validator.transfers.models import TaoStatsTransferResponse
from validator.transfers.models import TransferData
from validator.db.sql.transfers import get_or_create_coldkey_balance
from validator.db.sql.transfers import get_transfer_processing_state
from validator.db.sql.transfers import insert_transfer
from validator.db.sql.transfers import update_coldkey_balance
from validator.db.sql.transfers import update_transfer_processing_state
from core.logging import get_logger


logger = get_logger(__name__)

RAO_TO_TAO_DIVISOR = 1_000_000_000  # 1 TAO = 1,000,000,000 RAO


def rao_to_tao(rao_amount: int) -> float:
    """
    Convert RAO amount to TAO (for display purposes only)

    Args:
        rao_amount: Amount in RAO as integer

    Returns:
        float: Amount in TAO
    """
    return rao_amount / RAO_TO_TAO_DIVISOR


async def fetch_taostats_transfers(
    api_key: str,
    network: str = "finney",
    to_address: Optional[str] = None,
    from_address: Optional[str] = None,
    timestamp_start: Optional[int] = None,
    timestamp_end: Optional[int] = None,
    limit: int = 200,
    page: int = 1,
) -> TaoStatsTransferResponse:
    """
    Fetch transfer data from TaoStats API

    Args:
        api_key: TaoStats API key
        network: Network name (default: "finney")
        to_address: Recipient address filter
        from_address: Sender address filter
        timestamp_start: Start timestamp (Unix seconds)
        timestamp_end: End timestamp (Unix seconds)
        limit: Number of results per page (default: 200)
        page: Page number (default: 1)

    Returns:
        TaoStatsTransferResponse: Parsed response data
    """
    base_url = "https://api.taostats.io/api/transfer/v1"
    headers = {"Authorization": api_key, "accept": "application/json"}
    params = {"network": network, "limit": limit, "page": page}

    if to_address:
        params["to"] = to_address
    if from_address:
        params["from"] = from_address
    if timestamp_start:
        params["timestamp_start"] = timestamp_start
    if timestamp_end:
        params["timestamp_end"] = timestamp_end

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.get(base_url, headers=headers, params=params))
        response.raise_for_status()

        return TaoStatsTransferResponse.model_validate(response.json())

    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")
    except Exception as e:
        raise Exception(f"Failed to parse response: {e}")


def convert_taostats_to_internal(taostats_transfer: TaoStatsTransferData) -> TransferData:
    """
    Convert TaoStats transfer data to internal transfer data format

    Args:
        taostats_transfer: Transfer data from TaoStats API

    Returns:
        TransferData: Internal transfer data format
    """
    return TransferData(
        id=taostats_transfer.id,
        to_ss58=taostats_transfer.to.ss58,
        to_hex=taostats_transfer.to.hex,
        from_ss58=taostats_transfer.from_.ss58,
        from_hex=taostats_transfer.from_.hex,
        network=taostats_transfer.network,
        block_number=taostats_transfer.block_number,
        timestamp=datetime.fromisoformat(taostats_transfer.timestamp.replace("Z", "+00:00")),
        amount_rao=int(taostats_transfer.amount),  # Convert string to int
        fee_rao=int(taostats_transfer.fee),  # Convert string to int
        transaction_hash=taostats_transfer.transaction_hash,
        extrinsic_id=taostats_transfer.extrinsic_id,
    )


async def process_transfer_for_balance(psql_db, transfer: TransferData) -> None:
    """
    Process a transfer for balance tracking

    Args:
        psql_db: Database connection
        transfer: Transfer data to process
    """
    try:
        amount_rao = transfer.amount_rao

        sender_balance = await get_or_create_coldkey_balance(psql_db, transfer.from_ss58)
        if sender_balance:
            await update_coldkey_balance(
                psql_db,
                transfer.from_ss58,
                amount_rao,  # Amount sent to our target address in RAO
                transfer.timestamp,
            )
            amount_tao = rao_to_tao(amount_rao)
            logger.debug(
                f"Updated sender balance for {transfer.from_ss58}: +{amount_rao} RAO ({amount_tao:.9f} TAO) sent to target"
            )

    except Exception as e:
        logger.error(f"Failed to process transfer {transfer.id} for balance: {e}")


async def fetch_and_process_transfers(config: Config) -> None:
    """
    Fetch new transfers from TaoStats API and process them

    Args:
        config: Validator configuration
    """
    if not config.taostats_api_key or not config.transfer_target_address:
        logger.warning("Transfer monitoring not configured - missing API key or target address")
        return

    try:
        processing_state = await get_transfer_processing_state(config.psql_db)
        if not processing_state:
            logger.error("Failed to get transfer processing state")
            return

        logger.info(f"Starting transfer processing for address {config.transfer_target_address}")
        logger.info(f"Last processed: {processing_state.last_processed_timestamp}")

        timestamp_start = int(processing_state.last_processed_timestamp.timestamp())
        timestamp_end = int(datetime.now().timestamp())

        logger.info(f"Fetching transfers from {processing_state.last_processed_timestamp} to now")

        response = await fetch_taostats_transfers(
            api_key=config.taostats_api_key,
            network=config.transfer_network,
            to_address=config.transfer_target_address,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            limit=200,
        )

        logger.info(f"Fetched {len(response.data)} transfers from API")

        new_transfers_count = 0
        skipped_transfers_count = 0
        balance_updates_count = 0
        latest_timestamp = processing_state.last_processed_timestamp
        latest_block = processing_state.last_processed_block

        for taostats_transfer in response.data:
            transfer = convert_taostats_to_internal(taostats_transfer)

            was_inserted = await insert_transfer(config.psql_db, transfer)
            if was_inserted:
                new_transfers_count += 1
                logger.debug(
                    f"Inserted new transfer: {transfer.id} from {transfer.from_ss58[:20]}... for {transfer.amount_rao} RAO"
                )

                if transfer.to_ss58 == config.transfer_target_address:
                    await process_transfer_for_balance(config.psql_db, transfer)
                    balance_updates_count += 1
                    logger.debug(
                        f"Updated balance for sender {transfer.from_ss58[:20]}... (+{transfer.amount_rao} RAO sent to target)"
                    )
            else:
                skipped_transfers_count += 1
                logger.debug(f"Skipped existing transfer: {transfer.id}")

            if transfer.timestamp > latest_timestamp:
                latest_timestamp = transfer.timestamp
            if transfer.block_number > latest_block:
                latest_block = transfer.block_number

        await update_transfer_processing_state(
            config.psql_db,
            latest_timestamp,
            latest_block,
            config.transfer_target_address,
            config.transfer_network,
            config.transfer_processing_interval_hours,
        )

        total_rao_processed = sum(
            convert_taostats_to_internal(taostats_transfer).amount_rao
            for taostats_transfer in response.data
            if convert_taostats_to_internal(taostats_transfer).to_ss58 == config.transfer_target_address
        )
        total_rao_tao = rao_to_tao(total_rao_processed) if total_rao_processed > 0 else 0

        logger.info("Transfer processing completed successfully:")
        logger.info(f"  📊 API Response: {len(response.data)} transfers fetched")
        logger.info(f"  ✅ New transfers: {new_transfers_count}")
        logger.info(f"  ⏭️  Skipped (existing): {skipped_transfers_count}")
        logger.info(f"  💰 Balance updates: {balance_updates_count}")
        logger.info(f"  🪙 Total RAO to target: {total_rao_processed:,} RAO ({total_rao_tao:.6f} TAO)")
        logger.info(f"  🕐 Updated to timestamp: {latest_timestamp}")
        logger.info(f"  📦 Updated to block: {latest_block}")

        if balance_updates_count > 0:
            logger.info("📋 Sender addresses with transfers to target (for endpoint testing):")
            for taostats_transfer in response.data:
                transfer = convert_taostats_to_internal(taostats_transfer)
                if transfer.to_ss58 == config.transfer_target_address:
                    tao_amount = rao_to_tao(transfer.amount_rao)
                    logger.info(f"   💰 {transfer.from_ss58}: {transfer.amount_rao:,} RAO ({tao_amount:.6f} TAO)")

    except Exception as e:
        logger.error(f"Failed to fetch and process transfers: {e}")


async def transfer_monitoring_cycle(config: Config):
    """
    Continuous transfer monitoring cycle

    Args:
        config: Validator configuration
    """
    while True:
        try:
            await fetch_and_process_transfers(config)
        except Exception as e:
            logger.error(f"Error in transfer monitoring cycle: {e}")

        await asyncio.sleep(300)  # 5 minutes


async def main():
    """
    Main function for standalone transfer monitoring check
    """
    from validator.app.config import load_config
    from validator.tasks.details import try_db_connections

    logger.info("🚀 Starting standalone transfer monitoring check...")

    # Load configuration
    config = load_config()

    await try_db_connections(config)

    if not config.taostats_api_key:
        logger.error("❌ TAOSTATS_API_KEY not configured in environment")
        return

    if not config.transfer_target_address:
        logger.error("❌ TRANSFER_TARGET_ADDRESS not configured in environment")
        return

    logger.info("✅ Configuration loaded:")
    logger.info(f"   📡 API Key: {'*' * 20}{config.taostats_api_key[-10:] if config.taostats_api_key else 'None'}")
    logger.info(f"   🎯 Target Address: {config.transfer_target_address}")
    logger.info(f"   🌐 Network: {config.transfer_network}")
    logger.info(f"   ⏰ Processing Interval: {config.transfer_processing_interval_hours} hours")

    try:
        logger.info("🔄 Running transfer monitoring check...")
        await fetch_and_process_transfers(config)
        logger.info("✅ Transfer monitoring check completed successfully!")

    except Exception as e:
        logger.error(f"❌ Transfer monitoring check failed: {e}")
        raise

    finally:
        await config.psql_db.close()
        await config.redis_db.close()
        logger.info("🔌 Database connections closed")


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv(os.getenv("ENV_FILE", ".vali.env"), override=True)

    asyncio.run(main())
