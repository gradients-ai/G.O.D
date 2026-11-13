import asyncio
import math
import os
from datetime import datetime

from dotenv import load_dotenv

from validator.db.sql.auditing import store_latest_scores_url
from validator.db.sql.tournaments import get_active_tournament_participants
from validator.db.sql.tournaments import get_boss_rank_comparisons
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_full_results
from validator.db.sql.tournaments import get_tournament_participants
from validator.db.sql.tournaments import get_weekly_task_participation_data
from validator.scoring.tournaments import get_quant_tournament_weights_from_data
from validator.scoring.tournaments import get_tournament_weights_from_data
from validator.tournament.models import BossRankComparison
from validator.tournament.models import NodeWeightsResult
from validator.tournament.models import TournamentAuditData
from validator.tournament.models import TournamentBurnData
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentResults
from validator.tournament.models import TournamentResultsWithWinners
from validator.tournament.models import TournamentType


load_dotenv(os.getenv("ENV_FILE", ".vali.env"))

import json
from uuid import UUID

from async_substrate_interface import SubstrateInterface
from fiber.chain import fetch_nodes
from fiber.chain import weights
from fiber.chain.chain_utils import query_substrate
from fiber.chain.models import Node

import core.constants.network as ccst
import validator.scoring.constants as cts
from core.constants.credentials import BUCKET_NAME
from core.logging import get_logger
from validator.app.config import Config
from validator.app.config import load_config
from validator.db.sql.nodes import get_vali_node_id
from validator.scoring.emission_balance import get_dynamic_base_weights
from validator.scoring.emission_balance import max_weight_by_type
from validator.tasks.details import save_json_to_temp_file
from validator.tasks.details import try_db_connections
from validator.tasks.details import upload_file_to_minio


logger = get_logger(__name__)


async def _upload_results_to_s3(config: Config, tournament_audit_data: TournamentAuditData) -> None:
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, UUID):
                return str(obj)
            return super().default(obj)

    upload_data = {
        "tournament_audit_data": tournament_audit_data.model_dump(),
    }

    scores_json = json.dumps(upload_data, indent=2, cls=DateTimeEncoder)

    temp_file, _ = await save_json_to_temp_file(scores_json, "latest_scores", dump_json=False)
    datetime_of_upload = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    presigned_url = await upload_file_to_minio(temp_file, BUCKET_NAME, f"latest_scores_{datetime_of_upload}.json")
    os.remove(temp_file)
    await store_latest_scores_url(presigned_url, config)
    return presigned_url


async def get_tournament_burn_details(psql_db) -> TournamentBurnData:
    """
    Build the per-type tournament pools.

    Each type's pool is its participation-driven base share (guaranteed minimum plus flex). The bases sum to
    exactly 1.0, so the pools fill the whole emission pie and nothing is renormalised. The burn proportion of a
    type is the fraction of its cap (minimum plus the whole flex pool) its pool leaves unused.

    Returns:
        TournamentBurnData with the pool of each type and the burn proportions
    """
    logger.info("=== CALCULATING TOURNAMENT POOLS ===")

    dynamic_base_weights = await get_dynamic_base_weights(psql_db)
    max_weights = max_weight_by_type()
    text_weight = dynamic_base_weights[TournamentType.TEXT]
    quant_weight = dynamic_base_weights[TournamentType.QUANT]
    image_weight = dynamic_base_weights[TournamentType.IMAGE]

    text_burn_proportion = (max_weights[TournamentType.TEXT] - text_weight) / max_weights[TournamentType.TEXT]
    quant_max_weight = max_weights[TournamentType.QUANT]
    quant_burn_proportion = (quant_max_weight - quant_weight) / quant_max_weight if quant_max_weight > 0 else 0.0
    image_burn_proportion = (max_weights[TournamentType.IMAGE] - image_weight) / max_weights[TournamentType.IMAGE]

    logger.info(f"Pools (no burn) - Text: {text_weight}, Quant: {quant_weight}, Image: {image_weight}")

    pool_values = {
        "text_weight": text_weight,
        "quant_weight": quant_weight,
        "image_weight": image_weight,
        "text_burn_proportion": text_burn_proportion,
        "quant_burn_proportion": quant_burn_proportion,
        "image_burn_proportion": image_burn_proportion,
    }
    non_finite = [name for name, value in pool_values.items() if not math.isfinite(value)]
    if non_finite:
        logger.error(f"Refusing to set weights from a non-finite pool: {pool_values}")
        raise ValueError(f"Non-finite tournament pool values: {non_finite}")

    return TournamentBurnData(
        text_burn_proportion=text_burn_proportion,
        image_burn_proportion=image_burn_proportion,
        text_tournament_weight=text_weight,
        image_tournament_weight=image_weight,
        burn_weight=0.0,
        quant_burn_proportion=quant_burn_proportion,
        quant_tournament_weight=quant_weight,
    )


def _paid_node_id(hotkey: str, hotkey_to_node_id: dict[str, int]) -> int | None:
    """The node a tournament share is added to; None leaves the share undistributed. The burn hotkey's share (a kept boss with
    no known defending champion, and any contender share paid like it) is left undistributed so it reaches the burn node with
    the rest of the undistributed weight: the burn node's weight is set, not added to, after the tournament shares."""
    if hotkey == cts.EMISSION_BURN_HOTKEY:
        return None
    return hotkey_to_node_id.get(hotkey)


def _apply_pool(
    label: str,
    tournament_weights: dict[str, float],
    hotkey_to_node_id: dict[str, int],
    all_node_weights: list[float],
    scaled_pool_weight: float,
) -> float:
    """Pay each hotkey its share of one type's pool. Returns the part of the pool left undistributed."""
    distributed = 0.0
    logger.info(f"Processing {len(tournament_weights)} {label} tournament winners")
    for hotkey, weight in tournament_weights.items():
        node_id = _paid_node_id(hotkey, hotkey_to_node_id)
        if node_id is not None:
            contribution = weight * scaled_pool_weight
            all_node_weights[node_id] = all_node_weights[node_id] + contribution
            distributed += contribution

            logger.info(
                f"Node ID {node_id} (hotkey: {hotkey[:8]}...): "
                f"{label.upper()} TOURNAMENT - weight={weight:.6f}, scaled_pool={scaled_pool_weight:.6f}, "
                f"contribution={contribution:.6f}, total_weight={all_node_weights[node_id]:.6f}"
            )

    undistributed = scaled_pool_weight - distributed
    logger.info(
        f"{label.capitalize()} tournament: allocated={scaled_pool_weight:.10f}, "
        f"distributed={distributed:.10f}, undistributed={undistributed:.10f}"
    )
    return undistributed


def apply_tournament_weights(
    text_tournament_weights: dict[str, float],
    image_tournament_weights: dict[str, float],
    hotkey_to_node_id: dict[str, int],
    all_node_weights: list[float],
    scaled_text_tournament_weight: float,
    scaled_image_tournament_weight: float,
) -> float:
    """Pay the text and image pools by place. Returns the total undistributed weight that should go to burn."""
    logger.info("=== TOURNAMENT WEIGHT CALCULATIONS ===")

    text_undistributed = _apply_pool(
        "text", text_tournament_weights, hotkey_to_node_id, all_node_weights, scaled_text_tournament_weight
    )
    image_undistributed = _apply_pool(
        "image", image_tournament_weights, hotkey_to_node_id, all_node_weights, scaled_image_tournament_weight
    )

    total_undistributed = text_undistributed + image_undistributed
    logger.info(f"Total undistributed weight to add to burn: {total_undistributed:.10f}")

    return total_undistributed


def apply_quant_tournament_weights(
    quant_tournament_weights: dict[str, float],
    hotkey_to_node_id: dict[str, int],
    all_node_weights: list[float],
    scaled_quant_tournament_weight: float,
) -> float:
    """Pay the quant pool the way apply_tournament_weights pays text and image. Returns the undistributed weight."""
    return _apply_pool(
        "quant", quant_tournament_weights, hotkey_to_node_id, all_node_weights, scaled_quant_tournament_weight
    )


async def get_node_weights_from_tournament_audit_data(
    substrate: SubstrateInterface,
    netuid: int,
    tournament_audit_data: TournamentAuditData,
) -> NodeWeightsResult:
    all_nodes: list[Node] = fetch_nodes._get_nodes_for_uid(substrate, netuid)
    hotkey_to_node_id: dict[str, int] = {node.hotkey: node.node_id for node in all_nodes}

    all_node_ids: list[int] = [node.node_id for node in all_nodes]
    all_node_weights: list[float] = [0.0 for _ in all_nodes]

    logger.info("=== USING BURN DATA FROM AUDIT ===")

    logger.info(f"Text tournament weight: {tournament_audit_data.text_tournament_weight:.6f}")
    logger.info(f"Quant tournament weight: {tournament_audit_data.quant_tournament_weight:.6f}")
    logger.info(f"Image tournament weight: {tournament_audit_data.image_tournament_weight:.6f}")
    logger.info(f"Total burn weight: {tournament_audit_data.burn_weight:.6f}")

    # Check that base weights sum to 1.0
    base_weight_sum = (
        tournament_audit_data.text_tournament_weight
        + tournament_audit_data.quant_tournament_weight
        + tournament_audit_data.image_tournament_weight
        + tournament_audit_data.burn_weight
    )
    logger.info(f"Base weights sum (text + quant + image + burn): {base_weight_sum:.10f}")
    logger.info(f"Base weights sum to 1.0? {abs(base_weight_sum - 1.0) < 0.0001}")

    participants: list[str] = tournament_audit_data.participants
    participation_total: float = len(participants) * cts.TOURNAMENT_PARTICIPATION_WEIGHT
    scale_factor: float = 1.0 - participation_total if participation_total > 0 else 1.0

    logger.info(f"Number of participants: {len(participants)}")
    logger.info(f"Participation total weight: {participation_total:.10f}")
    logger.info(f"Scale factor (1.0 - participation_total): {scale_factor:.10f}")

    scaled_text_tournament_weight: float = tournament_audit_data.text_tournament_weight * scale_factor
    scaled_quant_tournament_weight: float = tournament_audit_data.quant_tournament_weight * scale_factor
    scaled_image_tournament_weight: float = tournament_audit_data.image_tournament_weight * scale_factor
    scaled_burn_weight: float = tournament_audit_data.burn_weight * scale_factor

    # Check that scaled weights + participation still sum to 1.0
    scaled_weight_sum = (
        scaled_text_tournament_weight
        + scaled_quant_tournament_weight
        + scaled_image_tournament_weight
        + scaled_burn_weight
        + participation_total
    )
    logger.info(
        "Scaled weights sum (scaled_text + scaled_quant + scaled_image + scaled_burn + participation): "
        f"{scaled_weight_sum:.10f}"
    )
    logger.info(f"Scaled weights sum to 1.0? {abs(scaled_weight_sum - 1.0) < 0.0001}")

    text_tournament_weights, image_tournament_weights = get_tournament_weights_from_data(
        tournament_audit_data.text_tournament_data,
        tournament_audit_data.image_tournament_data,
    )

    quant_tournament_weights = get_quant_tournament_weights_from_data(tournament_audit_data.quant_tournament_data)

    undistributed_weight = apply_tournament_weights(
        text_tournament_weights,
        image_tournament_weights,
        hotkey_to_node_id,
        all_node_weights,
        scaled_text_tournament_weight,
        scaled_image_tournament_weight,
    )
    undistributed_weight += apply_quant_tournament_weights(
        quant_tournament_weights,
        hotkey_to_node_id,
        all_node_weights,
        scaled_quant_tournament_weight,
    )

    # Check sum after tournament weights applied
    weight_sum_after_tournament = sum(all_node_weights)
    logger.info(f"Weight sum after tournament weights applied: {weight_sum_after_tournament:.10f}")

    for hotkey in participants:
        node_id = hotkey_to_node_id.get(hotkey)
        if node_id is not None:
            all_node_weights[node_id] += cts.TOURNAMENT_PARTICIPATION_WEIGHT

    # Check sum after participation weights added
    weight_sum_after_participation = sum(all_node_weights)
    logger.info(f"Weight sum after participation weights added: {weight_sum_after_participation:.10f}")

    # Add undistributed tournament weight to burn: the part of each pool the paid places do not
    # take (unfilled places, the burn hotkey's share). This ensures total weights sum to exactly 1.0.
    burn_node_id: int | None = hotkey_to_node_id.get(cts.EMISSION_BURN_HOTKEY)
    if burn_node_id is not None:
        all_node_weights[burn_node_id] = scaled_burn_weight + undistributed_weight
        logger.info(
            f"Burn weight: base={scaled_burn_weight:.10f} + undistributed={undistributed_weight:.10f} "
            f"= total={all_node_weights[burn_node_id]:.10f}"
        )

    # Final weight sum check
    final_weight_sum = sum(all_node_weights)
    logger.info("=== FINAL WEIGHT SUM CHECK ===")
    logger.info(f"Total weight sum (before normalization): {final_weight_sum:.10f}")
    logger.info("Expected: 1.0")
    logger.info(f"Difference from 1.0: {abs(final_weight_sum - 1.0):.10f}")
    logger.info(f"Weights sum to 1.0? {abs(final_weight_sum - 1.0) < 0.0001}")
    logger.info(f"Number of non zero node weights: {sum(1 for weight in all_node_weights if weight != 0)}")

    if abs(final_weight_sum - 1.0) >= 0.0001:
        logger.warning(f"⚠️  WARNING: Weights DO NOT sum to 1.0! Sum is {final_weight_sum:.10f}")
    else:
        logger.info("✅ Weights correctly sum to 1.0")

    return NodeWeightsResult(node_ids=all_node_ids, node_weights=all_node_weights)


async def stored_boss_rank_comparisons(tournament_id: str, psql_db) -> list[BossRankComparison]:
    """The tournament's stored boss rank comparisons; none (every contender paid) when they cannot be read."""
    try:
        return await get_boss_rank_comparisons(tournament_id, psql_db)
    except Exception as e:
        logger.warning(
            f"Boss rank comparisons of tournament {tournament_id} could not be read ({type(e).__name__}: {e}); "
            f"every contender is paid"
        )
        return []


async def build_tournament_audit_data(psql_db) -> TournamentAuditData:
    """
    Build TournamentAuditData with all necessary tournament information.

    This is the central function for gathering tournament data used by both
    the validator (for weight setting) and auditor (for verification).

    Args:
        psql_db: Database connection

    Returns:
        TournamentAuditData with all tournament information populated
    """
    tournament_audit_data = TournamentAuditData()

    # Fetch text tournament data
    text_tournament: TournamentData = await get_latest_completed_tournament(psql_db, TournamentType.TEXT)
    if text_tournament:
        tournament_results: TournamentResults = await get_tournament_full_results(text_tournament.tournament_id, psql_db)
        participants = await get_tournament_participants(text_tournament.tournament_id, psql_db)
        tournament_audit_data.text_tournament_data = TournamentResultsWithWinners(
            tournament_id=tournament_results.tournament_id,
            rounds=tournament_results.rounds,
            base_winner_hotkey=text_tournament.base_winner_hotkey,
            winner_hotkey=text_tournament.winner_hotkey,
            final_positions={p.hotkey: p.final_position for p in participants if p.final_position is not None},
            boss_rank_comparisons=await stored_boss_rank_comparisons(text_tournament.tournament_id, psql_db),
        )

    # Fetch quant tournament data
    quant_tournament = await get_latest_completed_tournament(psql_db, TournamentType.QUANT)
    if quant_tournament:
        tournament_results = await get_tournament_full_results(quant_tournament.tournament_id, psql_db)
        participants = await get_tournament_participants(quant_tournament.tournament_id, psql_db)
        tournament_audit_data.quant_tournament_data = TournamentResultsWithWinners(
            tournament_id=tournament_results.tournament_id,
            rounds=tournament_results.rounds,
            base_winner_hotkey=quant_tournament.base_winner_hotkey,
            winner_hotkey=quant_tournament.winner_hotkey,
            final_positions={p.hotkey: p.final_position for p in participants if p.final_position is not None},
            boss_rank_comparisons=await stored_boss_rank_comparisons(quant_tournament.tournament_id, psql_db),
        )

    # Fetch image tournament data
    image_tournament = await get_latest_completed_tournament(psql_db, TournamentType.IMAGE)
    if image_tournament:
        tournament_results = await get_tournament_full_results(image_tournament.tournament_id, psql_db)
        participants = await get_tournament_participants(image_tournament.tournament_id, psql_db)
        tournament_audit_data.image_tournament_data = TournamentResultsWithWinners(
            tournament_id=tournament_results.tournament_id,
            rounds=tournament_results.rounds,
            base_winner_hotkey=image_tournament.base_winner_hotkey,
            winner_hotkey=image_tournament.winner_hotkey,
            final_positions={p.hotkey: p.final_position for p in participants if p.final_position is not None},
        )

    # Fetch participants
    tournament_audit_data.participants = await get_active_tournament_participants(psql_db)

    # Fetch burn weights
    burn_data: TournamentBurnData = await get_tournament_burn_details(psql_db)
    tournament_audit_data.text_tournament_weight = burn_data.text_tournament_weight
    tournament_audit_data.quant_tournament_weight = burn_data.quant_tournament_weight
    tournament_audit_data.image_tournament_weight = burn_data.image_tournament_weight
    tournament_audit_data.burn_weight = burn_data.burn_weight

    # Fetch weekly participation data
    tournament_audit_data.weekly_participation = await get_weekly_task_participation_data(psql_db)

    return tournament_audit_data


async def set_weights(config: Config, all_node_ids: list[int], all_node_weights: list[float], validator_node_id: int) -> bool:
    try:
        success = await asyncio.to_thread(
            weights.set_node_weights,
            substrate=config.substrate,
            keypair=config.keypair,
            node_ids=all_node_ids,
            node_weights=all_node_weights,
            netuid=config.netuid,
            version_key=ccst.VERSION_KEY,
            validator_node_id=int(validator_node_id),
            wait_for_inclusion=False,
            wait_for_finalization=False,
            max_attempts=3,
        )
    except Exception as e:
        logger.error(f"Failed to set weights: {e}")
        return False

    if success:
        logger.info("Weights set successfully.")

        return True
    else:
        logger.error("Failed to set weights :(")
        return False


async def _get_and_set_weights(config: Config, validator_node_id: int) -> bool:
    # Build tournament audit data using the centralized function
    tournament_audit_data: TournamentAuditData = await build_tournament_audit_data(config.psql_db)

    result = await get_node_weights_from_tournament_audit_data(config.substrate, config.netuid, tournament_audit_data)
    all_node_ids = result.node_ids
    all_node_weights = result.node_weights
    logger.info("Weights calculated, about to set...")

    success = await set_weights(config, all_node_ids, all_node_weights, validator_node_id)
    if success:
        # Upload both task results and tournament data
        url = await _upload_results_to_s3(config, tournament_audit_data)
        logger.info(f"Uploaded the scores and tournament data to s3 for auditing - url: {url}")

    return success


async def _set_metagraph_weights(config: Config) -> None:
    nodes: list[Node] = fetch_nodes._get_nodes_for_uid(config.substrate, config.netuid)
    node_ids = [node.node_id for node in nodes]
    node_weights = [node.incentive for node in nodes]
    validator_node_id = await get_vali_node_id(config.substrate, config.keypair.ss58_address)
    if validator_node_id is None:
        raise ValueError("Validator node id not found")

    await asyncio.to_thread(
        weights.set_node_weights,
        substrate=config.substrate,
        keypair=config.keypair,
        node_ids=node_ids,
        node_weights=node_weights,
        netuid=config.netuid,
        version_key=ccst.VERSION_KEY,
        validator_node_id=int(validator_node_id),
        wait_for_inclusion=False,
        wait_for_finalization=False,
        max_attempts=3,
    )


# To improve: use activity cutoff & The epoch length to set weights at the perfect times
async def set_weights_periodically(config: Config, just_once: bool = False) -> None:
    substrate = config.substrate
    substrate, uid = query_substrate(
        substrate,
        "SubtensorModule",
        "Uids",
        [config.netuid, config.keypair.ss58_address],
        return_value=True,
    )

    if uid is None:
        raise ValueError(f"Can't find hotkey {config.keypair.ss58_address} for our keypair on netuid: {config.netuid}.")

    consecutive_failures = 0
    while True:
        substrate, current_block = query_substrate(substrate, "System", "Number", [], return_value=True)
        substrate, last_updated_value = query_substrate(
            substrate, "SubtensorModule", "LastUpdate", [config.netuid], return_value=False
        )
        updated: int = current_block - last_updated_value[uid]
        substrate, weights_set_rate_limit = query_substrate(
            substrate, "SubtensorModule", "WeightsSetRateLimit", [config.netuid], return_value=True
        )
        logger.info(
            f"My Validator Node ID: {uid}. Last updated {updated} blocks ago. Weights set rate limit: {weights_set_rate_limit}."
        )

        if updated < weights_set_rate_limit:
            logger.info("Sleeping for a bit as we set recently...")
            await asyncio.sleep((weights_set_rate_limit - updated + 1) * 12)
            continue

        if os.getenv("ENV", "prod").lower() == "dev":
            success = await _get_and_set_weights(config, uid)
        else:
            try:
                success = await _get_and_set_weights(config, uid)
            except Exception as e:
                logger.error(f"Failed to set weights with error: {e}")
                logger.exception(e)
                success = False

        if success:
            consecutive_failures = 0
            logger.info("Successfully set weights! Sleeping for 25 blocks before next check...")
            if just_once:
                return
            await asyncio.sleep(12 * 25)
            continue

        consecutive_failures += 1
        if just_once:
            logger.info("Failed to set weights, will try again...")
            await asyncio.sleep(12 * 1)
        else:
            logger.info(f"Failed to set weights {consecutive_failures} times in a row - sleeping for a bit...")
            await asyncio.sleep(12 * 25)  # Try again in 25 blocks

        if consecutive_failures == 1 or updated < 3000:
            continue

        if just_once or config.set_metagraph_weights_with_high_updated_to_not_dereg:
            logger.warning("Setting metagraph weights as our updated value is getting too high!")
            if just_once:
                logger.warning("Please exit if you do not want to do this!!!")
                await asyncio.sleep(4)
            try:
                success = await _set_metagraph_weights(config)
            except Exception as e:
                logger.error(f"Failed to set metagraph weights: {e}")
                success = False

            if just_once:
                return

            if success:
                consecutive_failures = 0
                continue


async def main():
    config = load_config()
    await try_db_connections(config)
    await set_weights_periodically(config)


if __name__ == "__main__":
    asyncio.run(main())
