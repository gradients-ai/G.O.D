import asyncio
import json

import numpy as np
from fiber.chain.chain_utils import query_substrate
from fiber.chain.weights import _normalize_and_quantize_weights

from core.constants.network import NETUID
from core.downloads import download_s3_file
from core.logging import get_logger
from validator.app.config import Config
from validator.app.config import load_config
from validator.scoring.weights import get_node_weights_from_tournament_audit_data
from validator.scoring.weights import set_weights
from validator.tournament.models import TournamentAuditData


logger = get_logger(__name__)


def _normalised_vector_dot_product(a, b):
    return sum(a[i] * b[i] for i in range(len(a))) / (np.linalg.norm(a) * np.linalg.norm(b))


async def _get_7_day_task_results_for_rayon_validator(config: Config) -> TournamentAuditData:
    """Get task results and tournament data for a rayon validator."""
    url_to_get_latest_scores_url = "https://api.gradients.io/auditing/scores-url"
    response = await config.httpx_client.get(url_to_get_latest_scores_url)

    if response.status_code != 200:
        logger.error(f"Failed to get latest scores url: {response.status_code} {response.text} :(")
        raise Exception(f"Failed to get latest scores url: {response.status_code} {response.text} :(")

    url = response.json()["url"]

    logger.info(f"Getting task results from {url}")
    result_filepath = await download_s3_file(url)
    with open(result_filepath, "r") as f:
        upload_data = json.load(f)

    tournament_audit_data = upload_data.get("tournament_audit_data")
    return TournamentAuditData(**tournament_audit_data) if tournament_audit_data else TournamentAuditData()


async def get_task_results_from_s3(config: Config) -> TournamentAuditData:
    tournament_audit_data = await _get_7_day_task_results_for_rayon_validator(config)
    return tournament_audit_data


async def get_similarity_score_for_rayon_weights(
    config: Config, tournament_audit_data: TournamentAuditData | None = None
) -> float:
    if not tournament_audit_data:
        logger.warning("No tournament data found in S3, cannot calculate weights without tournament information")
        raise ValueError("Tournament data is required for weight calculation")

    result = await get_node_weights_from_tournament_audit_data(config.substrate, config.netuid, tournament_audit_data)
    node_ids = result.node_ids
    node_weights = result.node_weights

    node_ids_formatted, node_weights_formatted = _normalize_and_quantize_weights(node_ids, node_weights)

    rayon_weights = [0 for i in range(256)]
    for node_id, weight in zip(node_ids_formatted, node_weights_formatted):
        rayon_weights[node_id] = weight

    substrate, weights = query_substrate(config.substrate, "SubtensorModule", "Weights", [56, 190])
    weights_values = [0 for i in range(256)]
    for node_id, weight_value in weights:
        weights_values[node_id] = weight_value

    similarity_between_scores = _normalised_vector_dot_product(rayon_weights, weights_values)

    return similarity_between_scores, node_ids_formatted, node_weights_formatted


async def audit_weights(config: Config, set_weights_on_chain: bool = True) -> bool:
    """Check that scores are calculated correctly by the validator.

    Receive details of every task that occurred in the past 7 days.
    Check that the scores for these tasks correctly add up to the weights set by the validator
    for all miners on chain.

    This helps to audit validators by:
        - Ensuring the weights set are backed up by scores for tasks
        - Preventing the validator from being able to manipulate scores to set weights on chain, without a task being completed
        - Ensuring miners are rewarded for fair work

    This is one tool, used in conjunction with others.
    Miners (well, everyone) can see every task that runs through the subnet,
    and see their scores using the dashboards on gradients.io. Details about the models trained by
    miners are available on there too, so anyone can check the evaluation was fair.

    To make this more robust: we can add a function which sends tasks through to the api, to ensure all organic
    jobs are indeed included in the scoring. For the short term, miners can of course always check this, by comparing the work
    they have done to the tasks on the dashboards - it's only a minor improvement.

    """
    tournament_audit_data = await get_task_results_from_s3(config)

    similarity_between_scores, node_ids_formatted, node_weights_formatted = await get_similarity_score_for_rayon_weights(
        config, tournament_audit_data
    )

    if similarity_between_scores > 0.98:
        logger.info(f"✅ Yay! The scores are similar to the weights set on chain!! Similarity: {similarity_between_scores}")

    else:
        logger.error(
            f"Dear Auditor, the similarity between the scores and the weights set on chain is {similarity_between_scores}."
            "This is quite low, and you might want to look into this!"
        )

    if set_weights_on_chain:
        logger.info("Setting the weights on chain...")
        _, my_vali_uid = query_substrate(
            config.substrate, "SubtensorModule", "Uids", [NETUID, config.keypair.ss58_address], return_value=True
        )
        success = await set_weights(config, node_ids_formatted, node_weights_formatted, my_vali_uid)
        return success
    else:
        return True


async def main():
    config = load_config()
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
            sleep_duration = (weights_set_rate_limit - updated) * 12
            logger.info(f"Sleeping for {sleep_duration} seconds [{sleep_duration / 12} blocks] as we set weights recently...")
            await asyncio.sleep(sleep_duration)
            continue

        success = await audit_weights(config)
        if success:
            break


if __name__ == "__main__":
    asyncio.run(main())
