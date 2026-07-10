import time
from typing import Any

import httpx
from fiber import Keypair
from fiber.chain import chain_utils
from fiber.chain.models import Node
from fiber.validator import client

from core.constants.network import NETUID
from core.logging import get_logger
from validator.app.config import Config
from validator.tasks.details import retry_http_with_backoff
from validator.tasks.synthetics.constants import IMAGE_GEN_ENDPOINT
from validator.tasks.synthetics.constants import NINETEEN_API_KEY
from validator.tasks.synthetics.constants import PROMPT_GEN_ENDPOINT


logger = get_logger(__name__)


def _get_headers_for_signed_https_request(keypair: Keypair):
    nonce = f"{time.time_ns()}"

    signature = chain_utils.sign_message(keypair, nonce)

    headers = {
        "validator-hotkey": keypair.ss58_address,
        "signature": signature,
        "nonce": nonce,
        "netuid": str(NETUID),
        "Content-Type": "application/json",
    }

    return headers


async def process_non_stream_fiber_get(endpoint: str, config: Config, node: Node) -> dict[str, Any] | None:
    server_address = client.construct_server_address(
        node=node,
        replace_with_docker_localhost=False,
        replace_with_localhost=False,
    )
    logger.info(f"Attempting to hit a GET {server_address} endpoint {endpoint}")
    try:
        response = await client.make_non_streamed_get(
            httpx_client=config.httpx_client,
            validator_ss58_address=config.keypair.ss58_address,
            miner_ss58_address=node.hotkey,
            keypair=config.keypair,
            server_address=server_address,
            endpoint=endpoint,
            timeout=10,
        )
    except Exception as e:
        logger.error(f"Failed to communicate with node {node.node_id} ({server_address}{endpoint}): {repr(e)}")
        return None

    if response.status_code != 200:
        logger.warning(
            f"Node {node.node_id} ({server_address}{endpoint}) returned HTTP {response.status_code}: {response.text[:200]}"
        )
        return None

    return response.json()


async def process_non_stream_fiber(
    endpoint: str, config: Config, node: Node, payload: dict[str, Any], timeout: int = 10
) -> dict[str, Any] | None:
    server_address = client.construct_server_address(
        node=node,
        replace_with_docker_localhost=False,
        replace_with_localhost=False,
    )
    try:
        response = await client.make_non_streamed_post(
            httpx_client=config.httpx_client,
            server_address=server_address,
            validator_ss58_address=config.keypair.ss58_address,
            miner_ss58_address=node.hotkey,
            keypair=config.keypair,
            endpoint=endpoint,
            payload=payload,
            timeout=timeout,
        )
    except Exception as e:
        logger.debug(f"Failed to communicate with node {node.node_id} ({server_address}{endpoint}): {repr(e)}")
        return None

    if response.status_code != 200:
        logger.debug(
            f"Node {node.node_id} ({server_address}{endpoint}) returned HTTP {response.status_code}: {response.text[:200]}"
        )
        return None

    return response.json()


async def post_to_nineteen_chat(payload: dict[str, Any], keypair: Keypair) -> str | None:
    response = await _post_to_nineteen_ai(PROMPT_GEN_ENDPOINT, payload, keypair)
    response_json = response.json()
    try:
        return response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        logger.error(f"Error in nineteen ai chat response: {response_json}")
        logger.exception(e)
        return None


async def post_to_nineteen_image(payload: dict[str, Any], keypair: Keypair) -> str | None:
    response = await _post_to_nineteen_ai(IMAGE_GEN_ENDPOINT, payload, keypair)
    return response


@retry_http_with_backoff
async def _post_to_nineteen_ai(url: str, payload: dict[str, Any], keypair: Keypair) -> httpx.Response:
    if NINETEEN_API_KEY is None:
        headers = _get_headers_for_signed_https_request(keypair)
    else:
        headers = {
            "Authorization": f"Bearer {NINETEEN_API_KEY}",
            "Content-Type": "application/json",
        }

    model = payload.get("model", "unknown")
    started = time.monotonic()
    logger.info(f"Calling Nineteen/Chutes endpoint={url} model={model}")
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(url=url, json=payload, headers=headers)
        elapsed = time.monotonic() - started
        logger.info(f"Nineteen/Chutes endpoint={url} model={model} returned status={response.status_code} in {elapsed:.2f}s")
        if response.status_code != 200:
            # NOTE: What do to about these as they pollute everywhere
            logger.error(f"Error in nineteen ai response: {response.content}")
            response.raise_for_status()

        return response


@retry_http_with_backoff
async def call_content_service(endpoint: str, keypair: Keypair, params: dict = None) -> dict[str, Any] | list[dict[str, Any]]:
    """Make a signed request to the content service."""
    headers = _get_headers_for_signed_https_request(keypair)

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(url=endpoint, headers=headers, params=params)
        if response.status_code != 200:
            logger.error(f"Error in content service response. Status code: {response.status_code} and response: {response.text}")
            response.raise_for_status()
        return response.json()


async def call_content_service_fast(
    endpoint: str, keypair: Keypair, params: dict = None
) -> dict[str, Any] | list[dict[str, Any]]:
    """Make a signed request to the content service with fast retries for LLM endpoints."""
    from validator.tasks.details import retry_http_fast

    @retry_http_fast
    async def _make_request():
        headers = _get_headers_for_signed_https_request(keypair)
        logger.info(f"Making request to {endpoint} with params: {params}")
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url=endpoint, headers=headers, params=params)
            if response.status_code != 200:
                logger.error(
                    f"Error in content service response. URL: {endpoint}, "
                    f"Status code: {response.status_code} and response: {response.text}"
                )
                response.raise_for_status()
            logger.info(f"Successful request to {endpoint}, response size: {len(response.text)} chars")
            return response.json()

    return await _make_request()
