import os

from core.logging import get_logger
from validator.infrastructure.service_constants import TRAINER_AUTH_HEADER


logger = get_logger(__name__)


def trainer_auth_headers() -> dict[str, str]:
    """Headers every validator -> trainer request carries. The trainer rejects any request without the shared key."""
    api_key = os.getenv("TRAINER_API_KEY")
    if not api_key:
        logger.error("TRAINER_API_KEY is not set on the validator: trainers will reject its requests with 403")
        return {}
    return {TRAINER_AUTH_HEADER: api_key}
