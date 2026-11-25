"""Access control for the trainer API: an IP allowlist plus a shared key header, both required on every route."""

import hmac
import os

from fastapi import HTTPException
from fastapi import Request

from core.logging import get_logger
from validator.infrastructure.service_constants import TRAINER_AUTH_HEADER


logger = get_logger(__name__)

LOCALHOST_IP = "127.0.0.1"


def allowed_orchestrator_ips() -> list[str]:
    """IPs allowed to call the trainer: ORCHESTRATOR_IPS (or the older ORCHESTRATOR_IP), plus localhost."""
    allowed_ips_str = os.getenv("ORCHESTRATOR_IPS") or os.getenv("ORCHESTRATOR_IP") or ""
    allowed_ips = [ip.strip() for ip in allowed_ips_str.split(",") if ip.strip()]
    allowed_ips.append(LOCALHOST_IP)
    return allowed_ips


def check_trainer_auth_config() -> None:
    """Logs an error for each missing setting. Without them the trainer rejects every request from the validator."""
    if not os.getenv("TRAINER_API_KEY"):
        logger.error(
            "TRAINER_API_KEY is not set: every trainer request will be rejected with 403. Set it in .trainer.env to the "
            "same value as TRAINER_API_KEY in the validator's .vali.env."
        )
    if not (os.getenv("ORCHESTRATOR_IPS") or os.getenv("ORCHESTRATOR_IP")):
        logger.error(
            f"ORCHESTRATOR_IPS is not set: only {LOCALHOST_IP} may call the trainer. Set it in .trainer.env to the "
            f"comma-separated validator IPs."
        )


async def verify_orchestrator_ip(request: Request):
    """Verify the request comes from an allowed orchestrator IP and carries the shared trainer API key."""
    client_ip = request.client.host if request.client else None
    if client_ip not in allowed_orchestrator_ips():
        raise HTTPException(status_code=403, detail="Access forbidden")

    expected_key = os.getenv("TRAINER_API_KEY")
    if not expected_key:
        logger.error("Rejected a trainer request: TRAINER_API_KEY is not set on this trainer")
        raise HTTPException(status_code=403, detail="Access forbidden")
    provided_key = request.headers.get(TRAINER_AUTH_HEADER, "")
    if not hmac.compare_digest(provided_key.encode(), expected_key.encode()):
        raise HTTPException(status_code=403, detail="Access forbidden")
    return client_ip
