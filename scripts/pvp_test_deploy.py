"""Standalone PvP eval deploy script — no DB, no production side effects.

Deploys two models against each other on Basilica using a custom image,
then streams logs until the container finishes or is killed.

Usage:
    python scripts/pvp_test_deploy.py

Set BASILICA_API_KEY in env (or .vali.env) before running.
"""

import asyncio
import logging
import os
import sys
import uuid

# Load .vali.env if present (picks up BASILICA_API_KEY, HF_TOKEN, etc.)
_env_file = os.path.join(os.path.dirname(__file__), "..", ".vali.env")
if os.path.exists(_env_file):
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

import basilica

from core.constants import EnvironmentName
from core.models.pvp_models import PvPEvalConfig, PvPMatchupConfig, PvPModelSpec, PvPMode
from validator.core import constants as vcst
from validator.evaluation.utils import create_basilica_eval_runner_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

IMAGE = "weightswandering/pvp-evaluator:pvp-time-budget"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

REPO_A = "gradients-io-tournaments/tournament-tourn_358aca49563e214e_20260622-8f3e2873-6032-4791-94ab-6ae979584920-5DfdHDKN"
REPO_B = "gradients-io-tournaments/tournament-tourn_358aca49563e214e_20260622-8f3e2873-6032-4791-94ab-6ae979584920-5Eh6F11Z"

ENVS = [EnvironmentName.OTHELLO, EnvironmentName.GOOFSPIEL]

LOG_POLL_INTERVAL = 15  # seconds between log fetches while running

# ── build config ──────────────────────────────────────────────────────────────

pvp_config = PvPEvalConfig(
    mode=PvPMode.PAIR,
    model_a=PvPModelSpec(repo=REPO_A, original_model=BASE_MODEL),
    model_b=PvPModelSpec(repo=REPO_B, original_model=BASE_MODEL),
    matchups={
        env: PvPMatchupConfig(time_budget_seconds=vcst.PVP_MATCHUP_TIME_BUDGET_SECONDS)
        for env in ENVS
    },
    seed=42,
    temperature=0.0,
)

# ── deploy ────────────────────────────────────────────────────────────────────

deployment_name = f"pvp-test-{uuid.uuid4().hex[:8]}"

env_vars = {
    vcst.PVP_CONFIG_ENV_VAR: pvp_config.model_dump_json(),
    "EVAL_LOG_LEVEL": "DEBUG",
    **vcst.HF_CONTAINER_ENV,
}


async def main(client: basilica.BasilicaClient | None = None):
    client = client or basilica.BasilicaClient()

    logger.info("Deploying %s → image=%s", deployment_name, IMAGE)
    logger.info("Model A: %s", REPO_A)
    logger.info("Model B: %s", REPO_B)
    logger.info("Environments: %s", [e.value for e in ENVS])
    logger.info("Budget per env: %.0fs", vcst.PVP_MATCHUP_TIME_BUDGET_SECONDS)

    source = create_basilica_eval_runner_source(
        ["python", "-m", "validator.evaluation.pvp"],
        vcst.PVP_RESULTS_PATH,
    )

    deployment = await asyncio.to_thread(
        client.deploy,
        name=deployment_name,
        image=IMAGE,
        env=env_vars,
        gpu_count=2,
        gpu_models=["A100"],
        min_gpu_memory_gb=80,
        source=source,
        timeout=3600,
    )

    name = getattr(deployment, "name", deployment_name)
    logger.info("Deployed: %s", name)
    logger.info("Streaming logs — Ctrl-C to stop watching (container keeps running)\n")

    seen_bytes = 0
    while True:
        await asyncio.sleep(LOG_POLL_INTERVAL)

        # Refresh deployment object
        deployments = await asyncio.to_thread(client.list)
        by_name = {getattr(d, "name", None): d for d in deployments}
        dep = by_name.get(name)
        if dep is None:
            logger.info("Deployment %s no longer listed — finished or deleted.", name)
            break

        try:
            raw = dep.logs()
        except Exception as e:
            logger.warning("Log fetch failed: %s", e)
            continue

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")

        if raw:
            new = raw[seen_bytes:]
            if new:
                print(new, end="", flush=True)
                seen_bytes = len(raw)

        status = getattr(dep, "status", None) or getattr(dep, "state", None)
        if status and str(status).lower() in ("completed", "failed", "stopped", "error"):
            logger.info("Container status: %s — done.", status)
            # Final log flush
            try:
                raw = dep.logs()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                if raw and len(raw) > seen_bytes:
                    print(raw[seen_bytes:], end="", flush=True)
            except Exception:
                pass
            break


async def tail(deployment_name: str):
    """Attach to an existing deployment and stream its logs."""
    client = basilica.BasilicaClient()
    logger.info("Tailing logs for: %s", deployment_name)

    seen_bytes = 0
    while True:
        deployments = await asyncio.to_thread(client.list)
        by_name = {getattr(d, "name", None): d for d in deployments}
        dep = by_name.get(deployment_name)
        if dep is None:
            logger.info("Deployment %s not found.", deployment_name)
            break

        try:
            raw = dep.logs()
        except Exception as e:
            logger.warning("Log fetch failed: %s", e)
            await asyncio.sleep(LOG_POLL_INTERVAL)
            continue

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")

        if raw:
            new = raw[seen_bytes:]
            if new:
                print(new, end="", flush=True)
                seen_bytes = len(raw)

        status = getattr(dep, "status", None) or getattr(dep, "state", None)
        if status and str(status).lower() in ("completed", "failed", "stopped", "error"):
            logger.info("Container status: %s — done.", status)
            break

        await asyncio.sleep(LOG_POLL_INTERVAL)


async def kill_pvp_test_deployments(client: basilica.BasilicaClient) -> None:
    """Delete all deployments whose name starts with 'pvp-test-'."""
    deployments = await asyncio.to_thread(client.list)
    victims = [d for d in deployments if (getattr(d, "name", "") or "").startswith("pvp-test-")]
    if not victims:
        logger.info("No existing pvp-test-* deployments to kill.")
        return
    for d in victims:
        name = getattr(d, "name", "?")
        logger.info("Killing deployment: %s", name)
        await asyncio.to_thread(client.delete_deployment, name)
    logger.info("Killed %d deployment(s).", len(victims))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--tail":
        try:
            asyncio.run(tail(sys.argv[2]))
        except KeyboardInterrupt:
            sys.exit(0)
    elif "--kill-all" in sys.argv:
        async def _kill():
            await kill_pvp_test_deployments(basilica.BasilicaClient())
        asyncio.run(_kill())
    else:
        async def _deploy():
            client = basilica.BasilicaClient()
            if "--fresh" in sys.argv:
                await kill_pvp_test_deployments(client)
            await main()
        try:
            asyncio.run(_deploy())
        except KeyboardInterrupt:
            logger.info("\nStopped watching. Container is still running on Basilica.")
            sys.exit(0)
