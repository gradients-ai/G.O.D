"""Fallback image builder on Shadeform, used only when no trainer can take queued builds.

The instance runs the trainer service in builder mode (no GPUs needed). Leak protection, in order:
Shadeform-side ``auto_delete`` (time and spend caps) set at creation, idle deletion, a max lifetime, and
re-discovery by tag after a validator restart.
"""

import base64
import os
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import httpx

from core.image_registry import ENV_ACCESS_KEY
from core.image_registry import ENV_BUCKET
from core.image_registry import ENV_ENDPOINT
from core.image_registry import ENV_PREFIX
from core.image_registry import ENV_SECRET_KEY
from core.logging import get_logger
from validator.tournament.image_builds import constants as cst


logger = get_logger(__name__)

API_KEY_ENV = "SHADEFORM_API_KEY"
ENABLED_ENV = "SHADEFORM_BUILDER_ENABLED"
ORCHESTRATOR_IPS_ENV = "SHADEFORM_BUILDER_ORCHESTRATOR_IPS"
REPO_ENV = "SHADEFORM_BUILDER_GOD_REPO"
REF_ENV = "SHADEFORM_BUILDER_GOD_REF"
ALLOW_GPU_ENV = "SHADEFORM_BUILDER_ALLOW_GPU_TYPES"

BOOTSTRAP_SCRIPT = r"""#!/usr/bin/env bash
# G.O.D fallback image builder bootstrap (validator/tournament/image_builds/shadeform.py).
set -euo pipefail
exec > >(tee -a /var/log/god-image-builder.log) 2>&1
export DEBIAN_FRONTEND=noninteractive
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh
apt-get update -y && apt-get install -y git python3-venv python3-pip
mkdir -p /etc/docker
python3 - <<'PY'
import json, pathlib
path = pathlib.Path("/etc/docker/daemon.json")
config = json.loads(path.read_text()) if path.exists() and path.read_text().strip() else {}
config.setdefault("features", {})["containerd-snapshotter"] = True
config["dns"] = ["1.1.1.1", "8.8.8.8"]
config["max-concurrent-downloads"] = 10
path.write_text(json.dumps(config, indent=2))
PY
systemctl restart docker
if [ -n "${DOCKERHUB_USERNAME:-}" ] && [ -n "${DOCKERHUB_TOKEN:-}" ]; then
  echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
fi
rm -rf /opt/god && git clone "$GOD_REPO" /opt/god && cd /opt/god && git checkout "$GOD_REF"
python3 -m venv /opt/god/.venv && . /opt/god/.venv/bin/activate
pip install --upgrade pip && pip install -e ".[organic,validator]"
umask 077
cat > /opt/god/.trainer.env <<EOF
IMAGE_BUILDER_ENABLED=true
ORCHESTRATOR_IPS=$ORCHESTRATOR_IPS
IMAGE_REGISTRY_ENDPOINT=$IMAGE_REGISTRY_ENDPOINT
IMAGE_REGISTRY_BUCKET=$IMAGE_REGISTRY_BUCKET
IMAGE_REGISTRY_ACCESS_KEY=$IMAGE_REGISTRY_ACCESS_KEY
IMAGE_REGISTRY_SECRET_KEY=$IMAGE_REGISTRY_SECRET_KEY
IMAGE_REGISTRY_PREFIX=${IMAGE_REGISTRY_PREFIX:-registry}
EOF
nohup /opt/god/.venv/bin/uvicorn trainer.asgi:app --host 0.0.0.0 --port 8001 --env-file /opt/god/.trainer.env \
  >> /var/log/god-image-builder.log 2>&1 &
"""


def shadeform_enabled() -> bool:
    return os.getenv(ENABLED_ENV, "false").lower() == "true" and bool(os.getenv(API_KEY_ENV))


@dataclass
class FallbackInstance:
    instance_id: str
    created_at: datetime
    status: str
    ip: str | None

    @property
    def address(self) -> str | None:
        return f"{self.ip}:{cst.DEFAULT_TRAINER_PORT}" if self.ip and self.status == "active" else None


class ShadeformClient:
    def __init__(self, api_key: str):
        self._headers = {"X-API-KEY": api_key}

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        async with httpx.AsyncClient(timeout=30, headers=self._headers) as client:
            response = await client.request(method, f"{cst.SHADEFORM_API_URL}{path}", **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}

    async def list_builders(self) -> list[FallbackInstance]:
        payload = await self._request("GET", "/instances", params={"search": cst.SHADEFORM_INSTANCE_NAME_PREFIX})
        instances = []
        for item in payload.get("instances") or []:
            if cst.SHADEFORM_TAG not in (item.get("tags") or []):
                continue
            if item.get("status") in ("deleting", "deleted"):
                continue
            instances.append(
                FallbackInstance(
                    instance_id=item["id"],
                    created_at=_parse_time(item.get("created_at")),
                    status=item.get("status") or "unknown",
                    ip=item.get("ip"),
                )
            )
        return instances

    async def instance_types(self) -> list[dict]:
        payload = await self._request(
            "GET", "/instances/types", params={"available": "true", "sort": "price", "shade_cloud": "true"}
        )
        return payload.get("instance_types") or []

    async def create(self, body: dict) -> str:
        return (await self._request("POST", "/instances/create", json=body))["id"]

    async def delete(self, instance_id: str) -> None:
        try:
            await self._request("POST", f"/instances/{instance_id}/delete")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def pick_instance_type(types: list[dict], allow_gpu: bool) -> tuple[dict, str] | None:
    """Cheapest on-demand type that is CPU-only (or any, if allowed) and big enough to build images."""
    options = []
    for item in types:
        config = item.get("configuration") or {}
        num_gpus = config.get("num_gpus", item.get("num_gpus", 0)) or 0
        vcpus = config.get("vcpus") or item.get("vcpus") or 0
        storage = config.get("storage_in_gb") or item.get("storage_in_gb") or 0
        price = item.get("hourly_price") or 0
        if (num_gpus and not allow_gpu) or vcpus < cst.SHADEFORM_MIN_VCPUS or storage < cst.SHADEFORM_MIN_STORAGE_GB:
            continue
        if price <= 0 or price > cst.SHADEFORM_MAX_HOURLY_PRICE_CENTS:
            continue
        for region in item.get("availability") or []:
            if region.get("available") and region.get("rental_type", "on_demand") == "on_demand":
                options.append((num_gpus > 0, price, item, region["region"]))
                break
    if not options:
        return None
    _, _, item, region = min(options, key=lambda option: (option[0], option[1]))
    return item, region


def _launch_envs() -> list[dict]:
    names = {
        "ORCHESTRATOR_IPS": os.getenv(ORCHESTRATOR_IPS_ENV, ""),
        "GOD_REPO": os.getenv(REPO_ENV, "https://github.com/gradients-ai/G.O.D.git"),
        "GOD_REF": os.getenv(REF_ENV, "main"),
        ENV_ENDPOINT: os.getenv(ENV_ENDPOINT, ""),
        ENV_BUCKET: os.getenv(ENV_BUCKET, ""),
        ENV_ACCESS_KEY: os.getenv(ENV_ACCESS_KEY, ""),
        ENV_SECRET_KEY: os.getenv(ENV_SECRET_KEY, ""),
        ENV_PREFIX: os.getenv(ENV_PREFIX, "registry"),
        "DOCKERHUB_USERNAME": os.getenv("DOCKERHUB_USERNAME", ""),
        "DOCKERHUB_TOKEN": os.getenv("DOCKERHUB_TOKEN", ""),
    }
    return [{"name": name, "value": value} for name, value in names.items() if value]


class FallbackBuilderManager:
    def __init__(self, client: ShadeformClient | None = None):
        self.client = client or (ShadeformClient(os.environ[API_KEY_ENV]) if shadeform_enabled() else None)
        self.instances: dict[str, FallbackInstance] = {}
        self._adopted = False
        self._idle_since: datetime | None = None

    @property
    def addresses(self) -> list[str]:
        return [address for instance in self.instances.values() if (address := instance.address)]

    async def refresh(self) -> None:
        if self.client is None:
            return
        try:
            found = await self.client.list_builders()
        except Exception as exc:
            logger.warning(f"Shadeform list failed: {exc}")
            return
        if not self._adopted and found:
            logger.info(f"Adopted {len(found)} existing Shadeform builder(s)")
        self._adopted = True
        self.instances = {instance.instance_id: instance for instance in found}

    async def tick(self, queued_due: int, building: int, oldest_queued: datetime | None, trainer_free_slots: int) -> None:
        """Create a builder when queued work is starving; delete builders that are idle, stuck or too old."""
        if self.client is None:
            return
        await self.refresh()
        now = datetime.now(timezone.utc)

        for instance in list(self.instances.values()):
            too_old = now - instance.created_at > timedelta(hours=cst.SHADEFORM_MAX_LIFETIME_HOURS)
            stuck_booting = instance.status != "active" and now - instance.created_at > timedelta(
                seconds=cst.SHADEFORM_BOOT_TIMEOUT_SECONDS
            )
            if too_old or stuck_booting:
                await self._delete(instance, "max lifetime reached" if too_old else "did not become active")

        busy = queued_due > 0 or building > 0
        self._idle_since = None if busy else (self._idle_since or now)
        if self._idle_since and now - self._idle_since > timedelta(seconds=cst.SHADEFORM_IDLE_SECONDS):
            for instance in list(self.instances.values()):
                await self._delete(instance, "idle")

        starving = (
            queued_due > 0
            and trainer_free_slots == 0
            and oldest_queued is not None
            and now - oldest_queued > timedelta(seconds=cst.SHADEFORM_FALLBACK_AFTER_SECONDS)
        )
        if starving and not self.instances:
            await self._create(now)

    async def _create(self, now: datetime) -> None:
        if not os.getenv(ORCHESTRATOR_IPS_ENV):
            logger.error(f"{ORCHESTRATOR_IPS_ENV} must be set for the Shadeform fallback builder")
            return
        try:
            choice = pick_instance_type(
                await self.client.instance_types(), allow_gpu=os.getenv(ALLOW_GPU_ENV, "false").lower() == "true"
            )
            if choice is None:
                logger.warning("No suitable Shadeform instance type available for a fallback image builder")
                return
            item, region = choice
            body = {
                "cloud": item["cloud"],
                "region": region,
                "shade_instance_type": item["shade_instance_type"],
                "shade_cloud": True,
                "name": f"{cst.SHADEFORM_INSTANCE_NAME_PREFIX}-{now:%Y%m%d%H%M%S}",
                "tags": [cst.SHADEFORM_TAG],
                "launch_configuration": {
                    "type": "script",
                    "script_configuration": {"base64_script": base64.b64encode(BOOTSTRAP_SCRIPT.encode()).decode()},
                },
                "envs": _launch_envs(),
                "auto_delete": {
                    "date_threshold": (now + timedelta(hours=cst.SHADEFORM_MAX_LIFETIME_HOURS)).isoformat(),
                    "spend_threshold": cst.SHADEFORM_MAX_SPEND_USD,
                },
            }
            instance_id = await self.client.create(body)
            self.instances[instance_id] = FallbackInstance(instance_id=instance_id, created_at=now, status="pending", ip=None)
            logger.info(
                f"Created Shadeform fallback image builder {instance_id} "
                f"({item['cloud']} {item['shade_instance_type']} in {region}, {item.get('hourly_price')}c/h)"
            )
        except Exception as exc:
            logger.error(f"Could not create Shadeform fallback builder: {exc}")

    async def _delete(self, instance: FallbackInstance, reason: str) -> None:
        try:
            await self.client.delete(instance.instance_id)
            self.instances.pop(instance.instance_id, None)
            logger.info(f"Deleted Shadeform fallback builder {instance.instance_id} ({reason})")
        except Exception as exc:
            logger.error(f"Could not delete Shadeform fallback builder {instance.instance_id}: {exc}")
