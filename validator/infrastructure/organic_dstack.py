"""Chain-free dstack helpers for organic training orchestration."""

from __future__ import annotations

import re
import string
import uuid
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from core.models.payload_models import DstackRunStatus
from validator.infrastructure.dstack_client import DstackClient


_IMAGE_DIGEST = re.compile(r"^[^@\s]+@sha256:[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class OrganicRunSpec:
    task_id: str
    model: str
    task_type: str
    hours_to_complete: float
    docker_image: str
    gpu_name: str
    gpu_count: int
    regions: tuple[str, ...]
    expected_repo_name: str
    training_data: str
    dataset_type: dict[str, Any] | None = None
    model_type: str | None = None
    trigger_word: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    disk_size: str = "1000GB"
    provisioning_grace_seconds: int = 3600


def validate_digest_pinned_image(image: str) -> str:
    """Reject mutable image tags before a remote run can be submitted."""
    if not _IMAGE_DIGEST.fullmatch(image):
        raise ValueError("dstack images must be pinned as 'repository@sha256:<64 hex characters>'")
    return image


def generate_run_name() -> str:
    run_name = uuid.uuid4().hex
    if run_name[0].isdigit():
        run_name = f"{string.ascii_lowercase[int(run_name[0])]}{run_name[1:]}"
    return run_name


def build_run_plan(spec: OrganicRunSpec, *, run_name: str | None = None) -> dict[str, Any]:
    """Build a dstack plan without reading process environment or validator state."""
    validate_digest_pinned_image(spec.docker_image)
    if spec.gpu_count < 1:
        raise ValueError("gpu_count must be at least one")
    if spec.hours_to_complete <= 0:
        raise ValueError("hours_to_complete must be positive")

    task_env = {
        "TASK_ID": spec.task_id,
        "MODEL": spec.model,
        "TASK_TYPE": spec.task_type,
        "EXPECTED_REPO_NAME": spec.expected_repo_name,
        "HOURS_TO_COMPLETE": str(spec.hours_to_complete),
        **spec.environment,
    }
    if spec.task_type == "ImageTask":
        task_env["DATASET_ZIP"] = spec.training_data
        if spec.model_type:
            task_env["MODEL_TYPE"] = spec.model_type
        if spec.trigger_word:
            task_env["TRIGGER_WORD"] = spec.trigger_word
    else:
        task_env["DATASET"] = spec.training_data
        task_env["FILE_FORMAT"] = "s3"
        if spec.dataset_type is not None:
            import json

            task_env["DATASET_TYPE"] = json.dumps(spec.dataset_type)

    return {
        "plan": {
            "run_spec": {
                "run_name": run_name or generate_run_name(),
                "configuration": {
                    "type": "task",
                    "name": "organic",
                    "image": spec.docker_image,
                    "env": task_env,
                    "resources": {
                        "gpu": {
                            "name": [spec.gpu_name],
                            "count": {"min": spec.gpu_count, "max": spec.gpu_count},
                        },
                        "disk": {"size": spec.disk_size},
                    },
                    "regions": list(spec.regions),
                    "max_duration": int(spec.hours_to_complete * 3600) + spec.provisioning_grace_seconds,
                },
            }
        },
        "force": False,
    }


async def submit_run(plan: dict[str, Any], client: DstackClient) -> str:
    result = await client.apply_run(plan)
    run_name = result.get("run_spec", {}).get("run_name") or result.get("run_name")
    if not run_name:
        raise RuntimeError("dstack accepted the run but returned no run name")
    return str(run_name)


async def get_run_status(run_name: str, client: DstackClient) -> DstackRunStatus:
    return await client.get_run_status(run_name)
