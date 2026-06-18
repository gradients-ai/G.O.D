"""
Dispatch model prep (augmentation + baseline stats) to a trainer with GPU.
Called during task prep, before miners are assigned.
"""

import httpx

from core.constants.environments import ENVIRONMENT_CONFIGS
from core.constants.environments import EnvironmentName
from core.logging import get_logger
from core.models.model_prep_models import AugmentationConfig
from core.models.payload_models import EnvConfig
from core.models.payload_models import ModelPrepRequest
from core.models.payload_models import ModelPrepResponse
from validator.infrastructure.service_constants import MODEL_PREP_ENDPOINT


logger = get_logger(__name__)

# Must comfortably exceed summed per-env baseline budgets plus SGLang startup
# and any LoRA merge.
MODEL_PREP_TIMEOUT_SECONDS = 5400


def _build_env_configs(
    environment_names: list[EnvironmentName] | None = None,
) -> dict[EnvironmentName, EnvConfig]:
    """Build env_configs payload from the canonical ENVIRONMENT_CONFIGS."""
    selected = {EnvironmentName(env_name) for env_name in environment_names} if environment_names else None
    return {
        env_name: EnvConfig(
            env_image=cfg.env_image,
            env_server_command=cfg.env_server_command,
            task_id_min=cfg.task_id_min,
            task_id_max=cfg.task_id_max,
            num_episodes=cfg.num_baseline_episodes,
            eval_payload_extra=cfg.eval_payload_extra,
        )
        for env_name, cfg in ENVIRONMENT_CONFIGS.items()
        if selected is None or env_name in selected
    }


async def dispatch_augmentation_and_stats(
    task_id: str,
    model_id: str,
    training_data_url: str,
    augmentation_config: AugmentationConfig | None,
    task_type,
    trainer_ip: str,
    gpu_ids: list[int],
    reward_functions=None,
    is_env_task: bool = False,
    hotkey: str | None = None,
    environment_names: list[EnvironmentName] | None = None,
) -> ModelPrepResponse | None:
    """Dispatch augmentation and stats collection to a trainer with GPU.

    The caller is responsible for GPU allocation and reservation.
    Returns ModelPrepResponse with augmented_model_id and baseline_stats,
    or None on failure.
    """
    if ":" not in trainer_ip:
        trainer_ip_with_port = f"{trainer_ip}:8001"
    else:
        trainer_ip_with_port = trainer_ip

    task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)
    request = ModelPrepRequest(
        task_id=task_id,
        model_id=model_id,
        training_data_url=training_data_url,
        task_type=task_type_str,
        augmentation_config=augmentation_config,
        gpu_ids=gpu_ids,
        reward_functions=reward_functions,
        env_configs=_build_env_configs(environment_names) if is_env_task else None,
        hotkey=hotkey,
    )

    url = f"http://{trainer_ip_with_port}{MODEL_PREP_ENDPOINT}"
    logger.info(f"Dispatching model prep to {url}")

    try:
        async with httpx.AsyncClient(timeout=MODEL_PREP_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=request.model_dump())
            response.raise_for_status()
            result = ModelPrepResponse.model_validate(response.json())
            logger.info(
                f"Model prep complete: augmented_model_id={result.augmented_model_id}, "
                f"baseline_stats={result.baseline_stats}"
            )
            return result
    except httpx.HTTPStatusError as e:
        body = e.response.text[:500] if e.response else "no response body"
        logger.error(f"Model prep dispatch failed (HTTP {e.response.status_code}): {body}")
        return None
    except Exception as e:
        logger.error(f"Model prep dispatch failed: {e}")
        return None
