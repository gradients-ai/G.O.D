"""
Dispatch model prep (augmentation + baseline stats) to a trainer with GPU.
Called during task prep, before miners are assigned.
"""

import httpx

from core.constants import ENVIRONMENT_CONFIGS
from core.constants import EnvironmentName
from core.models.model_prep_models import AugmentationConfig
from core.models.payload_models import EnvConfig
from core.models.payload_models import ModelPrepRequest
from core.models.payload_models import ModelPrepResponse
from core.models.tournament_models import GpuRequirement
from validator.core.config import Config
from validator.core.constants import MODEL_PREP_ENDPOINT
from validator.tournament.orchestrator import _check_suitable_gpus
from validator.utils.logging import get_logger


logger = get_logger(__name__)

MODEL_PREP_TIMEOUT_SECONDS = 1800



def _gpu_requirement_for_model_prep(num_params: int) -> GpuRequirement:
    """Select GPU tier for model prep based on parameter count.

    Model prep loads the model in fp16 (~2 bytes/param) plus overhead for
    activations, GPT-2 reference model (BPB), and gradient computation.
    Each H100 has 80GB VRAM.

    <10B  → ~20GB  → 1x H100
    10-35B → ~70GB  → 2x H100
    35-70B → ~140GB → 4x H100
    70B+   → ~140GB+→ 8x H100
    """
    if num_params < 10_000_000_000:
        return GpuRequirement.H100_1X
    elif num_params < 35_000_000_000:
        return GpuRequirement.H100_2X
    elif num_params < 70_000_000_000:
        return GpuRequirement.H100_4X
    return GpuRequirement.H100_8X


def _build_env_configs() -> dict[EnvironmentName, EnvConfig]:
    """Build env_configs payload from the canonical ENVIRONMENT_CONFIGS."""
    return {
        env_name: EnvConfig(
            env_image=cfg.env_image,
            task_id_min=cfg.task_id_min,
            task_id_max=cfg.task_id_max,
            num_episodes=cfg.num_baseline_episodes,
            eval_payload_extra=cfg.eval_payload_extra,
        )
        for env_name, cfg in ENVIRONMENT_CONFIGS.items()
    }


async def dispatch_augmentation_and_stats(
    task_id: str,
    model_id: str,
    training_data_url: str,
    augmentation_config: AugmentationConfig | None,
    model_params_count: int,
    task_type,
    config: Config,
    reward_functions=None,
    is_env_task: bool = False,
) -> ModelPrepResponse | None:
    """Dispatch augmentation and stats collection to a trainer with GPU.

    Returns ModelPrepResponse with augmented_model_id and baseline_stats,
    or None if no trainer is available.
    """
    gpu_req = _gpu_requirement_for_model_prep(model_params_count or 0)
    suitable = await _check_suitable_gpus(config, gpu_req)

    if suitable is None:
        logger.warning(f"No suitable GPUs for model prep of {model_id}, skipping")
        return None

    trainer_ip, gpu_ids = suitable

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
        env_configs=_build_env_configs() if is_env_task else None,
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
