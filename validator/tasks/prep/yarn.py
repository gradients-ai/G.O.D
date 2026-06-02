"""
YaRN (Yet another RoPE extensioN) extension utilities for extending context length of models.
"""

import json
import tempfile
import uuid

import requests
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub import repo_exists
from huggingface_hub.utils import build_hf_headers
from huggingface_hub.utils import hf_raise_for_status

from core.constants.credentials import YARN_HUGGINGFACE_TOKEN
from core.constants.credentials import YARN_HUGGINGFACE_USERNAME
from core.logging import get_logger
from validator.tasks.prep.constants import MODEL_COPY_ENDPOINT


logger = get_logger(__name__)


def duplicate_model_to_yarn_repo(source_repo: str, dst_repo: str) -> str | None:
    """
    Duplicate a HuggingFace model to a new repository for YaRN extension.

    Args:
        source_repo: Source model repository (e.g., "meta-llama/Llama-2-7b")
        dst_repo: Destination repository name (without username, e.g., "Llama-2-7b-YaRN-extended-4")

    Returns:
        The URL of the duplicated repository, or None if duplication failed
    """
    try:
        url = MODEL_COPY_ENDPOINT.format(source_repo=source_repo)
        headers = build_hf_headers(token=YARN_HUGGINGFACE_TOKEN)

        r = requests.post(
            url,
            headers=headers,
            json={"repository": dst_repo},
            timeout=300,  # 5 minute timeout for large models
        )
        hf_raise_for_status(r)

        result_url = r.json().get("url")
        logger.info(f"Successfully duplicated model from {source_repo} to {dst_repo}")
        return result_url

    except Exception as e:
        logger.error(f"Failed to duplicate model from {source_repo} to {dst_repo}: {e}")
        raise


def apply_yarn_extension_to_config(config: dict, yarn_factor: int) -> dict:
    original_max_pos = config.get("max_position_embeddings", 4096)
    new_max_pos = original_max_pos * yarn_factor
    config["max_position_embeddings"] = new_max_pos

    config["rope_scaling"] = {
        "type": "yarn",
        "factor": float(yarn_factor),
        "original_max_position_embeddings": original_max_pos,
    }

    return config


async def prepare_yarn_extended_model(original_model_id: str, yarn_factor: int) -> str:
    """
    Prepare a YaRN extended model by duplicating and modifying the config.

    This function:
    1. Checks if the YaRN extended model already exists
    2. If not, duplicates the original model
    3. Downloads and modifies the config.json with YaRN parameters
    4. Uploads the modified config back to HuggingFace

    Returns:
        The YaRN extended model ID to use for training
    """
    model_name = original_model_id.split("/")[-1]
    yarn_model_id = f"{YARN_HUGGINGFACE_USERNAME}/{model_name}-YaRN-extended-{yarn_factor}"

    if repo_exists(yarn_model_id, token=YARN_HUGGINGFACE_TOKEN):
        logger.info(f"YaRN extended model {yarn_model_id} already exists, skipping duplication")
        return yarn_model_id

    logger.info(f"Creating YaRN extended model: {yarn_model_id} with factor {yarn_factor}")

    duplicate_model_to_yarn_repo(original_model_id, yarn_model_id)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            unique_dir = f"{tmp_dir}/{uuid.uuid4().hex}"
            config_path = hf_hub_download(
                repo_id=yarn_model_id,
                filename="config.json",
                token=YARN_HUGGINGFACE_TOKEN,
                local_dir=unique_dir,
            )

            with open(config_path, "r") as f:
                config = json.load(f)

            modified_config = apply_yarn_extension_to_config(config, yarn_factor)

            modified_config_path = f"{unique_dir}/config_modified.json"
            with open(modified_config_path, "w") as f:
                json.dump(modified_config, f, indent=2)

            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(modified_config_path),
                path_in_repo="config.json",
                repo_id=yarn_model_id,
                token=YARN_HUGGINGFACE_TOKEN,
                commit_message=f"Apply YaRN extension with factor {yarn_factor}",
            )

        logger.info(f"Successfully created YaRN extended model: {yarn_model_id}")
        return yarn_model_id

    except Exception as e:
        logger.error(f"Failed to apply YaRN extension to model {yarn_model_id}: {e}")
        raise
