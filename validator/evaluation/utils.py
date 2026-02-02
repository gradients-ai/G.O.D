import base64
import os
import shutil
import subprocess
import tempfile
import time
from io import BytesIO

import basilica
import requests
from datasets import get_dataset_config_names
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoConfig
from transformers import AutoModelForCausalLM

from validator.core import constants as cst
from validator.utils.logging import get_logger
from validator.utils.retry_utils import retry_on_5xx


logger = get_logger(__name__)
hf_api = HfApi()


def model_is_a_finetune(original_repo: str, finetuned_model: AutoModelForCausalLM, local_files_only: bool = False) -> bool:
    max_retries = 3
    base_delay = 2

    # For local files, try to load config directly from snapshot
    if local_files_only:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        cache_path = os.path.join(cache_dir, "hub", f"models--{original_repo.replace('/', '--')}")

        if os.path.exists(cache_path):
            snapshots_dir = os.path.join(cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = sorted(os.listdir(snapshots_dir))

                for snapshot in snapshots:
                    snapshot_path = os.path.join(snapshots_dir, snapshot)
                    if ".no_exist" in snapshot_path:
                        continue
                    config_path = os.path.join(snapshot_path, "config.json")

                    if os.path.exists(config_path) and os.path.getsize(config_path) > 0:
                        logger.info(f"Loading original model config from snapshot: {snapshot}")
                        try:
                            original_config = AutoConfig.from_pretrained(snapshot_path, local_files_only=True)
                            logger.info("Successfully loaded config from snapshot")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load config from snapshot {snapshot}: {e}")
                            continue
                else:
                    logger.error(f"No valid config found in snapshots for {original_repo}")
                    return False
            else:
                logger.error(f"No snapshots directory found for {original_repo}")
                return False
        else:
            logger.error(f"No cache found for {original_repo}")
            return False
    else:
        # Standard online loading with retries
        for attempt in range(max_retries):
            try:
                kwargs = {"token": os.environ.get("HUGGINGFACE_TOKEN")}

                original_config = AutoConfig.from_pretrained(original_repo, **kwargs)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e

                error_msg = str(e).lower()
                if any(
                    pattern in error_msg for pattern in ["connection", "timeout", "5xx", "too many requests", "couldn't connect"]
                ):
                    delay = base_delay * (2**attempt)
                    logger.info(
                        f"HuggingFace connection issue (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    raise e
    finetuned_config = finetuned_model.config

    try:
        architecture_classes_match = finetuned_config.architectures == original_config.architectures
    except Exception as e:
        logger.debug(f"There is an issue with checking the architecture classes {e}")
        architecture_classes_match = False

    attrs_to_compare = [
        "architectures",
        "hidden_size",
        "n_layer",
        "intermediate_size",
        "head_dim",
        "hidden_act",
        "model_type",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    ]
    architecture_same = True
    for attr in attrs_to_compare:
        if getattr(original_config, attr, None) is not None:
            if not hasattr(finetuned_config, attr):
                architecture_same = False
                break
            if getattr(original_config, attr) != getattr(finetuned_config, attr):
                architecture_same = False
                break

    logger.info(f"Architecture same: {architecture_same}, Architecture classes match: {architecture_classes_match}")
    return architecture_same and architecture_classes_match


@retry_on_5xx()
def check_for_lora(model_id: str, local_files_only: bool = False) -> bool:
    """
    Check if a Hugging Face model has LoRA adapters by looking for adapter_config.json.

    Args:
        model_id (str): The Hugging Face model ID (e.g., 'username/model-name') or path
        local_files_only (bool): If True, only check local files without making API calls

    Returns:
        bool: True if it's a LoRA adapter, False otherwise
    """
    LORA_CONFIG_FILE = "adapter_config.json"
    try:
        if local_files_only:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            repo_path = os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))
            if os.path.exists(repo_path):
                for root, dirs, files in os.walk(repo_path):
                    if ".no_exist" in root:
                        continue
                    if LORA_CONFIG_FILE in files:
                        config_path = os.path.join(root, LORA_CONFIG_FILE)
                        if os.path.getsize(config_path) > 0:
                            return True
            return False
        else:
            return LORA_CONFIG_FILE in hf_api.list_repo_files(model_id)
    except Exception as e:
        logger.error(f"Error checking for LoRA adapters: {e}")
        return False


def get_default_dataset_config(dataset_name: str) -> str | None:
    try:
        logger.info(dataset_name)
        config_names = get_dataset_config_names(dataset_name)
    except Exception:
        return None
    if config_names:
        logger.info(f"Taking the first config name: {config_names[0]} for dataset: {dataset_name}")
        # logger.info(f"Dataset {dataset_name} has configs: {config_names}. Taking the first config name: {config_names[0]}")
        return config_names[0]
    else:
        return None


def adjust_image_size(image: Image.Image) -> Image.Image:
    width, height = image.size

    if width > height:
        new_width = 1024
        new_height = int((height / width) * 1024)
    else:
        new_height = 1024
        new_width = int((width / height) * 1024)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    width, height = image.size
    crop_width = min(width, new_width)
    crop_height = min(height, new_height)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    image = image.crop((left, top, right, bottom))

    return image


def base64_to_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image_stream = BytesIO(image_data)
    image = Image.open(image_stream)
    return image


def download_from_huggingface(repo_id: str, filename: str, local_dir: str) -> str:
    # Use a temp folder to ensure correct file placement
    try:
        local_filename = f"models--{repo_id.replace('/', '--')}.safetensors"
        final_path = os.path.join(local_dir, local_filename)
        if os.path.exists(final_path):
            logger.info(f"File {filename} already exists. Skipping download.")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=temp_dir)
                shutil.move(temp_file_path, final_path)
            logger.info(f"File {filename} downloaded successfully")
        return final_path
    except Exception as e:
        logger.error(f"Error downloading file: {e}")


def list_supported_images(dataset_path: str, extensions: tuple) -> list[str]:
    return [file_name for file_name in os.listdir(dataset_path) if file_name.lower().endswith(extensions)]


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    img_format = image.format if image.format else "PNG"
    image.save(buffer, format=img_format)
    return base64.b64encode(buffer.getvalue()).decode()


def read_prompt_file(text_file_path: str) -> str:
    if os.path.exists(text_file_path):
        with open(text_file_path, "r", encoding="utf-8") as text_file:
            return text_file.read()
    return None


def create_sglang_deployment_source(base_model: str, lora_model: str | None = None, seed: int = 42) -> str:
    """Create the source code for SGLang deployment with model download.
    
    Args:
        base_model: Base model name/path
        lora_model: Optional LoRA model name/path
        seed: Random seed for determinism
    """
    # Build SGLang command
    sglang_args = [
        "python3 -m sglang.launch_server",
        f"--model-path {base_model}",
    ]
    if lora_model:
        sglang_args.append("--enable-lora --lora-paths trained_lora=/tmp/lora/trained_lora --lora-backend triton")
    sglang_args.extend([
        "--host 0.0.0.0 --port 8000",
        "--tensor-parallel-size 1 --dtype float16",
        "--enable-deterministic-inference",
        f"--random-seed {seed}",
    ])
    sglang_cmd = " ".join(sglang_args)
    
    # Download section
    download_lora_code = f'''
# Download LoRA
print("Downloading LoRA: {lora_model}")
from huggingface_hub import snapshot_download
snapshot_download("{lora_model}", local_dir="/tmp/lora/trained_lora", local_dir_use_symlinks=False)
print("LoRA downloaded")
''' if lora_model else ''
    
    server_type = "with LoRA" if lora_model else ""
    
    return f'''import subprocess
import sys
import os

# HuggingFace cache configuration
os.environ['HF_HOME'] = '/root/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/root/.cache/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/.cache/huggingface'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# Determinism environment variables
os.environ['PYTHONHASHSEED'] = '{seed}'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

# Download base model
print("Downloading base model: {base_model}")
from huggingface_hub import snapshot_download
snapshot_download("{base_model}", local_files_only=False)
print("Base model downloaded")
{download_lora_code}
# Start SGLang server {server_type}
print("Starting SGLang server {server_type}...")
subprocess.Popen("{sglang_cmd}", shell=True).wait()
'''


def deploy_sglang_basilica(
    base_model: str,
    lora_model: str | None,
    deployment_name: str,
    seed: int = 42,
) -> basilica.Deployment:
    """Deploy SGLang server to Basilica.
    
    Args:
        base_model: Base model name/path
        lora_model: Optional LoRA model name/path
        deployment_name: Name for the deployment
        seed: Random seed for determinism (default: 42)
    """

    func_source = create_sglang_deployment_source(base_model, lora_model, seed)
    
    client = basilica.BasilicaClient()
    deployment = client.deploy(
        name=deployment_name,
        source=func_source,
        image=cst.BASILICA_SGLANG_IMAGE,
        port=8000,
        gpu_count=cst.BASILICA_SGLANG_GPU_COUNT,
        gpu_models=cst.BASILICA_SGLANG_GPU_MODELS,
        min_gpu_memory_gb=cst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
        memory=cst.BASILICA_SGLANG_MEMORY,
        ttl_seconds=cst.BASILICA_SGLANG_TTL_SECONDS,
        timeout=cst.BASILICA_SGLANG_TIMEOUT,
        env={
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONHASHSEED": str(seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0",
        },
    )
    
    try:
        deployment.wait_until_ready(timeout=cst.BASILICA_SGLANG_TIMEOUT)
    except Exception as e:
        error_msg = f"[{deployment_name}] Deployment wait failed: {e}"
        logger.warning(error_msg)
    
    return deployment


def deploy_env_basilica(
    deployment_name: str,
    env_image: str | None = None
) -> basilica.Deployment:
    """Deploy Environment Server to Basilica.
    
    Args:
        deployment_name: Name for the deployment
        env_image: Optional custom image
    """
    
    client = basilica.BasilicaClient()

    image_to_use = env_image if env_image else cst.BASILICA_AGENTGYM_IMAGE
    
    deployment = client.deploy(
        name=deployment_name,
        image=image_to_use,
        port=8000,
        cpu=cst.BASILICA_ENV_CPU,
        memory=cst.BASILICA_ENV_MEMORY,
        ttl_seconds=cst.BASILICA_ENV_TTL_SECONDS,
        timeout=cst.BASILICA_ENV_TIMEOUT
    )
    
    return deployment


def wait_for_basilica_health(url: str, timeout: int = 600, path: str = "/v1/models") -> bool:
    """Wait for Basilica service to be healthy."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}{path}", timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(5)
    
    error_msg = f"Service at {url} did not become healthy within {timeout} seconds"
    raise TimeoutError(error_msg)
