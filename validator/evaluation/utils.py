import asyncio
import base64
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from io import BytesIO
from typing import TYPE_CHECKING
from typing import Any
from uuid import UUID

import basilica
import requests
from datasets import get_dataset_config_names
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoConfig
from transformers import AutoModelForCausalLM

from core.models.payload_models import DockerEvaluationResults
from core.models.payload_models import EvaluationResultImage
from core.models.payload_models import EvaluationResultText
from validator.core import constants as vcst
from validator.utils.logging import get_logger
from validator.utils.retry_utils import retry_on_5xx

if TYPE_CHECKING:
    from core.models.utility_models import TaskType
    from validator.core.config import Config


logger = get_logger(__name__)
hf_api = HfApi()

EVAL_RESULT_STATUS_PATH = "/result"
_BASILICA_LOG_LINE_OFFSETS: dict[str, int] = {}


async def notify_evaluation_exception(
    config: "Config | None",
    *,
    task_id: str,
    task_type: "TaskType | Any",
    context: str,
    error: Exception | str,
    hotkeys: list[str] | None = None,
    repos: list[str] | None = None,
    deployment_ids: list[str] | None = None,
) -> None:
    if config is None or not config.discord_url:
        return

    try:
        import httpx

        details = str(error)
        if len(details) > 900:
            details = f"{details[:900]}..."
        task_type_value = getattr(task_type, "value", str(task_type))

        def _format_items(items: list[str], limit: int = 12) -> str:
            shown = items[:limit]
            suffix = f", ... (+{len(items) - limit} more)" if len(items) > limit else ""
            return f"{', '.join(shown)}{suffix}"

        lines = [
            "Evaluation exception",
            f"Task: {task_id}",
            f"Type: {task_type_value}",
            f"Context: {context}",
        ]
        if hotkeys:
            lines.append(f"Hotkeys: {_format_items(hotkeys)}")
        if repos:
            lines.append(f"Repos: {_format_items(repos)}")
        if deployment_ids:
            lines.append(f"Deployment IDs: {_format_items(deployment_ids)}")
        lines.append(f"Error: {details}")

        async with httpx.AsyncClient() as client:
            await client.post(config.discord_url, json={"content": "\n".join(lines)}, timeout=10)
    except Exception as notify_exc:
        logger.error(f"Failed to send Discord evaluation exception notification: {notify_exc}")


async def task_deployment_ids_for_hotkeys(
    task_id: UUID,
    config: "Config | None",
    hotkeys: list[str],
) -> list[str]:
    if config is None or config.psql_db is None or not hotkeys:
        return []

    hotkey_set = set(hotkeys)
    try:
        from validator.db.sql.tasks import get_task_evaluation_rows

        rows = await get_task_evaluation_rows(task_id, config.psql_db)
    except Exception as exc:
        logger.warning(f"Failed to load evaluation deployment IDs for task {task_id}: {exc}")
        return []

    deployment_ids = {
        row.get("deployment_id")
        for row in rows
        if row.get("hotkey") in hotkey_set and row.get("deployment_id")
    }
    return sorted(deployment_ids)


def configure_eval_logging() -> None:
    """Configure root logger for eval containers (stderr, replaces existing handlers)."""
    level_name = os.getenv("EVAL_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(level)
    for existing in root.handlers[:]:
        root.removeHandler(existing)
        try:
            existing.close()
        except Exception:
            pass
    root.addHandler(handler)


def stop_process(proc: subprocess.Popen | None, name: str) -> None:
    """Gracefully stop a subprocess, escalating to SIGKILL if needed."""
    if proc is None:
        return
    try:
        if proc.poll() is None:
            logger.info("Stopping %s (pid=%s)", name, proc.pid)
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=10)
    except Exception as exc:
        logger.warning("Failed to stop %s cleanly: %s", name, exc)


def _log_eval_step(eval_logger: logging.Logger, step: str, **fields) -> None:
    field_text = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    eval_logger.info(f"eval_step={step} {field_text}".rstrip())


def normalize_rewards_and_compute_loss(evaluation_results: dict) -> dict:
    """
    Normalize rewards across repos and compute final evaluation loss with KL penalty.

    Steps:
    1. For each reward type, normalize values across repos by dividing by max (after shifting if negative)
    2. Apply weights to normalized rewards (weights sum to 1)
    3. Sum weighted rewards to get final score in [0,1] range
    4. Apply KL penalty: score - (BETA_GRPO * kl_divergence)

    Special case: 2 repos with negative rewards map to [0.25, 0.75] to avoid extreme scores.
    """
    repo_keys = [key for key in evaluation_results.keys() if key != "model_params_count"]

    if len(repo_keys) < 2:
        return evaluation_results

    reward_collections = {}
    for repo_key in repo_keys:
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):
            continue

        final_raw_rewards = repo_data.get("final_raw_rewards", {})

        for reward_name, reward_value in final_raw_rewards.items():
            if reward_name not in reward_collections:
                reward_collections[reward_name] = []
            reward_collections[reward_name].append((repo_key, reward_value))

    normalized_rewards_per_repo = {repo_key: {} for repo_key in repo_keys}

    for reward_name, repo_value_pairs in reward_collections.items():
        if len(repo_value_pairs) < 2:
            for repo_key, value in repo_value_pairs:
                normalized_rewards_per_repo[repo_key][reward_name] = 1.0
            continue

        values = [value for _, value in repo_value_pairs]
        min_value = min(values)
        has_negatives = min_value < 0
        shifted_values = [(repo, value - min_value) for repo, value in repo_value_pairs] if has_negatives else repo_value_pairs
        max_shifted = max(value for _, value in shifted_values)

        if len(repo_value_pairs) == 2 and has_negatives:
            sorted_pairs = sorted(shifted_values, key=lambda x: x[1])
            normalized_rewards_per_repo[sorted_pairs[0][0]][reward_name] = 0.25
            normalized_rewards_per_repo[sorted_pairs[1][0]][reward_name] = 0.75
        elif max_shifted > 0:
            for repo, shifted_value in shifted_values:
                normalized_rewards_per_repo[repo][reward_name] = shifted_value / max_shifted
        else:
            for repo, _ in repo_value_pairs:
                normalized_rewards_per_repo[repo][reward_name] = 1.0

    final_scores = []
    for repo_key in repo_keys:
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):
            continue

        weights = repo_data.get("weights", {})
        normalized_rewards = normalized_rewards_per_repo.get(repo_key, {})
        weighted_sum = 0.0
        for reward_name, normalized_value in normalized_rewards.items():
            weight = weights.get(reward_name, 1.0)
            weighted_sum += normalized_value * weight

        final_scores.append(weighted_sum)

    for i, repo_key in enumerate(repo_keys):
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):
            continue

        if i < len(final_scores):
            kl_divergence = repo_data.get("kl_divergence", 0.0)
            repo_data["eval_loss"] = final_scores[i] - (vcst.BETA_GRPO * kl_divergence)

    return evaluation_results


def process_evaluation_results(results: dict, is_image: bool = False) -> DockerEvaluationResults:
    model_params_count = results.pop("model_params_count", 0)

    processed_results = {}
    for repo, result in results.items():
        if isinstance(result, str) and not isinstance(result, dict):
            processed_results[repo] = Exception(result)
        else:
            if is_image:
                result["is_finetune"] = True
                processed_results[repo] = EvaluationResultImage.model_validate(result)
            else:
                processed_results[repo] = EvaluationResultText.model_validate(result)

    return DockerEvaluationResults(
        results=processed_results,
        base_model_params_count=model_params_count,
    )


def clean_basilica_log_line(raw_line: str) -> str:
    line = raw_line.strip()
    if not line:
        return ""
    line = re.sub(r"^data:\s*", "", line).rstrip(", ")
    for _ in range(2):
        try:
            parsed = json.loads(line)
        except Exception:
            break

        if isinstance(parsed, dict):
            extracted = parsed.get("message") or parsed.get("log") or parsed.get("data")
            if isinstance(extracted, str) and extracted.strip():
                line = extracted.strip()
                continue
            line = str(parsed)
            break

        if isinstance(parsed, str):
            line = parsed.strip()
            continue

        line = str(parsed)
        break
    if "\\u001b" in line or "\\x1b" in line:
        try:
            line = bytes(line, "utf-8").decode("unicode_escape")
        except Exception:
            pass

    line = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", line)
    line = re.sub(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?\]\s*", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def log_basilica_logs_block(eval_logger: logging.Logger, repo: str, deployment_name: str, deployment) -> None:
    try:
        raw_logs = deployment.logs()
    except Exception as e:
        eval_logger.warning(f"[BASILICA_LOG_FETCH_FAILED] repo={repo} deployment={deployment_name} error={e}")
        return

    if not raw_logs:
        eval_logger.info(f"[BASILICA_LOGS] repo={repo} deployment={deployment_name} lines=0 message=\"no logs returned\"")
        return

    if isinstance(raw_logs, bytes):
        raw_logs = raw_logs.decode("utf-8", errors="replace")

    lines = []
    for raw_line in str(raw_logs).splitlines():
        cleaned = clean_basilica_log_line(raw_line)
        if cleaned:
            lines.append(cleaned)

    if not lines:
        eval_logger.info(
            f"[BASILICA_LOGS] repo={repo} deployment={deployment_name} lines=0 "
            "message=\"log payload present but no parsable lines\""
        )
        return

    previous_count = _BASILICA_LOG_LINE_OFFSETS.get(deployment_name, 0)
    if previous_count > len(lines):
        previous_count = 0
    new_lines = lines[previous_count:]
    _BASILICA_LOG_LINE_OFFSETS[deployment_name] = len(lines)

    if not new_lines:
        eval_logger.info(
            f"[BASILICA_LOGS] repo={repo} deployment={deployment_name} new_lines=0 total_lines={len(lines)}"
        )
        return

    eval_logger.info(
        f"[BASILICA_LOGS] repo={repo} deployment={deployment_name} "
        f"new_lines={len(new_lines)} total_lines={len(lines)}"
    )
    for line_number, line in enumerate(new_lines, start=previous_count + 1):
        eval_logger.info(f"[BASILICA_LOG] repo={repo} deployment={deployment_name} line={line_number} | {line}")


def deployment_is_healthy(deployment, health_path: str = "/health", timeout: int = 8) -> bool:
    try:
        response = requests.get(f"{deployment.url}{health_path}", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


async def delete_deployment_if_exists(deployment_name: str) -> None:
    try:
        client = basilica.BasilicaClient()
        deployments = await asyncio.to_thread(client.list)
        for dep in deployments:
            if getattr(dep, "name", None) == deployment_name:
                await asyncio.to_thread(dep.delete)
                return
    except Exception:
        return


async def cleanup_basilica_deployments_by_name(deployment_names: set[str]) -> None:
    """Cleanup specific Basilica deployments by name."""
    if not deployment_names:
        return
    try:
        client = basilica.BasilicaClient()
        deployments = await asyncio.to_thread(client.list)
    except Exception as e:
        logger.warning(f"Failed to list deployments for final cleanup: {e}")
        return

    by_name = {getattr(dep, "name", None): dep for dep in deployments}
    cleaned = 0
    for name in deployment_names:
        dep = by_name.get(name)
        if dep is None:
            continue
        try:
            await asyncio.to_thread(dep.delete)
            cleaned += 1
        except Exception as e:
            logger.warning(f"Failed final cleanup for deployment {name}: {e}")

    if cleaned:
        logger.info(f"Final cleanup removed {cleaned} lingering deployments for this evaluation batch")


async def cleanup_all_basilica_deployments() -> None:
    """Delete every visible Basilica deployment after the evaluation queue drains."""
    try:
        client = basilica.BasilicaClient()
        deployments = await asyncio.to_thread(client.list)
    except Exception as e:
        logger.warning(f"Failed to list deployments for drained evaluation cleanup: {e}")
        return

    cleaned = 0
    for dep in deployments:
        deployment_name = getattr(dep, "name", "unknown")
        try:
            await asyncio.to_thread(dep.delete)
            cleaned += 1
        except Exception as e:
            logger.warning(f"Failed drained evaluation cleanup for deployment {deployment_name}: {e}")

    logger.info(f"Drained evaluation cleanup removed {cleaned}/{len(deployments)} Basilica deployments")


def create_basilica_eval_runner_source(command: list[str], result_path: str) -> str:
    """Create a generic eval runner source with health and result endpoints.

    The runner executes a single eval command, then serves the parsed
    `evaluation_results.json` payload on `/result`.
    """
    command_json = json.dumps(command)
    result_path_json = json.dumps(result_path)
    return f"""import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

COMMAND = {command_json}
RESULT_PATH = {result_path_json}
RESULT_STATUS_PATH = "{EVAL_RESULT_STATUS_PATH}"

_state = {{
    "status": "running",
    "result": None,
    "error": None,
}}

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{{"status":"ok"}}')
            return
        if self.path == RESULT_STATUS_PATH:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(_state).encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return

def _run_eval():
    try:
        proc = subprocess.run(COMMAND, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Eval command failed with exit code {{proc.returncode}}")
        with open(RESULT_PATH, "r", encoding="utf-8") as f:
            _state["result"] = json.load(f)
        _state["status"] = "completed"
    except Exception as e:
        if _state["status"] != "completed":
            _state["status"] = "failed"
            _state["error"] = str(e)

def main():
    server = HTTPServer(("0.0.0.0", 8000), _Handler)
    worker = threading.Thread(target=_run_eval, daemon=True)
    worker.start()
    server.serve_forever()

if __name__ == "__main__":
    main()
"""


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


@retry_on_5xx()
def check_lora_has_added_tokens(model_id: str, local_files_only: bool = False) -> bool:
    """
    Check if a LoRA repo includes added_tokens.json.

    This is used to decide whether we need to merge LoRA into base model
    before launching SGLang.
    """
    ADDED_TOKENS_FILE = "added_tokens.json"
    try:
        if local_files_only:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            repo_path = os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))
            if os.path.exists(repo_path):
                for root, dirs, files in os.walk(repo_path):
                    if ".no_exist" in root:
                        continue
                    if ADDED_TOKENS_FILE in files:
                        token_file = os.path.join(root, ADDED_TOKENS_FILE)
                        if os.path.getsize(token_file) > 0:
                            return True
            return False
        return ADDED_TOKENS_FILE in hf_api.list_repo_files(model_id)
    except Exception as e:
        logger.error(f"Error checking for added_tokens.json in LoRA repo: {e}")
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


def wait_for_basilica_health(url: str, timeout: int = 3600, path: str = "/v1/models") -> bool:
    """Wait for Basilica service to be healthy."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}{path}", timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)

    error_msg = f"Service at {url} did not become healthy within {timeout} seconds"
    raise TimeoutError(error_msg)
