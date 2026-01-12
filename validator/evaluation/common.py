import json
import os
import re
import time
from math import ceil

import psutil
import torch
import torch.nn.functional as F
import yaml
from accelerate.utils import find_executable_batch_size
from axolotl.utils.dict import DictDefault
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TrainerCallback

from core.config.config_handler import create_dataset_entry
from validator.core import constants as cst
from validator.core.models import EvaluationArgs
from validator.utils.logging import get_logger
from validator.utils.retry_utils import retry_on_5xx


logger = get_logger(__name__)


def log_memory_stats():
    """Log detailed memory statistics for debugging."""
    logger.info("===== MEMORY STATS =====")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**2
            logger.info(
                f"GPU {i} Memory: Allocated: {allocated:.2f} MB, "
                f"Reserved: {reserved:.2f} MB, "
                f"Max Allocated: {max_allocated:.2f} MB"
            )
    else:
        logger.info("No CUDA devices available")

    ram = psutil.Process().memory_info()
    system_memory = psutil.virtual_memory()
    logger.info(f"RAM Usage: RSS: {ram.rss / 1024**2:.2f} MB, VMS: {ram.vms / 1024**2:.2f} MB")
    logger.info(
        f"System Memory: Total: {system_memory.total / 1024**2:.2f} MB, "
        f"Available: {system_memory.available / 1024**2:.2f} MB, "
        f"Used: {(system_memory.total - system_memory.available) / 1024**2:.2f} MB "
        f"({system_memory.percent}%)"
    )
    logger.info("========================")


class ProgressLoggerCallback(TrainerCallback):
    """A callback that logs the progress of the evaluation every log_interval_seconds seconds."""

    def __init__(self, log_interval_seconds):
        self.step = 0
        self.last_log_time = time.time()
        self.log_interval_seconds = log_interval_seconds
        logger.info(f"Initialized ProgressLoggerCallback with log interval of {log_interval_seconds} seconds")

    def on_prediction_step(self, args, state, control, **kwargs):
        self.step += 1
        current_time = time.time()

        if current_time - self.last_log_time >= self.log_interval_seconds:
            self.last_log_time = current_time
            logger.info(f"Evaluation step: {self.step}")

        return control


def create_finetuned_cache_dir():
    """Create and return a dedicated cache directory for finetuned models."""
    finetuned_cache_dir = os.path.join(cst.DOCKER_EVAL_HF_CACHE_DIR, "finetuned_repos")
    os.makedirs(finetuned_cache_dir, exist_ok=True)
    return finetuned_cache_dir


def patch_base_model_config_if_needed(base_model_name: str, cache_dir: str, context: str = "") -> bool:
    """
    Patch base model config.json if head_dim or partial_rotary_factor is None.
    
    This fixes issues with Yarn models where head_dim is None in the config,
    which causes TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'
    during model loading.
    
    Args:
        base_model_name: The base model name (e.g., "NousResearch/Yarn-Mistral-7b-128k")
        cache_dir: The HuggingFace cache directory
        context: Optional context string for logging (e.g., "fallback path")
    
    Returns:
        True if a patch was applied, False otherwise
    """
    try:
        base_cache_path = os.path.join(cache_dir, "hub", f"models--{base_model_name.replace('/', '--')}")
        
        if not os.path.exists(base_cache_path):
            return False
        
        base_snapshots_dir = os.path.join(base_cache_path, "snapshots")
        if not os.path.exists(base_snapshots_dir):
            return False
        
        base_snapshots = sorted(os.listdir(base_snapshots_dir))
        if not base_snapshots:
            return False
        
        base_snapshot_path = os.path.join(base_snapshots_dir, base_snapshots[-1])
        base_config_file = os.path.join(base_snapshot_path, "config.json")
        
        if not os.path.exists(base_config_file):
            return False
        
        with open(base_config_file, 'r') as cfg_f:
            base_config_dict = json.load(cfg_f)
        
        needs_patch = False
        
        if base_config_dict.get("head_dim") is None:
            if base_config_dict.get("hidden_size") and base_config_dict.get("num_attention_heads"):
                calculated_head_dim = base_config_dict["hidden_size"] // base_config_dict["num_attention_heads"]
                base_config_dict["head_dim"] = calculated_head_dim
                context_str = f" ({context})" if context else ""
                logger.info(f"Patching head_dim={calculated_head_dim} in base model config{context_str}")
                needs_patch = True
        
        if base_config_dict.get("partial_rotary_factor") is None and base_config_dict.get("rope_scaling", {}).get("type") == "yarn":
            base_config_dict["partial_rotary_factor"] = 1.0
            context_str = f" ({context})" if context else ""
            logger.info(f"Patching partial_rotary_factor=1.0 in base model config{context_str}")
            needs_patch = True
        
        if needs_patch:
            with open(base_config_file, 'w') as cfg_f:
                json.dump(base_config_dict, cfg_f, indent=2)
            context_str = f" ({context})" if context else ""
            logger.info(f"Patched base model config.json at {base_config_file}{context_str}")
            return True
        
        return False
    except Exception as e:
        logger.warning(f"Failed to patch base model config for {base_model_name}: {e}", exc_info=True)
        return False


@retry_on_5xx()
def load_model(model_name_or_path: str, is_base_model: bool = False, local_files_only: bool = False) -> AutoModelForCausalLM:
    try:
        # For local files, try to use the snapshot path directly
        if local_files_only:
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            cache_path = os.path.join(cache_dir, "hub", f"models--{model_name_or_path.replace('/', '--')}")

            if os.path.exists(cache_path):
                snapshots_dir = os.path.join(cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = sorted(os.listdir(snapshots_dir))  # Sort to get most recent

                    for snapshot in snapshots:
                        snapshot_path = os.path.join(snapshots_dir, snapshot)
                        files = os.listdir(snapshot_path)

                        has_model_files = any(f.endswith((".bin", ".safetensors")) for f in files)
                        has_config = "config.json" in files

                        if has_model_files and has_config:
                            try:
                                model = AutoModelForCausalLM.from_pretrained(
                                    snapshot_path, device_map="auto", torch_dtype=torch.bfloat16, local_files_only=True
                                )
                                return model
                            except Exception as e:
                                logger.warning(f"Failed to load from snapshot {snapshot}: {e}")
                                continue

        # Fallback to standard loading
        # Set cache_dir based on whether it's base model or finetuned
        if local_files_only:
            cache_dir = os.path.expanduser("~/.cache/huggingface")
        elif not is_base_model:
            cache_dir = create_finetuned_cache_dir()
        else:
            cache_dir = None

        kwargs = {
            "device_map": "auto",
            "cache_dir": cache_dir,
            "torch_dtype": torch.bfloat16,
            "local_files_only": local_files_only,
        }
        if not local_files_only:
            kwargs["token"] = os.environ.get("HUGGINGFACE_TOKEN")

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        return model
    except RuntimeError as e:
        error_msg = str(e)
        if "size mismatch for" in error_msg and ("lm_head.weight" in error_msg or "model.embed_tokens.weight" in error_msg):
            pattern = re.search(r"shape torch\.Size\(\[(\d+), (\d+)\]\).*shape.*torch\.Size\(\[(\d+), \2\]\)", error_msg)
            if pattern and abs(int(pattern.group(1)) - int(pattern.group(3))) == 1:
                logger.info("Detected vocabulary size off-by-one error, attempting to load with ignore_mismatched_sizes=True")
                kwargs["ignore_mismatched_sizes"] = True
                return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise  # Re-raise the exception to trigger retry


@retry_on_5xx()
def load_tokenizer(original_model: str, local_files_only: bool = False) -> AutoTokenizer:
    try:
        # For local files, try to use the snapshot path directly
        if local_files_only:
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            cache_path = os.path.join(cache_dir, "hub", f"models--{original_model.replace('/', '--')}")

            if os.path.exists(cache_path):
                snapshots_dir = os.path.join(cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = sorted(os.listdir(snapshots_dir))  # Sort to get most recent

                    for snapshot in snapshots:
                        snapshot_path = os.path.join(snapshots_dir, snapshot)
                        files = os.listdir(snapshot_path)
                        tokenizer_files = [f for f in files if "tokenizer" in f.lower() or f.endswith(".model")]

                        if tokenizer_files:
                            try:
                                tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
                                return tokenizer
                            except Exception as e:
                                logger.warning(f"Failed to load from snapshot {snapshot}: {e}")
                                continue

        # Fallback to standard loading
        kwargs = {
            "local_files_only": local_files_only,
            "cache_dir": os.path.expanduser("~/.cache/huggingface") if local_files_only else None,
        }
        if not local_files_only:
            kwargs["token"] = os.environ.get("HUGGINGFACE_TOKEN")

        tokenizer = AutoTokenizer.from_pretrained(original_model, **kwargs)
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        raise  # Re-raise the exception to trigger retry


@retry_on_5xx()
def load_finetuned_model(repo: str, local_files_only: bool = False) -> AutoPeftModelForCausalLM:
    try:
        # For local files, try to use the snapshot path directly
        if local_files_only:
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            cache_path = os.path.join(cache_dir, "hub", f"models--{repo.replace('/', '--')}")

            if os.path.exists(cache_path):
                snapshots_dir = os.path.join(cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = sorted(os.listdir(snapshots_dir))  # Sort to get most recent

                    for snapshot in snapshots:
                        snapshot_path = os.path.join(snapshots_dir, snapshot)
                        files = os.listdir(snapshot_path)

                        has_adapter = any("adapter" in f.lower() for f in files)

                        if has_adapter:
                            try:
                                adapter_config_path = os.path.join(snapshot_path, "adapter_config.json")
                                if os.path.exists(adapter_config_path):
                                    with open(adapter_config_path) as f:
                                        adapter_config = json.load(f)
                                        base_model_name = adapter_config.get("base_model_name_or_path")
                                        
                                        if base_model_name:
                                            patch_base_model_config_if_needed(base_model_name, cache_dir)
                                
                                model = AutoPeftModelForCausalLM.from_pretrained(
                                    snapshot_path,
                                    is_trainable=False,
                                    device_map="auto",
                                    torch_dtype=torch.bfloat16,
                                    local_files_only=True,
                                )
                                return model
                            except Exception as e:
                                logger.warning(f"Failed to load from snapshot {snapshot}: {e}", exc_info=True)
                                continue

        # Fallback to standard loading
        # Set cache_dir based on local_files_only
        if local_files_only:
            cache_dir = os.path.expanduser("~/.cache/huggingface")
        else:
            cache_dir = create_finetuned_cache_dir()

        kwargs = {
            "is_trainable": False,
            "device_map": "auto",
            "cache_dir": cache_dir,
            "torch_dtype": torch.bfloat16,
            "local_files_only": local_files_only,
        }
        if not local_files_only:
            kwargs["token"] = os.environ.get("HUGGINGFACE_TOKEN")

        model = AutoPeftModelForCausalLM.from_pretrained(repo, **kwargs)
        return model
    except RuntimeError as e:
        error_msg = str(e)
        if "size mismatch for" in error_msg and ("lm_head.weight" in error_msg or "model.embed_tokens.weight" in error_msg):
            pattern = re.search(r"shape torch\.Size\(\[(\d+), (\d+)\]\).*shape.*torch\.Size\(\[(\d+), \2\]\)", error_msg)
            if pattern and abs(int(pattern.group(1)) - int(pattern.group(3))) == 1:
                logger.info("Detected vocabulary size off-by-one error, attempting to load with ignore_mismatched_sizes=True")
                kwargs["ignore_mismatched_sizes"] = True
                return AutoPeftModelForCausalLM.from_pretrained(repo, **kwargs)

        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise  # Re-raise the exception to trigger retry


def count_model_parameters(model):
    """Count the total number of parameters in a model."""
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception as e:
        logger.error(f"Failed to count model parameters: {e}")
        return 0


def load_results_dict():
    """Load existing evaluation results or create an empty dict if not found."""
    results_dict = {}
    output_dir = os.path.dirname(cst.CONTAINER_EVAL_RESULTS_PATH)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(cst.CONTAINER_EVAL_RESULTS_PATH):
        try:
            with open(cst.CONTAINER_EVAL_RESULTS_PATH, "r") as f:
                results_dict = json.load(f)
        except Exception as e:
            logger.error(f"Could not read existing results from {cst.CONTAINER_EVAL_RESULTS_PATH}, starting fresh: {e}")

    return results_dict


def save_results_dict(results_dict, model_id=None):
    """Save evaluation results to file."""
    with open(cst.CONTAINER_EVAL_RESULTS_PATH, "w") as f:
        json.dump(results_dict, f, indent=2)

    msg = "Saved evaluation results"
    if model_id:
        msg += f" for {model_id}"

    logger.info(msg)
    logger.info(json.dumps(results_dict, indent=2))


def check_env_variables(required_vars):
    """Check for required environment variables."""
    env_vars = {var: os.environ.get(var, "") for var in required_vars}
    missing = [var for var, value in env_vars.items() if not value]

    if missing:
        logger.error("Missing required environment variables: " + ", ".join(missing))
        return False, env_vars

    return True, env_vars


def _log_dataset_and_model_info(
    eval_dataset: Dataset,
    language_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> None:
    logger.info(f"Eval dataset sample: {eval_dataset[0]}")
    logger.info(f"Model type: {type(language_model)}")
    logger.info(f"Model config: {language_model.config}")
    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
    logger.info(f"Model vocabulary size: {language_model.config.vocab_size}")


def log_cuda_info():
    """Log information about CUDA availability and devices."""
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA not available")


def _load_and_update_evaluation_config(
    evaluation_args: EvaluationArgs,
    finetuned_model: AutoModelForCausalLM,
    config_path: str,
) -> DictDefault:
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    dataset_entry = create_dataset_entry(
        dataset=evaluation_args.dataset,
        dataset_type=evaluation_args.dataset_type,
        file_format=evaluation_args.file_format,
        is_eval=True,
    )
    config_dict["datasets"] = [dataset_entry]

    max_embeddings = getattr(finetuned_model.config, "max_position_embeddings", None)

    if max_embeddings and max_embeddings < 2 * config_dict["sequence_len"]:
        config_dict["sequence_len"] = ceil(max_embeddings / 2)

    return DictDefault(config_dict)


def check_and_log_base_model_size(original_model: str) -> None:
    """Check if base model size is logged in results, if not load and log it."""
    results_dict = load_results_dict()

    if "model_params_count" not in results_dict:
        logger.info("Base model size not logged, loading base model to calculate size")
        base_model = load_model(original_model, is_base_model=True)
        results_dict["model_params_count"] = count_model_parameters(base_model)
        save_results_dict(results_dict)
        logger.info(f"Logged base model size: {results_dict['model_params_count']} parameters")
    else:
        logger.info(f"Base model size already logged: {results_dict['model_params_count']} parameters")


def calculate_kl_divergence(
    original_model: AutoModelForCausalLM,
    finetuned_model: AutoModelForCausalLM,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
) -> float:
    """
    Calculate KL divergence between original and finetuned model outputs on a dataset.

    Args:
        original_model: The original/base model
        finetuned_model: The finetuned model
        dataset: Dataset to evaluate on
        tokenizer: Tokenizer for text processing

    Returns:
        Average KL divergence across the dataset
    """
    logger.info("Starting KL divergence calculation...")

    # Calculate max_length using same logic as GRPO evaluation
    max_length = cst.GRPO_KL_SEQUENCE_LENGTH
    max_embeddings = getattr(finetuned_model.config, "max_position_embeddings", None)
    if max_embeddings and max_embeddings < 2 * max_length:
        max_length = ceil(max_embeddings / 2)

    original_model.eval()
    finetuned_model.eval()

    @find_executable_batch_size(starting_batch_size=cst.GRPO_KL_BATCH_SIZE)
    def calculate_kl_with_batch_size(batch_size):
        logger.info(f"Attempting KL divergence calculation with batch size: {batch_size}")

        total_kl_div = 0.0
        total_samples = 0

        # Process dataset in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            prompts = batch[cst.TRL_GRPO_FIELD_PROMPT]

            try:
                inputs = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
            except Exception as e:
                logger.warning(f"Failed to tokenize batch starting at index {i}: {e}")
                continue

            with torch.no_grad():
                try:
                    # Get logits from both models
                    original_outputs = original_model(**inputs)
                    finetuned_outputs = finetuned_model(**inputs)

                    original_logits = original_outputs.logits
                    finetuned_logits = finetuned_outputs.logits

                    # Convert logits to probabilities
                    original_probs = F.softmax(original_logits, dim=-1)
                    finetuned_log_probs = F.log_softmax(finetuned_logits, dim=-1)

                    # Calculate KL divergence: KL(original || finetuned)
                    kl_div = F.kl_div(finetuned_log_probs, original_probs, reduction="none")

                    # Average over sequence length and vocabulary, sum over batch
                    batch_kl = kl_div.sum(dim=-1).mean(dim=-1).sum().item()

                    total_kl_div += batch_kl
                    total_samples += len(prompts)

                    if (i // batch_size) % 10 == 0:
                        logger.info(f"Processed {i + len(prompts)} samples, current batch KL: {batch_kl / len(prompts):.6f}")

                except Exception as e:
                    logger.warning(f"Failed to compute KL divergence for batch starting at index {i}: {e}")
                    continue
                finally:
                    torch.cuda.empty_cache()

        if total_samples == 0:
            logger.error("No samples were successfully processed for KL divergence calculation")
            raise ValueError("No samples were successfully processed for KL divergence calculation")

        avg_kl_div = total_kl_div / total_samples
        logger.info(f"KL divergence calculation completed. Average KL divergence: {avg_kl_div:.6f} over {total_samples} samples")

        return avg_kl_div

    return calculate_kl_with_batch_size()
