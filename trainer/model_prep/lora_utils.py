import importlib.util
import logging
import os
import subprocess
import sys
import time

from huggingface_hub import HfApi
from huggingface_hub import snapshot_download


logger = logging.getLogger(__name__)


def check_for_lora(model_id: str, hf_token: str | None = None, local_files_only: bool = False) -> bool:
    """Check whether a Hugging Face model repo or local cache contains LoRA adapter config."""
    lora_config_file = "adapter_config.json"
    try:
        if local_files_only:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            repo_path = os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))
            if os.path.exists(repo_path):
                for root, _, files in os.walk(repo_path):
                    if ".no_exist" in root:
                        continue
                    if lora_config_file in files:
                        config_path = os.path.join(root, lora_config_file)
                        if os.path.getsize(config_path) > 0:
                            return True
            return False

        return lora_config_file in HfApi(token=hf_token).list_repo_files(model_id)
    except Exception as exc:
        logger.error("Error checking for LoRA adapters: %s", exc)
        return False


def download_model_with_retry(repo_id: str, hf_token: str | None = None, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("model_prep download base model (attempt %s/%s): %s", attempt, max_retries, repo_id)
            start = time.time()
            path = snapshot_download(repo_id, token=hf_token, local_files_only=False)
            logger.info("model_prep base model snapshot_download done in %.1fs -> %s", time.time() - start, path)
            return path
        except Exception as exc:
            logger.warning("Download attempt %s failed: %s", attempt, exc)
            if attempt < max_retries:
                wait = 30 * attempt
                logger.info("Retrying in %ss...", wait)
                time.sleep(wait)
            else:
                logger.error("All download attempts failed")
                raise


def download_lora_with_retry(repo_id: str, local_dir: str, hf_token: str | None = None, max_retries: int = 3) -> str:
    os.makedirs(local_dir, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("model_prep download LoRA (attempt %s/%s): %s -> %s", attempt, max_retries, repo_id, local_dir)
            start = time.time()
            snapshot_download(repo_id, token=hf_token, local_dir=local_dir, local_dir_use_symlinks=False)
            logger.info("model_prep LoRA snapshot_download done in %.1fs", time.time() - start)
            return local_dir
        except Exception as exc:
            logger.warning("Download attempt %s failed: %s", attempt, exc)
            if attempt < max_retries:
                wait = 30 * attempt
                logger.info("Retrying in %ss...", wait)
                time.sleep(wait)
            else:
                logger.error("All download attempts failed")
                raise


def merge_base_and_lora(base_model_path: str, lora_dir: str, output_dir: str = "/tmp/merged_model") -> str:
    needs_install = importlib.util.find_spec("peft") is None or importlib.util.find_spec("accelerate") is None
    if needs_install:
        logger.info("Installing merge dependencies at runtime...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "peft", "accelerate"], check=True)
        logger.info("Merge dependencies installed")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    merge_t0 = time.time()
    logger.info("model_prep merge: start base=%s lora=%s out=%s", base_model_path, lora_dir, output_dir)
    logger.info("model_prep merge: loading tokenizers...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    lora_tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)

    t0 = time.time()
    logger.info("model_prep merge: loading base weights (AutoModelForCausalLM.from_pretrained)...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    logger.info("model_prep merge: base weights in memory in %.1fs", time.time() - t0)

    base_vocab_size = base.get_input_embeddings().weight.shape[0]
    target_tokenizer = lora_tokenizer if len(lora_tokenizer) >= base_vocab_size else base_tokenizer
    target_vocab_size = len(target_tokenizer)
    if target_vocab_size > base_vocab_size:
        logger.info("Resizing token embeddings from %s to %s", base_vocab_size, target_vocab_size)
        base.resize_token_embeddings(target_vocab_size)
    elif target_vocab_size < base_vocab_size:
        logger.info("LoRA tokenizer smaller than base (%s < %s); keeping base vocab size.", target_vocab_size, base_vocab_size)

    t1 = time.time()
    logger.info("model_prep merge: attaching LoRA (PeftModel.from_pretrained)...")
    model = PeftModel.from_pretrained(base, lora_dir)
    logger.info("model_prep merge: LoRA attached in %.1fs", time.time() - t1)

    t2 = time.time()
    logger.info("model_prep merge: merge_and_unload...")
    merged = model.merge_and_unload(safe_merge=False)
    logger.info("model_prep merge: merge_and_unload done in %.1fs", time.time() - t2)

    os.makedirs(output_dir, exist_ok=True)
    t3 = time.time()
    logger.info("model_prep merge: saving merged model to disk...")
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    target_tokenizer.save_pretrained(output_dir)
    logger.info("model_prep merge: saved to %s in %.1fs (total merge wall %.1fs)", output_dir, time.time() - t3, time.time() - merge_t0)
    return output_dir
