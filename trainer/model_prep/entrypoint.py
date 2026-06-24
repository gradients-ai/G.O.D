"""
Model prep container entrypoint.
Handles LoRA detection + merge, augmentation, baseline stats, and HF upload.
Outputs JSON result on the last line of stdout for the caller to parse.
"""

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import time

import torch
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub import repo_exists
from peft import PeftModel
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from core.constants.environments import EnvironmentName
from core.constants.paths import LORA_ADAPTER_CONFIG_FILE
from core.downloads import download_s3_file
from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from core.models.model_prep_models import EnvBaselineConfig
from core.models.model_prep_models import ModelPrepResult
from core.models.task_models import TaskType
from trainer.model_prep.augmentation import augment_model
from trainer.model_prep.env_stats import compute_env_stats
from trainer.model_prep.stats import compute_text_stats


def detect_and_merge_lora(model_id: str, hf_token: str) -> ModelPrepResult:
    """Auto-detect LoRA adapter and merge with base if needed.

    model_id can be a local path or HF repo. Checks for adapter_config.json
    locally first, falls back to HF API for remote repos.
    """
    adapter_config_path = os.path.join(model_id, LORA_ADAPTER_CONFIG_FILE)
    is_local = os.path.isdir(model_id)

    if is_local:
        if not os.path.exists(adapter_config_path):
            return ModelPrepResult(effective_model_path=model_id)
    else:
        try:
            api = HfApi(token=hf_token)
            repo_files = api.list_repo_files(model_id, token=hf_token)
            if LORA_ADAPTER_CONFIG_FILE not in repo_files:
                return ModelPrepResult(effective_model_path=model_id)
        except Exception as exc:
            print(f"Could not check for LoRA: {exc}, loading as full weights", flush=True)
            return ModelPrepResult(effective_model_path=model_id)

    print(f"LoRA adapter detected: {model_id}", flush=True)

    try:
        if is_local:
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
        else:
            config_path = hf_hub_download(model_id, LORA_ADAPTER_CONFIG_FILE, token=hf_token)
            with open(config_path) as f:
                adapter_config = json.load(f)

        base_model_id = adapter_config.get("base_model_name_or_path")
        if not base_model_id:
            print("WARNING: adapter_config missing base_model_name_or_path, loading as-is", flush=True)
            return ModelPrepResult(effective_model_path=model_id)

        # Walk the LoRA chain and merge all adapters bottom-to-top. The immediate base
        # may itself be an adapter, so loading it directly can fail or drop deltas.
        chain: list[str] = []
        real_base = base_model_id
        for _ in range(10):
            try:
                cfg = hf_hub_download(real_base, LORA_ADAPTER_CONFIG_FILE, token=hf_token)
                with open(cfg) as f:
                    parent_base = json.load(f).get("base_model_name_or_path")
            except Exception:
                break
            if not parent_base:
                break
            chain.append(real_base)
            real_base = parent_base

        print(f"Merging LoRA chain into base: {real_base} (depth {len(chain)})", flush=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            real_base, torch_dtype="auto", token=hf_token,
            device_map="cuda:0" if torch.cuda.is_available() else "auto",
        )
        base_tokenizer = AutoTokenizer.from_pretrained(real_base, token=hf_token)

        def _merge_adapter(model, adapter_src):
            try:
                tok = AutoTokenizer.from_pretrained(adapter_src, token=hf_token)
            except Exception:
                tok = base_tokenizer
            if len(tok) > model.get_input_embeddings().weight.shape[0]:
                model.resize_token_embeddings(len(tok))
            pm = PeftModel.from_pretrained(model, adapter_src, token=hf_token)
            return pm.merge_and_unload(safe_merge=False), tok

        lora_tokenizer = base_tokenizer
        for adapter_repo in reversed(chain):
            base_model, lora_tokenizer = _merge_adapter(base_model, adapter_repo)
        merged, top_tokenizer = _merge_adapter(base_model, model_id)
        if top_tokenizer is not base_tokenizer:
            lora_tokenizer = top_tokenizer

        merge_dir = "/cache/merged_model"
        os.makedirs(merge_dir, exist_ok=True)
        merged.save_pretrained(merge_dir, safe_serialization=True)
        target_tokenizer = lora_tokenizer if len(lora_tokenizer) >= len(base_tokenizer) else base_tokenizer
        target_tokenizer.save_pretrained(merge_dir)

        del base_model, merged
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"LoRA merge complete → {merge_dir}", flush=True)
        return ModelPrepResult(
            effective_model_path=merge_dir,
            base_model_id=base_model_id,
            was_lora=True,
        )
    except Exception as exc:
        print(f"WARNING: LoRA merge failed ({exc}), falling back to full-weight loading", flush=True)
        return ModelPrepResult(effective_model_path=model_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--training-data", required=True, help="S3 URL or local path to training data")
    parser.add_argument(
        "--task-type", default=TaskType.INSTRUCTTEXTTASK.value,
        choices=[t.value for t in TaskType],
        help="Task type",
    )
    parser.add_argument("--aug-type", choices=[t.value for t in AugmentationType], default=None)
    parser.add_argument("--scope", choices=[s.value for s in AugmentationScope], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--intensity", type=float, default=None)
    parser.add_argument("--reward-functions", default=None, help="JSON list of reward function objects (for GRPO)")
    parser.add_argument(
        "--env-configs",
        default=None,
        help=(
            "JSON dict of {env_name: {url, task_id_min, task_id_max, "
            "num_episodes, eval_payload_extra}}"
        ),
    )
    return parser.parse_args()


def build_augmentation_config(args) -> AugmentationConfig | None:
    if args.aug_type is None:
        return None
    return AugmentationConfig(
        aug_type=AugmentationType(args.aug_type),
        scope=AugmentationScope(args.scope),
        seed=args.seed,
        intensity=args.intensity,
    )


def generate_anonymous_repo_name(model_id: str, seed: int) -> str:
    """Generate an opaque repo name that doesn't leak the original model identity."""
    hf_username = os.environ.get("HUGGINGFACE_USERNAME", "gradients-io")
    hash_input = f"{model_id}:{seed}"
    repo_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    return f"{hf_username}/augmented-{repo_hash}"


def load_training_data(path: str) -> list[dict]:
    """Load all training data records from a JSON file.

    Stats functions subsample internally for expensive operations, but need the
    full record count for total-token estimates.
    """
    if path.startswith("http"):
        local_path = asyncio.run(download_s3_file(path))
    else:
        local_path = path

    with open(local_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    return []


def upload_augmented_model(model, tokenizer, repo_id: str, hf_token: str) -> None:
    """Upload augmented model to HuggingFace, scrubbing identity."""
    print(f"Uploading augmented model to {repo_id}")

    model.config._name_or_path = repo_id
    model.push_to_hub(repo_id, token=hf_token, private=False)
    tokenizer.push_to_hub(repo_id, token=hf_token, private=False)

    # Scrub _name_or_path from config
    api = HfApi(token=hf_token)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = api.hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=tmp, token=hf_token)
        with open(config_path, "r") as f:
            config = json.load(f)
        if "_name_or_path" in config:
            del config["_name_or_path"]
            modified_path = os.path.join(tmp, "config_clean.json")
            with open(modified_path, "w") as f:
                json.dump(config, f, indent=2)
            api.upload_file(
                path_or_fileobj=modified_path,
                path_in_repo="config.json",
                repo_id=repo_id,
            )

    print(f"Upload complete: {repo_id}")


def _load_config_with_yarn_fix(model_path: str, hf_token: str):
    """Load model config while avoiding a transformers YaRN head_dim=None crash."""
    config = AutoConfig.from_pretrained(model_path, token=hf_token)
    head_dim = config.head_dim if hasattr(config, "head_dim") else None
    if head_dim is None and hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
        config.head_dim = config.hidden_size // config.num_attention_heads
        print(f"[model_prep] set head_dim={config.head_dim} (was None) to avoid YaRN rope crash", flush=True)
    return config


def main():
    t_total = time.time()
    args = parse_args()
    aug_config = build_augmentation_config(args)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")

    # Auto-detect LoRA and merge if needed (for model continuation between rounds)
    prep_result = detect_and_merge_lora(args.model, hf_token)
    model_path = prep_result.effective_model_path
    if prep_result.was_lora:
        print(f"Using merged model from {model_path} (base: {prep_result.base_model_id})", flush=True)

    # --- Model loading ---
    t0 = time.time()
    n_gpus = torch.cuda.device_count()
    print(f"[model_prep] Loading model: {model_path} (gpus={n_gpus})", flush=True)
    model_config = _load_config_with_yarn_fix(model_path, hf_token)
    # Load in the model's native dtype ("auto" reads config.torch_dtype) rather than forcing
    # fp16: bf16-native models can overflow in fp16, producing NaN baseline stats.
    if n_gpus > 1:
        print(f"[model_prep] Multi-GPU detected ({n_gpus}), using device_map=auto", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=model_config, torch_dtype="auto", token=hf_token, device_map="auto",
        )
    elif torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=model_config, torch_dtype="auto", token=hf_token,
        )
        model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config, torch_dtype="auto", token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[model_prep] Model loaded in {time.time() - t0:.1f}s ({num_params / 1e9:.1f}B params)", flush=True)

    # --- Augmentation ---
    augmented_model_id = None
    if aug_config is not None:
        repo_id = generate_anonymous_repo_name(args.model, aug_config.seed)

        if repo_exists(repo_id, token=hf_token):
            print(f"[model_prep] Augmented model already exists at {repo_id}, skipping", flush=True)
            augmented_model_id = repo_id
        else:
            t0 = time.time()
            print(f"[model_prep] Applying augmentation: {aug_config.aug_type.value}", flush=True)
            augment_model(model, aug_config)
            print(f"[model_prep] Augmentation done in {time.time() - t0:.1f}s, uploading...", flush=True)
            t0 = time.time()
            upload_augmented_model(model, tokenizer, repo_id, hf_token)
            print(f"[model_prep] Upload done in {time.time() - t0:.1f}s", flush=True)
            augmented_model_id = repo_id

    # --- Baseline stats ---
    print("[model_prep] Computing baseline stats...", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t0 = time.time()
    try:
        if args.env_configs:
            raw_configs: dict[str, dict] = json.loads(args.env_configs)
            env_configs = {
                EnvironmentName(k): EnvBaselineConfig.model_validate(v)
                for k, v in raw_configs.items()
            }
            stats = asyncio.run(compute_env_stats(
                # Use the merged path for LoRA continuation; a bare adapter dir
                # cannot be served directly by SGLang.
                model_path=model_path,
                model=model,
                env_configs=env_configs,
            ))
        else:
            data_records = load_training_data(args.training_data)
            reward_functions = json.loads(args.reward_functions) if args.reward_functions else None

            if data_records:
                task_type_enum = TaskType(args.task_type)
                print(f"[model_prep] {len(data_records)} data records, task_type={args.task_type}", flush=True)
                stats = compute_text_stats(
                    model, tokenizer, data_records,
                    task_type=task_type_enum,
                    reward_functions=reward_functions,
                )
            else:
                print("[model_prep] Warning: no training data available for stats", flush=True)
                stats = None
    except torch.cuda.OutOfMemoryError:
        print("[model_prep] CUDA OOM during stats computation, returning partial results", flush=True)
        torch.cuda.empty_cache()
        stats = None

    print(f"[model_prep] Stats computation done in {time.time() - t0:.1f}s", flush=True)

    if stats and hasattr(stats, "training"):
        print(f"[model_prep] loss={stats.training.init_loss:.4f}, entropy={stats.training.output_entropy:.4f}", flush=True)
    elif stats and hasattr(stats, "env_stats"):
        for env_name, env_stat in stats.env_stats.items():
            print(
                f"[model_prep]   {env_name.value}: {env_stat.num_episodes} episodes, "
                f"mean={env_stat.mean_score:.3f}",
                flush=True,
            )

    print(f"[model_prep] Total time: {time.time() - t_total:.1f}s", flush=True)

    # Output result as JSON on last line (parsed by caller)
    result = {
        "augmented_model_id": augmented_model_id,
        "baseline_stats": stats.model_dump() if stats else None,
        "lora_merge": prep_result.model_dump() if prep_result.was_lora else None,
    }
    def sanitize_floats(obj):
        """Replace NaN/Inf with None for JSON compliance."""
        if isinstance(obj, float):
            if obj != obj or obj == float("inf") or obj == float("-inf"):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: sanitize_floats(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_floats(v) for v in obj]
        return obj

    print(json.dumps(sanitize_floats(result)), flush=True)


if __name__ == "__main__":
    main()
