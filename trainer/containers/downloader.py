import argparse
import asyncio
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

import trainer.training_paths as train_paths
from core.downloads import download_s3_file
from core.models.dataset_models import FileFormat
from core.models.image_models import ImageModelType
from core.models.task_models import TaskType
from core.tokenizer_utils import ensure_chat_template
from core.tokenizer_utils import read_chat_template
from core.tokenizer_utils import sanitize_tokenizer_config
from trainer import constants as cst
from trainer.model_artifacts import get_anonymous_model_dir
from trainer.model_artifacts import scrub_model_identity


LORA_ADAPTER_CONFIG = "adapter_config.json"
IDEOGRAM4_TEXT_ENCODER_REPO = "Qwen/Qwen3-VL-8B-Instruct"
KREA2_TEXT_ENCODER_REPO = "Qwen/Qwen3-VL-4B-Instruct"
DIFFUSERS_COMPONENT_DIRS = {
    "scheduler",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "transformer",
    "unet",
    "vae",
}
SHARDED_CHECKPOINT_PATTERN = re.compile(r"-\d{5}-of-\d{5}\.safetensors$")
WEIGHT_INDEX_SUFFIXES = (".bin.index.json", ".safetensors.index.json")


hf_api = HfApi()


@dataclass(frozen=True)
class RepoFileMetadata:
    path: str
    size: int | None


async def download_text_dataset(task_id, dataset_url, file_format, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)

    if file_format == FileFormat.S3.value:
        input_data_path = train_paths.get_text_dataset_path(task_id)

        if not os.path.exists(input_data_path):
            local_path = await download_s3_file(dataset_url)
            shutil.copy(local_path, input_data_path)

    elif file_format == FileFormat.HF.value:
        repo_name = dataset_url.replace("/", "--")
        input_data_path = os.path.join(dataset_dir, repo_name)

        if not os.path.exists(input_data_path):
            snapshot_download(repo_id=dataset_url, repo_type="dataset", local_dir=input_data_path, local_dir_use_symlinks=False)

    return input_data_path, file_format


async def download_image_dataset(dataset_zip_url, task_id, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    local_zip_path = train_paths.get_image_training_zip_save_path(task_id)
    print(f"Downloading image dataset for task {task_id}")
    local_path = await download_s3_file(dataset_zip_url, local_zip_path)
    print(f"Downloaded image dataset to: {local_path}")
    return local_path


def _huggingface_token() -> str | None:
    return os.environ.get("HUGGINGFACE_TOKEN") or None


def _repo_file_metadata(repo_id: str) -> tuple[RepoFileMetadata, ...]:
    repo_files = []
    for entry in hf_api.list_repo_tree(
        repo_id=repo_id,
        repo_type="model",
        recursive=True,
        token=_huggingface_token(),
    ):
        if not hasattr(entry, "size"):
            continue

        path = str(entry.path)
        if path.startswith("./"):
            path = path[2:]
        parsed_path = PurePosixPath(path)
        if parsed_path.is_absolute() or ".." in parsed_path.parts:
            raise ValueError(f"Unsafe path returned for Hugging Face repository {repo_id}")
        repo_files.append(RepoFileMetadata(path=path, size=entry.size))

    if not repo_files:
        raise RuntimeError(f"Hugging Face repository {repo_id} contains no model files")
    return tuple(repo_files)


def _is_flux_model_type(model_type: ImageModelType | str) -> bool:
    return model_type == ImageModelType.FLUX or model_type == ImageModelType.FLUX.value


def _standalone_flux_checkpoint(
    model_type: ImageModelType | str,
    repo_files: tuple[RepoFileMetadata, ...],
) -> RepoFileMetadata | None:
    if not _is_flux_model_type(model_type):
        return None

    paths = [PurePosixPath(repo_file.path) for repo_file in repo_files]
    root_checkpoints = [
        repo_file
        for repo_file, path in zip(repo_files, paths)
        if len(path.parts) == 1 and path.suffix == ".safetensors"
    ]
    if len(root_checkpoints) != 1:
        return None
    if any(path == PurePosixPath("model_index.json") for path in paths):
        return None
    if any(path.parts[0] in DIFFUSERS_COMPONENT_DIRS for path in paths if len(path.parts) > 1):
        return None
    if any(repo_file.path.endswith(WEIGHT_INDEX_SUFFIXES) for repo_file in repo_files):
        return None
    if SHARDED_CHECKPOINT_PATTERN.search(root_checkpoints[0].path):
        return None
    return root_checkpoints[0]


def _local_repo_path(root: Path, repo_path: str) -> Path:
    return root.joinpath(*PurePosixPath(repo_path).parts)


def _file_matches_metadata(path: Path, expected_size: int | None) -> bool:
    if not path.is_file():
        return False
    if expected_size is None:
        return True
    return path.stat().st_size == expected_size


def _standalone_checkpoint_is_complete(cache_path: Path, checkpoint: RepoFileMetadata) -> bool:
    checkpoint_path = _local_repo_path(cache_path, checkpoint.path)
    return (
        cache_path.is_dir()
        and not checkpoint_path.is_symlink()
        and _file_matches_metadata(checkpoint_path, checkpoint.size)
        and (checkpoint.size is not None or checkpoint_path.stat().st_size > 0)
    )


def _weight_index_references_exist(cache_path: Path) -> bool:
    for index_path in cache_path.rglob("*.index.json"):
        if not index_path.name.endswith(WEIGHT_INDEX_SUFFIXES):
            continue
        try:
            with index_path.open(encoding="utf-8") as index_file:
                weight_map = json.load(index_file).get("weight_map")
        except (AttributeError, json.JSONDecodeError, OSError):
            return False
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        for shard_path in set(weight_map.values()):
            if not isinstance(shard_path, str):
                return False
            parsed_path = PurePosixPath(shard_path)
            if parsed_path.is_absolute() or ".." in parsed_path.parts:
                return False
            relative_parts = parsed_path.parts
            if not any(
                candidate.is_file()
                for candidate in (
                    index_path.parent.joinpath(*relative_parts),
                    cache_path.joinpath(*relative_parts),
                )
            ):
                return False
    return True


def _snapshot_is_complete(
    cache_path: Path,
    repo_files: tuple[RepoFileMetadata, ...],
    anonymized: bool,
) -> bool:
    if not cache_path.is_dir():
        return False
    for repo_file in repo_files:
        local_path = _local_repo_path(cache_path, repo_file.path)
        if not local_path.is_file():
            return False
        if anonymized and repo_file.path == "config.json":
            continue
        if repo_file.size is not None and local_path.stat().st_size != repo_file.size:
            return False
    return _weight_index_references_exist(cache_path)


def _normalize_standalone_checkpoint(cache_path: Path, checkpoint: RepoFileMetadata) -> None:
    checkpoint_name = PurePosixPath(checkpoint.path).name
    for entry in cache_path.iterdir():
        if entry.name != checkpoint_name and (entry.is_file() or entry.is_symlink()):
            entry.unlink()

    top_level_files = [entry for entry in cache_path.iterdir() if entry.is_file() or entry.is_symlink()]
    if len(top_level_files) != 1 or top_level_files[0].name != checkpoint_name or top_level_files[0].is_symlink():
        raise RuntimeError(f"Failed to normalize standalone FLUX checkpoint cache at {cache_path}")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _promote_directory(download_path: Path, save_path: Path) -> None:
    if not os.path.lexists(save_path):
        os.rename(download_path, save_path)
        return

    backup_path = Path(tempfile.mkdtemp(prefix=f".{save_path.name}.backup-", dir=save_path.parent))
    backup_path.rmdir()
    os.rename(save_path, backup_path)
    try:
        os.rename(download_path, save_path)
    except BaseException:
        os.rename(backup_path, save_path)
        raise
    else:
        _remove_path(backup_path)


def _download_standalone_checkpoint(repo_id: str, checkpoint: RepoFileMetadata, download_path: Path) -> None:
    checkpoint_path = _local_repo_path(download_path, checkpoint.path)
    downloaded_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint.path,
            local_dir=str(download_path),
            local_dir_use_symlinks=False,
            token=_huggingface_token(),
        )
    )
    if not checkpoint_path.is_file() and downloaded_path.is_file():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(downloaded_path, checkpoint_path)
    if not _standalone_checkpoint_is_complete(download_path, checkpoint):
        raise RuntimeError(f"Standalone FLUX checkpoint download is incomplete for {repo_id}")
    _normalize_standalone_checkpoint(download_path, checkpoint)


def write_environment_task_proxy_dataset(
    out_path: str,
    dataset_size: int = 1000,
    prompt_text: str = "Interact with this environment.",
    prompt_field: str = "prompt",
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = [{prompt_field: prompt_text} for _ in range(dataset_size)]

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} records to {out_path} with field '{prompt_field}'")


def _model_dir_name(repo_id: str, anonymize: bool) -> str:
    if anonymize:
        return get_anonymous_model_dir(repo_id)
    return repo_id.replace("/", "--")


async def download_base_model(
    repo_id: str,
    save_root: str,
    model_type: ImageModelType | str,
    anonymize: bool = True,
) -> str:
    model_name = _model_dir_name(repo_id, anonymize)
    save_root_path = Path(save_root)
    save_path = save_root_path / model_name
    save_root_path.mkdir(parents=True, exist_ok=True)

    repo_files = _repo_file_metadata(repo_id)
    standalone_checkpoint = _standalone_flux_checkpoint(model_type, repo_files)

    if standalone_checkpoint and _standalone_checkpoint_is_complete(save_path, standalone_checkpoint):
        _normalize_standalone_checkpoint(save_path, standalone_checkpoint)
        print(f"Standalone FLUX checkpoint cache is ready at {save_path}. Skipping download.", flush=True)
        return str(save_path)
    if standalone_checkpoint is None and _snapshot_is_complete(save_path, repo_files, anonymize):
        print(f"Model cache is complete at {save_path}. Skipping download.", flush=True)
        return str(save_path)

    if os.path.lexists(save_path):
        print(f"Model cache at {save_path} is incomplete. Replacing it safely.", flush=True)
    else:
        print(f"Downloading model to {save_path}.", flush=True)

    download_path = Path(tempfile.mkdtemp(prefix=f".{model_name}.download-", dir=save_root_path))
    try:
        if standalone_checkpoint:
            _download_standalone_checkpoint(repo_id, standalone_checkpoint, download_path)
        else:
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=str(download_path),
                local_dir_use_symlinks=False,
                token=_huggingface_token(),
            )
            if not _snapshot_is_complete(download_path, repo_files, anonymized=False):
                raise RuntimeError(f"Hugging Face snapshot download is incomplete for {repo_id}")

        if anonymize:
            scrub_model_identity(str(download_path))
        _promote_directory(download_path, save_path)
    except BaseException:
        _remove_path(download_path)
        raise

    print(f"Model cache is ready at {save_path}.", flush=True)
    return str(save_path)


def _detect_and_merge_lora(model_dir: str) -> None:
    """If model_dir contains a LoRA adapter, merge it into the base model in-place.

    After merge the directory contains full merged weights and the adapter
    files are removed so downstream code sees a normal model.
    """
    adapter_config_path = os.path.join(model_dir, LORA_ADAPTER_CONFIG)
    if not os.path.exists(adapter_config_path):
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    base_model_id = adapter_config.get("base_model_name_or_path")
    if not base_model_id:
        print(f"WARNING: {LORA_ADAPTER_CONFIG} missing base_model_name_or_path, skipping merge", flush=True)
        return

    # Collect every adapter in the chain and merge them all, bottom-to-top.
    chain: list[str] = []
    real_base = base_model_id
    for _ in range(10):  # max depth guard
        try:
            remote_adapter = hf_hub_download(real_base, LORA_ADAPTER_CONFIG)
            with open(remote_adapter) as f:
                parent_base = json.load(f).get("base_model_name_or_path")
        except Exception:
            break  # Not a LoRA repo; real_base is the foundation model.
        if not parent_base:
            break
        print(f"[downloader] Chained LoRA: {real_base} -> {parent_base}", flush=True)
        chain.append(real_base)
        real_base = parent_base

    print(f"[downloader] LoRA chain detected in {model_dir}: base={real_base}, depth={len(chain)}", flush=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    base_model = AutoModelForCausalLM.from_pretrained(
        real_base, torch_dtype=torch.float16, device_map=device,
    )
    base_tokenizer = AutoTokenizer.from_pretrained(real_base)

    def _merge_adapter(model, adapter_src):
        try:
            tok = AutoTokenizer.from_pretrained(adapter_src)
        except Exception:
            tok = base_tokenizer
        if len(tok) > model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tok))
        peft_model = PeftModel.from_pretrained(model, adapter_src)
        return peft_model.merge_and_unload(safe_merge=False), tok

    lora_tokenizer = base_tokenizer
    for adapter_repo in reversed(chain):
        base_model, lora_tokenizer = _merge_adapter(base_model, adapter_repo)
    merged, top_tokenizer = _merge_adapter(base_model, model_dir)
    if top_tokenizer is not base_tokenizer:
        lora_tokenizer = top_tokenizer

    # Disable peft hooks that break save_pretrained in newer transformers
    if hasattr(merged, "_hf_peft_config_loaded"):
        merged._hf_peft_config_loaded = False

    # Save merged model to a temp dir, then swap into model_dir
    merge_tmp = model_dir + ".merged_tmp"
    os.makedirs(merge_tmp, exist_ok=True)
    merged.save_pretrained(merge_tmp, safe_serialization=True)
    target_tokenizer = lora_tokenizer if len(lora_tokenizer) >= len(base_tokenizer) else base_tokenizer
    # Carry the adapter's chat template onto the saved tokenizer (base selection would drop it).
    ensure_chat_template(target_tokenizer, read_chat_template(model_dir), base_tokenizer.chat_template)
    target_tokenizer.save_pretrained(merge_tmp)
    # Keep the merged dir loadable by the read-only, any-version miner training container.
    sanitize_tokenizer_config(merge_tmp)

    del base_model, merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Replace adapter dir with merged weights
    shutil.rmtree(model_dir)
    os.rename(merge_tmp, model_dir)
    print(f"[downloader] LoRA merge complete → {model_dir}", flush=True)


async def download_axolotl_base_model(repo_id: str, save_dir: str, anonymize: bool = True) -> str:
    model_dir = os.path.join(save_dir, _model_dir_name(repo_id, anonymize))
    if os.path.exists(model_dir):
        print(f"Model already cached at {model_dir}.", flush=True)
        _detect_and_merge_lora(model_dir)
        print("Skipping download.", flush=True)
        return model_dir
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=model_dir, local_dir_use_symlinks=False)
    _detect_and_merge_lora(model_dir)
    if anonymize:
        scrub_model_identity(model_dir)
    return model_dir


async def download_huggingface_snapshot(repo_id: str, save_root: str) -> str:
    """Download a Hugging Face model snapshot to a stable cache directory."""
    save_path = os.path.join(save_root, repo_id.replace("/", "--"))
    os.makedirs(save_root, exist_ok=True)
    if os.path.exists(save_path):
        print(f"Model already cached at {save_path}. Skipping download.", flush=True)
        return save_path

    print(f"Downloading {repo_id} to {save_path}...", flush=True)
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=save_path, local_dir_use_symlinks=False)
    print(f"Downloaded {repo_id} to {save_path}", flush=True)
    return save_path


async def download_adapter(repo_id: str, filename: str, adapters_dir: str) -> str:
    """Download adapter file and save it in the adapters directory"""
    adapter_path = os.path.join(adapters_dir, filename)
    os.makedirs(adapters_dir, exist_ok=True)
    
    if os.path.exists(adapter_path):
        print(f"Adapter {filename} already exists at {adapter_path}. Skipping download.", flush=True)
        return adapter_path
    
    print(f"Downloading adapter {filename} from {repo_id}...", flush=True)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=temp_dir)
            shutil.move(temp_file_path, adapter_path)
        print(f"Adapter {filename} downloaded successfully to {adapter_path}", flush=True)
        return adapter_path
    except Exception as e:
        print(f"Error downloading adapter {filename}: {e}", flush=True)
        raise e


async def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    miner_ds_parser = subparsers.add_parser("download-miner-dataset")
    miner_ds_parser.add_argument("--repo-id", required=True)
    miner_ds_parser.add_argument("--cache-dir", required=True)

    parser.add_argument("--task-id")
    parser.add_argument("--model")
    parser.add_argument(
        "--task-type",
        choices=[
            TaskType.IMAGETASK.value,
            TaskType.INSTRUCTTEXTTASK.value,
            TaskType.DPOTASK.value,
            TaskType.GRPOTASK.value,
            TaskType.CHATTASK.value,
            TaskType.ENVIRONMENTTASK.value,
        ],
    )
    parser.add_argument("--dataset")
    parser.add_argument("--file-format")
    parser.add_argument(
        "--model-type",
        choices=[
            ImageModelType.FLUX.value,
            ImageModelType.Z_IMAGE.value,
            ImageModelType.QWEN_IMAGE.value,
            ImageModelType.IDEOGRAM4.value,
            ImageModelType.KREA2.value,
        ],
    )
    parser.add_argument("--anonymize", action="store_true", help="Anonymize model directory name and scrub identity")
    args = parser.parse_args()

    if args.command == "download-miner-dataset":
        download_miner_dataset(args.repo_id, args.cache_dir)
        return

    if not args.task_id or not args.model or not args.task_type or not args.dataset:
        parser.error("--task-id, --model, --task-type, and --dataset are required for training downloads")

    dataset_dir = cst.CACHE_DATASETS_DIR
    model_dir = cst.CACHE_MODELS_DIR
    adapters_dir = cst.HUGGINGFACE_CACHE_PATH
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(adapters_dir, exist_ok=True)

    print(f"Downloading datasets to: {dataset_dir}", flush=True)
    print(f"Downloading models to: {model_dir}", flush=True)

    if args.task_type == TaskType.IMAGETASK.value:
        await download_image_dataset(args.dataset, args.task_id, dataset_dir)
        model_path = await download_base_model(args.model, model_dir, args.model_type, anonymize=args.anonymize)

        if args.model_type == ImageModelType.Z_IMAGE.value:
            print("Downloading Z-Image adapter...", flush=True)
            zimage_adapter_path = await download_adapter(
                repo_id="ostris/zimage_turbo_training_adapter",
                filename="zimage_turbo_training_adapter_v2.safetensors",
                adapters_dir=adapters_dir
            )
            print(f"Z-Image adapter downloaded to: {zimage_adapter_path}", flush=True)
            
        elif args.model_type == ImageModelType.QWEN_IMAGE.value:
            print("Downloading Qwen-Image adapter...", flush=True)
            qwen_adapter_path = await download_adapter(
                repo_id="ostris/accuracy_recovery_adapters",
                filename="qwen_image_torchao_uint3.safetensors",
                adapters_dir=adapters_dir
            )
            print(f"Qwen-Image adapter downloaded to: {qwen_adapter_path}", flush=True)

        elif args.model_type == ImageModelType.IDEOGRAM4.value:
            print("Downloading Ideogram 4 text encoder...", flush=True)
            text_encoder_path = await download_huggingface_snapshot(
                IDEOGRAM4_TEXT_ENCODER_REPO,
                adapters_dir,
            )
            print(f"Ideogram 4 text encoder downloaded to: {text_encoder_path}", flush=True)

            print("Downloading Ideogram 4 unconditional LoRA...", flush=True)
            ideogram_adapter_path = await download_adapter(
                repo_id="ostris/ideogram_4_unconditional_lora",
                filename="ideogram_4_unconditional_lora_r16.safetensors",
                adapters_dir=adapters_dir
            )
            print(f"Ideogram 4 unconditional LoRA downloaded to: {ideogram_adapter_path}", flush=True)

        elif args.model_type == ImageModelType.KREA2.value:
            print("Downloading Krea 2 text encoder...", flush=True)
            text_encoder_path = await download_huggingface_snapshot(
                KREA2_TEXT_ENCODER_REPO,
                adapters_dir,
            )
            print(f"Krea 2 text encoder downloaded to: {text_encoder_path}", flush=True)

        from transformers import CLIPTokenizer

        print("Downloading clip models", flush=True)
        CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cst.HUGGINGFACE_CACHE_PATH)
        CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", cache_dir=cst.HUGGINGFACE_CACHE_PATH)
        snapshot_download(
            repo_id="google/t5-v1_1-xxl",
            repo_type="model",
            cache_dir=cst.HUGGINGFACE_CACHE_PATH,
            local_dir_use_symlinks=False,
            allow_patterns=["tokenizer_config.json", "spiece.model", "special_tokens_map.json", "config.json"],
        )
    elif args.task_type == TaskType.ENVIRONMENTTASK.value:
        model_path = await download_axolotl_base_model(args.model, model_dir, anonymize=args.anonymize)
        input_data_path = train_paths.get_text_dataset_path(args.task_id)
        write_environment_task_proxy_dataset(
            out_path=input_data_path,
            dataset_size=1000,
            prompt_text="Interact with this environment.",
            prompt_field="prompt",
        )
    else:
        dataset_path, _ = await download_text_dataset(args.task_id, args.dataset, args.file_format, dataset_dir)
        model_path = await download_axolotl_base_model(args.model, model_dir, anonymize=args.anonymize)

    print(f"Model path: {model_path}", flush=True)
    print(f"Dataset path: {dataset_dir}", flush=True)


def download_miner_dataset(repo_id: str, cache_dir: str) -> str:
    """Download a single HF dataset to the miner datasets cache."""
    cache_name = repo_id.replace("/", "--")
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        print(f"Dataset {repo_id} already cached at {cache_path}", flush=True)
        return cache_path

    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = cache_path + f".tmp.{os.getpid()}"
    try:
        print(f"Downloading dataset {repo_id} to {tmp_path}", flush=True)
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=tmp_path, local_dir_use_symlinks=False)
        os.rename(tmp_path, cache_path)
        print(f"Download complete: {repo_id}", flush=True)
    except BaseException:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    return cache_path


if __name__ == "__main__":
    asyncio.run(main())
