from huggingface_hub import repo_exists

from core.models.model_prep_models import AugmentationConfig
from trainer.model_prep.model_io import load_model_and_tokenizer
from trainer.model_prep.model_io import upload_augmented_model
from validator.evaluation.eval_environment import _download_lora_with_retry
from validator.evaluation.eval_environment import _download_model_with_retry
from validator.evaluation.eval_environment import _merge_base_and_lora
from validator.evaluation.utils import check_for_lora


def prepare_boss_base_model(aug_config: AugmentationConfig, repo_id: str, hf_token: str):
    if not aug_config.source_model_repo:
        raise ValueError("boss_base_model augmentation requires source_model_repo")
    if not aug_config.source_base_model_id:
        raise ValueError("boss_base_model augmentation requires source_base_model_id")

    source_repo = aug_config.source_model_repo
    if repo_exists(repo_id, token=hf_token):
        print(f"Prepared boss base model already exists at {repo_id}", flush=True)
        return load_model_and_tokenizer(repo_id, hf_token)

    is_lora = check_for_lora(source_repo, local_files_only=False)
    print(f"Preparing boss base model: repo={source_repo}, is_lora={is_lora}", flush=True)

    model_source = source_repo
    if is_lora:
        base_path = _download_model_with_retry(aug_config.source_base_model_id)
        lora_dir = f"/tmp/model_prep_lora/{aug_config.source_task_id or 'source'}"
        _download_lora_with_retry(source_repo, lora_dir)
        model_source = _merge_base_and_lora(base_path, lora_dir, "/tmp/model_prep_merged")

    model, tokenizer = load_model_and_tokenizer(model_source, hf_token)
    upload_augmented_model(model, tokenizer, repo_id, hf_token)
    return model, tokenizer
