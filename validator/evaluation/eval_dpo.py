import os
import subprocess
import traceback
import tempfile
import urllib.request


# Allow torch.load for transformers 4.46+ security check
os.environ["TRANSFORMERS_ALLOW_TORCH_LOAD"] = "true"

import torch
from accelerate.utils import find_executable_batch_size
from axolotl.utils.dict import DictDefault
from datasets import Dataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from trl import DPOConfig
from trl import DPOTrainer

from core.models.utility_models import DpoDatasetType
from validator.core import constants as cst
from validator.core.models import EvaluationArgs
from validator.evaluation.common import ProgressLoggerCallback
from validator.evaluation.common import _load_and_update_evaluation_config
from validator.evaluation.common import _log_dataset_and_model_info
from validator.evaluation.common import check_and_log_base_model_size
from validator.evaluation.common import count_model_parameters
from validator.evaluation.common import load_finetuned_model
from validator.evaluation.common import load_model
from validator.evaluation.common import load_results_dict
from validator.evaluation.common import load_tokenizer
from validator.evaluation.common import log_memory_stats
from validator.evaluation.common import save_results_dict
from validator.evaluation.utils import model_is_a_finetune
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def _adapt_dpo_columns_to_trl(dataset: Dataset, dataset_type: DpoDatasetType) -> Dataset:
    """
    Transform a DPO dataset to match trl's expected column names.

    Args:
        dataset: Hugging Face dataset object
        dataset_type: DpoDatasetType with field mappings
    """
    logger.info("Adapting DPO columns to standard format")

    chosen_field = dataset_type.field_chosen
    rejected_field = dataset_type.field_rejected

    if chosen_field in dataset.column_names and rejected_field in dataset.column_names:
        identical_count = 0
        sample_size = min(10, len(dataset))
        sample_indices = list(range(sample_size))

        for idx in sample_indices:
            example = dataset[idx]
            chosen = example[chosen_field]
            rejected = example[rejected_field]

            if chosen == rejected:
                identical_count += 1

        if identical_count > 0:
            logger.warning(
                f"CRITICAL: Found {identical_count}/{sample_size} samples with identical chosen/rejected, causing random predictions"
            )

            if identical_count > 0:
                example = dataset[sample_indices[0]]
                chosen = example[chosen_field]
                rejected = example[rejected_field]
                logger.warning(f"Example: Chosen/Rejected: '{chosen[:100]}...'")

    column_mapping = {
        dataset_type.field_prompt: cst.TRL_DPO_FIELD_PROMPT,
        dataset_type.field_chosen: cst.TRL_DPO_FIELD_CHOSEN,
        dataset_type.field_rejected: cst.TRL_DPO_FIELD_REJECTED,
    }
    for src_col, dst_col in column_mapping.items():
        if src_col in dataset.column_names and src_col != dst_col:
            dataset = dataset.rename_column(src_col, dst_col)

    columns_to_keep = [cst.TRL_DPO_FIELD_PROMPT, cst.TRL_DPO_FIELD_CHOSEN, cst.TRL_DPO_FIELD_REJECTED]
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    for col in columns_to_remove:
        dataset = dataset.remove_columns(col)

    return dataset


def _collate_dpo_batch(batch: list[dict[str, list[int]]], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    logger.debug(f"Collating batch of size {len(batch)}")
    try:
        prompt_ids = [torch.tensor(item["prompt_ids"]) for item in batch]
        prompt_attention_mask = [torch.tensor(item["prompt_attention_mask"]) for item in batch]
        chosen_ids = [torch.tensor(item["chosen_ids"]) for item in batch]
        chosen_attention_mask = [torch.tensor(item["chosen_attention_mask"]) for item in batch]
        rejected_ids = [torch.tensor(item["rejected_ids"]) for item in batch]
        rejected_attention_mask = [torch.tensor(item["rejected_attention_mask"]) for item in batch]

        if logger.isEnabledFor(10):
            logger.debug(f"Processing batch with {len(prompt_ids)} examples")

        prompt_ids = pad_sequence(prompt_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0)
        chosen_ids = pad_sequence(chosen_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        chosen_attention_mask = pad_sequence(chosen_attention_mask, batch_first=True, padding_value=0)
        rejected_ids = pad_sequence(rejected_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        rejected_attention_mask = pad_sequence(rejected_attention_mask, batch_first=True, padding_value=0)

        if logger.isEnabledFor(10):
            logger.debug(f"Padded tensors to shape {prompt_ids.shape[1]} tokens")

        return {
            "prompt_ids": prompt_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "chosen_ids": chosen_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_ids": rejected_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }
    except Exception as e:
        logger.error(f"Error in collate function: {e}")
        logger.error(traceback.format_exc())
        raise


def evaluate_dpo_model(
    evaluation_config: DictDefault,
    finetuned_model: AutoModelForCausalLM,
    reference_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    evaluation_args: EvaluationArgs,
) -> dict[str, float]:
    evaluation_config.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Config: {evaluation_config}")

    data_files = evaluation_config.datasets[0].get("data_files", [evaluation_config.datasets[0]["path"]])
    eval_dataset = load_dataset("json", data_files=data_files, split="train")
    eval_dataset = _adapt_dpo_columns_to_trl(eval_dataset, evaluation_args.dataset_type)

    _log_dataset_and_model_info(eval_dataset, finetuned_model, tokenizer)

    def custom_data_collator(features):
        logger.debug(f"Collating {len(features)} features")
        return _collate_dpo_batch(features, tokenizer)

    @find_executable_batch_size(starting_batch_size=evaluation_config.starting_batch_size)
    def evaluate_dpo_with_batch_size(batch_size):
        training_args = DPOConfig(
            output_dir=evaluation_config.output_dir,
            per_device_eval_batch_size=batch_size,
            report_to="none",
            bf16=True,
            beta=cst.BETA_DPO,
        )
        dpo_trainer = DPOTrainer(
            model=finetuned_model,
            ref_model=reference_model,
            args=training_args,
            train_dataset=Dataset.from_dict({col: [] for col in eval_dataset.column_names}),
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=[ProgressLoggerCallback(log_interval_seconds=evaluation_config.log_interval_seconds)],
        )

        results = dpo_trainer.evaluate()
        return results

    eval_results = evaluate_dpo_with_batch_size()
    logger.info(f"Final DPO evaluation results: {eval_results}")

    if abs(eval_results["eval_loss"] - 0.6931) < 0.0001:
        logger.error("CRITICAL: Loss value is approximately ln(2) ≈ 0.6931, suggesting models are making random predictions")

    evaluation_results = {
        "eval_loss": eval_results["eval_loss"],
    }
    return evaluation_results


def evaluate_finetuned_dpo_model(
    evaluation_args: EvaluationArgs,
    finetuned_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    reference_model: AutoModelForCausalLM,
) -> dict[str, float]:
    evaluation_config = _load_and_update_evaluation_config(
        evaluation_args=evaluation_args, finetuned_model=finetuned_model, config_path=cst.VALI_CONFIG_PATH
    )
    return evaluate_dpo_model(evaluation_config, finetuned_model, reference_model, tokenizer, evaluation_args)


def evaluate_dpo_repo(evaluation_args: EvaluationArgs) -> None:
    """Evaluate a single model repository and save results directly to file."""
    results_dict = load_results_dict()
    repo = evaluation_args.repo

    if repo in results_dict:
        logger.info(f"Skipping {repo} as it's already evaluated")
        return

    tokenizer = load_tokenizer(evaluation_args.original_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        logger.info(f"Loading reference model: {evaluation_args.original_model}")
        reference_model = load_model(evaluation_args.original_model, is_base_model=True)
        if reference_model is None:
            raise ValueError(f"Reference model {evaluation_args.original_model} failed to load")

        if "model_params_count" not in results_dict:
            results_dict["model_params_count"] = count_model_parameters(reference_model)

        try:
            logger.info(f"Loading finetuned model as LoRA adapter: {repo}")
            finetuned_model = load_finetuned_model(repo)
            is_finetune = True
        except Exception as lora_error:
            logger.info(f"Failed to load as LoRA adapter: {lora_error}")
            logger.info(f"Loading finetuned model as full model: {repo}")
            finetuned_model = load_model(repo, is_base_model=False)

            if finetuned_model is None:
                raise ValueError(f"Finetuned model {repo} failed to load as full model")

            try:
                is_finetune = model_is_a_finetune(evaluation_args.original_model, finetuned_model)
            except Exception as e:
                logger.warning(f"Problem with detection of finetune for {repo}: {e}")
                is_finetune = False

        log_memory_stats()
        finetuned_model.eval()
        reference_model.eval()

        results = evaluate_finetuned_dpo_model(
            evaluation_args=evaluation_args,
            finetuned_model=finetuned_model,
            tokenizer=tokenizer,
            reference_model=reference_model,
        )
        results["is_finetune"] = is_finetune
        results_dict[repo] = results
    except Exception as e:
        logger.error(f"Error evaluating {repo}: {e}", exc_info=True)
        results_dict[repo] = str(e)
    finally:
        save_results_dict(results_dict, repo)
        log_memory_stats()


def main():
    logger.info("=== DPO EVALUATION SCRIPT STARTING ===")
    dataset = os.environ.get("DATASET")
    dataset_url = os.environ.get("DATASET_URL")
    original_model = os.environ.get("ORIGINAL_MODEL")
    dataset_type_str = os.environ.get("DATASET_TYPE", "")
    file_format_str = os.environ.get("FILE_FORMAT")
    models_str = os.environ.get("MODELS", "")  # Comma-separated list of LoRA repos
    if not dataset and dataset_url:
        parsed_name = os.path.basename(dataset_url.split("?")[0]) or "dataset.json"
        dataset = os.path.join(tempfile.gettempdir(), parsed_name)
        urllib.request.urlretrieve(dataset_url, dataset)
        logger.info(f"Downloaded dataset from DATASET_URL to {dataset}")

    if not all([dataset, original_model, file_format_str, models_str]):
        logger.error("Missing required environment variables.")
        exit(1)

    repos = [m.strip() for m in models_str.split(",") if m.strip()]
    for repo in repos:
        try:
            evaluation_args = EvaluationArgs(
                dataset=dataset,
                original_model=original_model,
                dataset_type=dataset_type_str,
                file_format=file_format_str,
                repo=repo,
            )

            subprocess.run(
                ["python", "-m", "validator.evaluation.single_eval_dpo", evaluation_args.model_dump_json()], check=True
            )
            logger.info(f"Subprocess completed for {repo}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running subprocess for {repo}: {e}")
    try:
        check_and_log_base_model_size(original_model)
    except Exception as e:
        logger.error(f"Error checking and logging base model size: {e}")

    logger.info("=== DPO EVALUATION SCRIPT COMPLETED ===")


if __name__ == "__main__":
    main()
