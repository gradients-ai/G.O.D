import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from pydantic import TypeAdapter


# Allow torch.load for transformers 4.46+ security check
os.environ["TRANSFORMERS_ALLOW_TORCH_LOAD"] = "true"

import torch
import torch.nn.functional as F
from accelerate.utils import find_executable_batch_size
from axolotl.utils.dict import DictDefault
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

import core.constants as core_cst
from core.models.utility_models import TextDatasetType
from validator.core import constants as cst
from validator.core.models import EvaluationArgs
from validator.evaluation.common import ProgressLoggerCallback
from validator.evaluation.common import _load_and_update_evaluation_config
from validator.evaluation.common import _log_dataset_and_model_info
from validator.evaluation.common import check_and_log_base_model_size
from validator.evaluation.common import load_finetuned_model
from validator.evaluation.common import load_model
from validator.evaluation.common import load_results_dict
from validator.evaluation.common import load_tokenizer
from validator.evaluation.common import log_memory_stats
from validator.evaluation.common import sanitize_tokenizer_for_models
from validator.evaluation.common import save_results_dict
from validator.evaluation.utils import check_for_lora
from validator.evaluation.utils import model_is_a_finetune
from validator.utils.logging import get_logger


logger = get_logger(__name__)

try:
    from axolotl.utils.data import load_tokenized_prepared_datasets

    _USES_LEGACY_AXOLOTL_DATA_API = True
except ImportError:
    from axolotl.utils.data.sft import _load_tokenized_prepared_datasets as load_tokenized_prepared_datasets

    _USES_LEGACY_AXOLOTL_DATA_API = False


def _load_evaluation_dataset(evaluation_config: DictDefault, tokenizer: AutoTokenizer) -> Dataset:
    prepared_path = Path(evaluation_config.output_dir) / "prepared"
    if _USES_LEGACY_AXOLOTL_DATA_API:
        eval_dataset, _ = load_tokenized_prepared_datasets(tokenizer, evaluation_config, prepared_path)
    else:
        evaluation_config["dataset_prepared_path"] = str(prepared_path)
        eval_dataset, _ = load_tokenized_prepared_datasets(tokenizer, evaluation_config, split="train")

    original_length = len(eval_dataset)
    eval_dataset = [sample for sample in eval_dataset if any(label != -100 for label in sample["labels"])]
    filtered_length = len(eval_dataset)

    logger.info(f"Filtered out {original_length - filtered_length} samples with empty outputs")
    eval_dataset = sorted(eval_dataset, key=lambda x: len(x["input_ids"]))
    logger.info(f"Loaded evaluation dataset with {filtered_length} samples")
    return eval_dataset


def _max_eval_sequence_cap(tokenizer: AutoTokenizer, language_model: AutoModelForCausalLM) -> int:
    """Upper bound for eval sequence_len from model and tokenizer."""
    max_pos = getattr(language_model.config, "max_position_embeddings", None)
    if not isinstance(max_pos, int) or max_pos <= 0:
        max_pos = 131_072
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 1_000_000:
        return min(max_pos, tok_max)
    return max_pos


def _sequence_len_candidates(start: int, cap: int) -> list[int]:
    start, cap = int(start), int(cap)
    if start <= 0:
        start = 2048
    if cap <= 0:
        cap = 8192
    if start > cap:
        return [cap]
    out: list[int] = []
    s = start
    while True:
        if s not in out:
            out.append(s)
        if s >= cap:
            break
        nxt = min(s * 2, cap)
        if nxt <= s:
            break
        s = nxt
    return out


def _load_evaluation_dataset_with_sequence_retries(
    evaluation_config: DictDefault,
    tokenizer: AutoTokenizer,
    language_model: AutoModelForCausalLM,
) -> list:
    """
    Axolotl drops samples longer than sequence_len. If nothing survives, retry
    with larger sequence_len (cleared prepared cache) up to model limits.
    """
    prepared_path = Path(evaluation_config.output_dir) / "prepared"
    start_seq = int(evaluation_config.sequence_len)
    cap = _max_eval_sequence_cap(tokenizer, language_model)
    candidates = _sequence_len_candidates(start_seq, cap)
    logger.info(f"Eval sequence_len candidates (start={start_seq}, cap={cap}): {candidates}")

    last_dataset: list = []
    for seq_len in candidates:
        evaluation_config.sequence_len = seq_len
        if prepared_path.exists():
            shutil.rmtree(prepared_path, ignore_errors=True)
        last_dataset = _load_evaluation_dataset(evaluation_config, tokenizer)
        if len(last_dataset) > 0:
            logger.info(f"Using sequence_len={seq_len} with {len(last_dataset)} evaluation samples")
            return last_dataset
        logger.warning(
            f"No eval samples after tokenization at sequence_len={seq_len}; "
            f"retrying with a larger sequence_len if available"
        )

    raise ValueError(
        f"No evaluation samples after trying sequence_len values {candidates} "
        f"(all rows likely exceed cap {cap} or have empty trainable labels)"
    )


def _collate_evaluation_batch(batch: list[dict[str, list[int]]], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _calculate_instruct_kl_divergence(
    base_model: AutoModelForCausalLM,
    finetuned_model: AutoModelForCausalLM,
    eval_dataset: list,
    tokenizer: AutoTokenizer,
    batch_size: int,
) -> float:
    """
    Mean per-token KL(finetuned || base) over the trainable (label != -100) positions of the
    instruct eval set. Used to weight the eval loss when a task was trained with a KL term.
    """
    base_model.eval()
    finetuned_model.eval()

    total_kl = 0.0
    total_tokens = 0
    for i in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[i : i + batch_size]
        collated = _collate_evaluation_batch(batch, tokenizer)
        input_ids = collated["input_ids"].cuda()
        attention_mask = collated["attention_mask"].cuda()
        labels = collated["labels"].cuda()

        with torch.no_grad():
            base_logits = base_model(input_ids=input_ids, attention_mask=attention_mask).logits
            finetuned_logits = finetuned_model(input_ids=input_ids, attention_mask=attention_mask).logits

            base_log_probs = F.log_softmax(base_logits, dim=-1)
            finetuned_log_probs = F.log_softmax(finetuned_logits, dim=-1)
            finetuned_probs = finetuned_log_probs.exp()

            # KL(finetuned || base) per position, summed over vocab
            kl_per_token = (finetuned_probs * (finetuned_log_probs - base_log_probs)).sum(dim=-1)
            mask = (labels != -100).float()

            total_kl += (kl_per_token * mask).sum().item()
            total_tokens += int(mask.sum().item())

        torch.cuda.empty_cache()

    if total_tokens == 0:
        logger.warning("No trainable tokens found for KL divergence calculation; returning 0.0")
        return 0.0

    avg_kl = total_kl / total_tokens
    logger.info(f"Instruct KL divergence: {avg_kl:.6f} over {total_tokens} tokens")
    return avg_kl


def evaluate_instruct_text_model(
    evaluation_config: DictDefault,
    language_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_model: AutoModelForCausalLM | None = None,
    kl_coef: float | None = None,
) -> dict[str, float]:
    evaluation_config.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Config: {evaluation_config}")

    eval_dataset = _load_evaluation_dataset_with_sequence_retries(
        evaluation_config, tokenizer, language_model
    )

    _log_dataset_and_model_info(eval_dataset, language_model, tokenizer)

    def custom_data_collator(features):
        return _collate_evaluation_batch(features, tokenizer)

    @find_executable_batch_size(starting_batch_size=evaluation_config.starting_batch_size)
    def evaluate_with_batch_size(batch_size):
        training_args = TrainingArguments(
            output_dir=evaluation_config.output_dir,
            per_device_eval_batch_size=batch_size,
            report_to="none",
            bf16=True,
        )

        trainer = Trainer(
            model=language_model,
            args=training_args,
            processing_class=tokenizer,
            eval_dataset=eval_dataset,
            data_collator=custom_data_collator,
            callbacks=[ProgressLoggerCallback(log_interval_seconds=evaluation_config.log_interval_seconds)],
        )

        eval_results = trainer.evaluate()
        return eval_results

    eval_results = evaluate_with_batch_size()
    logger.info(f"Final evaluation results: {eval_results}")
    eval_loss = eval_results["eval_loss"]
    evaluation_results = {
        "eval_loss": eval_loss,
    }

    # When the task was trained with a KL term, weight the eval loss by the KL divergence
    # against the base model so the ranking metric rewards staying close to the base.
    if base_model is not None and kl_coef:
        kl_divergence = _calculate_instruct_kl_divergence(
            base_model=base_model,
            finetuned_model=language_model,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            batch_size=cst.GRPO_KL_BATCH_SIZE,
        )
        weighted_loss = eval_loss + kl_coef * kl_divergence
        logger.info(
            f"KL-weighted eval loss: {eval_loss:.6f} + {kl_coef} * {kl_divergence:.6f} = {weighted_loss:.6f}"
        )
        evaluation_results["eval_loss"] = weighted_loss
        evaluation_results["eval_loss_raw"] = eval_loss
        evaluation_results["kl_divergence"] = kl_divergence

    return evaluation_results


def evaluate_finetuned_model(
    evaluation_args: EvaluationArgs,
    finetuned_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_model: AutoModelForCausalLM | None = None,
    kl_coef: float | None = None,
) -> dict[str, float]:
    evaluation_config = _load_and_update_evaluation_config(
        evaluation_args=evaluation_args, finetuned_model=finetuned_model, config_path=cst.VALI_CONFIG_PATH
    )
    return evaluate_instruct_text_model(
        evaluation_config, finetuned_model, tokenizer, base_model=base_model, kl_coef=kl_coef
    )


def evaluate_repo(evaluation_args: EvaluationArgs) -> None:
    """Evaluate a single model repository and save results directly to file."""
    results_dict = load_results_dict()
    repo = evaluation_args.repo

    # Skip if duplicate
    if repo in results_dict:
        logger.info(f"Skipping {repo} as it's already evaluated")
        return

    tokenizer = load_tokenizer(evaluation_args.original_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # KL weighting is opt-in per task, signalled via container env vars.
    use_kl = os.environ.get(core_cst.USE_KL_ENV) == "1"
    kl_coef_env = os.environ.get(core_cst.KL_COEF_ENV)
    kl_coef = float(kl_coef_env) if kl_coef_env else None

    try:
        if check_for_lora(repo):
            logger.info("LoRA adapter detected. Loading as with Peft")
            finetuned_model = load_finetuned_model(repo)
            is_finetune = True
        else:
            logger.info("No LoRA adapter detected. Loading full model")
            finetuned_model = load_model(repo, is_base_model=False)
            try:
                is_finetune = model_is_a_finetune(evaluation_args.original_model, finetuned_model)
            except Exception as e:
                logger.info(f"Problem with detection of finetune for {repo}: {e}")
                logger.info("Assuming False")
                is_finetune = False
        log_memory_stats()
        finetuned_model.eval()

        base_model = None
        if use_kl and kl_coef:
            logger.info(f"KL weighting enabled (kl_coef={kl_coef}); loading base model {evaluation_args.original_model}")
            base_model = load_model(evaluation_args.original_model, is_base_model=True)
            base_model.eval()
            tokenizer = sanitize_tokenizer_for_models(tokenizer, finetuned_model, base_model)
        else:
            tokenizer = sanitize_tokenizer_for_models(tokenizer, finetuned_model)

        results = evaluate_finetuned_model(
            evaluation_args=evaluation_args,
            finetuned_model=finetuned_model,
            tokenizer=tokenizer,
            base_model=base_model,
            kl_coef=kl_coef,
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
    logger.info("=== INSTRUCT TEXT EVALUATION SCRIPT STARTING ===")
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

    model_adapter = TypeAdapter(TextDatasetType)
    dataset_type = model_adapter.validate_python(json.loads(dataset_type_str))

    repos = [m.strip() for m in models_str.split(",") if m.strip()]

    for repo in repos:
        try:
            evaluation_args = EvaluationArgs(
                dataset=dataset, original_model=original_model, dataset_type=dataset_type, file_format=file_format_str, repo=repo
            )

            # Launching subprocess to purge memory: https://github.com/huggingface/transformers/issues/26571
            subprocess.run(
                ["python", "-m", "validator.evaluation.single_eval_instruct_text", evaluation_args.model_dump_json()], check=True
            )
            logger.info(f"Subprocess completed for {repo}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running subprocess for {repo}: {e}")
    try:
        check_and_log_base_model_size(original_model)
    except Exception as e:
        logger.error(f"Error checking and logging base model size: {e}")

    logger.info("=== INSTRUCT TEXT EVALUATION SCRIPT COMPLETED ===")


if __name__ == "__main__":
    main()
