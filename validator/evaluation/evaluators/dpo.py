import math
import os
import subprocess
import tempfile
import traceback
import urllib.request


# Allow torch.load for transformers 4.46+ security check
os.environ["TRANSFORMERS_ALLOW_TORCH_LOAD"] = "true"

import torch
import torch.nn.functional as F
from accelerate.utils import find_executable_batch_size
from axolotl.utils.dict import DictDefault
from datasets import Dataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from trl import DPOConfig
from trl import DPOTrainer

import core.constants as core_cst
import validator.evaluation.constants as cst
from core.logging import get_logger
from core.models.dataset_models import DpoDatasetType
from validator.evaluation.common import ProgressLoggerCallback
from validator.evaluation.common import _load_and_update_evaluation_config
from validator.evaluation.common import _log_dataset_and_model_info
from validator.evaluation.common import check_and_log_base_model_size
from validator.evaluation.common import eval_set_fingerprint
from validator.evaluation.common import count_model_parameters
from validator.evaluation.common import load_finetuned_model
from validator.evaluation.common import load_model
from validator.evaluation.common import load_results_dict
from validator.evaluation.common import load_tokenizer
from validator.evaluation.common import log_memory_stats
from validator.evaluation.common import sanitize_tokenizer_for_models
from validator.evaluation.common import save_results_dict
from validator.evaluation.dataset_utils import build_dummy_train_dataset
from validator.evaluation.model_checks import model_is_a_finetune
from validator.evaluation.models import EvaluationArgs
from validator.infrastructure.service_constants import VALI_CONFIG_PATH


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


def _dpo_eval_set_fingerprint(eval_dataset: Dataset) -> str:
    """Fingerprint the preference pairs in the order they are scored, plus what shapes the scoring.

    Deliberately NOT keyed on max_length. It comes from the candidate's own
    max_position_embeddings, which two honest submissions are allowed to differ on - a larger
    declared context is legal - so including it would make their fingerprints disagree and cost the
    challenger the task outright, for a truncation that only bites on pairs longer than the context
    window. beta is mixed in so a change to the loss definition invalidates stored vectors rather
    than silently comparing across two of them.
    """

    def _parts():
        yield f"beta={cst.BETA_DPO}".encode("utf-8")
        for row in eval_dataset:
            yield "\x00".join(
                (row[cst.TRL_DPO_FIELD_PROMPT], row[cst.TRL_DPO_FIELD_CHOSEN], row[cst.TRL_DPO_FIELD_REJECTED])
            ).encode("utf-8")

    return eval_set_fingerprint(_parts())


def _tokenize_dpo_pair(
    tokenizer: AutoTokenizer, row: dict, dataset_index: int
) -> tuple[list[int], list[int], list[int]]:
    """Prompt / chosen-completion / rejected-completion token ids for one preference pair.

    Mirrors DPO preprocessing rather than tokenizing the three fields independently:

    - EOS is appended to a completion that lacks it, so the model is scored on stopping as well.
    - Prompt and prompt+completion are tokenized as whole strings with special tokens on, and the
      completion is taken as the suffix past the prompt's length. Tokenizing the completion on its
      own would differ wherever the tokenizer merges across the prompt/completion boundary, and
      would drop the leading BOS from the prompt.
    """
    prompt = row[cst.TRL_DPO_FIELD_PROMPT]
    chosen = row[cst.TRL_DPO_FIELD_CHOSEN]
    rejected = row[cst.TRL_DPO_FIELD_REJECTED]

    eos_token = tokenizer.eos_token
    if eos_token is not None:
        if not chosen.endswith(eos_token):
            chosen = chosen + eos_token
        if not rejected.endswith(eos_token):
            rejected = rejected + eos_token

    prompt_ids = tokenizer(text=prompt)["input_ids"]
    prompt_chosen_ids = tokenizer(text=prompt + chosen)["input_ids"]
    prompt_rejected_ids = tokenizer(text=prompt + rejected)["input_ids"]

    prompt_len = len(prompt_ids)
    if prompt_chosen_ids[:prompt_len] != prompt_ids or prompt_rejected_ids[:prompt_len] != prompt_ids:
        logger.warning(
            f"Row {dataset_index}: tokenized prompt is not a prefix of tokenized prompt+completion "
            "(tokenizer merged across the boundary); splitting on prompt length anyway"
        )

    return prompt_ids, prompt_chosen_ids[prompt_len:], prompt_rejected_ids[prompt_len:]


def _completion_logprob(
    model: AutoModelForCausalLM,
    prompt_ids: list[int],
    completion_ids: list[int],
    max_length: int,
) -> float | None:
    """Sum of log P(token) over the completion tokens only, prompt tokens masked out.

    SUMMED rather than averaged, matching TRL's default DPO loss (no length normalisation), so a
    pair's loss scales with its completion length. That is fine for pairing - example i is only
    ever compared against example i - but per-pair values are not comparable across pairs.
    """
    ids = (prompt_ids + completion_ids)[:max_length]
    prompt_len = min(len(prompt_ids), max_length)
    if len(ids) <= prompt_len:
        # Truncation ate the whole completion; nothing to score.
        return None

    input_ids = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits.float()

    # Position t predicts token t+1, so completion token at index prompt_len is predicted at t=prompt_len-1.
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # max() because prompt_len == 0 (empty prompt on a tokenizer that adds no BOS) would make this
    # [-1:], silently scoring only the final token instead of the whole completion.
    return token_log_probs[0, max(prompt_len - 1, 0) :].sum().item()


def _compute_per_pair_dpo_losses(
    finetuned_model: AutoModelForCausalLM,
    reference_model: AutoModelForCausalLM,
    eval_dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> list[float]:
    """DPO loss per preference pair, for the paired boss-round comparison.

    Computed directly from policy and reference log-probabilities rather than through TRL, because
    TRL is unpinned - it comes from the axolotl base image - so reaching into its internals would
    break silently on a base bump. The caller asserts the mean of this against TRL's reported
    eval_loss, which is what catches a wrong formula on code that cannot be run locally.
    """
    finetuned_model.eval()
    reference_model.eval()

    losses: list[float] = []
    for index, row in enumerate(eval_dataset):
        prompt_ids, chosen_ids, rejected_ids = _tokenize_dpo_pair(tokenizer, row, index)

        policy_chosen = _completion_logprob(finetuned_model, prompt_ids, chosen_ids, max_length)
        policy_rejected = _completion_logprob(finetuned_model, prompt_ids, rejected_ids, max_length)
        reference_chosen = _completion_logprob(reference_model, prompt_ids, chosen_ids, max_length)
        reference_rejected = _completion_logprob(reference_model, prompt_ids, rejected_ids, max_length)

        if None in (policy_chosen, policy_rejected, reference_chosen, reference_rejected):
            losses.append(float("nan"))
            continue

        logits = (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected)
        losses.append(-F.logsigmoid(torch.tensor(cst.BETA_DPO * logits)).item())

    # No empty_cache() per pair: it synchronises the device and forces reallocation, on top of the
    # four sequential forwards each pair already costs. That time lands against
    # EVAL_BASILICA_TIMEOUT, and a timeout fails the whole eval batch.
    return losses


def evaluate_dpo_model(
    evaluation_config: DictDefault,
    finetuned_model: AutoModelForCausalLM,
    reference_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    evaluation_args: EvaluationArgs,
) -> dict[str, float | list[float] | str]:
    evaluation_config.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Config: {evaluation_config}")

    data_files = evaluation_config.datasets[0].get("data_files", [evaluation_config.datasets[0]["path"]])
    eval_dataset = load_dataset("json", data_files=data_files, split="train")
    eval_dataset = _adapt_dpo_columns_to_trl(eval_dataset, evaluation_args.dataset_type)

    _log_dataset_and_model_info(eval_dataset, finetuned_model, tokenizer)

    def custom_data_collator(features):
        logger.debug(f"Collating {len(features)} features")
        return _collate_dpo_batch(features, tokenizer)

    max_length = getattr(finetuned_model.config, "max_position_embeddings", 8192)
    logger.info(f"DPO eval max_length set to {max_length} (from model config)")

    @find_executable_batch_size(starting_batch_size=evaluation_config.starting_batch_size)
    def evaluate_dpo_with_batch_size(batch_size):
        training_args = DPOConfig(
            output_dir=evaluation_config.output_dir,
            per_device_eval_batch_size=batch_size,
            report_to="none",
            bf16=True,
            beta=cst.BETA_DPO,
            max_length=max_length,
        )
        dpo_trainer = DPOTrainer(
            model=finetuned_model,
            ref_model=reference_model,
            args=training_args,
            train_dataset=build_dummy_train_dataset(eval_dataset),
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

    if os.environ.get(core_cst.EMIT_PER_EXAMPLE_LOSSES_ENV) != "1":
        return evaluation_results

    try:
        per_pair_losses = _compute_per_pair_dpo_losses(
            finetuned_model=finetuned_model,
            reference_model=reference_model,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_length=max_length,
        )
    except Exception as e:
        # Optional add-on: never let it discard the eval_loss the trainer already produced.
        logger.error(f"Per-pair DPO loss extraction failed, continuing without the vector: {e}", exc_info=True)
        return evaluation_results

    finite = [loss for loss in per_pair_losses if math.isfinite(loss)]
    if not finite:
        logger.error("PER_EXAMPLE_LOSSES: no finite per-pair DPO losses - the paired comparison will have nothing to use")
    else:
        mean_loss = sum(finite) / len(finite)
        # DPO loss is a flat per-pair mean and eval batch size is pinned to 1, so this should agree
        # closely. A divergence means the formula or the tokenization does not match TRL's.
        if not math.isclose(mean_loss, eval_results["eval_loss"], rel_tol=5e-3, abs_tol=1e-3):
            logger.error(
                f"PER_EXAMPLE_LOSSES MISMATCH: mean of {len(finite)} per-pair DPO losses is {mean_loss:.8f} "
                f"but TRL reported eval_loss={eval_results['eval_loss']:.8f}. The vector does not measure the "
                f"same quantity as the scalar - do not trust boss-round verdicts built on it."
            )
        else:
            logger.info(
                f"PER_EXAMPLE_LOSSES: {len(finite)} pairs, mean {mean_loss:.8f} matches "
                f"eval_loss {eval_results['eval_loss']:.8f}"
            )

    evaluation_results["per_example_losses"] = per_pair_losses
    evaluation_results["eval_set_fingerprint"] = _dpo_eval_set_fingerprint(eval_dataset)
    return evaluation_results


def evaluate_finetuned_dpo_model(
    evaluation_args: EvaluationArgs,
    finetuned_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    reference_model: AutoModelForCausalLM,
) -> dict[str, float]:
    evaluation_config = _load_and_update_evaluation_config(
        evaluation_args=evaluation_args, finetuned_model=finetuned_model, config_path=VALI_CONFIG_PATH
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
        tokenizer = sanitize_tokenizer_for_models(tokenizer, reference_model, finetuned_model)

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
                ["python", "-m", "validator.evaluation.evaluators.single_dpo", evaluation_args.model_dump_json()], check=True
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
