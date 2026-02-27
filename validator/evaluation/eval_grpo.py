import os
import subprocess
import time
import tempfile
import urllib.request


# Allow torch.load for transformers 4.46+ security check
os.environ["TRANSFORMERS_ALLOW_TORCH_LOAD"] = "true"

from accelerate.utils import find_executable_batch_size
from axolotl.utils.dict import DictDefault
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from trl import GRPOConfig
from trl import GRPOTrainer

from core.models.utility_models import GrpoDatasetType
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
from validator.evaluation.common import save_results_dict
from validator.evaluation.utils import check_for_lora
from validator.evaluation.utils import model_is_a_finetune
from validator.utils.logging import get_logger
from validator.utils.reward_functions import supports_extra_data
from validator.utils.reward_functions import validate_reward_function


logger = get_logger(__name__)


def _adapt_grpo_columns_to_trl(dataset: Dataset, dataset_type: GrpoDatasetType) -> Dataset:
    """
    Transform a GRPO dataset to match trl's expected column names.

    Args:
        dataset: Hugging Face dataset object
        dataset_type: GrpoDatasetType with field mappings
    """
    logger.info("Adapting GRPO columns to standard format")

    column_mapping = {
        dataset_type.field_prompt: cst.TRL_GRPO_FIELD_PROMPT,
    }

    if dataset_type.extra_column:
        column_mapping[dataset_type.extra_column] = cst.STANDARD_GRPO_EXTRA_COLUMN
    for src_col, dst_col in column_mapping.items():
        if src_col in dataset.column_names and src_col != dst_col:
            dataset = dataset.rename_column(src_col, dst_col)

    return dataset


def evaluate_grpo_model(
    evaluation_config: DictDefault,
    finetuned_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    evaluation_args: EvaluationArgs,
) -> dict[str, float]:
    evaluation_config.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Config: {evaluation_config}")

    data_files = evaluation_config.datasets[0].get("data_files", [evaluation_config.datasets[0]["path"]])
    eval_dataset = load_dataset("json", data_files=data_files, split="train")

    eval_dataset = _adapt_grpo_columns_to_trl(eval_dataset, evaluation_args.dataset_type)

    _log_dataset_and_model_info(eval_dataset, finetuned_model, tokenizer)

    reward_funcs_callable = []
    reward_func_names = []
    reward_weights = []

    logger.info(f"Processing {len(evaluation_args.dataset_type.reward_functions)} reward functions")

    reward_weights_list = [rf.reward_weight for rf in evaluation_args.dataset_type.reward_functions]
    logger.info(f"Using weights directly: {reward_weights_list}")

    for i, reward_function in enumerate(evaluation_args.dataset_type.reward_functions):
        reward_func_str = reward_function.reward_func

        sample_data = eval_dataset.to_list()[:10] if len(eval_dataset) > 0 else None
        is_valid, error_msg, reward_func_callable = validate_reward_function(reward_func_str, sample_data)
        if not is_valid:
            logger.error(f"Invalid reward function:\n{reward_func_str}")
            logger.error(f"Validation error message: {error_msg}")
            raise ValueError(f"Invalid reward function: {error_msg}")

        reward_weight = reward_weights_list[i]
        reward_funcs_callable.append(reward_func_callable)

        func_name = getattr(reward_function, "name", f"reward_func_{i}")
        weighted_name = f"{func_name}_weight_{reward_weight:.2f}"
        reward_func_names.append(weighted_name)
        reward_weights.append(reward_weight)

        logger.info(f"Using reward function {i}: {func_name} with weight {reward_weight:.4f}")

    captured_rewards = {name: [] for name in reward_func_names}
    raw_rewards = {name: [] for name in reward_func_names}
    wrapped_reward_funcs = []

    has_extra_column = evaluation_args.dataset_type.extra_column and cst.STANDARD_GRPO_EXTRA_COLUMN in eval_dataset.column_names
    extra_column_data = eval_dataset[cst.STANDARD_GRPO_EXTRA_COLUMN] if has_extra_column else None

    if extra_column_data is not None:
        import json

        parsed_extra_data = []
        for item in extra_column_data:
            if isinstance(item, str):
                try:
                    parsed_item = json.loads(item)
                    parsed_extra_data.append(parsed_item)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse extra_data as JSON: {item[:100]}...")
                    parsed_extra_data.append(item)
            else:
                parsed_extra_data.append(item)
        extra_column_data = parsed_extra_data

    for i, (original_func, func_name, weight) in enumerate(zip(reward_funcs_callable, reward_func_names, reward_weights)):

        def create_wrapper(original_func, func_name, weight):
            supports_extra = supports_extra_data(original_func)

            def call_reward_func(completions, kwargs):
                # Some reward functions accept **kwargs (and derive extra data from it),
                # while others only accept completions. Try kwargs first, then fallback.
                try:
                    return original_func(completions, **kwargs)
                except TypeError:
                    return original_func(completions)

            if supports_extra and has_extra_column:

                def wrapper(completions, **kwargs):
                    logger.debug(f"🔧 Calling {func_name} with {len(completions)} completions")

                    logger.info(f"🔍 {func_name}: TRL kwargs keys = {list(kwargs.keys())}")

                    # TRL already passes extra_data in kwargs - don't double-pass it
                    extra_data_from_trl = kwargs.get(cst.STANDARD_GRPO_EXTRA_COLUMN)
                    if extra_data_from_trl:
                        logger.info(f"🔍 {func_name}: extra_data from TRL = {str(extra_data_from_trl[0])[:100]}...")

                    raw_results = call_reward_func(completions, kwargs)

                    logger.info(f"🔍 {func_name}: returned scores = {raw_results[:3]}... (showing first 3)")
                    raw_rewards[func_name].extend(raw_results)
                    weighted_results = [r * weight for r in raw_results]
                    captured_rewards[func_name].extend(weighted_results)

                    if len(captured_rewards[func_name]) % 20 == 0:
                        avg_raw = sum(raw_results) / len(raw_results) if raw_results else 0
                        avg_weighted = sum(weighted_results) / len(weighted_results) if weighted_results else 0
                        total_count = len(captured_rewards[func_name])
                        logger.info(
                            f"🏆 {func_name}: batch_avg_raw={avg_raw:.4f}, batch_avg_weighted={avg_weighted:.4f}, total_samples={total_count}"
                        )

                    return weighted_results
            else:

                def wrapper(completions, **kwargs):
                    logger.debug(f"🔧 Calling {func_name} with {len(completions)} completions (no extra_data)")
                    raw_results = call_reward_func(completions, kwargs)
                    raw_rewards[func_name].extend(raw_results)
                    weighted_results = [r * weight for r in raw_results]
                    captured_rewards[func_name].extend(weighted_results)

                    if len(captured_rewards[func_name]) % 20 == 0:
                        avg_raw = sum(raw_results) / len(raw_results) if raw_results else 0
                        avg_weighted = sum(weighted_results) / len(weighted_results) if weighted_results else 0
                        total_count = len(captured_rewards[func_name])
                        logger.info(
                            f"🏆 {func_name}: batch_avg_raw={avg_raw:.4f}, batch_avg_weighted={avg_weighted:.4f}, total_samples={total_count}"
                        )

                    return weighted_results

            return wrapper

        wrapped_reward_funcs.append(create_wrapper(original_func, func_name, weight))

    @find_executable_batch_size(starting_batch_size=cst.GRPO_INITIAL_BATCH_SIZE)
    def evaluate_grpo_with_batch_size(batch_size):
        num_generations = max(2, cst.GRPO_DEFAULT_NUM_GENERATIONS)  # Ensure minimum of 2
        # Reduce generations if needed, but never below 2
        while batch_size < num_generations and num_generations > 2:
            num_generations = num_generations // 2
        logger.info(f"Using {num_generations} generations per prompt")
        training_args = GRPOConfig(
            output_dir=evaluation_config.output_dir,
            per_device_eval_batch_size=batch_size,
            report_to="none",
            bf16=True,
            beta=cst.BETA_GRPO,
            num_generations=num_generations,
        )
        grpo_trainer = GRPOTrainer(
            model=finetuned_model,
            reward_funcs=wrapped_reward_funcs,
            args=training_args,
            train_dataset=Dataset.from_dict({col: [] for col in eval_dataset.column_names}),
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=[ProgressLoggerCallback(log_interval_seconds=evaluation_config.log_interval_seconds)],
        )

        results = grpo_trainer.evaluate()
        return results

    eval_results = evaluate_grpo_with_batch_size()
    logger.info(f"Final GRPO evaluation results: {eval_results}")

    final_weighted_rewards = {}
    final_raw_rewards = {}

    logger.info("🎯 FINAL REWARD STATISTICS:")
    for name, captured_reward_list in captured_rewards.items():
        if captured_reward_list:
            final_weighted_rewards[name] = sum(captured_reward_list) / len(captured_reward_list)
            raw_reward_list = raw_rewards.get(name, [])
            final_raw_rewards[name] = sum(raw_reward_list) / len(raw_reward_list) if raw_reward_list else 0

            logger.info(
                f"🏆 {name}: avg_raw={final_raw_rewards[name]:.4f}, avg_weighted={final_weighted_rewards[name]:.4f}, samples={len(captured_reward_list)}"
            )

    total_avg_reward = sum(final_weighted_rewards.values())
    logger.info(f"🎯 TOTAL AVERAGE WEIGHTED REWARD: {total_avg_reward:.4f}")

    # Create weights mapping for normalization function
    weights_mapping = {}
    for name, weight in zip(reward_func_names, reward_weights):
        weights_mapping[name] = weight

    evaluation_results = {
        "eval_loss": total_avg_reward - eval_results.get("eval_loss", 0.0),
        "final_weighted_rewards": final_weighted_rewards,
        "final_raw_rewards": final_raw_rewards,
        "total_avg_reward": total_avg_reward,
        "weights": weights_mapping,
    }
    return evaluation_results


def evaluate_finetuned_grpo_model(
    evaluation_args: EvaluationArgs,
    finetuned_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> dict[str, float]:
    evaluation_config = _load_and_update_evaluation_config(
        evaluation_args=evaluation_args, finetuned_model=finetuned_model, config_path=cst.VALI_CONFIG_PATH
    )
    return evaluate_grpo_model(evaluation_config, finetuned_model, tokenizer, evaluation_args)


def evaluate_grpo_repo(evaluation_args: EvaluationArgs) -> None:
    """Evaluate a single model repository and save results directly to file."""
    results_dict = load_results_dict()
    repo = evaluation_args.repo

    # Skip if duplicate
    if repo in results_dict:
        logger.info(f"Skipping {repo} as it's already evaluated")
        return

    try:
        tokenizer = load_tokenizer(evaluation_args.original_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
        raise

    try:
        has_lora = check_for_lora(repo)

        if has_lora:
            finetuned_model = load_finetuned_model(repo)
            is_finetune = True
        else:
            finetuned_model = load_model(repo, is_base_model=False)
            try:
                is_finetune = model_is_a_finetune(evaluation_args.original_model, finetuned_model)
            except Exception as e:
                logger.info(f"Problem with detection of finetune for {repo}: {e}")
                logger.info("Assuming False")
                is_finetune = False
        log_memory_stats()
        finetuned_model.eval()

        results = evaluate_finetuned_grpo_model(
            evaluation_args=evaluation_args,
            finetuned_model=finetuned_model,
            tokenizer=tokenizer,
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
    dataset = os.environ.get("DATASET")
    dataset_url = os.environ.get("DATASET_URL")
    original_model = os.environ.get("ORIGINAL_MODEL")
    dataset_type_str = os.environ.get("DATASET_TYPE", "")
    file_format_str = os.environ.get("FILE_FORMAT")
    models_str = os.environ.get("MODELS", "")
    if not dataset and dataset_url:
        parsed_name = os.path.basename(dataset_url.split("?")[0]) or "dataset.json"
        dataset = os.path.join(tempfile.gettempdir(), parsed_name)
        urllib.request.urlretrieve(dataset_url, dataset)
        logger.info(f"Downloaded dataset from DATASET_URL to {dataset}")

    if not all([dataset, original_model, file_format_str, models_str]):
        logger.error("Missing required environment variables.")
        exit(1)

    repos = [m.strip() for m in models_str.split(",") if m.strip()]

    timeout_seconds = 18_000
    timeout_adjusted = False

    for repo in repos:
        try:
            evaluation_args = EvaluationArgs(
                dataset=dataset,
                original_model=original_model,
                dataset_type=dataset_type_str,
                file_format=file_format_str,
                repo=repo,
            )

            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    start_time = time.monotonic()
                    # Launching subprocess to purge memory
                    subprocess.run(
                        ["python", "-m", "validator.evaluation.single_eval_grpo", evaluation_args.model_dump_json()],
                        check=True,
                        timeout=timeout_seconds,
                    )
                    elapsed = time.monotonic() - start_time
                    logger.info(f"GRPO subprocess completed for {repo} in {elapsed:.2f} seconds")
                    break
                except subprocess.TimeoutExpired:
                    retry_count += 1
                    logger.warning(f"GRPO subprocess timed out for {repo} (attempt {retry_count}/{max_retries})")
                    if retry_count == max_retries:
                        logger.error(f"Max retries reached for GRPO evaluation of {repo}")
                        raise

            # Now run KL divergence calculation in separate subprocess
            logger.info(f"Starting KL divergence calculation for {repo}")
            subprocess.run(
                ["python", "-m", "validator.evaluation.single_eval_kl_divergence", evaluation_args.model_dump_json()],
                check=True,
                timeout=timeout_seconds,
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running GRPO subprocess for {repo}: {e}")
    try:
        check_and_log_base_model_size(original_model)
    except Exception as e:
        logger.error(f"Error checking and logging base model size: {e}")

    logger.info("=== GRPO EVALUATION SCRIPT COMPLETED ===")


if __name__ == "__main__":
    main()
