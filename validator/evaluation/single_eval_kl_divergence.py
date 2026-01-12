import sys

from datasets import load_dataset

from validator.core.models import EvaluationArgs
from validator.evaluation.common import calculate_kl_divergence
from validator.evaluation.common import load_finetuned_model
from validator.evaluation.common import load_model
from validator.evaluation.common import load_results_dict
from validator.evaluation.common import load_tokenizer
from validator.evaluation.common import log_memory_stats
from validator.evaluation.common import save_results_dict
from validator.evaluation.eval_grpo import _adapt_grpo_columns_to_trl
from validator.evaluation.utils import check_for_lora
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def evaluate_kl_divergence_repo(evaluation_args: EvaluationArgs) -> None:
    """Calculate KL divergence for a single model repository and update results."""
    results_dict = load_results_dict()
    repo = evaluation_args.repo

    # Check if repo exists in results
    if repo not in results_dict or not isinstance(results_dict[repo], dict):
        logger.error(f"No valid results for repository {repo}. Run GRPO evaluation first.")
        return

    # Skip if KL divergence already calculated
    if "kl_divergence" in results_dict[repo]:
        logger.info(f"KL divergence already calculated for {repo}")
        return

    try:
        logger.info(f"Loading tokenizer for {evaluation_args.original_model}")
        tokenizer = load_tokenizer(evaluation_args.original_model, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        logger.info(f"Loading original model: {evaluation_args.original_model}")
        original_model = load_model(evaluation_args.original_model, is_base_model=True, local_files_only=True)
        original_model.eval()

        logger.info(f"Loading finetuned model: {repo}")
        has_lora = check_for_lora(repo, local_files_only=True)
        if has_lora:
            finetuned_model = load_finetuned_model(repo, local_files_only=True)
        else:
            finetuned_model = load_model(repo, is_base_model=False, local_files_only=True)
        finetuned_model.eval()

        # Load evaluation dataset
        logger.info(f"Loading dataset: {evaluation_args.dataset}")
        dataset_path = evaluation_args.dataset
        eval_dataset = load_dataset("json", data_files=dataset_path, split="train")

        # Adapt columns to match GRPO format
        eval_dataset = _adapt_grpo_columns_to_trl(eval_dataset, evaluation_args.dataset_type)

        log_memory_stats()

        # Calculate KL divergence
        kl_divergence = calculate_kl_divergence(
            original_model=original_model,
            finetuned_model=finetuned_model,
            dataset=eval_dataset,
            tokenizer=tokenizer
        )

        # Update results with KL divergence
        results_dict[repo]["kl_divergence"] = kl_divergence
        logger.info(f"KL divergence for {repo}: {kl_divergence:.6f}")

    except Exception as e:
        logger.error(f"Error calculating KL divergence for {repo}: {e}", exc_info=True)
        results_dict[repo] = str(e)

    finally:
        save_results_dict(results_dict, repo)
        log_memory_stats()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error(f"Expected 1 argument, got {len(sys.argv) - 1}")
        logger.error(
            "Usage: python -m validator.evaluation.single_eval_kl_divergence <serialized_evaluation_args>"
        )
        sys.exit(1)

    evaluation_args = EvaluationArgs.model_validate_json(sys.argv[1])
    evaluate_kl_divergence_repo(evaluation_args)
