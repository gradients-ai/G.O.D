import sys

from validator.core.models import EvaluationArgs
from validator.evaluation.eval_grpo import evaluate_grpo_repo
from validator.utils.logging import get_logger


logger = get_logger(__name__)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error(f"Expected 1 argument, got {len(sys.argv) - 1}")
        logger.error(
            "Usage: python -m validator.evaluation.single_eval_grpo <serialized_evaluation_args>"
        )
        sys.exit(1)

    evaluation_args = EvaluationArgs.model_validate_json(sys.argv[1])
    evaluate_grpo_repo(evaluation_args)
