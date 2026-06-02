import sys

from validator.evaluation.models import EvaluationArgs
from validator.evaluation.evaluators.instruct_text import evaluate_repo


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python single_eval_instruct_text.py <serialized_evaluation_args>")
        sys.exit(1)

    evaluation_args = EvaluationArgs.model_validate_json(sys.argv[1])
    evaluate_repo(evaluation_args)
