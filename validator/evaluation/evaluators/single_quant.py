import logging
import sys
from pathlib import Path

from core.models.quant_models import QuantFixedSpec
from validator.evaluation.errors import evaluation_error_result
from validator.evaluation.evaluators.quant import FixedConfigCheck
from validator.evaluation.evaluators.quant import QuantEvaluationArgs
from validator.evaluation.evaluators.quant import evaluate_repo
from validator.evaluation.evaluators.quant import load_results
from validator.evaluation.evaluators.quant import record_base_model_params
from validator.evaluation.evaluators.quant import save_results
from validator.evaluation.quant_loss import load_examples
from validator.evaluation.quant_vllm import VllmPromptLogprobEngine
from validator.evaluation.quant_vllm import checked_kernel_classes
from validator.evaluation.quant_vllm import disable_humming_stream_k
from validator.evaluation.quant_vllm import expected_kernels_for
from validator.evaluation.quant_vllm import resolved_model_config


logger = logging.getLogger(__name__)


def _fixed_spec(evaluation_args: QuantEvaluationArgs) -> QuantFixedSpec | None:
    """The spec every layer must match, or None on the freeform track."""
    spec, quant_format = evaluation_args.quant_spec, evaluation_args.quant_format
    if quant_format is None:
        return spec
    if spec is not None and quant_format.spec() != spec:
        raise ValueError(f"QUANT_FORMAT {quant_format.value} does not match QUANT_SPEC {spec.name}")
    return quant_format.spec()


def run(evaluation_args: QuantEvaluationArgs) -> None:
    """Evaluate a single repo in vLLM and save its result to the shared results file."""
    results = load_results()
    repo, key = evaluation_args.repo, evaluation_args.saved_as
    if key in results:
        logger.info(f"Skipping {repo} as it's already evaluated")
        return

    try:
        # Before any engine is built, and outside evaluate_repo: a missing patch target is an
        # evaluation error, not the miner's scored fail.
        disable_humming_stream_k()
        spec = _fixed_spec(evaluation_args)
        expected_kernels = expected_kernels_for(spec) if spec is not None else None
        examples = load_examples(Path(evaluation_args.dataset))
        if not examples:
            raise ValueError(f"quant test set {evaluation_args.dataset} has no examples")
        max_prompt_len = max(len(example.input_ids) for example in examples)
        results[key] = evaluate_repo(
            repo=repo,
            examples=examples,
            engine_factory=lambda path: VllmPromptLogprobEngine(path, max_prompt_len, evaluation_args.original_model),
            layer_check=lambda engine: checked_kernel_classes(engine.llm, spec, expected_kernels),
            quant_spec=spec,
            config_check=FixedConfigCheck(base_model=evaluation_args.original_model, resolve=resolved_model_config),
            model_path=evaluation_args.model_path,
            eval_gpu=evaluation_args.eval_gpu,
        )
    except Exception as e:
        # evaluate_repo turns every fault of the submission into a quant_failure result, so whatever escapes it (the
        # hub unreachable, the Humming patch target missing, an empty test set) is an evaluation error, never the
        # miner's scored fail: the validator re-queues it.
        logger.error(f"Error evaluating {repo}: {e}", exc_info=True)
        results[key] = evaluation_error_result(e)
    finally:
        save_results(results)
        logger.info(f"Saved evaluation result for {repo}")

    try:
        record_base_model_params(evaluation_args.original_model)
    except Exception as e:
        logger.error(f"Error recording base model size: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python -m validator.evaluation.evaluators.single_quant <serialized_quant_evaluation_args>")
        sys.exit(1)

    run(QuantEvaluationArgs.model_validate_json(sys.argv[1]))
