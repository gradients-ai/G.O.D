import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from validator.evaluation.quant_loss import QuantEvalExample
from validator.evaluation.quant_loss import QuantKernelMismatch
from validator.evaluation.quant_loss import QuantScoringError
from validator.evaluation.quant_loss import assert_expected_kernels
from validator.evaluation.quant_loss import example_loss
from validator.evaluation.quant_loss import is_scored_fail
from validator.evaluation.quant_loss import load_examples
from validator.evaluation.quant_loss import quant_eval_set_fingerprint
from validator.evaluation.quant_loss import score_examples


class TableEngine:
    """Fake engine: log p(token_t | <t) is a fixed function of (t, token_t); entry 0 is None."""

    def prompt_logprobs(self, token_ids: list[int]) -> list[float | None]:
        return [None] + [table_logprob(t, token) for t, token in enumerate(token_ids) if t > 0]


def table_logprob(t: int, token: int) -> float:
    return -0.1 * t - 0.01 * token - 0.003


def hf_shifted_loss(input_ids: list[int], labels: list[int]) -> float:
    """HF ForCausalLMLoss at batch size 1: the label at t+1 is scored by the logits at t."""
    nll = [-table_logprob(t + 1, input_ids[t + 1]) for t in range(len(labels) - 1) if labels[t + 1] != -100]
    return sum(nll) / len(nll) if nll else float("nan")


PROMPT_MASKED = QuantEvalExample(input_ids=[5, 6, 7, 8, 9], labels=[-100, -100, 7, 8, 9])
FIRST_SUPERVISED = QuantEvalExample(input_ids=[3, 4, 5, 6], labels=[3, 4, 5, 6])
ALL_MASKED = QuantEvalExample(input_ids=[1, 2, 3], labels=[-100, -100, -100])


@pytest.mark.parametrize("example", [PROMPT_MASKED, FIRST_SUPERVISED])
def test_example_loss_matches_hf_shifted_mean(example):
    result = example_loss(TableEngine().prompt_logprobs(example.input_ids), example.labels)

    assert result.loss == pytest.approx(hf_shifted_loss(example.input_ids, example.labels), rel=1e-12)


def test_first_supervised_token_is_dropped_from_the_count():
    result = example_loss(TableEngine().prompt_logprobs(FIRST_SUPERVISED.input_ids), FIRST_SUPERVISED.labels)

    assert result.n_tokens == len(FIRST_SUPERVISED.labels) - 1


@pytest.mark.parametrize(
    "example",
    [ALL_MASKED, QuantEvalExample(input_ids=[7], labels=[7])],
    ids=["all-masked", "single-token"],
)
def test_example_without_scored_token_is_nan(example):
    result = example_loss(TableEngine().prompt_logprobs(example.input_ids), example.labels)

    assert math.isnan(result.loss)
    assert result.n_tokens == 0


def test_example_loss_rejects_missing_supervised_logprob_and_length_mismatch():
    with pytest.raises(QuantScoringError, match="no logprob"):
        example_loss([None, None, -1.0], [-100, 5, 6])
    with pytest.raises(QuantScoringError, match="2 logprobs for 3 labels"):
        example_loss([None, -1.0], [-100, 5, 6])


def test_score_is_flat_mean_over_finite_examples_with_nan_kept_in_place():
    examples = [PROMPT_MASKED, ALL_MASKED, FIRST_SUPERVISED]

    score = score_examples(TableEngine(), examples)

    expected = [hf_shifted_loss(e.input_ids, e.labels) for e in (PROMPT_MASKED, FIRST_SUPERVISED)]
    assert score.eval_loss == pytest.approx(sum(expected) / 2, rel=1e-12)
    assert score.per_example_losses[0] == pytest.approx(expected[0], rel=1e-12)
    assert math.isnan(score.per_example_losses[1])
    assert score.per_example_losses[2] == pytest.approx(expected[1], rel=1e-12)
    assert score.n_examples == 3
    assert score.eval_set_fingerprint == quant_eval_set_fingerprint(examples)
    assert is_scored_fail(score) is None


class NonFiniteOnSecondEngine:
    """Fake engine that returns finite logprobs for the first prompt and `bad` for every later one."""

    def __init__(self, bad: float):
        self.bad = bad
        self.calls = 0

    def prompt_logprobs(self, token_ids: list[int]) -> list[float | None]:
        self.calls += 1
        logprob = -2.0 if self.calls == 1 else self.bad
        return [None] + [logprob] * (len(token_ids) - 1)


@pytest.mark.parametrize("bad", [float("nan"), float("-inf")], ids=["nan", "neg-inf"])
def test_engine_non_finite_on_one_example_is_a_scored_fail(bad):
    score = score_examples(NonFiniteOnSecondEngine(bad), [PROMPT_MASKED, FIRST_SUPERVISED])

    assert score.per_example_losses[0] == pytest.approx(2.0)
    assert not math.isfinite(score.eval_loss)
    assert "not finite" in is_scored_fail(score)


def test_all_masked_set_is_a_scored_fail():
    score = score_examples(TableEngine(), [ALL_MASKED])

    assert math.isnan(score.eval_loss)
    assert "not finite" in is_scored_fail(score)


def test_fingerprint_is_stable_and_order_sensitive():
    forward = quant_eval_set_fingerprint([PROMPT_MASKED, FIRST_SUPERVISED])

    assert forward == quant_eval_set_fingerprint([PROMPT_MASKED.model_copy(), FIRST_SUPERVISED.model_copy()])
    assert forward != quant_eval_set_fingerprint([FIRST_SUPERVISED, PROMPT_MASKED])
    assert len(forward) == 32


def test_assert_expected_kernels_names_offending_layers():
    kernels = {"model.layers.0.q_proj": "MacheteLinearKernel", "model.layers.1.k_proj": "ExllamaLinearKernel"}

    with pytest.raises(QuantKernelMismatch) as exc_info:
        assert_expected_kernels(kernels, frozenset({"MacheteLinearKernel"}))

    assert "model.layers.1.k_proj" in str(exc_info.value)
    assert "model.layers.0.q_proj" not in str(exc_info.value)
    assert_expected_kernels({"model.layers.0.q_proj": "MacheteLinearKernel"}, frozenset({"MacheteLinearKernel"}))


def test_assert_expected_kernels_rejects_empty_map():
    with pytest.raises(QuantKernelMismatch, match="no quantised layers"):
        assert_expected_kernels({}, frozenset({"MacheteLinearKernel"}))


def test_load_examples_reads_jsonl_and_rejects_mismatched_lengths(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps(PROMPT_MASKED.model_dump()) + "\n" + json.dumps(FIRST_SUPERVISED.model_dump()) + "\n")
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"input_ids": [1, 2, 3], "labels": [-100, 2]}) + "\n")

    assert load_examples(good) == [PROMPT_MASKED, FIRST_SUPERVISED]
    with pytest.raises(ValidationError, match="labels has 2"):
        load_examples(bad)


@pytest.mark.parametrize(
    "fields",
    [{"input_ids": [], "labels": []}, {"input_ids": [1, 2], "labels": [-100, 3]}],
    ids=["empty", "label-differs-from-token"],
)
def test_example_rejects_malformed_rows(fields):
    with pytest.raises(ValidationError):
        QuantEvalExample(**fields)


def test_quant_loss_imports_no_gpu_stack():
    probe = (
        "import sys, validator.evaluation.quant_loss; "
        "heavy = {'torch', 'vllm', 'transformers', 'validator.evaluation.common'} & set(sys.modules); "
        "print(sorted(heavy))"
    )

    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, "-c", probe], cwd=repo_root, capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "[]"
