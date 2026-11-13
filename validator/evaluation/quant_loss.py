"""Quant-task test loss from prompt logprobs, and the fixed-track kernel check.

Pure Python + numpy so it runs on CPU with a fake engine. It deliberately imports nothing from
torch, vllm, transformers or validator.evaluation.common; the vLLM adapter lives in quant_vllm.py
and is only imported inside the quant eval image.

Test-set format (written by quant prep): JSONL, one tokenised example per line:

    {"input_ids": [int, ...], "labels": [int, ...]}

`labels` has the same length as `input_ids` and uses the axolotl convention: -100 masks a prompt
token, any other value is the supervised token id and equals `input_ids` at that position.
Examples are scored in file order.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel
from pydantic import model_validator


IGNORE_INDEX = -100


class QuantEvalExample(BaseModel):
    input_ids: list[int]
    labels: list[int]

    @model_validator(mode="after")
    def validate_alignment(self) -> "QuantEvalExample":
        if not self.input_ids:
            raise ValueError("input_ids must contain at least one token id")
        if len(self.input_ids) != len(self.labels):
            raise ValueError(f"input_ids has {len(self.input_ids)} ids but labels has {len(self.labels)}")
        # The scorer reads the logprob of input_ids[t], so a supervised label must name that token.
        if any(label not in (IGNORE_INDEX, token) for token, label in zip(self.input_ids, self.labels)):
            raise ValueError("every supervised label must equal the input id at its position")
        return self


class PromptLogprobEngine(Protocol):
    def prompt_logprobs(self, token_ids: list[int]) -> list[float | None]:
        """Entry i is log p(token_i | tokens_<i); entry 0 is None (vLLM's prompt_logprobs convention)."""
        ...


class ExampleLoss(BaseModel):
    loss: float  # NaN when the example has no supervised token
    n_tokens: int


class QuantScore(BaseModel):
    # Flat mean over examples with a supervised token, matching HF eval at batch size 1. A non-finite
    # loss on any scored example propagates here, so is_scored_fail fires instead of the example
    # being silently dropped.
    eval_loss: float
    per_example_losses: list[float]  # in example order, NaN kept in place
    eval_set_fingerprint: str
    n_examples: int


class QuantScoringError(ValueError):
    """The engine's logprobs cannot be scored against the example's labels."""


def load_examples(path: Path) -> list[QuantEvalExample]:
    with open(path) as f:
        return [QuantEvalExample.model_validate(json.loads(line)) for line in f if line.strip()]


def example_loss(logprobs: list[float | None], labels: list[int]) -> ExampleLoss:
    """Mean negative logprob over supervised positions, with HF's SHIFTED token count.

    HF predicts the label at t from the logits at t-1, so position 0 is never scored and a
    supervised first token is dropped (instruct_text._compute_per_example_losses).
    """
    if len(logprobs) != len(labels):
        raise QuantScoringError(f"engine returned {len(logprobs)} logprobs for {len(labels)} labels")
    supervised = [t for t in range(1, len(labels)) if labels[t] != IGNORE_INDEX]
    if not supervised:
        return ExampleLoss(loss=float("nan"), n_tokens=0)
    if any(logprobs[t] is None for t in supervised):
        raise QuantScoringError("engine returned no logprob for a supervised position")
    return ExampleLoss(loss=-sum(logprobs[t] for t in supervised) / len(supervised), n_tokens=len(supervised))


def quant_eval_set_fingerprint(examples: list[QuantEvalExample]) -> str:
    # Same construction as instruct_text._instruct_eval_set_fingerprint + common.eval_set_fingerprint,
    # reimplemented here because common imports torch.
    digest = hashlib.sha256()
    for example in examples:
        digest.update(
            np.asarray(example.input_ids, dtype=np.int64).tobytes() + b"|" + np.asarray(example.labels, dtype=np.int64).tobytes()
        )
        digest.update(b"\x00")
    return digest.hexdigest()[:32]


def score_examples(engine: PromptLogprobEngine, examples: list[QuantEvalExample]) -> QuantScore:
    results = [example_loss(engine.prompt_logprobs(example.input_ids), example.labels) for example in examples]
    scored = [result.loss for result in results if result.n_tokens > 0]
    return QuantScore(
        eval_loss=sum(scored) / len(scored) if scored else float("nan"),
        per_example_losses=[result.loss for result in results],
        eval_set_fingerprint=quant_eval_set_fingerprint(examples),
        n_examples=len(examples),
    )


class QuantKernelMismatch(ValueError):
    pass


def assert_expected_kernels(kernel_classes: dict[str, str], expected: frozenset[str]) -> None:
    if expected and not kernel_classes:
        raise QuantKernelMismatch("no quantised layers found; a fixed-track model must load through a quant kernel")
    offenders = [(layer, kernel) for layer, kernel in kernel_classes.items() if kernel not in expected]
    if offenders:
        raise QuantKernelMismatch(f"unexpected kernels (expected one of {sorted(expected)}): {offenders}")


def is_scored_fail(score: QuantScore) -> str | None:
    if not math.isfinite(score.eval_loss):
        return f"eval_loss is not finite ({score.eval_loss})"
    return None
