"""Held-out text rows -> the ordered list of examples an evaluator scores, without axolotl.

Replaces `axolotl.utils.data.sft._load_tokenized_prepared_datasets` plus G.O.D's own post-filter. The order of
operations is the one that produced every recorded eval loss: load the json with HF `datasets` (same pyarrow type
normalisation), tokenize rows in file order, drop rows outside [min_sequence_len, sequence_len], drop rows with no
scored token, then stable-sort by length. Ties keep file order, which the eval-set fingerprint depends on.
axolotl's on-disk "prepared" cache is gone; it only ever saved re-tokenizing a file of ~1k rows.
"""

import os
from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any

import datasets
from pydantic import BaseModel
from pydantic import Field

from core.logging import get_logger
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import InstructTextDatasetType
from validator.evaluation.instruct_labels import TruncatingTokenizer
from validator.evaluation.instruct_labels import instruct_format
from validator.evaluation.instruct_labels import tokenize_completion_row
from validator.evaluation.instruct_labels import tokenize_instruct_row
from validator.evaluation.text_eval_config import TextEvalConfig
from validator.evaluation.text_examples import TokenizedExample


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


logger = get_logger(__name__)


class HfTruncatingTokenizer:
    """Adapts a transformers tokenizer to `TruncatingTokenizer` with the exact call axolotl made.

    The kwargs matter: default add_special_tokens and the tokenizer's own truncation side are what make the ids
    identical to the axolotl path across BOS/EOS auto-adding tokenizers.
    """

    def __init__(self, tokenizer: "PreTrainedTokenizerBase"):
        self._tokenizer = tokenizer

    @property
    def bos_token_id(self) -> int | None:
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id

    def encode_truncated(self, text: str, max_length: int) -> list[int]:
        encoded = self._tokenizer(text, truncation=True, max_length=max_length, padding=False, return_tensors=None)
        return list(encoded["input_ids"])


class DropReason(str, Enum):
    EMPTY = "empty"
    SHORT = "short"
    LONG = "long"
    UNSCORED = "unscored"
    RENDER = "render"


class TextEvalDatasetBuild(BaseModel):
    """The scored examples plus why the rest were dropped.

    `n_rows` counts source rows; a completion row can yield several examples (one per sequence_len chunk), so
    `len(examples) + sum(dropped.values())` equals `n_rows` only for instruct and chat tasks.
    """

    examples: list[TokenizedExample]
    n_rows: int
    dropped: dict[DropReason, int] = Field(default_factory=lambda: {reason: 0 for reason in DropReason})
    straddles: int = 0


def load_eval_rows(dataset_path: str) -> datasets.Dataset:
    return datasets.load_dataset("json", data_files=[os.path.abspath(dataset_path)], split="train")


def _drop_reason(example: TokenizedExample, config: TextEvalConfig) -> DropReason | None:
    length = len(example.input_ids)
    if length == 0:
        return DropReason.EMPTY
    if length < config.min_sequence_len:
        return DropReason.SHORT
    if length > config.sequence_len:
        return DropReason.LONG
    if example.n_scored() == 0:
        return DropReason.UNSCORED
    return None


def build_instruct_eval_dataset(
    rows: Iterable[dict[str, Any]],
    dataset_type: InstructTextDatasetType,
    tokenizer: TruncatingTokenizer,
    config: TextEvalConfig,
) -> TextEvalDatasetBuild:
    fmt = instruct_format(dataset_type)
    dropped = {reason: 0 for reason in DropReason}
    kept: list[TokenizedExample] = []
    n_rows = 0
    for row in rows:
        n_rows += 1
        if fmt.is_completion:
            candidates = tokenize_completion_row(tokenizer, row, fmt, config.sequence_len)
            if not candidates:
                dropped[DropReason.EMPTY] += 1
        else:
            candidates = [tokenize_instruct_row(tokenizer, row, fmt, config.sequence_len)]
        for example in candidates:
            reason = _drop_reason(example, config)
            if reason is None:
                kept.append(example)
            else:
                dropped[reason] += 1

    examples = sorted(kept, key=lambda example: len(example.input_ids))
    logger.info(
        f"Built {len(examples)} eval examples from {n_rows} rows at sequence_len={config.sequence_len}; dropped "
        + ", ".join(f"{reason.value}={count}" for reason, count in dropped.items() if count)
    )
    return TextEvalDatasetBuild(examples=examples, n_rows=n_rows, dropped=dropped)


def build_text_eval_dataset(
    rows: Iterable[dict[str, Any]],
    dataset_type: InstructTextDatasetType | ChatTemplateDatasetType,
    tokenizer: TruncatingTokenizer,
    config: TextEvalConfig,
) -> TextEvalDatasetBuild:
    """Single entry point for the text evaluators, dispatching on the task's dataset type."""
    if isinstance(dataset_type, InstructTextDatasetType):
        return build_instruct_eval_dataset(rows, dataset_type, tokenizer, config)
    if isinstance(dataset_type, ChatTemplateDatasetType):
        # Chat rows are labelled by validator.evaluation.chat_labels, wired in with the evaluator switch-over (F5).
        raise NotImplementedError("chat eval datasets are not built here yet")
    raise TypeError(f"unsupported text dataset type: {type(dataset_type).__name__}")
