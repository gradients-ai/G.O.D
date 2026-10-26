"""Held-out text rows -> the ordered list of examples an evaluator scores.

This is the validator's own data path for instruct, completion and chat evals (it replaced axolotl's prompt strategies
and `_load_tokenized_prepared_datasets` plus G.O.D's post-filter). The order of operations is the one that produced
every recorded eval loss: load the file with HF `datasets` (same pyarrow type normalisation), tokenize rows in file
order, drop rows outside [min_sequence_len, sequence_len], drop rows with no scored token, then stable-sort by length.
Ties keep file order, which the eval-set fingerprint depends on. There is no on-disk "prepared" cache; it only ever
saved re-tokenizing a file of ~1k rows.

Instruct and completion labels are bit-identical to the old path (`instruct_labels`); chat labels follow the
validator-owned rule in `chat_labels` (every assistant turn's content and the template's turn end).
"""

import os
from collections.abc import Callable
from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any

import datasets
from pydantic import BaseModel
from pydantic import Field

from core.logging import get_logger
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from validator.evaluation.chat_labels import ChatLabelSkip
from validator.evaluation.chat_labels import ChatTokenizer
from validator.evaluation.chat_labels import chat_messages
from validator.evaluation.chat_labels import chat_role_map
from validator.evaluation.chat_labels import label_chat
from validator.evaluation.chat_templates import HfChatTokenizer
from validator.evaluation.chat_templates import resolve_chat_template
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


# Default chat field names when a task leaves one None (the defaults the old chat_template strategy fell back to).
DEFAULT_CHAT_COLUMN = "messages"
DEFAULT_CHAT_ROLE_FIELD = "role"
DEFAULT_CHAT_CONTENT_FIELD = "content"

# The eval container only ever sees a downloaded local file; the S3 transport format is resolved to the file's parser
# format before the container starts (docker_evaluation._effective_remote_file_format).
_DATASETS_BUILDER_FOR_FORMAT = {FileFormat.JSON: "json", FileFormat.CSV: "csv"}


def load_eval_rows(dataset_path: str, file_format: FileFormat = FileFormat.JSON) -> datasets.Dataset:
    builder = _DATASETS_BUILDER_FOR_FORMAT.get(file_format)
    if builder is None:
        raise ValueError(f"Text eval datasets must be local json or csv files, got file_format={file_format.value!r}")
    return datasets.load_dataset(builder, data_files=[os.path.abspath(dataset_path)], split="train")


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


_CHAT_SKIP_DROP_REASON = {ChatLabelSkip.EMPTY: DropReason.UNSCORED, ChatLabelSkip.RENDER: DropReason.RENDER}


def _finish_build(
    rows: Iterable[dict[str, Any]],
    examples_for_row: Callable[[dict[str, Any]], list[TokenizedExample] | DropReason],
    config: TextEvalConfig,
    kind: str,
) -> TextEvalDatasetBuild:
    """Tokenize every row in file order, apply the drop rules, stable-sort by length and log what was dropped."""
    dropped = {reason: 0 for reason in DropReason}
    kept: list[TokenizedExample] = []
    n_rows = 0
    for row in rows:
        n_rows += 1
        candidates = examples_for_row(row)
        if isinstance(candidates, DropReason):
            dropped[candidates] += 1
            continue
        for example in candidates:
            reason = _drop_reason(example, config)
            if reason is None:
                kept.append(example)
            else:
                dropped[reason] += 1

    examples = sorted(kept, key=lambda example: len(example.input_ids))
    logger.info(
        f"Built {len(examples)} {kind} eval examples from {n_rows} rows at sequence_len={config.sequence_len}; dropped "
        + (", ".join(f"{reason.value}={count}" for reason, count in dropped.items() if count) or "none")
    )
    return TextEvalDatasetBuild(examples=examples, n_rows=n_rows, dropped=dropped)


def build_instruct_eval_dataset(
    rows: Iterable[dict[str, Any]],
    dataset_type: InstructTextDatasetType,
    tokenizer: TruncatingTokenizer,
    config: TextEvalConfig,
) -> TextEvalDatasetBuild:
    fmt = instruct_format(dataset_type)

    def examples_for_row(row: dict[str, Any]) -> list[TokenizedExample] | DropReason:
        if fmt.is_completion:
            # An empty / null text yields no chunk at all (the old completion strategy emitted no row).
            return tokenize_completion_row(tokenizer, row, fmt, config.sequence_len) or DropReason.EMPTY
        return [tokenize_instruct_row(tokenizer, row, fmt, config.sequence_len)]

    return _finish_build(rows, examples_for_row, config, kind="completion" if fmt.is_completion else "instruct")


def build_chat_eval_dataset(
    rows: Iterable[dict[str, Any]],
    dataset_type: ChatTemplateDatasetType,
    tokenizer: ChatTokenizer,
    config: TextEvalConfig,
) -> TextEvalDatasetBuild:
    """Chat rows labelled by `chat_labels.label_chat`; input ids are the rendered conversation's natural tokenisation.

    A row the template itself refuses to render (gemma with a system turn, non-alternating roles) raises, failing the
    eval exactly as it did before; only the rule's own auxiliary renders turn into a per-row RENDER drop.
    """
    role_map = chat_role_map(dataset_type.chat_user_reference, dataset_type.chat_assistant_reference)
    chat_column = dataset_type.chat_column or DEFAULT_CHAT_COLUMN
    role_field = dataset_type.chat_role_field or DEFAULT_CHAT_ROLE_FIELD
    content_field = dataset_type.chat_content_field or DEFAULT_CHAT_CONTENT_FIELD
    straddles = 0

    def examples_for_row(row: dict[str, Any]) -> list[TokenizedExample] | DropReason:
        nonlocal straddles
        messages = chat_messages(row, chat_column, role_field, content_field, role_map)
        labelled = label_chat(tokenizer, messages)
        if isinstance(labelled, ChatLabelSkip):
            return _CHAT_SKIP_DROP_REASON[labelled]
        straddles += labelled.straddles
        return [
            TokenizedExample(
                input_ids=labelled.input_ids,
                attention_mask=[1] * len(labelled.input_ids),
                labels=labelled.labels,
            )
        ]

    build = _finish_build(rows, examples_for_row, config, kind="chat")
    # Straddles are counted over every labelled row, including ones later dropped for length: they describe the
    # tokenizer/template pair, not the kept set.
    logger.info(f"Chat labelling scored {straddles} boundary-straddling tokens")
    return build.model_copy(update={"straddles": straddles})


def build_text_eval_dataset(
    rows: Iterable[dict[str, Any]],
    dataset_type: InstructTextDatasetType | ChatTemplateDatasetType,
    tokenizer: "PreTrainedTokenizerBase",
    config: TextEvalConfig,
) -> TextEvalDatasetBuild:
    """Single entry point for the text evaluators, dispatching on the task's dataset type."""
    if isinstance(dataset_type, InstructTextDatasetType):
        return build_instruct_eval_dataset(rows, dataset_type, HfTruncatingTokenizer(tokenizer), config)
    if isinstance(dataset_type, ChatTemplateDatasetType):
        chat_tokenizer = HfChatTokenizer(tokenizer, resolve_chat_template(dataset_type.chat_template, tokenizer))
        return build_chat_eval_dataset(rows, dataset_type, chat_tokenizer, config)
    raise TypeError(f"unsupported text dataset type: {type(dataset_type).__name__}")
