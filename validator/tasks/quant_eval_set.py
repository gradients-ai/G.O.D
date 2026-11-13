"""Tokenise a quant task's held-out split into the scorer's test set (see validator/evaluation/quant_loss.py).

Pure Python + pydantic: the tokenizer is injected through QuantTokenizer, so this module never imports
transformers and runs on CPU with a fake tokenizer. The real one is validator/tasks/quant_tokenizer.py.

Chat rows are labelled by validator/evaluation/chat_labels.label_chat, the same rule the standard text
eval uses: every assistant turn is scored, through the template's own turn end.

Token ids are fixed here, from the validator's own download of the task's base tokenizer, so the
reference and every entrant are scored on byte-identical examples.
"""

from collections import Counter
from enum import Enum
from typing import Protocol

from pydantic import BaseModel

import validator.tasks.datasets.constants as cst
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import InstructTextDatasetType
from validator.evaluation.chat_labels import ChatLabelSkip
from validator.evaluation.chat_labels import ChatTokenizer
from validator.evaluation.chat_labels import chat_messages
from validator.evaluation.chat_labels import chat_role_map
from validator.evaluation.chat_labels import label_chat
from validator.evaluation.quant_loss import IGNORE_INDEX
from validator.evaluation.quant_loss import QuantEvalExample


# Same defaults as the standard text eval when a task leaves a chat field None (validator/evaluation/text_eval_dataset);
# the column default is the one dataset preparation samples into the test split.
DEFAULT_CHAT_ROLE_FIELD = "role"
DEFAULT_CHAT_CONTENT_FIELD = "content"


class QuantTokenizer(Protocol):
    bos_token_id: int | None
    eos_token_id: int | None
    vocab_size: int
    context_len: int | None  # the model's max_position_embeddings; None when its config does not say

    def encode(self, text: str, add_special_tokens: bool) -> list[int]: ...

    def chat_labeller(self, chat_template: str | None) -> ChatTokenizer:
        """The chat_labels tokenizer for the template a task's `chat_template` name selects."""
        ...


class QuantEvalSetBuild(BaseModel):
    examples: list[QuantEvalExample]
    n_rows: int
    n_skipped_empty: int  # see SkipReason
    n_skipped_long: int
    n_skipped_render: int


class InstructFields(BaseModel):
    """The column names axolotl's user_defined strategy reads: an unset field falls back to UserDefinedDatasetConfig's default
    name (axolotl/prompt_strategies/user_defined.py), because core/training_config drops None fields before axolotl sees them."""

    instruction: str
    input: str
    output: str
    system: str


def _instruct_fields(dataset_type: InstructTextDatasetType) -> InstructFields:
    return InstructFields(
        instruction=dataset_type.field_instruction or "instruction",
        input=dataset_type.field_input or "input",
        output=dataset_type.field_output or "output",
        system=dataset_type.field_system or "system",
    )


def _text(value: object) -> str:
    return "" if value is None else str(value)


def instruct_prompt_and_completion(row: dict, dataset_type: InstructTextDatasetType) -> tuple[str, str] | None:
    """The prompt and completion strings axolotl builds for this row on the training path.

    Mirrors SystemDataPrompter.build_prompt_w_system (axolotl/prompt_strategies/alpaca_w_system.py) fed by
    user_defined.load (axolotl/prompt_strategies/user_defined.py) with the dataset entry core/training_config
    writes: the system text is `system_format` of the row's system column (or the static `system_prompt`) glued
    straight onto the turn, and the turn is `format` when the row has a non-empty input, else `no_input_format`.
    Nothing is added at the prompt/completion boundary. None when there is no completion to supervise.
    """
    fields = _instruct_fields(dataset_type)
    completion = _text(row.get(fields.output))
    if not completion:
        return None
    instruction = _text(row.get(fields.instruction))
    input_text = _text(row.get(fields.input))
    if input_text:
        turn_format = dataset_type.format or ("{instruction} {input}" if dataset_type.field_input else "{instruction}")
    else:
        turn_format = dataset_type.no_input_format or "{instruction}"
    system = _text(row[fields.system]) if fields.system in row else _text(dataset_type.system_prompt)
    system_prefix = dataset_type.system_format.format(system=system) if system and dataset_type.system_format else ""
    return system_prefix + turn_format.format(instruction=instruction, input=input_text), completion


def _example(prompt_ids: list[int], completion_ids: list[int]) -> QuantEvalExample:
    return QuantEvalExample(
        input_ids=prompt_ids + completion_ids,
        labels=[IGNORE_INDEX] * len(prompt_ids) + completion_ids,
    )


def tokenize_instruct(tokenizer: QuantTokenizer, prompt: str, completion: str) -> QuantEvalExample:
    """The ids and labels axolotl's InstructionPromptTokenizingStrategy.tokenize_prompt produces with train_on_inputs false
    (axolotl/prompt_tokenizers.py, `_tokenize`).

    Prompt and completion are tokenised separately, each with the tokenizer's default special tokens: the prompt keeps
    its BOS, the completion loses a leading BOS and gains an EOS unless it already ends in one. Because they are
    tokenised apart no token ever spans the boundary, exactly as in training, and the prompt is masked.
    Truncation to sequence_len is not replicated: an over-long row is dropped (SkipReason.LONG), as axolotl drops it.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True) if prompt else []
    completion_ids = tokenizer.encode(completion, add_special_tokens=True)
    if completion_ids and completion_ids[0] == tokenizer.bos_token_id:
        completion_ids = completion_ids[1:]
    if tokenizer.eos_token_id is not None and (not completion_ids or completion_ids[-1] != tokenizer.eos_token_id):
        completion_ids.append(tokenizer.eos_token_id)
    return _example(prompt_ids, completion_ids)


class SkipReason(str, Enum):
    EMPTY = "empty"  # nothing to supervise: empty output, no assistant turn, a malformed chat row, or no scored label
    LONG = "long"  # longer than max_sequence_len
    RENDER = "render"  # the chat template rejects the row, or label_chat cannot locate its turns


_CHAT_SKIP_REASON = {ChatLabelSkip.EMPTY: SkipReason.EMPTY, ChatLabelSkip.RENDER: SkipReason.RENDER}


def tokenize_chat(
    chat_tokenizer: ChatTokenizer, row: dict, dataset_type: ChatTemplateDatasetType
) -> QuantEvalExample | SkipReason:
    """One chat row labelled by chat_labels.label_chat.

    Unlike the standard eval, which fails the whole eval, a row that is malformed or that the template
    refuses to render is skipped: prep would otherwise fail the task on one bad held-out row.
    """
    role_map = chat_role_map(dataset_type.chat_user_reference, dataset_type.chat_assistant_reference)
    try:
        messages = chat_messages(
            row,
            dataset_type.chat_column or cst.STANDARD_CHAT_MESSAGES_COLUMN,
            dataset_type.chat_role_field or DEFAULT_CHAT_ROLE_FIELD,
            dataset_type.chat_content_field or DEFAULT_CHAT_CONTENT_FIELD,
            role_map,
        )
    except ValueError:
        return SkipReason.EMPTY
    try:
        labelled = label_chat(chat_tokenizer, messages)
    except Exception:  # noqa: BLE001 - the template rejecting the row (system turn on gemma, None content, ...)
        return SkipReason.RENDER
    if isinstance(labelled, ChatLabelSkip):
        return _CHAT_SKIP_REASON[labelled]
    return QuantEvalExample(input_ids=labelled.input_ids, labels=labelled.labels)


def _n_scored_labels(example: QuantEvalExample) -> int:
    # The scorer never scores position 0 (quant_loss.example_loss), so a label there does not count.
    return sum(label != IGNORE_INDEX for label in example.labels[1:])


def _check_vocab(example: QuantEvalExample, vocab_size: int) -> None:
    # A broken tokenizer must fail prep, not score garbage.
    out_of_range = [token for token in example.input_ids if not 0 <= token < vocab_size]
    if out_of_range:
        raise ValueError(f"token ids {out_of_range[:5]} are outside the tokenizer vocabulary ({vocab_size})")


def _screen(example: QuantEvalExample, tokenizer: QuantTokenizer, max_sequence_len: int) -> QuantEvalExample | SkipReason:
    _check_vocab(example, tokenizer.vocab_size)
    if _n_scored_labels(example) == 0:
        return SkipReason.EMPTY
    if len(example.input_ids) > max_sequence_len:
        return SkipReason.LONG
    return example


def quant_eval_max_sequence_len(context_len: int | None, cap: int) -> int:
    """The longest example the eval set may keep: the scorer's engine holds it plus one generated token."""
    return cap if context_len is None else min(cap, context_len - 1)


def build_quant_eval_set(
    rows: list[dict],
    dataset_type: InstructTextDatasetType | ChatTemplateDatasetType,
    tokenizer: QuantTokenizer,
    max_sequence_len: int,
) -> QuantEvalSetBuild:
    """Deterministic in row order."""
    if isinstance(dataset_type, InstructTextDatasetType):

        def tokenize_row(row: dict) -> QuantEvalExample | SkipReason:
            pair = instruct_prompt_and_completion(row, dataset_type)
            return tokenize_instruct(tokenizer, *pair) if pair else SkipReason.EMPTY

    else:
        # Resolved once: an unknown template name is a task misconfiguration and fails prep.
        chat_tokenizer = tokenizer.chat_labeller(dataset_type.chat_template)

        def tokenize_row(row: dict) -> QuantEvalExample | SkipReason:
            return tokenize_chat(chat_tokenizer, row, dataset_type)

    examples = []
    skipped = Counter()
    for row in rows:
        outcome = tokenize_row(row)
        if isinstance(outcome, QuantEvalExample):
            outcome = _screen(outcome, tokenizer, max_sequence_len)
        if isinstance(outcome, SkipReason):
            skipped[outcome] += 1
        else:
            examples.append(outcome)
    return QuantEvalSetBuild(
        examples=examples,
        n_rows=len(rows),
        n_skipped_empty=skipped[SkipReason.EMPTY],
        n_skipped_long=skipped[SkipReason.LONG],
        n_skipped_render=skipped[SkipReason.RENDER],
    )


def quant_eval_set_jsonl(examples: list[QuantEvalExample]) -> str:
    """The JSONL text quant_loss.load_examples reads back unchanged."""
    return "".join(example.model_dump_json() + "\n" for example in examples)
