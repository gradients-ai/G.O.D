"""Instruct and completion rows -> tokens and labels, bit-identical to axolotl 0bda5a13.

The validator used to delegate this to axolotl's `user_defined` (alpaca_w_system) and `completion` prompt
strategies. These functions replicate them line for line rather than "improving" them, because every eval loss
ever recorded for instruct tasks was computed on exactly these tokens; a change here would silently move scores.
Where axolotl's behaviour looks odd it is kept and the oddity is named:

- a `field_system` column that exists but is null yields no system text, even when the task has a system_prompt
  (axolotl tests column membership, not the value);
- a null instruction is formatted as the string "None";
- the prompt and the response are tokenized separately, each truncated to sequence_len, and the response's
  leading BOS is stripped while the prompt's is kept;
- EOS is appended only when the last token is not already EOS and the text did not hit the truncation length.

The tokenizer is injected through `TruncatingTokenizer` so this module never imports transformers; the
adapter over a real tokenizer lives in `text_eval_dataset`.
"""

from typing import Any
from typing import Protocol

from pydantic import BaseModel

from core.models.dataset_models import InstructTextDatasetType
from validator.evaluation.text_examples import IGNORE_INDEX
from validator.evaluation.text_examples import TokenizedExample


# axolotl's completion strategy truncates the whole text at this multiple of sequence_len before chunking.
COMPLETION_MAX_LENGTH_MULTIPLIER = 64

# UserDefinedDatasetConfig dataclass defaults; they apply to every field G.O.D leaves as None because
# core.training_config strips Nones before handing the dict to axolotl.
_AXOLOTL_FIELD_SYSTEM = "system"
_AXOLOTL_FIELD_INSTRUCTION = "instruction"
_AXOLOTL_FIELD_INPUT = "input"
_AXOLOTL_SYSTEM_FORMAT = "{system}"


class InstructFormat(BaseModel):
    """The fully resolved prompt format of one instruct task; `field_output is None` means completion."""

    field_instruction: str
    field_input: str
    field_output: str | None
    field_system: str
    system_prompt: str
    system_format: str
    format: str
    no_input_format: str

    @property
    def is_completion(self) -> bool:
        return self.field_output is None


class TruncatingTokenizer(Protocol):
    bos_token_id: int | None
    eos_token_id: int | None

    def encode_truncated(self, text: str, max_length: int) -> list[int]:
        """`tokenizer(text, truncation=True, max_length=max_length, padding=False, return_tensors=None)["input_ids"]`."""
        ...


def _given(value: str | None, default: str) -> str:
    return default if value is None else value


def instruct_format(dataset_type: InstructTextDatasetType) -> InstructFormat:
    """Resolve the task's format the way core.training_config + axolotl's dataclass defaults did.

    Only None falls back to a default (an explicit "" is kept), mirroring `{k: v ... if v is not None}` followed
    by `setdefault`. A falsy `field_output` selects the completion strategy, which reads `field_instruction`.
    """
    if not dataset_type.field_output:
        if dataset_type.field_instruction is None:
            raise ValueError("completion-style instruct task (no field_output) needs field_instruction")
        field_output = None
        field_instruction = dataset_type.field_instruction
    else:
        field_output = dataset_type.field_output
        field_instruction = _given(dataset_type.field_instruction, _AXOLOTL_FIELD_INSTRUCTION)

    default_format = "{instruction} {input}" if dataset_type.field_input else "{instruction}"
    return InstructFormat(
        field_instruction=field_instruction,
        field_input=_given(dataset_type.field_input, _AXOLOTL_FIELD_INPUT),
        field_output=field_output,
        field_system=_given(dataset_type.field_system, _AXOLOTL_FIELD_SYSTEM),
        system_prompt=dataset_type.system_prompt or "",
        system_format=_given(dataset_type.system_format, _AXOLOTL_SYSTEM_FORMAT),
        format=_given(dataset_type.format, default_format),
        no_input_format=_given(dataset_type.no_input_format, "{instruction}"),
    )


def build_instruct_prompt(row: dict[str, Any], fmt: InstructFormat) -> tuple[str, str]:
    """(user_prompt, response) for one row, per alpaca_w_system.SystemDataPrompter.build_prompt_w_system."""
    if fmt.field_output is None:
        raise ValueError("build_instruct_prompt called with a completion format")
    instruction = row[fmt.field_instruction]
    input_ = row[fmt.field_input] if fmt.field_input in row else ""
    response = row[fmt.field_output] if fmt.field_output in row else ""
    system = row[fmt.field_system] if fmt.field_system in row else fmt.system_prompt

    system_text = fmt.system_format.format(system=system) if system and fmt.system_format else ""
    if input_:
        user_prompt = system_text + fmt.format.format(instruction=instruction, input=input_)
    else:
        user_prompt = system_text + fmt.no_input_format.format(instruction=instruction)
    return user_prompt, response if response else ""


def _tokenize(tokenizer: TruncatingTokenizer, text: str, max_length: int, add_eos: bool, strip_bos: bool) -> list[int]:
    """axolotl PromptTokenizingStrategy._tokenize, input_ids only (its attention mask is all ones)."""
    if not text:
        return []
    ids = list(tokenizer.encode_truncated(text, max_length))
    if not ids:
        return []
    if ids[-1] != tokenizer.eos_token_id and len(ids) < max_length and add_eos:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no eos_token_id; cannot terminate the scored sequence")
        ids.append(tokenizer.eos_token_id)
    if ids[0] == tokenizer.bos_token_id and strip_bos:
        ids = ids[1:]
    return ids


def tokenize_instruct_row(
    tokenizer: TruncatingTokenizer, row: dict[str, Any], fmt: InstructFormat, sequence_len: int
) -> TokenizedExample:
    """Prompt masked, response (+EOS) scored. May return an empty example; the builder drops it."""
    user_prompt, response = build_instruct_prompt(row, fmt)
    prompt_ids = _tokenize(tokenizer, user_prompt, sequence_len, add_eos=False, strip_bos=False)
    response_ids = _tokenize(tokenizer, response, sequence_len, add_eos=True, strip_bos=True)
    input_ids = prompt_ids + response_ids
    return TokenizedExample(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=[IGNORE_INDEX] * len(prompt_ids) + response_ids,
    )


def tokenize_completion_row(
    tokenizer: TruncatingTokenizer, row: dict[str, Any], fmt: InstructFormat, sequence_len: int
) -> list[TokenizedExample]:
    """Whole text scored, cut into consecutive sequence_len chunks; an empty text yields no chunk at all."""
    max_length = sequence_len * COMPLETION_MAX_LENGTH_MULTIPLIER
    ids = _tokenize(tokenizer, row[fmt.field_instruction], max_length, add_eos=True, strip_bos=False)
    return [
        TokenizedExample(
            input_ids=ids[start : start + sequence_len],
            attention_mask=[1] * len(ids[start : start + sequence_len]),
            labels=ids[start : start + sequence_len],
        )
        for start in range(0, len(ids), sequence_len)
    ]
