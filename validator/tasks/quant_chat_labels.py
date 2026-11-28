"""Which tokens of a chat row the standard text eval scores, reproduced without axolotl for the compression eval set.

The text eval tokenises chat rows with axolotl's `chat_template` strategy (`ChatTemplateStrategy` in
axolotl/prompt_strategies/chat_template.py) under the dataset entry core/training_config writes. That entry sets none
of the strategy's options, so their defaults apply: `roles_to_train=["assistant"]`, `train_on_eos="turn"` (and
`train_on_eot` follows it), `eot_tokens=[eos_token]`, and the config's `train_on_inputs` is unset (false).

The rule, for messages M (after role mapping) with input ids = tokenise(render(M)):

* every assistant turn i is located by tokenising render(M[:i+1]) twice, once as is and once with turn i's content
  replaced by `[[dummy_message]]`. The turn's tokens run from the first token where the two differ to the last one
  (comparing from the end). Those positions are scored in the full conversation's ids as they are, so a template that
  renders turn i differently when it is the last turn (Qwen3's empty think block) moves the scored positions of every
  earlier assistant turn by that difference. A turn whose two renders do not differ, or differ only in length, scores
  nothing;
* after each assistant turn, the first eos token at or after the turn's end position is scored when it lies within 3
  tokens of it. When the template ends turns with a token other than the tokenizer's eos (Qwen3 base: `<|im_end|>`
  vs eos `<|endoftext|>`) no turn end is scored;
* template decoration (role headers, generation prompts, think blocks the template inserts) is never scored.

When the row has a `system` column and its first message is not a system turn, that column becomes a leading system
turn, as axolotl's `get_conversation_thread` does.

Pure Python + pydantic: the tokenizer is injected through chat_labels.ChatTokenizer, of which only `render` and the ids
of `encode_with_offsets` are used.
"""

from pydantic import BaseModel

from validator.evaluation.chat_labels import ASSISTANT_ROLE
from validator.evaluation.chat_labels import ChatMessage
from validator.evaluation.chat_labels import ChatTokenizer
from validator.evaluation.quant_loss import IGNORE_INDEX
from validator.evaluation.quant_loss import QuantEvalExample


SYSTEM_ROLE = "system"
# axolotl's field_system: the row column read as a leading system turn.
SYSTEM_COLUMN = "system"
# The content axolotl's find_turn substitutes for the turn it locates.
DUMMY_CONTENT = "[[dummy_message]]"
# How far after a turn's end axolotl looks for its eos ("Allow for some template padding").
EOS_WINDOW = 3


class TurnBounds(BaseModel):
    """One assistant turn's content tokens: ids[start:end] of the render that ends with that turn."""

    start: int
    end: int


def conversation_with_row_system(row: dict, messages: list[ChatMessage]) -> list[ChatMessage]:
    """The row's `system` column as a leading system turn when the conversation does not start with one."""
    system = row.get(SYSTEM_COLUMN)
    if system is None or (messages and messages[0].role == SYSTEM_ROLE):
        return messages
    return [ChatMessage(role=SYSTEM_ROLE, content=system), *messages]


def label_chat_as_text_eval(
    tokenizer: ChatTokenizer, messages: list[ChatMessage], eos_token_id: int | None
) -> QuantEvalExample | None:
    """Input ids and labels the text eval produces for one conversation; None when it has no assistant turn.

    Rendering may raise (a template rejecting the row); the caller decides what that means.
    """
    rendered = [message.template_dict() for message in messages]
    assistant_turns = [index for index, message in enumerate(messages) if message.role == ASSISTANT_ROLE]
    if not assistant_turns:
        return None

    input_ids = _render_ids(tokenizer, rendered)
    labels = [IGNORE_INDEX] * len(input_ids)
    for turn_index in assistant_turns:
        bounds = _find_turn(tokenizer, rendered, turn_index)
        if bounds is None:
            turn_end = -1
        else:
            labels[bounds.start : bounds.end] = input_ids[bounds.start : bounds.end]
            turn_end = bounds.end
        eos_index = _first_eos(input_ids, turn_end, eos_token_id)
        if eos_index is not None and abs(eos_index - turn_end) <= EOS_WINDOW:
            labels[eos_index] = input_ids[eos_index]
    return QuantEvalExample(input_ids=input_ids, labels=labels)


def _render_ids(tokenizer: ChatTokenizer, messages: list[dict]) -> list[int]:
    ids, _ = tokenizer.encode_with_offsets(tokenizer.render(messages, add_generation_prompt=False))
    return list(ids)


def _find_turn(tokenizer: ChatTokenizer, rendered: list[dict], turn_index: int) -> TurnBounds | None:
    """axolotl's ChatTemplateStrategy.find_turn: the differing tokens of the conversation ending at the turn, rendered
    with its real content and with the dummy content. None where axolotl returns (-1, -1)."""
    dummy_turn = {"role": rendered[turn_index]["role"], "content": DUMMY_CONTENT}
    dummy_ids = _render_ids(tokenizer, rendered[:turn_index] + [dummy_turn])
    full_ids = _render_ids(tokenizer, rendered[: turn_index + 1])
    if not dummy_ids or not full_ids:
        return None
    min_len = min(len(dummy_ids), len(full_ids))
    start = next((i for i in range(min_len) if dummy_ids[i] != full_ids[i]), None)
    if start is None:
        return None
    end = next(
        (len(full_ids) - i for i in range(min_len) if dummy_ids[len(dummy_ids) - 1 - i] != full_ids[len(full_ids) - 1 - i]),
        None,
    )
    if end is None or end <= start:
        return None
    return TurnBounds(start=start, end=end)


def _first_eos(input_ids: list[int], start: int, eos_token_id: int | None) -> int | None:
    """axolotl's find_first_eos_token, including its Python indexing of a -1 start (a turn it could not locate)."""
    if eos_token_id is None:
        return None
    for index in range(start, len(input_ids)):
        if input_ids[index] == eos_token_id:
            return index
    return None
