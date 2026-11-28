"""The text eval's chat labelling rule (axolotl chat_template defaults) with fake tokenizers: no transformers needed."""

from validator.evaluation.chat_labels import ChatMessage
from validator.evaluation.quant_loss import IGNORE_INDEX
from validator.evaluation.quant_loss import QuantEvalExample
from validator.tasks.quant_chat_labels import DUMMY_CONTENT
from validator.tasks.quant_chat_labels import conversation_with_row_system
from validator.tasks.quant_chat_labels import label_chat_as_text_eval


EOS = "$"
EOS_ID = ord(EOS)


class CharChatTokenizer:
    """One id per character (ord). Each turn renders as `<role>content` + `turn_end`; `last_assistant_prefix` goes before
    the content of the conversation's last message when it is an assistant turn (as Qwen3's empty think block does)."""

    def __init__(self, turn_end: str = "|", last_assistant_prefix: str = ""):
        self.turn_end = turn_end
        self.last_assistant_prefix = last_assistant_prefix

    def render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        parts = []
        for index, message in enumerate(messages):
            last_assistant = index == len(messages) - 1 and message["role"] == "assistant"
            prefix = self.last_assistant_prefix if last_assistant else ""
            parts.append(f"<{message['role']}>{prefix}{message.get('content', '')}{self.turn_end}")
        return "".join(parts) + ("<assistant>" if add_generation_prompt else "")

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        return [ord(c) for c in text], [(i, i + 1) for i in range(len(text))]


def _messages(*turns: tuple[str, str]) -> list[ChatMessage]:
    return [ChatMessage(role=role, content=content) for role, content in turns]


def _scored(example: QuantEvalExample) -> str:
    return "".join(chr(label) if label != IGNORE_INDEX else "_" for label in example.labels)


def test_only_assistant_content_is_scored_when_the_template_never_emits_eos():
    example = label_chat_as_text_eval(
        CharChatTokenizer(), _messages(("system", "s"), ("user", "hi"), ("assistant", "yo")), EOS_ID
    )

    assert example.input_ids == [ord(c) for c in "<system>s|<user>hi|<assistant>yo|"]
    assert _scored(example) == "_" * len("<system>s|<user>hi|<assistant>") + "yo_"


def test_an_eos_within_three_tokens_of_the_turn_end_is_scored():
    near = label_chat_as_text_eval(CharChatTokenizer(turn_end="\n$"), _messages(("user", "q"), ("assistant", "a")), EOS_ID)
    far = label_chat_as_text_eval(CharChatTokenizer(turn_end="\n\n\n\n$"), _messages(("user", "q"), ("assistant", "a")), EOS_ID)

    assert _scored(near) == "_" * len("<user>q\n$<assistant>") + "a_$"
    assert _scored(far) == "_" * len("<user>q\n\n\n\n$<assistant>") + "a_____"


def test_every_assistant_turn_is_scored_with_its_eos():
    example = label_chat_as_text_eval(
        CharChatTokenizer(turn_end="$"), _messages(("user", "a"), ("assistant", "b"), ("user", "c"), ("assistant", "d")), EOS_ID
    )

    assert _scored(example) == "_" * len("<user>a$<assistant>") + "b$" + "_" * len("<user>c$<assistant>") + "d$"


def test_decoration_only_on_the_last_turn_moves_earlier_turns_scored_positions():
    """Each turn is located in the render that ends with it; those positions are scored in the full conversation."""
    tokenizer = CharChatTokenizer(last_assistant_prefix="<T>")
    example = label_chat_as_text_eval(
        tokenizer, _messages(("user", "a"), ("assistant", "bb"), ("user", "c"), ("assistant", "d")), EOS_ID
    )

    full = "<user>a|<assistant>bb|<user>c|<assistant><T>d|"
    assert example.input_ids == [ord(c) for c in full]
    first_turn_start = len("<user>a|<assistant>") + len("<T>")
    expected = ["_"] * len(full)
    for position in range(first_turn_start, first_turn_start + 2):
        expected[position] = full[position]
    expected[full.index("d")] = "d"
    assert _scored(example) == "".join(expected)


def test_a_turn_whose_content_is_the_dummy_scores_nothing():
    example = label_chat_as_text_eval(CharChatTokenizer(), _messages(("user", "q"), ("assistant", DUMMY_CONTENT)), EOS_ID)

    assert set(example.labels) == {IGNORE_INDEX}


def test_an_empty_answer_scores_nothing():
    example = label_chat_as_text_eval(CharChatTokenizer(), _messages(("user", "q"), ("assistant", "")), EOS_ID)

    assert set(example.labels) == {IGNORE_INDEX}


def test_a_conversation_without_an_assistant_turn_has_no_example():
    assert label_chat_as_text_eval(CharChatTokenizer(), _messages(("user", "q"), ("user", "r")), EOS_ID) is None
    assert label_chat_as_text_eval(CharChatTokenizer(), [], EOS_ID) is None


def test_without_an_eos_id_only_content_is_scored():
    example = label_chat_as_text_eval(CharChatTokenizer(turn_end="$"), _messages(("user", "q"), ("assistant", "a")), None)

    assert _scored(example) == "_" * len("<user>q$<assistant>") + "a_"


def test_the_row_system_column_leads_the_conversation_unless_it_starts_with_a_system_turn():
    messages = _messages(("user", "q"), ("assistant", "a"))

    assert conversation_with_row_system({"system": "be brief"}, messages)[0] == ChatMessage(role="system", content="be brief")
    assert conversation_with_row_system({"system": None}, messages) == messages
    assert conversation_with_row_system({}, messages) == messages
    with_system = _messages(("system", "own"), ("user", "q"), ("assistant", "a"))
    assert conversation_with_row_system({"system": "column"}, with_system) == with_system
