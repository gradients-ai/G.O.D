"""The chat labelling rule on real tokenizers and their chat templates (offline HF cache, CPU only).

Skipped where transformers is not installed (the dev venv); run under the axgold venv with HF_HUB_OFFLINE=1.
"""

import pytest

from validator.evaluation.chat_labels import IGNORE_INDEX
from validator.evaluation.chat_labels import SENTINEL
from validator.evaluation.chat_labels import ChatLabelled
from validator.evaluation.chat_labels import ChatMessage
from validator.evaluation.chat_labels import label_chat
from validator.evaluation.chat_templates import HfChatTokenizer
from validator.evaluation.chat_templates import resolve_chat_template


transformers = pytest.importorskip("transformers")


TOKENIZER_REPOS = [
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-4B-Base",
    "unsloth/Llama-3.2-1B",
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/gemma-2-2b",
    "unsloth/gemma-2-2b-it",
    "zake7749/gemma-2-2b-it-chinese-kyara-dpo",
    "unsloth/SmolLM-135M-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "NousResearch/Hermes-2-Theta-Llama-3-8B",
    "ibm-granite/granite-4.1-3b",
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "LiquidAI/LFM2.5-2.6B",
    "openbmb/MiniCPM5-1B",
    "openbmb/MiniCPM5-2B-Base",
    "Nanbeige/Nanbeige4.1-3B",
    "kakaocorp/kanana-2-3b-instruct",
]
# Turn ends seen across the surveyed templates; every scored span must end in the template's own one.
KNOWN_TURN_END_MARKERS = ("<|im_end|>", "</s>", "<end_of_turn>", "<|end_of_text|>", "<|eot_id|>")

PLAIN = [
    ChatMessage(role="user", content="What is the capital of France?"),
    ChatMessage(role="assistant", content="Paris is the capital."),
    ChatMessage(role="user", content="And Germany?"),
    ChatMessage(role="assistant", content="Berlin."),
]
SYSTEM_AND_NEWLINES = [
    ChatMessage(role="system", content="Be terse."),
    ChatMessage(role="user", content="Capital of France?"),
    ChatMessage(role="assistant", content="\nParis.\n"),
    ChatMessage(role="user", content="Germany?"),
    ChatMessage(role="assistant", content="Berlin"),
]
CONVERSATIONS = {"plain": PLAIN, "system+newlines": SYSTEM_AND_NEWLINES}


def _load_tokenizer(repo: str):
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - not cached / incomplete snapshot
        pytest.skip(f"{repo} not usable from the local HF cache: {type(exc).__name__}")
    if not tokenizer.chat_template:
        pytest.skip(f"{repo} has no chat template")
    return tokenizer


def _render(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _turn_end(tokenizer, messages: list[dict], turn_index: int) -> str:
    """The as-last render's text after the turn's content, found independently of chat_labels' internals."""
    as_last = _render(tokenizer, messages[: turn_index + 1])
    sentinel = _render(tokenizer, messages[:turn_index] + [{**messages[turn_index], "content": SENTINEL}])
    suffix = 0
    while suffix < min(len(as_last), len(sentinel)) and as_last[-1 - suffix] == sentinel[-1 - suffix]:
        suffix += 1
    return as_last[len(as_last) - suffix :]


def _label(repo: str, conversation: str):
    tokenizer = _load_tokenizer(repo)
    messages = CONVERSATIONS[conversation]
    dicts = [m.template_dict() for m in messages]
    try:
        full = _render(tokenizer, dicts)
    except Exception as exc:  # noqa: BLE001 - e.g. gemma rejects a system turn; axolotl failed the same way
        pytest.skip(f"{repo} template rejects the {conversation} conversation: {type(exc).__name__}")
    adapter = HfChatTokenizer(tokenizer, resolve_chat_template("tokenizer_default", tokenizer))
    result = label_chat(adapter, messages)
    assert isinstance(result, ChatLabelled), f"{repo} {conversation}: {result}"
    return tokenizer, dicts, full, result


@pytest.mark.parametrize("conversation", sorted(CONVERSATIONS))
@pytest.mark.parametrize("repo", TOKENIZER_REPOS)
def test_rule_on_real_templates(repo, conversation):
    tokenizer, dicts, full, result = _label(repo, conversation)

    expected_ids = tokenizer.apply_chat_template(dicts, tokenize=True, return_dict=False)
    assert result.input_ids == list(expected_ids)

    offsets = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    assistant_turns = [i for i, m in enumerate(dicts) if m["role"] == "assistant"]
    assert [span.turn_index for span in result.spans] == assistant_turns

    previous_end = 0
    for span in result.spans:
        assert previous_end <= span.start < span.end
        previous_end = span.end
        span_text = full[offsets[span.start][0] : offsets[span.end - 1][1]]
        turn_end = _turn_end(tokenizer, dicts, span.turn_index)
        assert any(marker in turn_end for marker in KNOWN_TURN_END_MARKERS), (repo, turn_end)
        assert span_text.endswith(turn_end), (repo, span.turn_index, span_text, turn_end)
        assert not span.straddle_end

    last_turn_end = _turn_end(tokenizer, dicts, assistant_turns[-1])
    if full.endswith(last_turn_end):
        assert result.spans[-1].end == len(result.input_ids)

    # Reconstruction: unscored gaps and scored spans, read back through the offsets in order, give the whole render.
    # Offsets tile the render except that the byte tokens of one split character share that character's range.
    assert offsets[0][0] == 0 and offsets[-1][1] == len(full)
    assert all(offsets[k][1] == offsets[k + 1][0] or offsets[k] == offsets[k + 1] for k in range(len(offsets) - 1))
    boundaries = [0] + [k for span in result.spans for k in (span.start, span.end)] + [len(offsets)]
    pieces = [full[offsets[a][0] : offsets[b - 1][1]] for a, b in zip(boundaries, boundaries[1:]) if b > a]
    assert "".join(pieces) == full

    scored = {k for span in result.spans for k in range(span.start, span.end)}
    for position, label in enumerate(result.labels):
        assert label == (result.input_ids[position] if position in scored else IGNORE_INDEX)

    if conversation == "plain":
        assert result.straddles == 0


def _scored_texts(tokenizer, result: ChatLabelled) -> list[str]:
    return [_decode(tokenizer, result.input_ids[span.start : span.end]) for span in result.spans]


def test_qwen3_base_scores_each_turns_own_content_and_turn_end():
    tokenizer, _, _, result = _label("Qwen/Qwen3-0.6B-Base", "plain")
    assert _scored_texts(tokenizer, result) == [
        "Paris is the capital.<|im_end|>\n",
        "<think>\n\n</think>\n\nBerlin.<|im_end|>\n",
    ]


def test_qwen25_base_scores_im_end_on_every_turn():
    tokenizer, _, _, result = _label("Qwen/Qwen2.5-0.5B", "plain")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert tokenizer.eos_token != "<|im_end|>"
    for span in result.spans:
        assert im_end in result.labels[span.start : span.end]
    assert _scored_texts(tokenizer, result) == ["Paris is the capital.<|im_end|>\n", "Berlin.<|im_end|>\n"]


def test_bundled_chatml_on_a_real_tokenizer():
    tokenizer = _load_tokenizer("Qwen/Qwen2.5-0.5B")
    adapter = HfChatTokenizer(tokenizer, resolve_chat_template("chatml", tokenizer))
    result = label_chat(adapter, PLAIN)
    assert isinstance(result, ChatLabelled)
    assert _scored_texts(tokenizer, result) == ["Paris is the capital.<|im_end|>\n", "Berlin.<|im_end|>\n"]
