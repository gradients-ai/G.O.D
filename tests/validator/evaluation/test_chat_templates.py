import hashlib

import pytest

from validator.evaluation.chat_templates import BUNDLED_TEMPLATES_DIR
from validator.evaluation.chat_templates import HfChatTokenizer
from validator.evaluation.chat_templates import bundled_chat_template
from validator.evaluation.chat_templates import resolve_chat_template


# sha256 of axolotl 0bda5a13 src/axolotl/utils/chat_templates/templates/chatml.jinja - the template API-created chat
# tasks have been evaluated with; the bundled copy must stay byte-identical.
AXOLOTL_CHATML_SHA256 = "e7a3df47d72feec173403ca6c63b3632edde3536fc0f5a4413ab432a2cfddb8d"
TOKENIZER_TEMPLATE = "{{ tokenizer's own template }}"


class FakeTokenizer:
    def __init__(self, chat_template: str | dict | None = TOKENIZER_TEMPLATE, is_fast: bool = True):
        self.chat_template = chat_template
        self.is_fast = is_fast
        self.calls: list[tuple] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered"

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": [5, 6], "offset_mapping": [[0, 3], [3, 8]]}


def test_bundled_chatml_is_byte_identical_to_axolotls():
    raw = (BUNDLED_TEMPLATES_DIR / "chatml.jinja").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == AXOLOTL_CHATML_SHA256


def test_bundled_chatml_renders_im_start_role_newline_content_im_end():
    template = bundled_chat_template("chatml")
    assert "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}" in template
    assert "{{ '<|im_start|>assistant\n' }}" in template


@pytest.mark.parametrize("name", ["tokenizer_default", None, ""])
def test_tokenizer_default_and_unset_use_the_tokenizers_template(name):
    assert resolve_chat_template(name, FakeTokenizer()) == TOKENIZER_TEMPLATE


def test_tokenizer_default_without_a_tokenizer_template_raises():
    with pytest.raises(ValueError, match="no chat_template"):
        resolve_chat_template("tokenizer_default", FakeTokenizer(chat_template=None))


def test_named_tokenizer_templates_use_the_default_one():
    tokenizer = FakeTokenizer(chat_template={"default": TOKENIZER_TEMPLATE, "tool_use": "{{ tools }}"})
    assert resolve_chat_template("tokenizer_default", tokenizer) == TOKENIZER_TEMPLATE
    with pytest.raises(ValueError, match="no 'default'"):
        resolve_chat_template("tokenizer_default", FakeTokenizer(chat_template={"tool_use": "{{ tools }}"}))


def test_chatml_ignores_the_tokenizers_template():
    assert resolve_chat_template("chatml", FakeTokenizer()) == bundled_chat_template("chatml")


def test_fallback_prefers_the_tokenizer_then_the_bundled_template():
    assert resolve_chat_template("tokenizer_default_fallback_chatml", FakeTokenizer()) == TOKENIZER_TEMPLATE
    assert resolve_chat_template("tokenizer_default_fallback_chatml", FakeTokenizer(chat_template=None)) == (
        bundled_chat_template("chatml")
    )


@pytest.mark.parametrize("name", ["llama3", "jinja", "tokenizer_default_fallback_llama3"])
def test_unknown_names_raise(name):
    with pytest.raises(ValueError, match="not bundled"):
        resolve_chat_template(name, FakeTokenizer(chat_template=None))


def test_hf_chat_tokenizer_renders_with_the_resolved_template_and_no_tokenisation():
    tokenizer = FakeTokenizer()
    adapter = HfChatTokenizer(tokenizer, "TEMPLATE")
    messages = [{"role": "user", "content": "hi"}]

    assert adapter.render(messages, add_generation_prompt=True) == "rendered"
    assert tokenizer.calls == [
        (messages, {"tokenize": False, "add_generation_prompt": True, "chat_template": "TEMPLATE"}),
    ]


def test_hf_chat_tokenizer_encodes_without_special_tokens_and_returns_offsets():
    tokenizer = FakeTokenizer()
    ids, offsets = HfChatTokenizer(tokenizer, "TEMPLATE").encode_with_offsets("abcdefgh")

    assert ids == [5, 6]
    assert offsets == [(0, 3), (3, 8)]
    assert tokenizer.calls == [("abcdefgh", {"add_special_tokens": False, "return_offsets_mapping": True})]


def test_hf_chat_tokenizer_rejects_slow_tokenizers():
    with pytest.raises(TypeError, match="fast tokenizer"):
        HfChatTokenizer(FakeTokenizer(is_fast=False), "TEMPLATE")
