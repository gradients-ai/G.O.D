"""Tests for the tokenizer-backed memory counter and its loader.

Uses a char-level stub tokenizer (DI) so the counter logic is exercised without
loading a real model. load_token_counter is tested via monkeypatch — no network.
"""

import core.pvp.tokenizer_counter as tc
from core.pvp.memory import WhitespaceTokenCounter
from core.pvp.tokenizer_counter import HFTokenCounter
from core.pvp.tokenizer_counter import load_token_counter


class StubTokenizer:
    """Char-level tokenizer: one token per character, deterministic."""

    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(ids)


class TestHFTokenCounter:
    def test_count_is_token_count(self):
        assert HFTokenCounter(StubTokenizer()).count("abcd") == 4

    def test_truncate_head_keeps_front(self):
        assert HFTokenCounter(StubTokenizer()).truncate("abcde", 2, "head") == "ab"

    def test_truncate_tail_keeps_end(self):
        assert HFTokenCounter(StubTokenizer()).truncate("abcde", 2, "tail") == "de"

    def test_truncate_noop_under_budget(self):
        assert HFTokenCounter(StubTokenizer()).truncate("ab", 5, "head") == "ab"


class TestLoadTokenCounter:
    def test_uses_hf_when_loadable(self, monkeypatch):
        monkeypatch.setattr(tc.AutoTokenizer, "from_pretrained", lambda repo: StubTokenizer())
        counter = load_token_counter("org/model")
        assert isinstance(counter, HFTokenCounter)
        assert counter.count("abc") == 3

    def test_falls_back_on_load_error(self, monkeypatch):
        def boom(repo):
            raise OSError("no such model")

        monkeypatch.setattr(tc.AutoTokenizer, "from_pretrained", boom)
        assert isinstance(load_token_counter("org/missing"), WhitespaceTokenCounter)

    def test_non_repo_id_skips_load_entirely(self, monkeypatch):
        def must_not_call(repo):
            raise AssertionError("from_pretrained should not be called for a non-repo id")

        monkeypatch.setattr(tc.AutoTokenizer, "from_pretrained", must_not_call)
        # No "/" and not a path -> straight to whitespace (e.g. "test", "claude-haiku-4-5")
        assert isinstance(load_token_counter("claude-haiku-4-5"), WhitespaceTokenCounter)
        assert isinstance(load_token_counter("test"), WhitespaceTokenCounter)
