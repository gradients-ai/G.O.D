"""Real-tokenizer round-trip tests for core.tokenizer_utils.

The pure-dict tests in test_tokenizer_utils.py prove the config transformations; these prove the
SEMANTIC guarantees against a real transformers loader — the ones the continuous-SFT / quasar
lineage depends on:

  1. sanitize_tokenizer_config makes a v5-serialized tokenizer loadable by a v4 consumer WITHOUT
     changing its vocab, special tokens, or chat template (a v4 loader crashes without it).
  2. the merge carry preserves the seed/base chat template and custom eos through save + sanitize.

Built from a tiny in-memory fast tokenizer so they need no network. Skipped where tokenizers/
transformers aren't installed (e.g. a validator-only box).
"""

import json
import os

import pytest


pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from tokenizers import Tokenizer  # noqa: E402
from tokenizers import models  # noqa: E402
from tokenizers import pre_tokenizers  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from core.constants.paths import CHAT_TEMPLATE_FILE  # noqa: E402
from core.tokenizer_utils import TOKENIZER_CONFIG_FILE  # noqa: E402
from core.tokenizer_utils import ensure_chat_template  # noqa: E402
from core.tokenizer_utils import read_chat_template  # noqa: E402
from core.tokenizer_utils import sanitize_tokenizer_config  # noqa: E402


# A quasar-style HUMAN/ASSISTANT-ish template with custom special tokens (not chatml default).
QUASAR_TEMPLATE = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"


def _tiny_tokenizer(chat_template=None):
    vocab = {"<unk>": 0, "<pad>": 1, "<|endoftext|>": 2, "<|im_start|>": 3, "<|im_end|>": 4, "hi": 5, "there": 6}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    ptf = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", eos_token="<|endoftext|>")
    if chat_template:
        ptf.chat_template = chat_template
    return ptf


def _inject_v5_quirks(dir_path):
    """Rewrite a saved tokenizer dir to look like a transformers-v5 save a v4 consumer chokes on."""
    cfg_path = os.path.join(dir_path, TOKENIZER_CONFIG_FILE)
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["tokenizer_class"] = "TokenizersBackend"
    cfg["extra_special_tokens"] = ["<|im_start|>", "<|im_end|>"]
    cfg.pop("chat_template", None)  # v5 pops the inline key, leaving only chat_template.jinja
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)


def test_sanitize_makes_v5_tokenizer_loadable_on_v4_without_changing_it(tmp_path):
    tok = _tiny_tokenizer(chat_template=QUASAR_TEMPLATE)
    d = str(tmp_path / "merged")
    tok.save_pretrained(d)
    orig_vocab = tok.get_vocab()
    _inject_v5_quirks(d)

    # negative control: a v4 loader crashes on the v5 serialization (this is the failure sanitize fixes)
    with pytest.raises(Exception):
        AutoTokenizer.from_pretrained(d)

    sanitize_tokenizer_config(d)

    reloaded = AutoTokenizer.from_pretrained(d)  # now loads
    # vocab + special tokens are untouched — sanitize only rewrites tokenizer_config.json, never tokenizer.json
    assert reloaded.get_vocab() == orig_vocab
    assert reloaded.eos_token == "<|endoftext|>"
    assert "<|im_start|>" in reloaded.get_vocab() and "<|im_end|>" in reloaded.get_vocab()
    # chat template preserved and renders identically
    assert reloaded.chat_template == QUASAR_TEMPLATE
    rendered = reloaded.apply_chat_template([{"role": "user", "content": "hi there"}], tokenize=False)
    assert "<|im_start|>user" in rendered and "<|im_end|>" in rendered
    # config is now v4-shaped
    with open(os.path.join(d, TOKENIZER_CONFIG_FILE)) as f:
        cfg = json.load(f)
    assert cfg["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert isinstance(cfg["extra_special_tokens"], dict)
    assert cfg["chat_template"] == QUASAR_TEMPLATE  # folded inline for pre-4.47 consumers


def test_continuous_sft_seed_template_and_eos_survive_merge_save(tmp_path):
    # Quasar lineage: the base/seed tokenizer owns the template + custom eos; the merge selects it as
    # target. The template and eos must survive save + sanitize so the carried base stays consistent.
    base_tok = _tiny_tokenizer(chat_template=QUASAR_TEMPLATE)
    adapter_dir = str(tmp_path / "adapter")
    os.makedirs(adapter_dir)  # bare adapter, no chat template of its own
    assert read_chat_template(adapter_dir) is None

    target = base_tok  # merge fell back to base
    ensure_chat_template(target, read_chat_template(adapter_dir), base_tok.chat_template)
    out = str(tmp_path / "merged")
    target.save_pretrained(out)
    sanitize_tokenizer_config(out)

    reloaded = AutoTokenizer.from_pretrained(out)
    assert reloaded.chat_template == QUASAR_TEMPLATE
    assert reloaded.eos_token == "<|endoftext|>"


def test_adapter_jinja_only_template_grafted_onto_templateless_target(tmp_path):
    # The primary feature: adapter ships its template only as chat_template.jinja; a base-derived
    # target that has no template of its own must end up serving the adapter's template.
    adapter_dir = str(tmp_path / "adapter")
    os.makedirs(adapter_dir)
    with open(os.path.join(adapter_dir, CHAT_TEMPLATE_FILE), "w") as f:
        f.write(QUASAR_TEMPLATE)

    target = _tiny_tokenizer()  # no template
    assert target.chat_template is None
    ensure_chat_template(target, read_chat_template(adapter_dir), None)
    out = str(tmp_path / "merged")
    target.save_pretrained(out)
    sanitize_tokenizer_config(out)

    reloaded = AutoTokenizer.from_pretrained(out)
    assert reloaded.chat_template == QUASAR_TEMPLATE
