"""Quant prep's real tokenizer on a Qwen3 checkpoint, offline: a local config.json + tokenizer fixture, no downloads.

Prep read the context length through AutoConfig, which refuses a model_type the installed transformers does not know
(transformers 4.46.2 and `qwen3`), so every compression task on a Qwen3 source failed prep. The fixture is laid out the
way a transformers-v5 trainer saves a Qwen3 tokenizer: tokenizer_class Qwen2Tokenizer, the chat template only in
chat_template.jinja. Skipped where transformers is not installed (the `gpu` extra).
"""

import json
from pathlib import Path

import httpx
import pytest
import requests as requests_lib
from huggingface_hub.errors import LocalEntryNotFoundError

import validator.tasks.prep.constants as prep_cst
from core.models.dataset_models import ChatTemplateDatasetType
from tests.validator.tasks.test_quant_calibration_prep import _freeform
from tests.validator.tasks.test_quant_calibration_prep import prep  # noqa: F401 - fixture
from validator.tasks import requests
from validator.tasks.datasets.fetch import DataFetchError
from validator.tasks.quant_shapes import QUANT_SUPPORTED_MODEL_TYPES


transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
quant_tokenizer = pytest.importorskip("validator.tasks.quant_tokenizer")


QWEN3_CONTEXT_LEN = 40960
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CHATML_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
QWEN3_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_hidden_layers": 2,
    "max_position_embeddings": QWEN3_CONTEXT_LEN,
    "vocab_size": 300,
    "torch_dtype": "bfloat16",
}


def _byte_level_tokenizer() -> "tokenizers.Tokenizer":
    """A byte-level BPE with no merges: every byte is one token, so any text encodes, with character offsets."""
    vocab = {char: index for index, char in enumerate(sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()))}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return tokenizer


@pytest.fixture
def qwen3_dir(tmp_path) -> Path:
    model_dir = tmp_path / "qwen3"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(QWEN3_CONFIG))
    _byte_level_tokenizer().save(str(model_dir / "tokenizer.json"))
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
                "bos_token": None,
                "model_max_length": 131072,
            }
        )
    )
    (model_dir / "chat_template.jinja").write_text(CHATML_TEMPLATE)
    return model_dir


def test_a_qwen3_tokenizer_loads_with_its_context_len_and_jinja_chat_template(qwen3_dir):
    tokenizer = quant_tokenizer.load_quant_tokenizer(str(qwen3_dir))

    assert tokenizer.context_len == QWEN3_CONTEXT_LEN
    assert tokenizer.eos_token_id == tokenizer._tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert tokenizer.bos_token_id is None
    assert tokenizer.vocab_size == 256 + len(SPECIAL_TOKENS)
    labeller = tokenizer.chat_labeller("tokenizer_default")
    assert labeller.render([{"role": "user", "content": "hi"}], add_generation_prompt=False) == (
        "<|im_start|>user\nhi<|im_end|>\n"
    )


async def test_quant_prep_of_a_qwen3_chat_task_completes_with_the_real_tokenizer(prep, qwen3_dir):  # noqa: F811
    turns = [("system", "Be terse."), ("user", "Capital of France?"), ("assistant", "Paris.")]
    rows = [{"conversations": [{"role": role, "content": f"{content} {i}"} for role, content in turns]} for i in range(400)]
    prep.test_rows, prep.calibration_rows = rows[:150], rows[150:]
    task = _freeform()
    task.model_id = str(qwen3_dir)
    task.dataset_type = ChatTemplateDatasetType(
        chat_template="tokenizer_default", chat_column="conversations", chat_role_field="role", chat_content_field="content"
    )

    task = await requests.run_quant_task_prep(task, keypair=None, psql_db="db")

    assert task.calibration_n_examples == 250
    assert task.test_data == "https://minio/quant_eval_set.jsonl"


def test_the_installed_transformers_knows_every_quant_model_type():
    """The validator's transformers pin must keep up with quant_shapes' allowlist: AutoTokenizer falls back to AutoConfig
    when a tokenizer_config.json names no tokenizer class."""
    unknown = [model_type for model_type in QUANT_SUPPORTED_MODEL_TYPES if model_type not in transformers.CONFIG_MAPPING]
    assert unknown == []


def test_prep_does_not_need_transformers_to_know_the_model_type(qwen3_dir):
    """A model_type newer than the installed transformers. The tokenizer names its own class, and
    the context length is read from config.json as JSON, so prep still loads."""
    (qwen3_dir / "config.json").write_text(json.dumps({**QWEN3_CONFIG, "model_type": "qwen9_not_released"}))

    assert quant_tokenizer.load_quant_tokenizer(str(qwen3_dir)).context_len == QWEN3_CONTEXT_LEN


def test_a_multimodal_config_reads_the_context_len_from_its_text_config():
    config = {"model_type": "gemma3", "text_config": {"model_type": "gemma3_text", "max_position_embeddings": 131072}}

    assert quant_tokenizer.config_context_len(config) == 131072
    assert quant_tokenizer.config_context_len({"model_type": "qwen3"}) is None


@pytest.mark.parametrize(
    "transport_error", [requests_lib.ConnectionError("offline"), httpx.ConnectError("refused"), httpx.ReadTimeout("slow")]
)
def test_an_unreachable_hub_is_a_fetch_error(monkeypatch, transport_error):
    def unreachable(*args, **kwargs):
        try:
            try:
                raise transport_error
            except Exception as e:
                raise LocalEntryNotFoundError("not in the cache") from e
        except LocalEntryNotFoundError as e:
            raise OSError("We couldn't connect to 'https://huggingface.co'") from e

    monkeypatch.setattr(quant_tokenizer.AutoTokenizer, "from_pretrained", unreachable)

    with pytest.raises(DataFetchError, match="org/base"):
        quant_tokenizer.load_quant_tokenizer("org/base")


def test_an_unknown_architecture_is_not_a_fetch_error(monkeypatch):
    def unknown(*args, **kwargs):
        raise ValueError("The checkpoint you are trying to load has model type `qwen9`")

    monkeypatch.setattr(quant_tokenizer.AutoTokenizer, "from_pretrained", unknown)

    with pytest.raises(ValueError, match="qwen9"):
        quant_tokenizer.load_quant_tokenizer("org/base")


def test_the_prep_max_sequence_len_is_capped_below_the_qwen3_context(qwen3_dir):
    tokenizer = quant_tokenizer.load_quant_tokenizer(str(qwen3_dir))

    max_len = requests.quant_eval_max_sequence_len(tokenizer.context_len, prep_cst.QUANT_EVAL_MAX_SEQUENCE_LEN)
    assert max_len == min(prep_cst.QUANT_EVAL_MAX_SEQUENCE_LEN, QWEN3_CONTEXT_LEN - 1)


def test_a_tokenizer_without_vocabulary_files_fails_prep(qwen3_dir):
    (qwen3_dir / "tokenizer.json").unlink()

    with pytest.raises(ValueError, match="no vocabulary"):
        quant_tokenizer.load_quant_tokenizer(str(qwen3_dir))
