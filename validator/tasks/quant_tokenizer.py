"""The real QuantTokenizer for quant prep: the task's base tokenizer, downloaded by the validator.

The only quant prep module that imports transformers (the `gpu` extra; the dev venv may not have it), so
run_quant_task_prep imports this module lazily and tests inject a fake tokenizer instead.

The model's context length is read from its config.json as JSON, not through AutoConfig: AutoConfig refuses any
model_type the installed transformers does not know, and prep needs one integer from the file.
"""

import json
import os

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from transformers.utils.import_utils import is_jinja_available

from core.constants.credentials import HUGGINGFACE_TOKEN
from core.constants.quant import QUANT_MODEL_CONFIG_FILENAME
from validator.evaluation.chat_labels import ChatTokenizer
from validator.evaluation.chat_templates import HfChatTokenizer
from validator.evaluation.chat_templates import resolve_chat_template
from validator.tasks.datasets.fetch import DataFetchError
from validator.tasks.datasets.fetch import is_transient_hub_error
from validator.tasks.quant_eval_set import QuantTokenizer


CONTEXT_LEN_KEY = "max_position_embeddings"
# Multimodal configs (gemma3, llama4, ...) keep the language model's fields under this key.
TEXT_CONFIG_KEY = "text_config"


def _read_model_config(model_id: str) -> dict:
    """The model's config.json as a dict: from model_id itself when it is a local directory, else from the hub."""
    if os.path.isdir(model_id):
        path = os.path.join(model_id, QUANT_MODEL_CONFIG_FILENAME)
    else:
        path = hf_hub_download(model_id, QUANT_MODEL_CONFIG_FILENAME, token=HUGGINGFACE_TOKEN)
    with open(path) as f:
        return json.load(f)


def config_context_len(config: dict) -> int | None:
    """max_position_embeddings from config.json, else from its text_config; None when neither says."""
    for section in (config, config.get(TEXT_CONFIG_KEY) or {}):
        value = section.get(CONTEXT_LEN_KEY) if isinstance(section, dict) else None
        if isinstance(value, int) and value > 0:
            return value
    return None


class HfQuantTokenizer:
    def __init__(self, model_id: str):
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_TOKEN, trust_remote_code=False)
            config = _read_model_config(model_id)
        except Exception as e:
            if is_transient_hub_error(e):
                raise DataFetchError(f"could not load the tokenizer of {model_id} from the hub: {type(e).__name__}: {e}") from e
            raise
        self.bos_token_id: int | None = self._tokenizer.bos_token_id
        self.eos_token_id: int | None = self._tokenizer.eos_token_id
        self.vocab_size: int = len(self._tokenizer)
        self.context_len: int | None = config_context_len(config)

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def chat_labeller(self, chat_template: str | None) -> ChatTokenizer:
        # Without jinja2, apply_chat_template raises NameError on every row, which tokenize_chat would
        # count as unrenderable rows: fail prep on the environment instead of blaming the data.
        if not is_jinja_available():
            raise RuntimeError("jinja2 is not installed: transformers cannot render chat templates")
        return HfChatTokenizer(self._tokenizer, resolve_chat_template(chat_template, self._tokenizer))


def load_quant_tokenizer(model_id: str) -> QuantTokenizer:
    return HfQuantTokenizer(model_id)
