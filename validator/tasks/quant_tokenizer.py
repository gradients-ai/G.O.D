"""The real QuantTokenizer for quant prep: the task's base tokenizer, downloaded by the validator.

The only quant prep module that imports transformers. The validator image (FROM axolotlai/axolotl)
carries it; the dev venv does not (pyproject puts it in the gpu extra), so run_quant_task_prep
imports this module lazily and tests inject a fake tokenizer instead.
"""

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers.utils.import_utils import is_jinja_available

from core.constants.credentials import HUGGINGFACE_TOKEN
from validator.evaluation.chat_labels import ChatTokenizer
from validator.evaluation.chat_templates import HfChatTokenizer
from validator.evaluation.chat_templates import resolve_chat_template
from validator.tasks.quant_eval_set import QuantTokenizer


class HfQuantTokenizer:
    def __init__(self, model_id: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_TOKEN, trust_remote_code=False)
        self.bos_token_id: int | None = self._tokenizer.bos_token_id
        self.eos_token_id: int | None = self._tokenizer.eos_token_id
        self.vocab_size: int = len(self._tokenizer)
        config = AutoConfig.from_pretrained(model_id, token=HUGGINGFACE_TOKEN, trust_remote_code=False)
        self.context_len: int | None = getattr(config, "max_position_embeddings", None)

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
