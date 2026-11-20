"""Compression prep's token ids and labels are what the quant scorer reads and what transformers itself produces.

The quant scorer does not tokenise text: it loads the test set prep wrote (quant_loss.load_examples) and feeds those
ids to the engine. So parity has three parts, checked here on a tokenizer saved by the pinned transformers (template in
chat_template.jinja), fully offline:

* the ids prep writes are the ids `apply_chat_template(tokenize=True)` gives on the same tokenizer;
* the labelled tokens are exactly each assistant turn's content plus the template's turn end (chat_labels' rule, the
  module the text eval and the miners' training share);
* the JSONL prep uploads loads back through the scorer's own loader unchanged, with the same fingerprint.

tests/test_transformers_pins.py holds the validator's transformers equal to the quant evaluator image's.
"""

import json
from pathlib import Path

import pytest

from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import InstructTextDatasetType
from validator.evaluation import chat_labels
from validator.evaluation.chat_labels import IGNORE_INDEX
from validator.evaluation.quant_loss import load_examples
from validator.evaluation.quant_loss import quant_eval_set_fingerprint
from validator.tasks import quant_eval_set
from validator.tasks.quant_eval_set import build_quant_eval_set
from validator.tasks.quant_eval_set import quant_eval_set_jsonl


transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
quant_tokenizer = pytest.importorskip("validator.tasks.quant_tokenizer")
chat_templates = pytest.importorskip("validator.evaluation.chat_templates")


MAX_SEQUENCE_LEN = 4096
TURN_END = "<|im_end|>\n"
LAST_TURN_DECORATION = "<think>\n\n</think>\n\n"
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<s>"]
# Decorates only the last assistant turn: label spans must still land on every turn.
CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{% if message['role'] == 'assistant' and loop.last %}<think>\n\n</think>\n\n{% endif %}"
    "{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
CHAT_DATASET = ChatTemplateDatasetType(
    chat_template="tokenizer_default", chat_column="conversations", chat_role_field="role", chat_content_field="content"
)
CHAT_ROWS = [
    {
        "conversations": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "What is two plus two?"},
            {"role": "assistant", "content": "Four."},
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "Name a prime."},
            {"role": "assistant", "content": "Seven."},
            {"role": "user", "content": "Another, über-quick?"},
            {"role": "assistant", "content": "Eleven — done."},
        ]
    },
    {"conversations": [{"role": "user", "content": "Say nothing."}, {"role": "assistant", "content": "Okay."}]},
]
INSTRUCT_DATASET = InstructTextDatasetType(field_instruction="question", field_output="answer")
INSTRUCT_ROWS = [{"question": "Spell cat.", "answer": "c-a-t"}, {"question": "Capital of Peru?", "answer": "Lima"}]


def _byte_level_tokenizer() -> "tokenizers.Tokenizer":
    """Byte-level BPE with no merges (every byte one token), BOS added by its post-processor as Llama-style tokenizers do."""
    vocab = {char: index for index, char in enumerate(sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()))}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    bos_id = tokenizer.token_to_id("<s>")
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="<s> $A", pair="<s> $A <s> $B", special_tokens=[("<s>", bos_id)]
    )
    return tokenizer


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    """A tokenizer saved by the installed (pinned) transformers, next to a config.json."""
    directory = tmp_path_factory.mktemp("parity-model")
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=_byte_level_tokenizer(), bos_token="<s>", eos_token="<|im_end|>", pad_token="<|endoftext|>"
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(str(directory))
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen3", "max_position_embeddings": 32768}))
    return directory


@pytest.fixture(scope="module")
def reference_tokenizer(model_dir):
    """transformers' own view of the saved tokenizer, independent of prep's loader."""
    return transformers.AutoTokenizer.from_pretrained(str(model_dir))


def _ids(encoded) -> list[int]:
    # apply_chat_template(tokenize=True) returns a BatchEncoding or a plain list depending on the transformers release.
    return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def _scored_runs(input_ids: list[int], labels: list[int]) -> list[list[int]]:
    """Maximal runs of labelled positions, as token id lists."""
    runs, current = [], []
    for token, label in zip(input_ids, labels):
        if label == IGNORE_INDEX:
            if current:
                runs.append(current)
            current = []
        else:
            assert label == token
            current.append(token)
    return runs + ([current] if current else [])


def test_the_fixture_is_saved_the_way_current_transformers_saves_a_tokenizer(model_dir):
    assert (model_dir / "chat_template.jinja").read_text() == CHAT_TEMPLATE
    assert "chat_template" not in json.loads((model_dir / "tokenizer_config.json").read_text())


def test_prep_labels_chat_rows_with_the_shared_chat_labels_module(model_dir):
    labeller = quant_tokenizer.load_quant_tokenizer(str(model_dir)).chat_labeller(CHAT_DATASET.chat_template)

    assert quant_eval_set.label_chat is chat_labels.label_chat
    assert isinstance(labeller, chat_templates.HfChatTokenizer)
    assert labeller.chat_template == CHAT_TEMPLATE


def test_prep_chat_ids_and_labels_match_transformers_on_the_same_rows(model_dir, reference_tokenizer):
    build = build_quant_eval_set(CHAT_ROWS, CHAT_DATASET, quant_tokenizer.load_quant_tokenizer(str(model_dir)), MAX_SEQUENCE_LEN)

    assert len(build.examples) == len(CHAT_ROWS)
    for row, example in zip(CHAT_ROWS, build.examples):
        messages = row["conversations"]
        assert example.input_ids == _ids(reference_tokenizer.apply_chat_template(messages, tokenize=True))
        # The scored text starts where the generation prompt ends, so the last turn's decoration is scored with it.
        expected_spans = [
            (LAST_TURN_DECORATION if index == len(messages) - 1 else "") + message["content"] + TURN_END
            for index, message in enumerate(messages)
            if message["role"] == "assistant"
        ]
        assert [reference_tokenizer.decode(run) for run in _scored_runs(example.input_ids, example.labels)] == expected_spans


def test_prep_instruct_ids_match_transformers_on_the_same_rows(model_dir, reference_tokenizer):
    build = build_quant_eval_set(
        INSTRUCT_ROWS, INSTRUCT_DATASET, quant_tokenizer.load_quant_tokenizer(str(model_dir)), MAX_SEQUENCE_LEN
    )

    eos = reference_tokenizer.eos_token_id
    bos = reference_tokenizer.bos_token_id
    for row, example in zip(INSTRUCT_ROWS, build.examples, strict=True):
        prompt_ids = reference_tokenizer(row["question"])["input_ids"]
        completion_ids = reference_tokenizer(row["answer"])["input_ids"]
        assert prompt_ids[0] == bos and completion_ids[0] == bos
        expected_completion = completion_ids[1:] + [eos]
        assert example.input_ids == prompt_ids + expected_completion
        assert example.labels == [IGNORE_INDEX] * len(prompt_ids) + expected_completion


def test_the_scorer_reads_back_exactly_what_prep_wrote(model_dir, tmp_path):
    tokenizer = quant_tokenizer.load_quant_tokenizer(str(model_dir))
    examples = build_quant_eval_set(CHAT_ROWS, CHAT_DATASET, tokenizer, MAX_SEQUENCE_LEN).examples
    examples += build_quant_eval_set(INSTRUCT_ROWS, INSTRUCT_DATASET, tokenizer, MAX_SEQUENCE_LEN).examples
    path = tmp_path / "quant_eval_set.jsonl"
    path.write_text(quant_eval_set_jsonl(examples))

    loaded = load_examples(path)

    assert loaded == examples
    assert quant_eval_set_fingerprint(loaded) == quant_eval_set_fingerprint(examples)
