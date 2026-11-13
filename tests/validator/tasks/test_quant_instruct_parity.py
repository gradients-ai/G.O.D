"""The quant scorer's instruct ids and labels equal what training (axolotl) builds, token for token.

`axolotl_instruct_example` is a transliteration of axolotl 0.18.0's
InstructionWSystemPromptTokenizingStrategy.tokenize_prompt (axolotl/prompt_strategies/alpaca_w_system.py:27-52) over
PromptTokenizingStrategy._tokenize (axolotl/prompt_tokenizers.py:73-105),
the strategy `user_defined.load` returns for the dataset entry core/training_config._process_instruct_dataset_fields writes,
with train_on_inputs false. The prompt string is axolotl's SystemDataPrompter.build_prompt_w_system (alpaca_w_system.py:68-85).
The instruct evaluator loads the same entry through the same strategy
(validator/evaluation/evaluators/instruct_text.py).

The real-tokenizer cases need transformers and the HF cache (HF_HUB_OFFLINE=1) and skip where they are absent.
"""

import pytest

from core.models.dataset_models import InstructTextDatasetType
from core.training_config import _process_instruct_dataset_fields
from validator.evaluation.quant_loss import IGNORE_INDEX
from validator.evaluation.quant_loss import QuantEvalExample
from validator.evaluation.quant_loss import quant_eval_set_fingerprint
from validator.tasks.quant_eval_set import QuantTokenizer
from validator.tasks.quant_eval_set import build_quant_eval_set


SEQUENCE_LEN = 4096
REPOS = [
    "deepseek-ai/deepseek-coder-6.7b-instruct",  # BPE, adds BOS
    "unsloth/Llama-3.2-1B-Instruct",  # tiktoken-style BPE, adds BOS
    "Qwen/Qwen2.5-1.5B-Instruct",  # byte-level BPE, no BOS
    "bigscience/bloomz-560m",  # byte-level BPE, no BOS
]

PLAIN = InstructTextDatasetType(field_instruction="q", field_output="a")
WITH_INPUT = InstructTextDatasetType(field_instruction="q", field_input="ctx", field_output="a", field_system="sys")
TEMPLATED = InstructTextDatasetType(
    field_instruction="q",
    field_input="ctx",
    field_output="a",
    field_system="sys",
    system_format="### System:\n{system}\n\n",
    format="### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    no_input_format="### Instruction:\n{instruction}\n\n### Response:\n",
)
# Prompts ending in a letter, a colon, a space and a newline; the completion starts with a word (joint tokenisation
# would merge a trailing space into it) and, in the last rows, with a space or a digit.
ROWS = [
    {"q": "Translate to French", "a": "bonjour"},
    {"q": "What is 2+2? Answer:", "a": "4"},
    {"q": "Say hi\n", "a": "Hello there"},
    {"q": "Capital of France? ", "a": "Paris is the capital."},
    {"q": "Summarise", "ctx": "The cat sat on the mat.", "a": " It sat."},
    {"q": "Fix the typo", "ctx": "", "a": "ok", "sys": "Be terse."},
    {"q": "Réponds", "ctx": "naïve café", "a": "日本語 and emoji 🙂", "sys": "Sys"},
    {"q": "Say <s> literally", "a": "</s> or <|endoftext|>"},
]


def axolotl_instruct_example(
    hf_tokenizer, row: dict, dataset_type: InstructTextDatasetType, sequence_len: int = SEQUENCE_LEN
) -> QuantEvalExample | None:
    """None when axolotl would drop the row: no completion tokens, or longer than sequence_len."""
    entry = _process_instruct_dataset_fields({k: v for k, v in dataset_type.model_dump().items() if v is not None})["type"]
    field = {
        "instruction": entry.get("field_instruction") or "instruction",
        "input": entry.get("field_input") or "input",
        "output": entry.get("field_output") or "output",
        "system": entry.get("field_system") or "system",
    }
    # UserDefinedPrompterType.parse_instruction_fields (prompt_strategies/user_defined.py:52-65)
    instruction = row[field["instruction"]]
    input_text = row[field["input"]] if field["input"] in row else ""
    response = row[field["output"]] if field["output"] in row else ""
    system = row[field["system"]] if field["system"] in row else entry.get("system_prompt", "")
    # SystemDataPrompter.build_prompt_w_system (prompt_strategies/alpaca_w_system.py:68-85)
    system_format = entry.get("system_format")
    sys_prompt = system_format.format(system=system) if system and system_format else ""
    if input_text:
        user_prompt = sys_prompt + entry["format"].format(instruction=instruction, input=input_text)
    else:
        user_prompt = sys_prompt + entry["no_input_format"].format(instruction=instruction)

    def tokenize(text: str, add_eos_token: bool = True, strip_bos_token: bool = False) -> list[int]:
        # PromptTokenizingStrategy._tokenize (prompt_tokenizers.py:73-105); truncation=True, max_length=sequence_len
        if not text:
            return []
        ids = hf_tokenizer(text, truncation=True, max_length=sequence_len, padding=False, return_tensors=None)["input_ids"]
        if not ids:
            return []
        if ids[-1] != hf_tokenizer.eos_token_id and len(ids) < sequence_len and add_eos_token:
            ids.append(hf_tokenizer.eos_token_id)
        if ids[0] == hf_tokenizer.bos_token_id and strip_bos_token:
            ids = ids[1:]
        return ids

    prompt_ids = tokenize(user_prompt, add_eos_token=False)
    completion_ids = tokenize(response, strip_bos_token=True, add_eos_token=True)
    example = QuantEvalExample(input_ids=prompt_ids + completion_ids, labels=[IGNORE_INDEX] * len(prompt_ids) + completion_ids)
    # the instruct evaluator drops rows with no label (instruct_text._load_evaluation_dataset);
    # axolotl drops rows over sequence_len.
    if not completion_ids or len(example.input_ids) > sequence_len:
        return None
    return example


@pytest.fixture(scope="module", params=REPOS)
def tokenizer(request) -> QuantTokenizer:
    pytest.importorskip("transformers")
    quant_tokenizer = pytest.importorskip("validator.tasks.quant_tokenizer")
    try:
        return quant_tokenizer.HfQuantTokenizer(request.param)
    except Exception as exc:  # noqa: BLE001 - not cached / incomplete snapshot
        pytest.skip(f"{request.param} not usable from the local HF cache: {type(exc).__name__}")


@pytest.mark.parametrize("dataset_type", [PLAIN, WITH_INPUT, TEMPLATED], ids=["plain", "input+system", "templated"])
def test_quant_instruct_ids_and_labels_equal_axolotls(tokenizer, dataset_type):
    rows = [row for row in ROWS if "a" in row]
    expected = [axolotl_instruct_example(tokenizer._tokenizer, row, dataset_type) for row in rows]

    build = build_quant_eval_set(rows, dataset_type, tokenizer, SEQUENCE_LEN)

    assert [e for e in expected if e is not None] == build.examples
    assert len(build.examples) == len(rows)


def test_a_trailing_space_is_never_merged_into_the_completion(tokenizer):
    """Axolotl tokenises prompt and completion apart, so `... ` + `Paris` is never the joint `... ` `Paris` merge."""
    row = {"q": "Capital of France? ", "a": "Paris is the capital."}
    hf = tokenizer._tokenizer
    joint = hf(row["q"] + row["a"])["input_ids"]

    example = build_quant_eval_set([row], PLAIN, tokenizer, SEQUENCE_LEN).examples[0]

    separate = hf(row["q"])["input_ids"] + [i for i in hf(row["a"])["input_ids"] if i != hf.bos_token_id]
    assert example.input_ids[: len(separate)] == separate
    assert example.input_ids != joint + [hf.eos_token_id]


def test_prompt_is_masked_and_completion_plus_eos_supervised(tokenizer):
    example = build_quant_eval_set([ROWS[1]], PLAIN, tokenizer, SEQUENCE_LEN).examples[0]
    n_prompt = len(tokenizer.encode(ROWS[1]["q"], add_special_tokens=True))

    assert set(example.labels[:n_prompt]) == {IGNORE_INDEX}
    assert example.labels[n_prompt:] == example.input_ids[n_prompt:]
    assert example.input_ids[-1] == tokenizer.eos_token_id


def test_construction_change_changes_the_fingerprint_and_unchanged_rows_keep_it(tokenizer):
    """The fingerprint hashes ids and labels, so a newline-appending construction cannot collide with the current one."""

    def old_construction(row: dict) -> QuantEvalExample:
        prompt = row["q"] if row["q"][-1:].isspace() else row["q"] + "\n"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        completion_ids = tokenizer.encode(row["a"], add_special_tokens=False) + [tokenizer.eos_token_id]
        return QuantEvalExample(input_ids=prompt_ids + completion_ids, labels=[IGNORE_INDEX] * len(prompt_ids) + completion_ids)

    needs_newline = [{"q": "What is 2+2? Answer:", "a": "4"}]
    ends_in_whitespace = [{"q": "Say hi\n", "a": "Hello"}]

    def new(rows):
        return build_quant_eval_set(rows, PLAIN, tokenizer, SEQUENCE_LEN).examples

    assert quant_eval_set_fingerprint(new(needs_newline)) != quant_eval_set_fingerprint(
        [old_construction(r) for r in needs_newline]
    )
    if tokenizer.bos_token_id is None:  # without a BOS to strip the two constructions agree on whitespace-ended prompts
        assert quant_eval_set_fingerprint(new(ends_in_whitespace)) == quant_eval_set_fingerprint(
            [old_construction(r) for r in ends_in_whitespace]
        )
