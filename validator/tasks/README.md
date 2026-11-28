# Validator tasks

Task models, dataset preparation, model prep, reward functions and synthetic task creation.

## Contents

| Path | Purpose |
| --- | --- |
| `datasets/` | Dataset column mapping, validation, splitting and the calibration carve. |
| `prep/` | Augmentation, model prep dispatch and YaRN helpers. |
| `rewards/` | Reward function validation, built-in reward functions and templates. |
| `synthetics/` | Synthetic task creation and training-hours budgets. |
| `config.py` | Per-task-type config models and `get_task_config`. |
| `details.py` | Conversion of a task to its public details, with sensitive fields hidden until it finishes. |
| `models.py` | Validator task models, one per task type, and the `Backend` enum. |
| `quant_chat_labels.py` | Which tokens of a chat row the standard text eval scores, reproduced without axolotl for compression eval sets. |
| `quant_eval_set.py` | Tokenises a compression task's held-out split into the scorer's test set. |
| `quant_input_repo.py` | `full_model_rejection`: whether a hub repo is a full model a compression task can take as input. |
| `quant_shapes.py` | Which base models a compression task may use, read from the validator's download of `config.json`. |
| `quant_tokenizer.py` | The task's base tokenizer for compression prep, and its context length read from config.json as JSON (no AutoConfig, so a model_type the installed transformers does not know still preps). The only compression prep module that imports transformers. |
| `requests.py` | Model parameter lookup, task prep for each task type and train-request construction. |

## Compression eval sets

`quant_eval_set.py` writes one JSONL line per example: `{"input_ids": [...], "labels": [...]}`. Labels are the ones the standard text eval sets on the same row, so a model scores the same tokens in both evals.

| Data | How labels are set |
| --- | --- |
| Instruct | Prompt and completion are tokenised apart. Only completion tokens are labelled. |
| Chat | `quant_chat_labels.label_chat_as_text_eval`, axolotl's `chat_template` rule with its default options: the content of every assistant turn, located in the render that ends with that turn, plus the tokenizer's eos when the template emits it within 3 tokens after the turn. Those positions are labelled in the full conversation, so a template that renders the last assistant turn with extra tokens (Qwen3's empty think block) moves the labels of earlier assistant turns by that many tokens. The think block itself is not labelled. |

Empty, over-long and unrenderable rows are skipped and counted. An out-of-vocabulary id raises. `validator/evaluation/quant_loss.quant_eval_set_fingerprint` hashes ids and labels, so sets built under different tokenisation never compare as equal. A prep that yields fewer than `QUANT_EVAL_MIN_EXAMPLES` examples fails.

## See also

- [Evaluation](../../docs/miner_compression.md#evaluation): the miner-facing description.
- [Evaluation internals](../evaluation/README.md): evaluation internals and failure classes.
