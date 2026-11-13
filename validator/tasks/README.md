# Validator Tasks

Task schemas, dataset preparation, model prep, reward validation, synthetic task generation, and task request helpers.

## Contents

- `datasets/`: dataset column mapping, dataset validation, and preparation.
- `prep/`: augmentation, model prep, and YaRN helpers.
- `rewards/`: reward function validation and affine reward helpers.
- `synthetics/`: synthetic task/data generation.
- `config.py`: task config models and task-type config helpers.
- `details.py`: task detail conversion and upload helpers.
- `models.py`: validator task models and task-owned enums such as backend selection.
- `quant_eval_set.py`: pure tokenisation of a quant task's held-out split into the scorer's JSONL (`{input_ids, labels}`, see `validator/evaluation/quant_loss.py`): instruct rows built exactly as axolotl's user_defined strategy builds them in training (the trainer's format defaults, system_format glued onto the turn, prompt and completion tokenised apart, BOS kept on the prompt and stripped from the completion, EOS appended; nothing is added at the boundary, see `tests/validator/tasks/test_quant_instruct_parity.py`), chat labelled by `validator/evaluation/chat_labels.label_chat` (the standard eval's rule: every assistant turn plus the template's turn end, template chosen by the task's `chat_template` name); skips (empty, too long, unrenderable) are counted and an out-of-vocab id raises. The tokenizer is injected through `QuantTokenizer`, so it never imports transformers. The fingerprint hashes the ids and labels, so every fingerprint comparison (reference reuse, calibration reference, paired losses) rejects a set built under different tokenisation rules.
- `quant_tokenizer.py`: `HfQuantTokenizer` / `load_quant_tokenizer`, the real `QuantTokenizer` (the task's base `AutoTokenizer`, no remote code). The only quant prep module importing transformers, which the validator image carries and the dev venv does not.
- `quant_input_repo.py`: whether a hub repo is a usable quant input (`full_model_rejection`): a config.json, no LoRA adapter, and loadable weights; every shard a `*.safetensors.index.json` maps must exist (vLLM follows the index), else at least one `*.safetensors`. Shared by the chained-input pick, seed replacement and the boss round's own-upstream checks (and the own-model R1 repo).
- `quant_shapes.py`: quant model-pool filter: dense, no remote code, a vLLM-supported `model_type`, and shapes divisible by group size 128 (read from the validator's own config.json download); `filter_has_chat_template` further keeps only bases whose tokenizer ships a chat template, for R3's `tokenizer_default` chat data.
- `requests.py`: model parameter lookup, task prep and train-request helpers. `run_quant_task_prep` splits and uploads like a text task (instruct columns renamed to the standard ones, `dataset_type` rewritten to match), then tokenises the test split with the task's base tokenizer and replaces `test_data` with the uploaded JSONL; a task with more test rows than `MAX_TEST_DATA_POINTS` (a boss-round continuous chunk) is first cut to that many by `_capped_eval_rows`, a sample seeded by the task's ds so tasks on one chunk score identical rows; fewer than `QUANT_EVAL_MIN_EXAMPLES` examples fail prep. `prepare_quant_task_request` is the organic `TrainRequestQuant`.
- `__init__.py`: package marker.
