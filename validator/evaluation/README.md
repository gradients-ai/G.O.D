# Validator Evaluation

Evaluation runtimes and helpers used after miner training completes.

## Contents

- `comfy_workflows/`: ComfyUI workflow JSON files for image evaluation.
- `evaluators/`: task-specific evaluator entrypoints for text, DPO, GRPO, and image tasks.
- `basilica.py`: Basilica client integration.
- `basilica_deployments.py`: Basilica deployment helpers.
- `chat_labels.py`: the chat labelling rule (which tokens of a conversation are scored: every assistant turn's content plus the template's turn end). Self-contained (stdlib + pydantic) because miners copy it verbatim.
- `chat_templates.py`: chat template resolution (`tokenizer_default`, `chatml`, `tokenizer_default_fallback_chatml`) and the transformers adapter for `chat_labels`.
- `common.py`: shared text evaluation helpers (eval config for a submission, base param counts via hub metadata, no weight reload).
- `constants.py`: evaluation constants (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on all remote eval containers).
- `remote_runner.py`: HTTP wrapper for Basilica/Runpod eval containers; serves `/result` even if a post-eval step exits non-zero when results already exist.
- `runpod.py`: Runpod evaluation backend via dstack services.
- `dataset_configs.py`: default dataset config discovery.
- `db_utils.py`: evaluation persistence helpers.
- `docker_evaluation.py`: Docker/Basilica evaluation orchestration.
- `evaluation_logging.py`: evaluation logging helpers.
- `image_io.py`: image loading and encoding helpers.
- `instruct_labels.py`: instruct and completion rows to tokens and labels, bit-identical to the axolotl strategies the eval used before.
- `local_evaluation.py`: local evaluation runner.
- `model_checks.py`: model architecture and LoRA checks.
- `models.py`: evaluation payload models.
- `reconcile.py`: evaluation deployment and GPU-reservation reconciliation, including release of leftover PvP reservations from historical environment tasks.
- `result_processing.py`: conversion of raw eval output into persisted results.
- `runtime.py`: evaluation runtime helpers.
- `text_eval_config.py`: typed text-eval settings (`TextEvalConfig`: sequence_len and its halving rule for short-context models, batch size 1).
- `text_eval_dataset.py`: held-out text rows to the ordered, filtered eval set (load, tokenise, drop, stable sort) for instruct, completion and chat tasks.
- `text_examples.py`: `TokenizedExample`, the tokenised unit every text evaluator scores.
- `utils.py`: evaluation utility helpers.

The text evaluators (instruct/chat, DPO, GRPO) do not use axolotl: data loading, tokenisation and label masks are the
validator's own (`text_eval_dataset.py`), DPO/GRPO tokenisation is TRL's.

Environment, InterCode, SWE Infinite, and PvP evaluation runtimes have been removed. `TaskType.ENVIRONMENTTASK` and stored PvP score rows remain readable for historical tournaments.
