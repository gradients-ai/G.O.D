# Validator Evaluation Tests

Tests for text/image evaluation runtimes, the text-eval data path, Basilica deployment tracking, and eval reconciliation.

## Contents

- `fixtures/`: committed reference data (the axolotl golden fixtures).
- `golden/`: one-off capture tooling for the golden fixtures (never run by pytest).
- `test_axolotl_golden_fixtures.py`: integrity of the golden fixtures and the two chat-masking defects they freeze.
- `test_axolotl_golden_equivalence.py`: the native text-eval data path against the golden fixtures: instruct/completion
  bit-identical, chat input ids/rows/order identical with only the intended label deltas (real tokenizers).
- `test_basilica_deployment_tracking.py`: Basilica eval deployment name resolution and readiness/delete lifecycle tests.
- `test_chat_labels.py`: the chat labelling rule on fake templates (Qwen3-, kanana-, LFM-like shapes, straddles, skips).
- `test_chat_labels_templates.py`: the chat labelling rule on every cached real chat template (real tokenizers).
- `test_chat_templates.py`: chat template name resolution and the bundled `chatml.jinja` hash.
- `test_eval_reconcile.py`: eval-deployment reconciler (orphan/ghost grace) tests.
- `test_instruct_labels.py`: instruct/completion prompt formatting, masking, EOS/BOS and chunking rules.
- `test_instruct_text_dataset_wiring.py`: what the instruct/chat evaluator hands the Trainer and its sequence_len retry ladder
  (imports the evaluator, so needs torch/transformers).
- `test_no_axolotl_imports.py`: static check that no validator code imports or references axolotl.
- `test_runpod_backend.py`: RunPod eval backend tests.
- `test_text_eval_config.py`: `TextEvalConfig` defaults and the short-context halving rule.
- `test_text_eval_dataset.py`: eval-set build order (drops, stable sort), dispatch and row loading.
- `__init__.py`: package marker.

Tests marked "real tokenizers" or importing the evaluator skip in the dev venv (no transformers/torch). Run them in a venv
with the eval image's stack (transformers 5.12.1, torch 2.9.1, trl 1.5.1, datasets 4.8.x) and the tokenizers in the local
HF cache, offline:

```bash
HF_HUB_OFFLINE=1 <venv>/bin/python -m pytest -q -p no:cacheprovider -o addopts='' tests/validator/evaluation
```
