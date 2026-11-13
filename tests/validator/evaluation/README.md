# Validator evaluation tests

Tests for `validator/evaluation`: evaluators, the compression scorer, remote backends and deployment reconciliation.

## Contents

| Path | Purpose |
| --- | --- |
| `fake_vllm.py` | CPU stand-ins for the vLLM and kernel surface that `quant_vllm` touches. |
| `fixtures/quant_configs/` | Model configs used by the config check tests. See its README. |
| `fixtures/quant_headers/` | Safetensors headers (dtype and shape, no weights) used by the structure tests. See its README. |
| `test_basilica_deployment_tracking.py` | Basilica deployment name resolution, readiness and deletion. |
| `test_chat_labels.py` | The chat labelling rule on fake templates. Imports only `chat_labels`. |
| `test_chat_labels_pin.py` | `chat_labels.py` matches its pinned bytes. |
| `test_chat_labels_templates.py` | The chat labelling rule on real tokenizers. Skipped without transformers. |
| `test_chat_templates.py` | Chat template resolution by name, the bundled ChatML bytes and the tokenizer adapter. |
| `test_eval_errors.py` | Whose fault a failure is: the infra predicate, error descriptions and the evaluator and backend channels. |
| `test_eval_reconcile.py` | Deployment reconciler: orphan and ghost grace. |
| `test_quant_artifacts.py` | Decoded model materialisation checks and artifact byte counting. |
| `test_quant_config_check.py` | `quantization_config` against the fixed-track spec, and config drift from the input model. |
| `test_quant_eval_dispatch.py` | Compression evaluator dispatch, per-repo scoring, scored fails and the same-deployment bf16 reference. |
| `test_quant_image_imports.py` | The scorer imports without torch, axolotl, peft or fiber. |
| `test_quant_loss.py` | Per-example loss from prompt logprobs and the flat mean over examples. |
| `test_quant_runpod_backend.py` | GPU models and instance types passed to the Runpod backend, and transient dstack errors. |
| `test_quant_structure.py` | The expected fixed-track tensor set, and submissions that deviate from it. |
| `test_quant_vllm.py` | Per-layer spec and kernel check, the deterministic kernel patch and the pinned engine inputs. |
| `test_runpod_backend.py` | Runpod evaluation backend. |

## See also

- [Evaluation internals](../../../validator/evaluation/README.md): evaluation internals and failure classes.
