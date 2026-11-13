# Validator Evaluation Tests

Tests for text/image evaluation runtimes, Basilica deployment tracking, and eval reconciliation.

## Contents

- `test_basilica_deployment_tracking.py`: Basilica eval deployment name resolution and readiness/delete lifecycle tests.
- `test_eval_reconcile.py`: eval-deployment reconciler (orphan/ghost grace) tests.
- `test_runpod_backend.py`: RunPod eval backend tests.
- `test_chat_labels.py`: the chat labelling rule on fake templates shaped like real ones (Qwen3 think blocks, Kanana/LFM generation prompts, turn ends that differ on the last turn, merged newline tokens); imports only `chat_labels`, so miners can copy it.
- `test_chat_labels_pin.py`: `chat_labels.py` is byte-identical to the upstream G.O.D standard eval's.
- `test_chat_labels_templates.py`: the labelling rule on real tokenizers and their chat templates (offline HF cache, CPU; skipped without transformers).
- `test_chat_templates.py`: chat template resolution by name, the bundled ChatML's bytes, and the `HfChatTokenizer` adapter.
- `test_quant_artifacts.py`: decoded freeform model materialisation (sha256/bytes/name checks) and HF byte counting.
- `fake_vllm.py`: CPU stand-ins for the vllm/humming surface `quant_vllm` touches (including the `LLM` kwargs and
  `transformers_utils.config.get_config`).
- `test_quant_config_check.py`: config.json `quantization_config` check against the fixed-track spec
  (each way a config can deviate from it); config drift from the input model (`rms_norm_eps`, nested RoPE theta,
  added RoPE scaling, dropped and new keys, allowlisted keys) and honest transformers-5 saves passing on resolved configs.
- `test_quant_eval_dispatch.py`: quant evaluator dispatch env, per-repo scoring and scored fails (including config
  drift from the input model, scored before any load), decoded-model routing in `main()`, the same-deployment bf16 reference (scored first on an uncached SKU, skipped on a cached one, scored once when it is one of the repos) and its return with the result.
- `test_quant_image_imports.py`: the quant scorer imports without torch, axolotl/peft or fiber.
- `test_quant_structure.py`: expected fixed-track tensor set vs recorded llmcompressor headers per launch format,
  structural mutants (extra tensor, group 64, fp32 scales, bf16 MLP, g_idx, input_scale, embedding shape, .bin
  only), embedding tying and hub-error mapping.
- `fixtures/quant_headers/`: safetensors headers (dtype + shape, no weights) of a base model and its quantised outputs;
  see its README for how they were produced.
- `fixtures/quant_configs/`: resolved and raw configs of two input models and their transformers-5 saves; see its README.
- `test_quant_vllm.py`: per-layer spec/kernel check, `apply_model` collection, the Humming stream-K patch, the pinned
  engine inputs (input model's tokenizer, so a changed chat template is never read; HF config; safetensors) and the
  resolved-config wiring.
- `__init__.py`: package marker.
