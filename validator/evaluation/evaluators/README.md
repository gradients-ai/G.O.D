# Evaluators

Task-specific evaluator entrypoints.

## Contents

- `diffusion.py`: image/diffusion evaluation.
- `dpo.py`: DPO evaluation.
- `environment.py`: environment-task evaluation.
- `grpo.py`: GRPO evaluation.
- `instruct_text.py`: instruct/chat text evaluation. Every result carries the eval-set fingerprint and, with CUDA, `eval_gpu` (name and SM count, no SKU check), so a boss-round instruct entry stored at seed time records which rows and hardware it was scored on; per-example losses stay opt-in (`EMIT_PER_EXAMPLE_LOSSES`).
- `intercode.py`: InterCode evaluation.
- `quant.py`: quant-task evaluation; per-repo scoring with an injectable engine, CPU-testable. A repo present in
  `DECODED_MODEL_URLS` is materialised from its decoded shards and loaded from that `model_path`; a repo absent
  from the map loads straight from HF (the bf16 reference). Every scored fail it
  records persists as `task_nodes.quant_failure`, which `process_miners_pool` treats as final on every later run;
  re-evaluating a hotkey means clearing that column first (see `validator/scoring/README.md`).
- `single_dpo.py`: single-repo DPO evaluation CLI.
- `single_grpo.py`: single-repo GRPO evaluation CLI.
- `single_instruct_text.py`: single-repo instruct text evaluation CLI.
- `single_kl_divergence.py`: KL-divergence evaluation CLI.
- `single_quant.py`: single-repo quant evaluation CLI; wires the vLLM engine into `quant.evaluate_repo`.
- `swe.py`: SWE Infinite evaluation against an external Affinetes server.
- `__init__.py`: package marker.
