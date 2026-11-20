# Evaluators

Evaluator entrypoints, one per task type. They run inside the evaluator images.

## Contents

| Path | Purpose |
| --- | --- |
| `diffusion.py` | Image evaluation: prediction error on shared test cases, with captioned and empty-caption losses. |
| `dpo.py` | DPO evaluation. |
| `grpo.py` | GRPO evaluation. |
| `instruct_text.py` | Instruct and chat text evaluation. Each result carries the eval-set fingerprint and, with CUDA, the GPU it ran on. |
| `quant.py` | Compression task evaluation: checks the GPU, scores each repo and records scored fails. |
| `single_dpo.py` | Single-repo DPO evaluation CLI. |
| `single_grpo.py` | Single-repo GRPO evaluation CLI. |
| `single_instruct_text.py` | Single-repo instruct text evaluation CLI. |
| `single_kl_divergence.py` | KL divergence evaluation CLI. |
| `single_quant.py` | Single-repo compression evaluation CLI. Wires the vLLM engine into `quant.evaluate_repo`. |

## See also

- [Evaluation internals](../README.md): evaluation internals and failure classes.
- [Compression scoring](../../scoring/README.md#compression-scoring): how compression losses are normalised and ranked.
