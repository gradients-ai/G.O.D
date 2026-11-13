# Synthetic Tasks

Synthetic data and synthetic task generation.

## Contents

- `constants.py`: synthetic generation constants.
- `diffusion.py`: image/diffusion synthetic task generation.
- `example_prompts.json`: prompt examples for synthetic generation.
- `scheduler.py`: synthetic text, GRPO, DPO, environment and quant task scheduling. `create_synthetic_quant_task` draws the fixed format (seeded `draw_quant_format`) once, then calls the retried `_create_synthetic_quant_task_pinned`.
- `__init__.py`: package marker.

## Quant hours

`quant_hours_breakdown` / `quant_training_hours` in `scheduler.py` price a quant task from its own model-prep stats
(the miner-facing description is "Hours" in `docs/miner_quant.md`; the constants and the per-track `QuantWorkload`s
are in `validator/tasks/datasets/constants.py`). The budget is the sum of four terms (forward-only token work,
forward+backward (training) token work, a term that scales with model size, and host / I/O time) times a headroom:

```
token_work = eval_passes * QUANT_EVAL_ROWS_PER_PASS * mean_seq * QUANT_FORWARD_TOKEN_FACTOR
           + forward_tokens * QUANT_FORWARD_TOKEN_FACTOR
           + min(train_tokens, train_max_epochs * split_tokens) * QUANT_TRAIN_TOKEN_FACTOR
seconds    = token_work / (tps * gpus) + model_seconds_per_b * params_b / gpus
           + host_seconds_per_b * params_b + host_fixed_seconds
hours      = clamp(QUANT_TRAINING_HOURS_MIN, MAX_TRAINING_HOURS, ceil_quarter(seconds * hours_slack / 3600 * boss_multiplier))
```

`token_work` is in probe-token equivalents: forward-only tokens (evaluation passes over the held-out split, plus
`forward_tokens`) count `QUANT_FORWARD_TOKEN_FACTOR` each, training tokens count `QUANT_TRAIN_TOKEN_FACTOR` each (the
epoch cap can only lower the budget). `tps` is the prep's measured per-GPU fwd+bwd rate
(`baseline_stats.device.train_tokens_per_sec`, the hardened median-of-windows probe; prep data without it falls
back to the 3-step `throughput.tokens_per_sec`), clamped like instruct's.

- **Where hours are set.** At creation and on a chained release with unusable upstream stats: the flat fallback
  (`QUANT_FREEFORM_EARLY_HOURS`, `quant_fixed_hours`). On a chained release with usable upstream stats: the formula on
  the upstream's stats. After the task's own prep (`quant_hours_after_prep`): the formula on its own stats, which is
  authoritative. Usable stats = a `BaselineStats` with `throughput` and `dataset.seq_length_distribution.mean > 0`.
- **No boss bump.** `QUANT_BOSS_HOURS_MULTIPLIER` is 1.0 on both tracks: boss and challenger play the same task with the
  same hours, so extra time is not a fairness lever.
- **Not priced.** Layer count beyond params (the per-block cost is flat in width) and host-contention stalls from two
  containers sharing a host.