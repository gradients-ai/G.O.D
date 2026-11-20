# Synthetic tasks

Creation of synthetic tasks and their training-hours budgets.

## Contents

| Path | Purpose |
| --- | --- |
| `scheduler.py` | Creates synthetic instruct, DPO, GRPO and compression tasks, and computes training hours. |
| `diffusion.py` | Creates synthetic image tasks. |
| `constants.py` | Synthetic generation constants. |
| `example_prompts.json` | Example prompts for synthetic generation. |
| `image_synth_prompts.json` | Prompt data for synthetic image tasks. |

## Compression task hours

At creation a compression task gets a flat budget from `quant_flat_hours` in `scheduler.py`. After model prep, `quant_hours_after_prep` recomputes it with `quant_training_hours` from the measured stats when they are usable. The constants are in `validator/tasks/datasets/constants.py`.

## See also

- [What your code receives](../../../docs/miner_compression.md#what-your-code-receives): the budget as a miner sees it.
- [Validator tournament](../../tournament/README.md): crown rule, seeds, orchestration.
