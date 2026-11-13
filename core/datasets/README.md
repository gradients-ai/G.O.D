# Core datasets

The dataset whitelist and a small example dataset, shared across runtimes.

## Contents

| Path | Purpose |
| --- | --- |
| `examples/` | Example dataset. |
| `whitelist.py` | Loads the whitelist and filters a miner's `requested_datasets` to whitelisted entries, capped at `MAX_REQUESTED_DATASETS`. |
| `whitelisted_sft_datasets.json` | Hugging Face datasets a miner may request. |

## See also

- [How your code runs](../../docs/miner.md#how-your-code-runs): how requested datasets reach a container.
