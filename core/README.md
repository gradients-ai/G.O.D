# Core

Code shared by more than one runtime: validator, trainer and miner. A module belongs here only when it is part of a cross-process contract.

## Contents

| Path | Purpose |
| --- | --- |
| `constants/` | Shared network, dataset, Docker, path, training and credential constants. |
| `datasets/` | The dataset whitelist miners may request from, and a small example dataset. |
| `models/` | Pydantic models that cross service boundaries. |
| `training_templates/` | Base training config templates copied into trainer containers. |
| `build_context.py` | `find_large_files`: lists files in a training repo over the review size. It lists and never rejects. |
| `downloads.py` | Guarded HTTP download helper with size and redirect limits. |
| `git.py` | Git clone URL helpers, including token-authenticated URLs. |
| `interfaces.py` | `KeypairLike`, the minimal signing interface used by chain-free task preparation. |
| `logging.py` | Logger setup, and `redact_url_queries`, which masks the query of each http(s) URL in a string before it is logged. |
| `merged_repo.py` | Deterministic name of the published flat merge of a LoRA adapter repo. |
| `near_duplicate.py` | Near-duplicate rate of a text dataset, shared by data prep and model prep. |
| `tokenizer_utils.py` | Chat template and tokenizer config helpers used by LoRA merges, the uploader and compression shape checks. |
| `training_config.py` | Axolotl dataset entry construction used by validator evaluation. |

## See also

- [Repository layout](../docs/developer.md#repository-layout): where each top-level package sits.
