# Trainer model prep

Code that runs in the model-prep container before selected training jobs: adapter merge, augmentation, baseline statistics and upload.

## Contents

| Path | Purpose |
| --- | --- |
| `augmentation.py` | Deterministic layer selection and weight modification. |
| `device_probes.py` | Small fixed workloads that time the prep device for compression tasks. A failed probe never fails prep. |
| `entrypoint.py` | Container entrypoint. Prints its JSON result on the last line of stdout. |
| `stats.py` | Dataset, weight-structure and training-dynamics statistics per task type. |

## Compression tasks

Compression tasks carry instruct or chat data. `text_extractor_for` in `stats.py` picks the chat extractor when the first record has a `conversations` column, and the instruct extractor otherwise.

## Adapter bases

When a task's base model is a LoRA adapter repo, model prep publishes a flat full-weight copy and returns it as `augmented_model_id`.

- The copy goes to a deterministic `merged-<hash>` repo.
- The hash is keyed on the source Hugging Face repo id (`--source-model-id`), not on the local cache path, so retries reuse one target.
- Training and evaluation use that flat base.
