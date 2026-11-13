# Task prep

Preparation that runs before miners are assigned to a task.

## Contents

| Path | Purpose |
| --- | --- |
| `augmentation.py` | Pure functions that decide a task's augmentation config. |
| `constants.py` | Prep and augmentation switches and probabilities, and the compression eval-set bounds. |
| `model.py` | Dispatches model prep (augmentation and baseline stats) to a trainer with a GPU. |
| `yarn.py` | YaRN context-length extension helpers. |

## See also

- [Model prep](../../../trainer/model_prep/README.md): the model prep job a trainer runs.
