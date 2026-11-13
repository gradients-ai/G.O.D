# Task Prep

Pre-training task preparation helpers.

## Contents

- `augmentation.py`: decides and applies augmentation configuration. Text
  augmentation is skipped for bases >= 35B so model-prep does not upload a
  full 70B copy.
- `constants.py`: task prep feature flags and probabilities, and the quant eval-set bounds:
  `QUANT_EVAL_MAX_SEQUENCE_LEN` (longer examples are dropped; bounds the scorer's `max_model_len`) and
  `QUANT_EVAL_MIN_EXAMPLES` (fewer fail prep, so the tournament replaces the task).
- `model.py`: model-prep API calls to trainer nodes.
- `yarn.py`: YaRN extension helpers.
- `__init__.py`: package marker.
