# Task Prep

Pre-training task preparation helpers.

## Contents

- `augmentation.py`: decides and applies augmentation configuration. Text
  augmentation is skipped for bases >= 35B so model-prep does not upload a
  full 70B copy.
- `constants.py`: task prep feature flags and probabilities.
- `model.py`: model-prep API calls to trainer nodes.
- `yarn.py`: YaRN extension helpers.
- `__init__.py`: package marker.
