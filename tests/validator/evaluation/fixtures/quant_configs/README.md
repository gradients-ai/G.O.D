# Compression config fixtures

Model configs that pin what `config_drift_failure` (`validator/evaluation/quant_config_check.py`) must accept. Each is a fixed-track submission whose `config.json`, once resolved as the evaluator resolves it, differs from the input model's only in the keys that are free to differ. The raw files differ in more keys.

## Contents

| Path | Purpose |
| --- | --- |
| `olmo2_1b.json` | Four configs for `allenai/OLMo-2-0425-1B`. |
| `gemma3_1b.json` | Four configs for `unsloth/gemma-3-1b-pt`. |

## Keys in each file

| Key | Content |
| --- | --- |
| `input_model_raw` | the input model's `config.json` as published |
| `input_model` | that config as the evaluator resolves it |
| `honest_save_raw` | a submission's `config.json` for the same model, with a `quantization_config` added |
| `honest_save` | that config resolved the same way |

`tests/validator/evaluation/test_quant_config_check.py` loads them. The resolved pairs must pass the drift check.
