# Core models

Pydantic models that cross runtime boundaries. Validator-only schemas live in `validator/`, trainer-only schemas in `trainer/`.

## Contents

| Path | Purpose |
| --- | --- |
| `dataset_models.py` | Dataset format enums, dataset type schemas and image/text pair schemas. |
| `image_models.py` | Image model type enum and FLUX checkpoint selection. |
| `model_prep_models.py` | Baseline stats and augmentation schemas produced by model prep. |
| `payload_models.py` | API payloads exchanged by miner, trainer and validator. |
| `pvp_models.py` | Input and result models for player-versus-player environment evaluation. |
| `quant_artifact_models.py` | Freeform decode request and result, the decoded model index and the miner-written `quant_manifest.json`. |
| `quant_models.py` | Compression formats and fixed-format specs, the miner target and freeform penalty, calibration-set models, the eval GPU record (`QuantEvalGpu`) and the bf16 reference score (`QuantReferenceScore`). |
| `reward_models.py` | Reward function schemas used by GRPO tasks and model prep. |
| `task_models.py` | Task type and status enums and task result schemas. |
| `tournament_models.py` | Tournament type enum shared by miner and validator. |
| `trainer_contract_models.py` | Trainer service responses read by validator orchestration. |
