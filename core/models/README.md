# Core Models

Pydantic models that cross runtime boundaries or are shared by multiple services.

## Contents

- `model_prep_models.py`: baseline stats and augmentation model-prep schemas.
- `payload_models.py`: API payloads shared by miner, trainer, and validator services.
- `dataset_models.py`: dataset format enums, dataset-type schemas, and image/text pair schemas.
- `image_models.py`: shared image model type enum.
- `reward_models.py`: reward function schemas used by GRPO tasks and model prep.
- `task_models.py`: shared task type/status enums and task result schemas.
- `tournament_models.py`: shared miner-validator tournament type enum.
- `trainer_contract_models.py`: trainer service response models consumed by validator orchestration.

Validator-only schemas should live in `validator/`; trainer-only schemas should live in `trainer/`.
