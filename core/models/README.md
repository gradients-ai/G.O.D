# Core Models

Pydantic models that cross runtime boundaries or are shared by multiple services.

## Contents

- `model_prep_models.py`: baseline stats and augmentation model-prep schemas.
- `payload_models.py`: API payloads shared by miner, trainer, and validator services.
- `pvp_models.py`: PvP evaluation rows, metadata, and result models.
- `utility_models.py`: shared enums and small utility schemas such as task types, file formats, image model types, and reward functions.

Validator-only schemas should live in `validator/`; trainer-only schemas should live in `trainer/`.
