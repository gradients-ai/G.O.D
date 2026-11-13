# Core Models

Pydantic models that cross runtime boundaries or are shared by multiple services.

## Contents

- `model_prep_models.py`: baseline stats and augmentation model-prep schemas.
- `quant_models.py`: quant-task fixed-format specs (`QuantFixedSpec`), the fixed-format rotation, the eval GPU
  record (`QuantEvalGpu`: what the container read plus the matched `QuantEvalSku` label, or `override`; its
  `reference_key` keys the per-SKU reference cache) and `QuantReferenceScore`, the bf16 reference scored in the
  same deployment as a submission (on `EvaluationResultText.quant_reference`).
- `quant_artifact_models.py`: freeform decode contract between validator and trainer: `QuantDecodeRequest` (on
  `TrainRequestQuant.decode`), `QuantDecodeResult` (on `TrainerTaskLog.quant_decode`; exactly one of `ok`,
  `quant_failure`, `infra_error`, plus an informational `build_warning`) and the uploaded `index.json` (`QuantDecodedIndex`); and the miner-written
  `quant_manifest.json` (`QuantManifest`, `QuantManifestFile`, `track_for`) the trainer checks before upload.
- `payload_models.py`: API payloads shared by miner, trainer, and validator services.
- `dataset_models.py`: dataset format enums, dataset-type schemas, and image/text pair schemas.
- `image_models.py`: shared image model type enum.
- `reward_models.py`: reward function schemas used by GRPO tasks and model prep.
- `task_models.py`: shared task type/status enums and task result schemas.
- `tournament_models.py`: shared miner-validator tournament type enum.
- `trainer_contract_models.py`: trainer service response models consumed by validator orchestration.

Validator-only schemas should live in `validator/`; trainer-only schemas should live in `trainer/`.
