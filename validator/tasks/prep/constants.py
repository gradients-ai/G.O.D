from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from core.models.task_models import TaskType


BASELINE_STATS_ENABLED_ORGANIC = False
MODEL_PREP_ENABLED_TEXT = True
MODEL_PREP_ENABLED_IMAGE = False
MODEL_PREP_ENABLED_ENV = False
MODEL_PREP_ENABLED_BY_TASK_TYPE: dict[TaskType, bool] = {
    TaskType.INSTRUCTTEXTTASK: MODEL_PREP_ENABLED_TEXT,
    TaskType.DPOTASK: MODEL_PREP_ENABLED_TEXT,
    TaskType.GRPOTASK: MODEL_PREP_ENABLED_TEXT,
    TaskType.CHATTASK: MODEL_PREP_ENABLED_TEXT,
    TaskType.IMAGETASK: MODEL_PREP_ENABLED_IMAGE,
    TaskType.ENVIRONMENTTASK: MODEL_PREP_ENABLED_ENV,
    TaskType.QUANTFIXEDTASK: MODEL_PREP_ENABLED_TEXT,
    TaskType.QUANTFREEFORMTASK: MODEL_PREP_ENABLED_TEXT,
}

YARN_EXTENSION_PROBABILITY = 0.0
YARN_TOURNAMENT_FACTORS = [2, 4]
MODEL_COPY_ENDPOINT = "https://huggingface.co/api/models/{source_repo}/duplicate"

AUGMENTATION_ENABLED_TEXT = True
AUGMENTATION_ENABLED_IMAGE = False
AUGMENTATION_ENABLED_ENV = False
AUGMENTATION_PROBABILITY = 0.5
# Skip publishing an augmented copy once the base is this large: model-prep would
# upload a full 35B–70B repo and regularly blow the 90-minute prep timeout.
AUGMENTATION_SKIP_MIN_SIZE_B = 35.0

AUGMENTATION_TYPE_WEIGHTS: dict[AugmentationType, float] = {
    AugmentationType.GAUSSIAN_NOISE: 0.20,
    AugmentationType.WEIGHT_SCALING: 0.40,
    AugmentationType.MAGNITUDE_PRUNING: 0.25,
    AugmentationType.LAYER_REINIT: 0.15,
}

AUGMENTATION_SCOPE_WEIGHTS: dict[AugmentationScope, float] = {
    AugmentationScope.SINGLE_LAYER: 0.10,
    AugmentationScope.LAYER_TYPE_GROUP: 0.15,
    AugmentationScope.MULTI_LAYER: 0.35,
    AugmentationScope.ALL_LAYERS: 0.40,
}

AUGMENTATION_INTENSITY_RANGES: dict[AugmentationType, tuple[float, float]] = {
    AugmentationType.GAUSSIAN_NOISE: (0.01, 0.3),
    AugmentationType.WEIGHT_SCALING: (0.5, 1.5),
    AugmentationType.MAGNITUDE_PRUNING: (0.25, 0.50),
    AugmentationType.LAYER_REINIT: (0.01, 0.10),
}

# Quant prep tokenises the held-out split once into the scorer's test set (validator/tasks/quant_eval_set.py).
# Longer examples are dropped: the scorer's max_model_len is the longest prompt + 1, so this bounds H100 memory.
QUANT_EVAL_MAX_SEQUENCE_LEN = 4096  # also capped at the model context - 1 (quant_eval_max_sequence_len)
# Fewer surviving examples fail prep (PREP_TASK_FAILURE -> the tournament replaces the task): on a tiny
# test set the freeform bar and the dead-zone margins are noise.
QUANT_EVAL_MIN_EXAMPLES = 100
# Quant prep drops calibration examples equal to an eval example; fewer than this survive and prep fails
# (CalibrationUnavailableError: a compression task is never handed to miners without a calibration set).
QUANT_CALIBRATION_MIN_EXAMPLES = 100
# Downloading, tokenising and uploading the calibration set is attempted this many times before prep fails; the
# wait between attempts is QUANT_CALIBRATION_PUBLISH_BACKOFF_SECONDS times the attempt number.
QUANT_CALIBRATION_PUBLISH_ATTEMPTS = 3
QUANT_CALIBRATION_PUBLISH_BACKOFF_SECONDS = 5.0
