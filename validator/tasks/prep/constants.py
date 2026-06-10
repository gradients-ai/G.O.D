from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from core.models.task_models import TaskType


BASELINE_STATS_ENABLED_ORGANIC = False
MODEL_PREP_ENABLED_TEXT = True
MODEL_PREP_ENABLED_IMAGE = False
MODEL_PREP_ENABLED_ENV = True
MODEL_PREP_ENABLED_BY_TASK_TYPE: dict[TaskType, bool] = {
    TaskType.INSTRUCTTEXTTASK: MODEL_PREP_ENABLED_TEXT,
    TaskType.DPOTASK: MODEL_PREP_ENABLED_TEXT,
    TaskType.GRPOTASK: MODEL_PREP_ENABLED_TEXT,
    TaskType.CHATTASK: MODEL_PREP_ENABLED_TEXT,
    TaskType.IMAGETASK: MODEL_PREP_ENABLED_IMAGE,
    TaskType.ENVIRONMENTTASK: MODEL_PREP_ENABLED_ENV,
}

YARN_EXTENSION_PROBABILITY = 0.0
YARN_TOURNAMENT_FACTORS = [2, 4]
MODEL_COPY_ENDPOINT = "https://huggingface.co/api/models/{source_repo}/duplicate"

AUGMENTATION_ENABLED_TEXT = True
AUGMENTATION_ENABLED_IMAGE = False
AUGMENTATION_ENABLED_ENV = False
AUGMENTATION_PROBABILITY = 0.5

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
