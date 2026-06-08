"""
Pure functions for deciding augmentation config.
No heavy dependencies — only stdlib, core models, and validator constants.
"""

import random
from typing import TypeVar

import validator.tasks.prep.constants as vcst
from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from core.models.task_models import TaskType


T = TypeVar("T", AugmentationType, AugmentationScope)


def weighted_choice(weights: dict[T, float], rng: random.Random) -> T:
    """Pick an enum member from a weighted dict, normalising weights at runtime."""
    keys = list(weights.keys())
    vals = list(weights.values())
    total = sum(vals)
    normalised = [v / total for v in vals]
    return rng.choices(keys, weights=normalised, k=1)[0]


def seeded_intensity(aug_type: AugmentationType, rng: random.Random) -> float:
    """Return a random intensity for each augmentation type, driven by the seeded RNG."""
    low, high = vcst.AUGMENTATION_INTENSITY_RANGES.get(aug_type, (0.01, 0.01))
    return rng.uniform(low, high)


def maybe_get_augmentation_config(task_type: TaskType) -> AugmentationConfig | None:
    """Randomly decide whether to augment a model and return the full config.

    All random choices after the initial coin flip are driven by a single seed,
    so the config is fully reproducible from {seed}.
    """
    if task_type == TaskType.IMAGETASK and not vcst.AUGMENTATION_ENABLED_IMAGE:
        return None
    elif task_type == TaskType.ENVIRONMENTTASK and not vcst.AUGMENTATION_ENABLED_ENV:
        return None
    elif task_type not in (TaskType.IMAGETASK, TaskType.ENVIRONMENTTASK) and not vcst.AUGMENTATION_ENABLED_TEXT:
        return None

    if random.random() >= vcst.AUGMENTATION_PROBABILITY:
        return None

    seed = random.randint(0, 2**32 - 1)
    rng = random.Random(seed)

    aug_type: AugmentationType = weighted_choice(vcst.AUGMENTATION_TYPE_WEIGHTS, rng)
    scope: AugmentationScope = weighted_choice(vcst.AUGMENTATION_SCOPE_WEIGHTS, rng)
    # LAYER_REINIT and WEIGHT_SCALING are the most damaging ops; constrain them to
    # MULTI_LAYER scope so they never hit a single isolated layer or every layer at once.
    if aug_type in (AugmentationType.LAYER_REINIT, AugmentationType.WEIGHT_SCALING):
        scope = AugmentationScope.MULTI_LAYER
    intensity = seeded_intensity(aug_type, rng)

    return AugmentationConfig(
        aug_type=aug_type,
        scope=scope,
        seed=seed,
        intensity=intensity,
    )
