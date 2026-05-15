"""
Model augmentation operations: layer selection and weight modification.
All operations are deterministic given the same seed.
"""

import re
import random
from collections import defaultdict

import torch
import numpy as np

from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType


def select_target_layers(
    named_params: list[str],
    scope: AugmentationScope,
    seed: int,
) -> list[str]:
    """Select which layers to augment based on scope and seed."""
    rng = random.Random(seed)

    weight_layers = [
        n for n in named_params
        if "weight" in n and "norm" not in n.lower() and "embed" not in n.lower()
    ]

    if not weight_layers:
        return named_params

    if scope == AugmentationScope.SINGLE_LAYER:
        return [rng.choice(weight_layers)]

    elif scope == AugmentationScope.LAYER_TYPE_GROUP:
        type_groups: dict[str, list[str]] = defaultdict(list)
        for name in weight_layers:
            match = re.search(r"\.(\w+)\.weight$", name)
            if match:
                type_groups[match.group(1)].append(name)

        if type_groups:
            chosen_type = rng.choice(list(type_groups.keys()))
            return type_groups[chosen_type]
        return [rng.choice(weight_layers)]

    elif scope == AugmentationScope.MULTI_LAYER:
        fraction = rng.uniform(0.25, 0.75)
        count = max(1, int(len(weight_layers) * fraction))
        return rng.sample(weight_layers, count)

    elif scope == AugmentationScope.ALL_LAYERS:
        return weight_layers

    return weight_layers


def apply_augmentation(
    param: torch.Tensor,
    aug_type: AugmentationType,
    intensity: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Apply augmentation to a parameter tensor in-place. Returns the modified tensor."""
    dtype = param.dtype
    device = param.device
    data = param.detach().float()

    if aug_type == AugmentationType.GAUSSIAN_NOISE:
        std = intensity * data.std().item()
        noise = torch.from_numpy(
            rng.normal(0, std, size=data.shape).astype(np.float32)
        ).to(device=device)
        data = data + noise
        del noise

    elif aug_type == AugmentationType.WEIGHT_SCALING:
        data = data * intensity

    elif aug_type == AugmentationType.MAGNITUDE_PRUNING:
        threshold = torch.quantile(data.abs().flatten(), intensity).item()
        data = data * (data.abs() >= threshold)

    elif aug_type == AugmentationType.LAYER_REINIT:
        mean_val = data.mean().item()
        std_val = data.std().item()
        mask = torch.from_numpy(
            (rng.random(size=data.shape) < intensity)
        ).to(device=device)
        reinit = torch.from_numpy(
            rng.normal(mean_val, std_val, size=data.shape).astype(np.float32)
        ).to(device=device)
        data = torch.where(mask, reinit, data)
        del mask, reinit

    return data.to(dtype=dtype)


def augment_model(model, config: AugmentationConfig) -> None:
    """Apply augmentation to a loaded model in-place."""
    rng = np.random.default_rng(config.seed)

    all_param_names = [name for name, _ in model.named_parameters()]
    target_layers = select_target_layers(all_param_names, config.scope, config.seed)

    print(f"Augmenting {len(target_layers)} layers with {config.aug_type.value} "
          f"(scope={config.scope.value}, intensity={config.intensity:.4f}, seed={config.seed})")

    for name, param in model.named_parameters():
        if name in target_layers:
            param.data = apply_augmentation(param.data, config.aug_type, config.intensity, rng)
            print(f"  Augmented: {name}")
