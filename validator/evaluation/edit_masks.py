"""Changed-region masks for image-edit objective evaluation.

The source image is part of an edit model's input, so unchanged pixels are
almost free to predict and, unweighted, the small edited area is lost in them.
A mask built from the held-out source/target difference splits the loss so the
edited area and the rest of the image carry equal weight mass
(MASK_TARGET_CHANGED_MASS): a candidate must make the edit and must not damage
what it should leave alone. The changed weight stays 1.0; the unchanged weight
is solved per image from the mask's coverage.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from PIL import ImageFilter


if TYPE_CHECKING:
    import torch


MASK_WORK_LONG_SIDE = 256
MASK_DIFF_BLUR_RADIUS = 2
MASK_CHANGE_THRESHOLD = 0.05
MASK_DILATE_SIZE = 9
MASK_MIN_COVERAGE = 0.005
MASK_MAX_COVERAGE = 0.85
# Cap on the unchanged-pixel weight: never above the changed weight.
MASK_UNCHANGED_WEIGHT = 1.0
MASK_UNCHANGED_WEIGHT_MIN = 0.005
# Share of total weight mass on the changed region: half, so edit quality and preserving the rest count equally.
MASK_TARGET_CHANGED_MASS = 0.5
# Soft mask above this counts as "changed" for region-split metrics.
MASK_REGION_THRESHOLD = 0.5


def _work_size(width: int, height: int) -> tuple[int, int]:
    scale = MASK_WORK_LONG_SIDE / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def build_edit_mask(source: bytes, target: bytes) -> Image.Image | None:
    """Return an L-mode mask at the target's size, or None when the edit is global or absent."""
    target_image = Image.open(io.BytesIO(target)).convert("RGB")
    source_image = Image.open(io.BytesIO(source)).convert("RGB")
    work_size = _work_size(*target_image.size)
    target_work = np.asarray(target_image.resize(work_size, Image.BICUBIC), dtype=np.float32) / 255.0
    source_work = np.asarray(source_image.resize(work_size, Image.BICUBIC), dtype=np.float32) / 255.0

    difference = np.abs(target_work - source_work).mean(axis=2)
    difference_image = Image.fromarray(np.round(difference * 255.0).astype(np.uint8), mode="L")
    difference_image = difference_image.filter(ImageFilter.GaussianBlur(radius=MASK_DIFF_BLUR_RADIUS))
    changed = np.asarray(difference_image, dtype=np.float32) / 255.0 > MASK_CHANGE_THRESHOLD

    changed_image = Image.fromarray(changed.astype(np.uint8) * 255, mode="L")
    changed_image = changed_image.filter(ImageFilter.MaxFilter(MASK_DILATE_SIZE))
    coverage = float(np.asarray(changed_image, dtype=np.float32).mean() / 255.0)
    if coverage < MASK_MIN_COVERAGE or coverage > MASK_MAX_COVERAGE:
        return None
    return changed_image.resize(target_image.size, Image.BILINEAR)


def resize_and_crop(mask: Image.Image, scale_size: tuple[int, int], crop_box: tuple[int, int, int, int]) -> Image.Image:
    return mask.resize(scale_size, Image.BILINEAR).crop(crop_box)


def cover_and_center_crop(mask: Image.Image, width: int, height: int) -> Image.Image:
    """Match Musubi's bucket transform: scale to cover the bucket, then centre-crop."""
    scale = max(width / mask.width, height / mask.height)
    scaled_width = int(mask.width * scale + 0.5)
    scaled_height = int(mask.height * scale + 0.5)
    left = (scaled_width - width) // 2
    top = (scaled_height - height) // 2
    return resize_and_crop(mask, (scaled_width, scaled_height), (left, top, left + width, top + height))


def mask_as_float(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask.convert("L"), dtype=np.float32) / 255.0


def unchanged_weight_for_coverage(coverage: float) -> float:
    """Unchanged-pixel weight so changed pixels own ~MASK_TARGET_CHANGED_MASS of mass.

    With changed weight fixed at 1 and soft coverage ``c = mean(mask)``:
    ``c / (c + (1 - c) * w_lo) ≈ MASK_TARGET_CHANGED_MASS``, solved for ``w_lo``
    and clipped to ``[MASK_UNCHANGED_WEIGHT_MIN, MASK_UNCHANGED_WEIGHT]``.
    """
    coverage = float(np.clip(coverage, 0.0, 1.0))
    if coverage >= 1.0 - 1e-6:
        return MASK_UNCHANGED_WEIGHT
    if coverage <= 0.0:
        return MASK_UNCHANGED_WEIGHT
    alpha = MASK_TARGET_CHANGED_MASS
    weight = (coverage / (1.0 - coverage)) * ((1.0 - alpha) / alpha)
    return float(np.clip(weight, MASK_UNCHANGED_WEIGHT_MIN, MASK_UNCHANGED_WEIGHT))


def mask_weights(mask: Image.Image) -> np.ndarray:
    """Map a 0-255 mask to loss weights in [unchanged_weight, 1].

    The unchanged weight is set from the mask's coverage so the changed and
    unchanged regions each carry about half of the weight mass.
    """
    values = mask_as_float(mask)
    floor = unchanged_weight_for_coverage(float(values.mean()))
    return floor + values * (1.0 - floor)


def region_mean_losses(
    per_pixel_loss: "torch.Tensor",
    mask_01: "torch.Tensor",
    threshold: float = MASK_REGION_THRESHOLD,
) -> tuple[float | None, float | None, float]:
    """Unweighted mean loss on changed vs unchanged pixels.

    ``per_pixel_loss`` and ``mask_01`` must be broadcastable to the same spatial
    shape (typically ``[B, 1, H, W]``). Returns ``(changed, unchanged, coverage)``;
    a side is ``None`` when that region is empty.
    """
    import torch

    if per_pixel_loss.ndim == 4 and per_pixel_loss.shape[1] > 1:
        per_pixel_loss = per_pixel_loss.mean(dim=1, keepdim=True)
    mask = mask_01.to(device=per_pixel_loss.device, dtype=per_pixel_loss.dtype)
    if mask.shape[-2:] != per_pixel_loss.shape[-2:]:
        mask = torch.nn.functional.interpolate(mask, size=per_pixel_loss.shape[-2:], mode="nearest")
    if mask.shape[1] != 1 and mask.shape[1] == per_pixel_loss.shape[1]:
        mask = mask[:, :1]
    changed = mask >= threshold
    unchanged = ~changed
    coverage = float(changed.float().mean().item())
    changed_loss = float(per_pixel_loss[changed].mean().item()) if bool(changed.any()) else None
    unchanged_loss = float(per_pixel_loss[unchanged].mean().item()) if bool(unchanged.any()) else None
    return changed_loss, unchanged_loss, coverage
