"""Changed-region masks for image-edit objective evaluation.

The source image is part of an edit model's input, so unchanged pixels are
almost free to predict and dominate an unweighted loss. A mask built from the
held-out source/target difference weights the edited area more, while a floor
weight keeps penalising candidates that damage the rest of the image.
"""

import io

import numpy as np
from PIL import Image
from PIL import ImageFilter


MASK_WORK_LONG_SIDE = 256
MASK_DIFF_BLUR_RADIUS = 2
MASK_CHANGE_THRESHOLD = 0.05
MASK_DILATE_SIZE = 9
MASK_MIN_COVERAGE = 0.005
MASK_MAX_COVERAGE = 0.85
MASK_UNCHANGED_WEIGHT = 0.2


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


def mask_weights(mask: Image.Image) -> np.ndarray:
    """Map a 0-255 mask to loss weights in [MASK_UNCHANGED_WEIGHT, 1]."""
    values = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return MASK_UNCHANGED_WEIGHT + values * (1.0 - MASK_UNCHANGED_WEIGHT)
