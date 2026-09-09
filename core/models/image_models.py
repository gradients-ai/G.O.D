import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class ImageModelType(str, Enum):
    FLUX = "flux"
    SDXL = "sdxl"
    Z_IMAGE = "z-image"
    QWEN_IMAGE = "qwen-image"
    IDEOGRAM4 = "ideogram4"
    KREA2 = "krea2"


class EvaluationResultImage(BaseModel):
    """Versioned image prediction L2; legacy pixel-reconstruction payloads are rejected."""

    eval_loss: float = Field(ge=0, allow_inf_nan=False)
    is_finetune: bool | None = None
    metric_version: Literal["image-denoising-l2-v1"]
    metric: Literal["flow_prediction_mse"]
    text_weight: Literal[0.5]
    text_guided_losses: list[float] = Field(min_length=1)
    no_text_losses: list[float] = Field(min_length=1)
    image_ids: list[str] = Field(min_length=1)
    eval_set_fingerprint: str = Field(min_length=64, max_length=64)
    strata: int = Field(ge=1)
    noises: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_prediction_loss(self):
        if len(self.text_guided_losses) != len(self.no_text_losses) or len(self.image_ids) != len(self.text_guided_losses):
            raise ValueError("Image loss vectors must be aligned")
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("Image identities must be unique")
        losses = self.text_guided_losses + self.no_text_losses
        if any(not math.isfinite(v) or v < 0 for v in losses):
            raise ValueError("Image losses must be finite and nonnegative")
        expected = 0.5 * (sum(self.text_guided_losses) + sum(self.no_text_losses)) / len(self.image_ids)
        if not math.isclose(self.eval_loss, expected, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("Image score differs from its 50/50 prediction losses")
        return self
