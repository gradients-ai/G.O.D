"""Training-mode selection for synthetic image tasks."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from core.models.dataset_models import ImageTextPair
from core.models.image_models import ImageModelType
from core.models.image_models import ImageTrainingMode
from core.models.payload_models import ImageModelInfo
from validator.tasks.synthetics import diffusion


async def _models(model_type: ImageModelType):
    yield ImageModelInfo(model_id=f"owner/{model_type.value}", model_type=model_type)


def _pairs(count: int) -> list[ImageTextPair]:
    return [
        ImageTextPair(image_url=f"https://example.com/{index}.png", text_url=f"https://example.com/{index}.txt")
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_create_synthetic_image_task_honors_forced_training_mode(monkeypatch):
    pairs = _pairs(diffusion.synth_cst.MIN_IMAGE_SYNTH_PAIRS)
    monkeypatch.setattr(diffusion, "generate_edit_synthetic", AsyncMock(return_value=([], "edit", None)))
    monkeypatch.setattr(
        diffusion,
        "generate_image_synthetic_by_category",
        AsyncMock(return_value=(pairs, "t2i", "trigger")),
    )
    monkeypatch.setattr(diffusion, "pick_image_synth_category", lambda: "person")
    monkeypatch.setattr(diffusion, "maybe_get_augmentation_config", lambda _task_type: None)
    monkeypatch.setattr(diffusion, "add_task", AsyncMock(side_effect=lambda task, _db: task))
    monkeypatch.setattr(diffusion.random, "randint", lambda _a, _b: diffusion.synth_cst.MIN_IMAGE_SYNTH_PAIRS)

    with patch.object(diffusion.Path, "mkdir", return_value=None):
        task = await diffusion.create_synthetic_image_task(
            MagicMock(),
            _models(ImageModelType.FLUX2_KLEIN_4B),
            training_mode=ImageTrainingMode.TEXT_TO_IMAGE,
        )

    assert task.training_mode == ImageTrainingMode.TEXT_TO_IMAGE
    diffusion.generate_edit_synthetic.assert_not_awaited()
    diffusion.generate_image_synthetic_by_category.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_synthetic_image_task_rejects_unsupported_forced_mode():
    with pytest.raises(ValueError, match="does not support text-to-image"):
        await diffusion.create_synthetic_image_task(
            MagicMock(),
            _models(ImageModelType.QWEN_IMAGE_EDIT_2511),
            training_mode=ImageTrainingMode.TEXT_TO_IMAGE,
        )


def test_pick_image_training_mode_edit_only_models():
    assert diffusion.pick_image_training_mode(ImageModelType.QWEN_IMAGE_EDIT_2511) == ImageTrainingMode.EDIT
