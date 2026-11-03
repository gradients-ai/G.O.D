from pathlib import Path

import pytest
import yaml

from core.models.image_models import ImageModelType
from core.models.image_models import ImageTrainingMode
from core.models.image_models import supports_image_training_mode
from core.models.payload_models import TrainRequestImage
from trainer import training_paths


@pytest.mark.parametrize(
    ("model_type", "training_mode", "template_name"),
    [
        ("flux2-klein-4b", "text-to-image", "base_diffusion_flux2_klein_4b.yaml"),
        ("flux2-klein-4b", "edit", "base_diffusion_flux2_klein_4b_edit.yaml"),
        ("qwen-image-edit-2511", "edit", "base_diffusion_qwen_image_edit_2511.yaml"),
        ("hidream-o1", "text-to-image", "base_diffusion_hidream_o1.yaml"),
        ("moss-image-base", "text-to-image", "base_diffusion_moss_image_base.yaml"),
        ("moss-image-base", "edit", "base_diffusion_moss_image_base_edit.yaml"),
    ],
)
def test_new_ai_toolkit_template_routing(monkeypatch, tmp_path, model_type, training_mode, template_name):
    monkeypatch.setattr(training_paths.train_cst, "IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH", str(tmp_path))

    result = training_paths.get_image_training_config_template_path(model_type, training_mode)

    assert result == str(tmp_path / template_name)


def test_edit_only_model_rejects_text_to_image_template():
    with pytest.raises(ValueError, match="Unsupported image training combination"):
        training_paths.get_image_training_config_template_path("qwen-image-edit-2511", "text-to-image")


def test_hidream_edit_uses_explicit_non_ai_toolkit_fallback():
    assert supports_image_training_mode(ImageModelType.HIDREAM_O1, ImageTrainingMode.EDIT)
    with pytest.raises(ValueError, match="Unsupported image training combination"):
        training_paths.get_image_training_config_template_path("hidream-o1", "edit")


def test_image_train_request_defaults_to_text_to_image():
    request = TrainRequestImage(
        task_id="task",
        model="owner/model",
        dataset_zip="https://example.com/data.zip",
        expected_repo_name="result",
        hours_to_complete=1,
    )

    assert request.training_mode == ImageTrainingMode.TEXT_TO_IMAGE
    assert Path(training_paths.get_image_training_config_template_path("flux")).name == "base_diffusion_flux.yaml"


def test_image_train_request_rejects_unsupported_model_mode():
    with pytest.raises(ValueError, match="does not support text-to-image"):
        TrainRequestImage(
            task_id="task",
            model="Qwen/Qwen-Image-Edit-2511",
            dataset_zip="https://example.com/data.zip",
            expected_repo_name="result",
            hours_to_complete=1,
            model_type=ImageModelType.QWEN_IMAGE_EDIT_2511,
        )


@pytest.mark.parametrize(
    "template_name",
    [
        "base_diffusion_flux2_klein_4b.yaml",
        "base_diffusion_flux2_klein_4b_edit.yaml",
        "base_diffusion_qwen_image_edit_2511.yaml",
        "base_diffusion_hidream_o1.yaml",
        "base_diffusion_moss_image_base.yaml",
        "base_diffusion_moss_image_base_edit.yaml",
    ],
)
def test_new_training_templates_are_valid_ai_toolkit_jobs(template_name):
    template_path = Path(__file__).parents[2] / "core" / "training_templates" / template_name
    config = yaml.safe_load(template_path.read_text())
    process = config["config"]["process"][0]

    assert config["job"] == "extension"
    assert process["type"] == "diffusion_trainer"
    assert process["network"]["type"] == "lora"
    assert process["model"]["arch"]
