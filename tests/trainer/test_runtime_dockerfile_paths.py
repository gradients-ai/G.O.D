from types import SimpleNamespace

import pytest

from core.models.image_models import ImageModelType
from core.models.task_models import TaskType
from trainer.runtime import get_dockerfile_path


def test_get_dockerfile_path_prefers_reorganized_text_path(tmp_path):
    preferred = tmp_path / "ops/docker/standalone-text-trainer.dockerfile"
    legacy = tmp_path / "dockerfiles/standalone-text-trainer.dockerfile"
    preferred.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    preferred.touch()
    legacy.touch()

    assert get_dockerfile_path(TaskType.INSTRUCTTEXTTASK, SimpleNamespace(), str(tmp_path)) == str(preferred)


def test_get_dockerfile_path_supports_legacy_text_path(tmp_path):
    legacy = tmp_path / "dockerfiles/standalone-text-trainer.dockerfile"
    legacy.parent.mkdir(parents=True)
    legacy.touch()

    assert get_dockerfile_path(TaskType.GRPOTASK, SimpleNamespace(), str(tmp_path)) == str(legacy)


def test_get_dockerfile_path_supports_legacy_image_path(tmp_path):
    legacy = tmp_path / "dockerfiles/standalone-image-trainer.dockerfile"
    legacy.parent.mkdir(parents=True)
    legacy.touch()

    training_data = SimpleNamespace(model_type=ImageModelType.SDXL)

    assert get_dockerfile_path(TaskType.IMAGETASK, training_data, str(tmp_path)) == str(legacy)


def test_get_dockerfile_path_supports_legacy_image_toolkit_path(tmp_path):
    legacy = tmp_path / "dockerfiles/standalone-image-toolkit-trainer.dockerfile"
    legacy.parent.mkdir(parents=True)
    legacy.touch()

    training_data = SimpleNamespace(model_type=ImageModelType.QWEN_IMAGE)

    assert get_dockerfile_path(TaskType.IMAGETASK, training_data, str(tmp_path)) == str(legacy)


def test_get_dockerfile_path_errors_when_no_supported_path_exists(tmp_path):
    with pytest.raises(FileNotFoundError, match="ops/docker/standalone-text-trainer.dockerfile"):
        get_dockerfile_path(TaskType.DPOTASK, SimpleNamespace(), str(tmp_path))
