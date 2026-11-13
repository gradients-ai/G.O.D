"""get_task_type routes quant requests by their own task_type before the text dataset_type branch."""

import pytest

from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import DpoDatasetType
from core.models.dataset_models import EnvironmentDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import GrpoDatasetType
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestQuant
from core.models.payload_models import TrainRequestText
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskType
from trainer.runtime import get_task_type


BASE = {"model": "org/model", "task_id": "task-1", "hours_to_complete": 2.0}
DATASET = {"dataset": "s3://bucket/train.json", "file_format": FileFormat.S3}


def _proxy(training_data) -> TrainerProxyRequest:
    return TrainerProxyRequest(training_data=training_data, github_repo="https://github.com/org/repo", gpu_ids=[0], hotkey="hk")


@pytest.mark.parametrize(
    ("task_type", "quant_spec"),
    [(TaskType.QUANTFIXEDTASK, QuantFormat.W4A16_SYM_G128.spec()), (TaskType.QUANTFREEFORMTASK, None)],
)
def test_quant_request_returns_its_task_type(task_type, quant_spec):
    training_data = TrainRequestQuant(
        **BASE, **DATASET, dataset_type=InstructTextDatasetType(), task_type=task_type, quant_spec=quant_spec
    )
    assert get_task_type(_proxy(training_data)) == task_type


@pytest.mark.parametrize(
    ("dataset_type", "task_type"),
    [
        (InstructTextDatasetType(field_instruction="instruction"), TaskType.INSTRUCTTEXTTASK),
        (DpoDatasetType(field_prompt="prompt"), TaskType.DPOTASK),
        (GrpoDatasetType(field_prompt="prompt"), TaskType.GRPOTASK),
        (ChatTemplateDatasetType(), TaskType.CHATTASK),
        (EnvironmentDatasetType(), TaskType.ENVIRONMENTTASK),
    ],
)
def test_text_request_returns_dataset_task_type(dataset_type, task_type):
    training_data = TrainRequestText(**BASE, **DATASET, dataset_type=dataset_type)
    assert get_task_type(_proxy(training_data)) == task_type


def test_image_request_returns_image_task_type():
    training_data = TrainRequestImage(**BASE, dataset_zip="s3://bucket/data.zip")
    assert get_task_type(_proxy(training_data)) == TaskType.IMAGETASK
