from typing import Callable

from pydantic import BaseModel
from pydantic import Field

from core.models.utility_models import TaskType
from validator.core import constants as cst
from validator.core.models import AnyTypeRawTask
from validator.core.models import ChatRawTask
from validator.core.models import DpoRawTask
from validator.core.models import GrpoRawTask
from validator.core.models import EnvRawTask
from validator.core.models import ImageRawTask
from validator.core.models import InstructTextRawTask
from validator.cycle.util_functions import get_fake_text_dataset_size
from validator.cycle.util_functions import get_total_image_dataset_size
from validator.cycle.util_functions import prepare_image_task_request
from validator.cycle.util_functions import prepare_text_task_request
from validator.cycle.util_functions import run_image_task_prep
from validator.cycle.util_functions import run_text_task_prep


# TODO
# being lazy here with everything as callable, can we look at the signatures and use the same for the diff
# data types
class TaskConfig(BaseModel):
    task_type: TaskType = Field(..., description="The type of task.")
    data_size_function: Callable = Field(..., description="The function used to determine the dataset size")
    task_prep_function: Callable = Field(
        ..., description="What we call in order to do the prep work - train test split and whatnot"
    )

    task_request_prepare_function: Callable = Field(..., description="Namoray will come up with a better var name for sure")
    start_training_endpoint: str = Field(..., description="The endpoint to start training")


class ImageTaskConfig(TaskConfig):
    task_type: TaskType = TaskType.IMAGETASK
    data_size_function: Callable = get_total_image_dataset_size
    task_prep_function: Callable = run_image_task_prep
    task_request_prepare_function: Callable = prepare_image_task_request
    start_training_endpoint: str = cst.START_TRAINING_IMAGE_ENDPOINT


class InstructTextTaskConfig(TaskConfig):
    task_type: TaskType = TaskType.INSTRUCTTEXTTASK
    data_size_function: Callable = get_fake_text_dataset_size
    task_prep_function: Callable = run_text_task_prep
    task_request_prepare_function: Callable = prepare_text_task_request
    start_training_endpoint: str = cst.START_TRAINING_ENDPOINT


class DpoTaskConfig(TaskConfig):
    task_type: TaskType = TaskType.DPOTASK
    data_size_function: Callable = get_fake_text_dataset_size
    task_prep_function: Callable = run_text_task_prep
    task_request_prepare_function: Callable = prepare_text_task_request
    start_training_endpoint: str = cst.START_TRAINING_ENDPOINT


class GrpoTaskConfig(TaskConfig):
    task_type: TaskType = TaskType.GRPOTASK
    data_size_function: Callable = get_fake_text_dataset_size
    task_prep_function: Callable = run_text_task_prep
    task_request_prepare_function: Callable = prepare_text_task_request
    start_training_endpoint: str = cst.START_TRAINING_GRPO_ENDPOINT


class EnvTaskConfig(TaskConfig):
    task_type: TaskType = TaskType.ENVIRONMENTTASK
    data_size_function: Callable = get_fake_text_dataset_size
    task_prep_function: Callable = run_text_task_prep
    task_request_prepare_function: Callable = prepare_text_task_request
    start_training_endpoint: str = cst.START_TRAINING_GRPO_ENDPOINT


class ChatTaskConfig(TaskConfig):
    task_type: TaskType = TaskType.CHATTASK
    data_size_function: Callable = get_fake_text_dataset_size
    task_prep_function: Callable = run_text_task_prep
    task_request_prepare_function: Callable = prepare_text_task_request
    start_training_endpoint: str = cst.START_TRAINING_ENDPOINT


def get_task_config(task: AnyTypeRawTask) -> TaskConfig:
    if isinstance(task, InstructTextRawTask):
        return InstructTextTaskConfig()
    elif isinstance(task, ImageRawTask):
        return ImageTaskConfig()
    elif isinstance(task, DpoRawTask):
        return DpoTaskConfig()
    elif isinstance(task, GrpoRawTask):
        return GrpoTaskConfig()
    elif isinstance(task, ChatRawTask):
        return ChatTaskConfig()
    elif isinstance(task, EnvRawTask):
        return EnvTaskConfig()
    else:
        raise ValueError(f"Unsupported task type: {type(task).__name__}")
