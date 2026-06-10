import re

from fiber import Keypair
from huggingface_hub import HfApi

import validator.tasks.datasets.constants as cst
from core.logging import get_logger
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import DpoDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import GrpoDatasetType
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.task_models import TaskStatus
from validator.tasks.datasets.preparation import prepare_image_task
from validator.tasks.datasets.preparation import prepare_text_task
from validator.tasks.models import AnyTextTypeRawTask
from validator.tasks.models import ChatRawTask
from validator.tasks.models import DpoRawTask
from validator.tasks.models import GrpoRawTask
from validator.tasks.models import ImageRawTask
from validator.tasks.models import InstructTextRawTask
from validator.tasks.prep.yarn import prepare_yarn_extended_model


logger = get_logger(__name__)
hf_api = HfApi()


async def get_fake_text_dataset_size(task: AnyTextTypeRawTask) -> int:
    return 100_000


def get_model_num_params(model_id: str) -> int:
    try:
        model_info = hf_api.model_info(model_id)
        size = model_info.safetensors.total
        return size
    except Exception as e:
        logger.warning(f"Error getting model size from safetensors: {e}")
        model_size = re.search(r"(\d+)(?=[bB])", model_id)
        model_size = int(model_size.group(1)) * 1_000_000_000 if model_size else None
        logger.info(f"Model size from regex: {model_size}")
        return model_size


async def get_total_image_dataset_size(task: ImageRawTask) -> int:
    if task.image_text_pairs:
        return len(task.image_text_pairs)
    # Zip-backed image tasks are validated and counted during task prep.
    return 0


async def run_image_task_prep(task: ImageRawTask, keypair: Keypair, psql_db=None) -> ImageRawTask:
    test_url, train_url = await prepare_image_task(
        image_text_pairs=task.image_text_pairs,
        dataset_zip_url=task.ds if not task.image_text_pairs else None,
    )
    task.training_data = train_url
    task.test_data = test_url
    task.status = TaskStatus.LOOKING_FOR_NODES
    logger.info(
        "Data creation is complete - now time to find some miners",
    )
    return task


async def run_text_task_prep(task: AnyTextTypeRawTask, keypair: Keypair, psql_db=None) -> AnyTextTypeRawTask:
    if task.yarn_factor is not None:
        logger.info(f"Applying YaRN extension with factor {task.yarn_factor} to model {task.model_id}")
        try:
            yarn_model_id = await prepare_yarn_extended_model(task.model_id, task.yarn_factor)
            task.model_id = yarn_model_id
            logger.info(f"YaRN extension applied, new model_id: {yarn_model_id}")
        except Exception as e:
            logger.error(f"Failed to apply YaRN extension: {e}, using original model")
            # Clear yarn_factor to indicate extension was not applied
            task.yarn_factor = None

    test_data, train_data = await prepare_text_task(task, keypair=keypair, psql_db=psql_db)
    task.training_data = train_data
    task.status = TaskStatus.LOOKING_FOR_NODES
    task.synthetic_data = None  # TODO: remove from db schema
    task.test_data = test_data

    if isinstance(task, InstructTextRawTask):
        task.field_instruction = cst.STANDARD_INSTRUCT_COLUMN
        task.field_output = cst.STANDARD_OUTPUT_COLUMN
        task.field_input = cst.STANDARD_INPUT_COLUMN if task.field_input else None
        task.field_system = cst.STANDARD_SYSTEM_COLUMN if task.field_system else None
    elif isinstance(task, DpoRawTask):
        task.field_prompt = cst.STANDARD_DPO_PROMPT_COLUMN
        task.field_chosen = cst.STANDARD_DPO_CHOSEN_COLUMN
        task.field_rejected = cst.STANDARD_DPO_REJECTED_COLUMN
        task.field_system = cst.STANDARD_SYSTEM_COLUMN if task.field_system else None
    elif isinstance(task, GrpoRawTask):
        task.field_prompt = cst.STANDARD_GRPO_PROMPT_COLUMN

    logger.info("Data creation is complete - now time to find some miners")
    return task


def prepare_text_task_request(task: AnyTextTypeRawTask) -> TrainRequestText:
    if isinstance(task, InstructTextRawTask):
        dataset_type = InstructTextDatasetType(
            field_system=task.field_system,
            field_input=task.field_input,
            field_output=task.field_output,
            field_instruction=task.field_instruction,
            format=task.format,
            no_input_format=task.no_input_format,
        )
    elif isinstance(task, DpoRawTask):
        dataset_type = DpoDatasetType(
            field_prompt=task.field_prompt,
            field_system=task.field_system,
            field_chosen=task.field_chosen,
            field_rejected=task.field_rejected,
            prompt_format=task.prompt_format,
            chosen_format=task.chosen_format,
            rejected_format=task.rejected_format,
        )
    elif isinstance(task, GrpoRawTask):
        dataset_type = GrpoDatasetType(
            field_prompt=task.field_prompt,
            reward_functions=task.reward_functions,
            extra_column=task.extra_column,
        )
    elif isinstance(task, ChatRawTask):
        dataset_type = ChatTemplateDatasetType(
            chat_template=task.chat_template,
            chat_column=task.chat_column,
            chat_role_field=task.chat_role_field,
            chat_content_field=task.chat_content_field,
            chat_user_reference=task.chat_user_reference,
            chat_assistant_reference=task.chat_assistant_reference,
        )

    dataset = task.training_data if task.training_data else "dataset error"
    use_kl, kl_coef = (task.use_kl, task.kl_coef) if isinstance(task, InstructTextRawTask) else (False, None)
    task_request_body = TrainRequestText(
        dataset=dataset,
        model=task.model_id,
        dataset_type=dataset_type,
        file_format=FileFormat.S3,
        task_id=str(task.task_id),
        hours_to_complete=task.hours_to_complete,
        use_kl=use_kl,
        kl_coef=kl_coef,
    )

    return task_request_body


def prepare_image_task_request(task: ImageRawTask) -> TrainRequestImage:
    return TrainRequestImage(
        model=task.model_id,
        task_id=str(task.task_id),
        hours_to_complete=task.hours_to_complete,
        dataset_zip=task.training_data,
        model_type=task.model_type,
        trigger_word=task.trigger_word,
    )
