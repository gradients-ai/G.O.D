import json
import math
import os
import tempfile

import httpx
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from core.models.payload_models import AnyTypeTaskDetails
from core.models.payload_models import ChatTaskDetails
from core.models.payload_models import DpoTaskDetails
from core.models.payload_models import GrpoTaskDetails
from core.models.payload_models import EnvironmentTaskDetails
from core.models.payload_models import ImageTaskDetails
from core.models.payload_models import InstructTextTaskDetails
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.config import Config
from validator.core.models import AnyTypeTask
from validator.core.models import ImageTextPair
from validator.utils.logging import get_logger
from validator.utils.minio import async_minio_client


logger = get_logger(__name__)

retry_http_with_backoff = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    reraise=True,
)

retry_with_backoff = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)

retry_http_fast = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, min=0.1, max=1),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    reraise=True,
)


async def try_db_connections(config: Config) -> None:
    logger.info("Attempting to connect to PostgreSQL...")
    await config.psql_db.connect()
    await config.psql_db.pool.execute("SELECT 1=1 as one")
    logger.info("PostgreSQL connected successfully")

    logger.info("Attempting to connect to Redis")
    await config.redis_db.ping()
    logger.info("Redis connected successfully")


async def save_json_to_temp_file(data: list[dict], prefix: str, dump_json: bool = True) -> tuple[str, int]:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix=prefix)
    if dump_json:
        with open(temp_file.name, "w") as f:
            json.dump(data, f)
    else:
        with open(temp_file.name, "w") as f:
            f.write(data)
    file_size = os.path.getsize(temp_file.name)
    return temp_file.name, file_size


async def upload_file_to_minio(file_path: str, bucket_name: str, object_name: str) -> str | None:
    """
    Uploads a file to MinIO and returns the presigned URL for the uploaded file.
    """
    result = await async_minio_client.upload_file(bucket_name, object_name, file_path)
    if result:
        return await async_minio_client.get_presigned_url(bucket_name, object_name)
    else:
        return None


def convert_task_to_task_details(task: AnyTypeTask) -> AnyTypeTaskDetails:
    if task.task_type == TaskType.INSTRUCTTEXTTASK:
        return InstructTextTaskDetails(
            id=task.task_id,
            account_id=task.account_id,
            status=task.status,
            base_model_repository=task.model_id,
            ds_repo=task.ds,
            field_input=task.field_input,
            field_system=task.field_system,
            field_instruction=task.field_instruction,
            field_output=task.field_output,
            format=task.format,
            no_input_format=task.no_input_format,
            system_format=task.system_format,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.termination_at,
            hours_to_complete=task.hours_to_complete,
            trained_model_repository=task.trained_model_repository,
            task_type=task.task_type,
            result_model_name=task.result_model_name,
        )
    elif task.task_type == TaskType.CHATTASK:
        return ChatTaskDetails(
            id=task.task_id,
            account_id=task.account_id,
            status=task.status,
            base_model_repository=task.model_id,
            ds_repo=task.ds,
            chat_template=task.chat_template,
            chat_column=task.chat_column,
            chat_role_field=task.chat_role_field,
            chat_content_field=task.chat_content_field,
            chat_user_reference=task.chat_user_reference,
            chat_assistant_reference=task.chat_assistant_reference,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.termination_at,
            hours_to_complete=task.hours_to_complete,
            trained_model_repository=task.trained_model_repository,
            task_type=task.task_type,
            result_model_name=task.result_model_name,
        )
    elif task.task_type == TaskType.IMAGETASK:
        return ImageTaskDetails(
            id=task.task_id,
            account_id=task.account_id,
            status=task.status,
            base_model_repository=task.model_id,
            image_text_pairs=task.image_text_pairs,
            model_type=task.model_type,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.termination_at,
            hours_to_complete=task.hours_to_complete,
            trained_model_repository=task.trained_model_repository,
            task_type=task.task_type,
            result_model_name=task.result_model_name,
        )
    elif task.task_type == TaskType.DPOTASK:
        return DpoTaskDetails(
            id=task.task_id,
            account_id=task.account_id,
            status=task.status,
            base_model_repository=task.model_id,
            ds_repo=task.ds,
            field_prompt=task.field_prompt,
            field_system=task.field_system,
            field_chosen=task.field_chosen,
            field_rejected=task.field_rejected,
            prompt_format=task.prompt_format,
            chosen_format=task.chosen_format,
            rejected_format=task.rejected_format,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.termination_at,
            hours_to_complete=task.hours_to_complete,
            trained_model_repository=task.trained_model_repository,
            task_type=task.task_type,
            result_model_name=task.result_model_name,
        )
    elif task.task_type == TaskType.GRPOTASK:
        return GrpoTaskDetails(
            id=task.task_id,
            account_id=task.account_id,
            status=task.status,
            base_model_repository=task.model_id,
            ds_repo=task.ds,
            field_prompt=task.field_prompt,
            reward_functions=task.reward_functions,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.termination_at,
            hours_to_complete=task.hours_to_complete,
            trained_model_repository=task.trained_model_repository,
            task_type=task.task_type,
            result_model_name=task.result_model_name,
        )
    elif task.task_type == TaskType.ENVIRONMENTTASK:
        return EnvironmentTaskDetails(
            id=task.task_id,
            account_id=task.account_id,
            status=task.status,
            base_model_repository=task.model_id,
            ds_repo=task.ds,
            environment_name=task.environment_name,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.termination_at,
            hours_to_complete=task.hours_to_complete,
            trained_model_repository=task.trained_model_repository,
            task_type=task.task_type,
            result_model_name=task.result_model_name,
        )


def is_task_in_flight(task: AnyTypeTask) -> bool:
    return task.status not in [
        TaskStatus.SUCCESS,
        TaskStatus.FAILURE,
        TaskStatus.FAILURE_FINDING_NODES,
        TaskStatus.PREP_TASK_FAILURE,
    ]


def hide_sensitive_data_till_finished(task: AnyTypeTask) -> AnyTypeTask:
    if is_task_in_flight(task):
        if task.task_type == TaskType.IMAGETASK:
            task.image_text_pairs = [ImageTextPair(image_url="hidden", text_url="hidden")]
        task.test_data = None

        task.training_data = None
        task.ds = "Hidden"
    return task


def normalise_float(float: float | None) -> float | None:
    if float is None:
        return 0.0

    if math.isnan(float):
        return None

    if math.isinf(float):
        float = 1e100 if float > 0 else -1e100
    return float
