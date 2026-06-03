import os

from fiber.logging_utils import get_logger

from core.constants.datasets import DPO_DEFAULT_DATASET_TYPE
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import DpoDatasetType
from core.models.dataset_models import EnvironmentDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import GrpoDatasetType
from core.models.dataset_models import InstructTextDatasetType
from core.models.dataset_models import TextDatasetType


logger = get_logger(__name__)


def create_dataset_entry(
    dataset: str,
    dataset_type: TextDatasetType,
    file_format: FileFormat,
    is_eval: bool = False,
) -> dict:
    dataset_entry = {"path": dataset}

    logger.info(dataset_type)

    if file_format == FileFormat.JSON:
        if is_eval:
            dataset_entry = {"path": os.path.dirname(dataset)}
        else:
            dataset_entry = {"path": "/workspace/input_data/"}

    if isinstance(dataset_type, EnvironmentDatasetType):
        dataset_entry.update(_process_environment_dataset_fields(dataset_type))
    elif isinstance(dataset_type, InstructTextDatasetType):
        instruct_type_dict = {key: value for key, value in dataset_type.model_dump().items() if value is not None}
        dataset_entry.update(_process_instruct_dataset_fields(instruct_type_dict))
    elif isinstance(dataset_type, DpoDatasetType):
        dataset_entry.update(_process_dpo_dataset_fields(dataset_type))
    elif isinstance(dataset_type, GrpoDatasetType):
        dataset_entry.update(_process_grpo_dataset_fields(dataset_type))
    elif isinstance(dataset_type, ChatTemplateDatasetType):
        dataset_entry.update(_process_chat_template_dataset_fields(dataset_type))
    else:
        raise ValueError("Invalid dataset_type provided.")

    if file_format != FileFormat.HF:
        dataset_entry["ds_type"] = file_format.value
        if is_eval:
            dataset_entry["data_files"] = [os.path.abspath(dataset)]
        else:
            dataset_entry["data_files"] = [os.path.basename(dataset)]

    return dataset_entry


def _process_grpo_dataset_fields(dataset_type: GrpoDatasetType) -> dict:
    return {"split": "train"}


def _process_environment_dataset_fields(dataset_type: EnvironmentDatasetType) -> dict:
    return {"split": "train"}


def _process_dpo_dataset_fields(dataset_type: DpoDatasetType) -> dict:
    # Enable below when https://github.com/axolotl-ai-cloud/axolotl/issues/1417 is fixed
    # context: https://discord.com/channels/1272221995400167588/1355226588178022452/1356982842374226125

    # dpo_type_dict = dataset_type.model_dump()
    # dpo_type_dict["type"] = "user_defined.default"
    # if not dpo_type_dict.get("prompt_format"):
    #     if dpo_type_dict.get("field_system"):
    #         dpo_type_dict["prompt_format"] = "{system} {prompt}"
    #     else:
    #         dpo_type_dict["prompt_format"] = "{prompt}"
    # return dpo_type_dict

    # Fallback to https://axolotl-ai-cloud.github.io/axolotl/docs/rlhf.html#chatml.intel
    # Column names are hardcoded in axolotl: "DPO_DEFAULT_FIELD_SYSTEM",
    # "DPO_DEFAULT_FIELD_PROMPT", "DPO_DEFAULT_FIELD_CHOSEN", "DPO_DEFAULT_FIELD_REJECTED"
    return {"type": DPO_DEFAULT_DATASET_TYPE, "split": "train"}


def _process_instruct_dataset_fields(instruct_type_dict: dict) -> dict:
    if not instruct_type_dict.get("field_output"):
        return {
            "type": "completion",
            "field": instruct_type_dict.get("field_instruction"),
        }

    processed_dict = instruct_type_dict.copy()
    processed_dict.setdefault("no_input_format", "{instruction}")
    if processed_dict.get("field_input"):
        processed_dict.setdefault("format", "{instruction} {input}")
    else:
        processed_dict.setdefault("format", "{instruction}")

    return {"format": "custom", "type": processed_dict}


def _process_chat_template_dataset_fields(dataset_dict: ChatTemplateDatasetType) -> dict:
    processed_dict = {}

    processed_dict["chat_template"] = dataset_dict.chat_template
    processed_dict["type"] = "chat_template"
    processed_dict["field_messages"] = dataset_dict.chat_column
    processed_dict["message_field_role"] = dataset_dict.chat_role_field
    processed_dict["message_field_content"] = dataset_dict.chat_content_field
    processed_dict["roles"] = {
        "assistant": [dataset_dict.chat_assistant_reference],
        "user": [dataset_dict.chat_user_reference],
    }

    processed_dict["message_property_mappings"] = {
        "role": dataset_dict.chat_role_field,
        "content": dataset_dict.chat_content_field,
    }

    return processed_dict
