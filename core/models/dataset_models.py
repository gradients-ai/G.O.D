from enum import Enum

from pydantic import BaseModel
from pydantic import Field

from core.constants.environments import EnvironmentName
from core.models.reward_models import RewardFunction


class FileFormat(str, Enum):
    CSV = "csv"  # needs to be local file
    JSON = "json"  # needs to be local file
    HF = "hf"  # Hugging Face dataset
    S3 = "s3"


class InstructTextDatasetType(BaseModel):
    system_prompt: str | None = ""
    system_format: str | None = "{system}"
    field_system: str | None = None
    field_instruction: str | None = None
    field_input: str | None = None
    field_output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    field: str | None = None


class GrpoDatasetType(BaseModel):
    field_prompt: str | None = None
    reward_functions: list[RewardFunction] | None = Field(default_factory=list)
    extra_column: str | None = None


class EnvironmentDatasetType(BaseModel):
    environment_names: list[EnvironmentName] | None = None


class DpoDatasetType(BaseModel):
    field_prompt: str | None = None
    field_system: str | None = None
    field_chosen: str | None = None
    field_rejected: str | None = None
    prompt_format: str | None = "{prompt}"
    chosen_format: str | None = "{chosen}"
    rejected_format: str | None = "{rejected}"


class ChatTemplateDatasetType(BaseModel):
    chat_template: str | None = "chatml"
    chat_column: str | None = "conversations"
    chat_role_field: str | None = "from"
    chat_content_field: str | None = "value"
    chat_user_reference: str | None = "user"
    chat_assistant_reference: str | None = "assistant"


TextDatasetType = InstructTextDatasetType | DpoDatasetType | GrpoDatasetType | ChatTemplateDatasetType | EnvironmentDatasetType


class ImageTextPair(BaseModel):
    image_url: str = Field(..., description="Presigned URL for the image file")
    text_url: str = Field(..., description="Presigned URL for the text file")
