"""Payload models for validator evaluation runtimes."""

import json

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator

from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import DpoDatasetType
from core.models.dataset_models import EnvironmentDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import GrpoDatasetType
from core.models.dataset_models import InstructTextDatasetType


class TokenizerConfig(BaseModel):
    bos_token: str | None = None
    eos_token: str | None = None
    pad_token: str | None = None
    unk_token: str | None = None
    chat_template: str | None = None
    use_default_system_prompt: bool | None = None


class ModelConfig(BaseModel):
    architectures: list[str]
    model_type: str
    tokenizer_config: TokenizerConfig

    model_config = ConfigDict(protected_namespaces=())


class ModelData(BaseModel):
    model_id: str
    downloads: int | None = None
    likes: int | None = None
    private: bool | None = None
    trending_score: int | None = None
    tags: list[str] | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    created_at: str | None = None
    config: dict
    parameter_count: int | None = None

    model_config = ConfigDict(protected_namespaces=())


class Img2ImgPayload(BaseModel):
    ckpt_name: str
    lora_name: str
    steps: int
    cfg: float
    denoise: float
    comfy_template: dict
    height: int = 1024
    width: int = 1024
    model_type: str = "sdxl"
    seed: int | None = None
    is_safetensors: bool = True
    prompt: str | None = None
    base_image: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class EvaluationArgs(BaseModel):
    dataset: str
    original_model: str
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType | ChatTemplateDatasetType | EnvironmentDatasetType
    file_format: FileFormat
    repo: str

    @field_validator("file_format", mode="before")
    def parse_file_format(cls, value):
        if isinstance(value, str):
            return FileFormat(value)
        return value

    @field_validator("dataset_type", mode="before")
    def parse_dataset_type(cls, value):
        if isinstance(value, str):
            try:
                data = json.loads(value)
                if "field_instruction" in data and "field_input" in data:
                    return InstructTextDatasetType.model_validate(data)
                elif "chat_column" in data:
                    return ChatTemplateDatasetType.model_validate(data)  # TODO correct?
                elif "field_chosen" in data:
                    return DpoDatasetType.model_validate(data)
                elif "reward_functions" in data:
                    return GrpoDatasetType.model_validate(data)
                elif "rollout_function" in data:
                    return EnvironmentDatasetType.model_validate(data)
            except Exception as e:
                raise ValueError(f"Failed to parse dataset type: {e}")
        return value
