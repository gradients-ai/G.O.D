import uuid
from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class FileFormat(str, Enum):
    CSV = "csv"  # needs to be local file
    JSON = "json"  # needs to be local file
    HF = "hf"  # Hugging Face dataset
    S3 = "s3"


class JobStatus(str, Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    NOT_FOUND = "Not Found"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PREPARING_DATA = "preparing_data"
    PREP_TASK_FAILURE = "prep_task_failure"
    LOOKING_FOR_NODES = "looking_for_nodes"
    FAILURE_FINDING_NODES = "failure_finding_nodes"
    DELAYED = "delayed"
    READY = "ready"
    TRAINING = "training"
    PREEVALUATION = "preevaluation"
    EVALUATING = "evaluating"
    SUCCESS = "success"
    FAILURE = "failure"


class WinningSubmission(BaseModel):
    hotkey: str
    score: float
    model_repo: str

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())


class MinerSubmission(BaseModel):
    repo: str
    model_hash: str | None = None


class MinerTaskResult(BaseModel):
    hotkey: str
    quality_score: float
    test_loss: float | None
    synth_loss: float | None
    score_reason: str | None


# NOTE: Confusing name with the class above
class TaskMinerResult(BaseModel):
    task_id: UUID
    quality_score: float


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


class RewardFunction(BaseModel):
    """Model representing a reward function with its metadata"""

    reward_id: str | None = Field(None, description="UUID of the reward function in the database")
    reward_func: str = Field(
        ...,
        description="String with the python code of the reward function to use",
        examples=[
            "def reward_func_conciseness(completions, **kwargs):",
            '"""Reward function that favors shorter, more concise answers."""',
            "    return [100.0/(len(completion.split()) + 10) for completion in completions]",
        ],
    )
    reward_weight: float = Field(..., ge=0)
    func_hash: str | None = None
    is_generic: bool | None = None
    is_manual: bool | None = None


class GrpoDatasetType(BaseModel):
    field_prompt: str | None = None
    reward_functions: list[RewardFunction] | None = []
    extra_column: str | None = None


class EnvironmentDatasetType(BaseModel):
    environment_name: str | None = None


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


class ImageModelType(str, Enum):
    FLUX = "flux"
    SDXL = "sdxl"
    Z_IMAGE = "z-image"
    QWEN_IMAGE = "qwen-image"


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model: str
    status: JobStatus = JobStatus.QUEUED
    error_message: str | None = None
    expected_repo_name: str | None = None


TextDatasetType = InstructTextDatasetType | DpoDatasetType | GrpoDatasetType | ChatTemplateDatasetType | EnvironmentDatasetType


class TextJob(Job):
    dataset: str
    dataset_type: TextDatasetType
    file_format: FileFormat


class DiffusionJob(Job):
    model_config = ConfigDict(protected_namespaces=())
    dataset_zip: str = Field(
        ...,
        description="Link to dataset zip file",
        min_length=1,
    )
    model_type: ImageModelType = ImageModelType.SDXL


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class Prompts(BaseModel):
    input_output_reformulation_sys: str
    input_output_reformulation_user: str
    input_reformulation_sys: str
    input_reformulation_user: str
    reward_function_generation_sys: str
    reward_function_generation_user: str


class TaskType(str, Enum):
    INSTRUCTTEXTTASK = "InstructTextTask"
    IMAGETASK = "ImageTask"
    DPOTASK = "DpoTask"
    GRPOTASK = "GrpoTask"
    CHATTASK = "ChatTask"
    ENVIRONMENTTASK = "EnvTask"

    def __hash__(self):
        return hash(str(self))


class ImageTextPair(BaseModel):
    image_url: str = Field(..., description="Presigned URL for the image file")
    text_url: str = Field(..., description="Presigned URL for the text file")


class GPUType(str, Enum):
    H100 = "H100"
    A100 = "A100"
    A6000 = "A6000"


class TrainingStatus(str, Enum):
    PENDING = "pending"
    TRAINING = "training"
    SUCCESS = "success"
    FAILURE = "failure"


class Backend(str, Enum):
    OBLIVUS = "oblivus"
    RUNPOD = "runpod"


class GPUInfo(BaseModel):
    gpu_id: int = Field(..., description="GPU ID")
    gpu_type: GPUType = Field(..., description="GPU Type")
    vram_gb: int = Field(..., description="GPU VRAM in GB")
    available: bool = Field(..., description="GPU Availability")
    used_until: datetime | None = Field(default=None, description="GPU Used Until")


class TrainerInfo(BaseModel):
    trainer_ip: str = Field(..., description="Trainer IP address")
    gpus: list[GPUInfo] = Field(..., description="List of GPUs available on this trainer")
