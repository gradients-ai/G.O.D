from datetime import datetime
from uuid import UUID
from uuid import uuid4

from fiber.logging_utils import get_logger
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from core import constants as cst
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import ImageTextPair
from core.models.utility_models import JobStatus
from core.models.utility_models import MinerTaskResult
from core.models.utility_models import RewardFunction
from core.models.utility_models import TaskMinerResult
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from core.models.utility_models import TextDatasetType
from core.models.utility_models import EnvironmentDatasetType
from validator.core.models import AllNodeStats


logger = get_logger(__name__)


class MinerTaskOffer(BaseModel):
    ds_size: int | None = None
    model: str
    hours_to_complete: float
    task_id: str
    task_type: TaskType
    model_params_count: int | None = None

    model_config = ConfigDict(protected_namespaces=())


class TrainRequest(BaseModel):
    model: str = Field(..., description="Name or path of the model to be trained", min_length=1)
    task_id: str
    hours_to_complete: float
    expected_repo_name: str | None = None


class TrainRequestText(TrainRequest):
    dataset: str = Field(
        ...,
        description="Path to the dataset file or Hugging Face dataset name",
        min_length=1,
    )
    dataset_type: TextDatasetType
    file_format: FileFormat


class TrainRequestGrpo(TrainRequest):
    dataset: str = Field(
        ...,
        description="Path to the dataset file or Hugging Face dataset name",
        min_length=1,
    )
    dataset_type: GrpoDatasetType
    file_format: FileFormat

class TrainRequestEnvironment(TrainRequest):
    dataset: str = Field(
        ...,
        description="Path to the dataset file or Hugging Face dataset name",
        min_length=1,
    )
    dataset_type: EnvironmentDatasetType
    file_format: FileFormat


class TrainRequestImage(TrainRequest):
    model_config = ConfigDict(protected_namespaces=())
    dataset_zip: str = Field(
        ...,
        description="Link to dataset zip file",
        min_length=1,
    )
    model_type: ImageModelType = ImageModelType.SDXL
    trigger_word: str | None = None


class TrainerProxyRequest(BaseModel):
    training_data: TrainRequestImage | TrainRequestText
    github_repo: str
    gpu_ids: list[int]
    hotkey: str
    github_branch: str | None = None
    github_commit_hash: str | None = None


class TrainerTaskLog(TrainerProxyRequest):
    status: TaskStatus
    started_at: datetime | None
    finished_at: datetime | None
    wandb_url: str | None = None
    logs: list[str] = []


class TrainResponse(BaseModel):
    message: str
    task_id: UUID


class TrainingRepoResponse(BaseModel):
    github_repo: str = Field(..., description="The GitHub repository URL")
    commit_hash: str = Field(..., description="The commit hash of the repository")


class JobStatusPayload(BaseModel):
    task_id: UUID


class JobStatusResponse(BaseModel):
    task_id: UUID
    status: JobStatus


class EvaluationRequest(TrainRequest):
    original_model: str


class EvaluationRequestDiffusion(BaseModel):
    test_split_url: str
    original_model_repo: str
    models: list[str]


class DiffusionLosses(BaseModel):
    text_guided_losses: list[float]
    no_text_losses: list[float]


class EvaluationResultImage(BaseModel):
    eval_loss: DiffusionLosses | float
    is_finetune: bool | None = None


class EvaluationResultText(BaseModel):
    is_finetune: bool
    eval_loss: float


class DockerEvaluationResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    results: dict[str, EvaluationResultText | EvaluationResultImage | Exception]
    base_model_params_count: int = 0


class MinerTaskResponse(BaseModel):
    message: str
    accepted: bool


class DpoDatasetColumnsResponse(BaseModel):
    field_prompt: str
    field_chosen: str | None = None
    field_rejected: str | None = None


class InstructTextDatasetColumnsResponse(BaseModel):
    field_instruction: str
    field_input: str | None = None
    field_output: str | None = None


class NewTaskRequest(BaseModel):
    account_id: UUID
    hours_to_complete: float = Field(..., description="The number of hours to complete the task", examples=[1])
    result_model_name: str | None = Field(None, description="The name to give to a model that is created by this task")
    backend: str = Field(default="oblivus", description="The backend to use for training: 'oblivus' or 'runpod'", examples=["runpod", "oblivus"])
    yarn_factor: int | None = Field(None, description=f"YaRN extension factor for extending context length (powers of 2: {cst.YARN_VALID_FACTORS})", examples=[2, 4, 8, 16])

    @field_validator("yarn_factor")
    @classmethod
    def validate_yarn_factor(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if not isinstance(v, int):
            raise ValueError("yarn_factor must be an integer")
        if v not in cst.YARN_VALID_FACTORS:
            raise ValueError(f"yarn_factor must be a power of 2: {cst.YARN_VALID_FACTORS}")
        return v


class NewTaskRequestInstructText(NewTaskRequest):
    field_instruction: str = Field(..., description="The column name for the instruction", examples=["instruction"])
    field_input: str | None = Field(None, description="The column name for the input", examples=["input"])
    field_output: str | None = Field(None, description="The column name for the output", examples=["output"])
    field_system: str | None = Field(None, description="The column name for the system (prompt)", examples=["system"])

    ds_repo: str = Field(..., description="The repository for the dataset", examples=["yahma/alpaca-cleaned"])
    file_format: FileFormat = Field(
        FileFormat.HF, description="The format of the dataset", examples=[FileFormat.HF, FileFormat.S3]
    )
    model_repo: str = Field(..., description="The repository for the model", examples=["Qwen/Qwen2.5-Coder-32B-Instruct"])
    format: None = None
    no_input_format: None = None

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    def convert_empty_strings(cls, values: dict) -> dict:
        string_fields = ["field_instruction", "field_input", "field_output", "field_system"]
        for field in string_fields:
            if field in values and isinstance(values[field], str):
                values[field] = values[field].strip() or None
        return values


class NewTaskRequestChat(NewTaskRequest):
    chat_template: str = Field(..., description="The chat template of the dataset", examples=["chatml"])
    chat_column: str | None = Field(None, description="The column name containing the conversations", examples=["conversations"])
    chat_role_field: str | None = Field(None, description="The column name for the role", examples=["from"])
    chat_content_field: str | None = Field(None, description="The column name for the content", examples=["value"])
    chat_user_reference: str | None = Field(None, description="The user reference", examples=["user"])
    chat_assistant_reference: str | None = Field(None, description="The assistant reference", examples=["assistant"])

    ds_repo: str = Field(..., description="The repository for the dataset", examples=["Magpie-Align/Magpie-Pro-300K-Filtered"])
    file_format: FileFormat = Field(
        FileFormat.HF, description="The format of the dataset", examples=[FileFormat.HF, FileFormat.S3]
    )
    model_repo: str = Field(..., description="The repository for the model", examples=["Qwen/Qwen2.5-Coder-32B-Instruct"])

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    def convert_empty_strings(cls, values):
        string_fields = [
            "chat_column",
            "chat_role_field",
            "chat_content_field",
            "chat_user_reference",
            "chat_assistant_reference",
        ]
        for field in string_fields:
            if field in values and isinstance(values[field], str):
                values[field] = values[field].strip() or None
        return values
    
class NewTaskRequestEnvironment(NewTaskRequest):
    environment_name: str = Field(..., description="The name of the specific environment we are training for.", examples=["alfworld"])


    ds_repo: str = Field(..., description="The repository for the dataset", examples=["Magpie-Align/Magpie-Pro-300K-Filtered"])
    model_repo: str = Field(..., description="The repository for the model", examples=["Qwen/Qwen2.5-Coder-32B-Instruct"])

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    def convert_empty_strings(cls, values):
        string_fields = [
            "environment_name",
        ]
        for field in string_fields:
            if field in values and isinstance(values[field], str):
                values[field] = values[field].strip() or None
        return values


class NewTaskRequestDPO(NewTaskRequest):
    field_prompt: str = Field(..., description="The column name for the prompt", examples=["prompt"])
    field_system: str | None = Field(None, description="The column name for the system (prompt)", examples=["system"])
    field_chosen: str = Field(..., description="The column name for the chosen response", examples=["chosen"])
    field_rejected: str = Field(..., description="The column name for the rejected response", examples=["rejected"])

    prompt_format: str | None = Field(None, description="The format of the prompt", examples=["{system} {prompt}"])
    chosen_format: str | None = Field(None, description="The format of the chosen response", examples=["{chosen} <|endoftext|>"])
    rejected_format: str | None = Field(
        None, description="The format of the rejected response", examples=["{rejected} <|endoftext|>"]
    )

    ds_repo: str = Field(..., description="The repository for the dataset", examples=["Intel/orca_dpo_pairs"])
    file_format: FileFormat = Field(
        FileFormat.HF, description="The format of the dataset", examples=[FileFormat.HF, FileFormat.S3]
    )
    model_repo: str = Field(..., description="The repository for the model", examples=["Qwen/Qwen2.5-Coder-32B-Instruct"])

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    def convert_empty_strings(cls, values: dict) -> dict:
        string_fields = ["field_prompt", "field_system", "field_chosen", "field_rejected"]
        for field in string_fields:
            if field in values and isinstance(values[field], str):
                values[field] = values[field].strip() or None
        return values


class RewardFunctionReference(BaseModel):
    """Model representing a reference to a reward function by ID"""

    reward_id: str = Field(
        ..., description="UUID of the reward function in the database", examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    reward_weight: float = Field(..., ge=0, description="Weight for this reward function")


class NewTaskRequestGrpo(NewTaskRequest):
    field_prompt: str = Field(..., description="The column name for the prompt", examples=["prompt"])
    extra_column: str | None = Field(None, description="The column name for the extra data", examples=["extra_data"])

    ds_repo: str = Field(..., description="The repository for the dataset", examples=["trl-lib/tldr"])
    file_format: FileFormat = Field(
        FileFormat.HF, description="The format of the dataset", examples=[FileFormat.HF, FileFormat.S3]
    )
    model_repo: str = Field(..., description="The repository for the model", examples=["Qwen/Qwen2.5-Coder-32B-Instruct"])

    reward_functions: list[RewardFunctionReference]

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    def convert_empty_strings(cls, values: dict) -> dict:
        string_fields = ["field_prompt"]
        for field in string_fields:
            if field in values and isinstance(values[field], str):
                values[field] = values[field].strip() or None
        return values

    @model_validator(mode="after")
    def validate_reward_lists(self) -> "NewTaskRequestGrpo":
        if len(self.reward_functions) == 0:
            raise ValueError("reward_functions must not be empty")
        return self


class NewTaskRequestImage(NewTaskRequest):
    model_config = ConfigDict(protected_namespaces=())
    model_repo: str = Field(..., description="The model repository to use")
    image_text_pairs: list[ImageTextPair] = Field(
        ...,
        description="List of image and text file pairs",
        min_length=cst.MIN_IMAGE_TEXT_PAIRS,
        max_length=cst.MAX_IMAGE_TEXT_PAIRS,
    )
    ds_id: str = Field(
        default_factory=lambda: str(uuid4()), description="A ds name. The actual dataset is provided via the image_text_pairs"
    )
    model_type: ImageModelType = ImageModelType.SDXL


class NewTaskWithCustomDatasetRequest(NewTaskRequestInstructText):
    ds_repo: str | None = Field(None, description="Optional: The original repository of the dataset")
    training_data: str = Field(..., description="The prepared training dataset")
    test_data: str | None = Field(None, description="The prepared test dataset")
    file_format: FileFormat = Field(
        FileFormat.S3, description="The format of the dataset", examples=[FileFormat.HF, FileFormat.S3]
    )


class NewTaskWithCustomDatasetRequestChat(NewTaskRequestChat):
    ds_repo: str | None = Field(None, description="Optional: The original repository of the dataset")
    training_data: str = Field(..., description="The prepared training dataset")
    test_data: str | None = Field(None, description="The prepared test dataset")
    file_format: FileFormat = Field(
        FileFormat.S3, description="The format of the dataset", examples=[FileFormat.HF, FileFormat.S3]
    )


class NewTaskResponse(BaseModel):
    success: bool = Field(..., description="Whether the task was created successfully")
    task_id: UUID | None = Field(..., description="The ID of the task")
    created_at: datetime = Field(..., description="The creation time of the task")
    account_id: UUID | None = Field(..., description="The account ID who owns the task")


class TaskResultResponse(BaseModel):
    id: UUID
    miner_results: list[MinerTaskResult] | None


class AllOfNodeResults(BaseModel):
    success: bool
    hotkey: str
    task_results: list[TaskMinerResult] | None


class TaskDetails(BaseModel):
    id: UUID
    account_id: UUID
    status: TaskStatus
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    hours_to_complete: float
    trained_model_repository: str | None
    task_type: TaskType
    result_model_name: str | None = None


class InstructTextTaskDetails(TaskDetails):
    task_type: TaskType = TaskType.INSTRUCTTEXTTASK
    base_model_repository: str
    ds_repo: str

    field_system: str | None = Field(None, description="The column name for the `system (prompt)`", examples=["system"])
    field_instruction: str = Field(
        ..., description="The column name for the instruction - always needs to be provided", examples=["instruction"]
    )
    field_input: str | None = Field(None, description="The column name for the `input`", examples=["input"])
    field_output: str | None = Field(None, description="The column name for the `output`", examples=["output"])

    # NOTE: ATM can not be defined by the user, but should be able to in the future
    format: None = Field(None, description="The column name for the `format`", examples=["{instruction} {input}"])
    no_input_format: None = Field(
        None, description="If the field_input is not provided, what format should we use? ", examples=["{instruction}"]
    )
    system_format: None = Field(None, description="How to format the `system (prompt)`", examples=["{system}"])

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())


class ChatTaskDetails(TaskDetails):
    task_type: TaskType = TaskType.CHATTASK
    base_model_repository: str
    ds_repo: str

    chat_template: str = Field(..., description="The chat template used", examples=["chatml"])
    chat_column: str | None = Field(None, description="The column name for the chat conversations", examples=["conversations"])
    chat_role_field: str = Field(..., description="The column name to specify the role in the conversation ", examples=["from"])
    chat_content_field: str = Field(..., description="The column name to specify the text content", examples=["value"])
    chat_user_reference: str | None = Field(None, description="The column name to specify the user", examples=["user"])
    chat_assistant_reference: str | None = Field(
        None, description="The column name to specify the assistant", examples=["assistant"]
    )

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())


class DpoTaskDetails(TaskDetails):
    task_type: TaskType = TaskType.DPOTASK
    base_model_repository: str
    ds_repo: str

    field_prompt: str = Field(..., description="The column name for the prompt", examples=["prompt"])
    field_system: str | None = Field(None, description="The column name for the `system (prompt)`", examples=["system"])
    field_chosen: str = Field(..., description="The column name for the chosen response", examples=["chosen"])
    field_rejected: str = Field(..., description="The column name for the rejected response", examples=["rejected"])

    prompt_format: str | None = Field(None, description="The format of the prompt", examples=["{system} {prompt}"])
    chosen_format: str | None = Field(None, description="The format of the chosen response", examples=["{chosen} <|endoftext|>"])
    rejected_format: str | None = Field(
        None, description="The format of the rejected response", examples=["{rejected} <|endoftext|>"]
    )

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())


class GrpoTaskDetails(TaskDetails):
    task_type: TaskType = TaskType.GRPOTASK
    base_model_repository: str
    ds_repo: str

    field_prompt: str = Field(..., description="The column name for the prompt", examples=["prompt"])
    reward_functions: list[RewardFunction]

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())

class EnvironmentTaskDetails(TaskDetails):
    task_type: TaskType = TaskType.ENVIRONMENTTASK
    environment_name: str
    base_model_repository: str
    ds_repo: str

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())


class ImageTaskDetails(TaskDetails):
    task_type: TaskType = TaskType.IMAGETASK
    image_text_pairs: list[ImageTextPair]
    base_model_repository: str = Field(..., description="The repository for the model")
    model_type: ImageModelType = ImageModelType.SDXL

    model_config = ConfigDict(protected_namespaces=())


class TaskListResponse(BaseModel):
    success: bool
    task_id: UUID
    status: TaskStatus


class LeaderboardRow(BaseModel):
    hotkey: str
    stats: AllNodeStats


class ImageModelInfo(BaseModel):
    model_id: str
    model_type: ImageModelType

    model_config = ConfigDict(protected_namespaces=())


class ImageModelsResponse(BaseModel):
    models: list[ImageModelInfo]


class GpuRequirementSummary(BaseModel):
    gpu_type: str
    count: int
    total_hours: float


class TournamentGpuRequirementsResponse(BaseModel):
    gpu_requirements: list[GpuRequirementSummary]
    total_tasks: int
    total_hours: float


class BenchmarkResult(BaseModel):
    """Individual benchmark result for a participant"""

    copy_task_id: str
    participant_hotkey: str
    tournament_id: str | None
    quality_score: float
    test_loss: float | None
    synth_loss: float | None
    repo: str | None
    completed_at: datetime | None
    created_at: datetime
    model_id: str
    dataset: str
    task_type: str


class BenchmarkRootTaskResults(BaseModel):
    """Results for a specific benchmark root task"""

    root_task_id: str
    model_id: str
    dataset: str
    task_type: str
    results: list[BenchmarkResult]


class RewardFunctionInfo(BaseModel):
    reward_id: str = Field(..., description="UUID of the reward function in the database")
    name: str
    description: str
    code: str


class RewardFunctionsResponse(BaseModel):
    reward_functions: dict[str, RewardFunctionInfo]


class AddRewardFunctionRequest(BaseModel):
    name: str
    description: str
    code: str
    reward_weight: float | None = None


# Type alias for task details types
AnyTypeTaskDetails = InstructTextTaskDetails | ChatTaskDetails| ImageTaskDetails | DpoTaskDetails | GrpoTaskDetails | EnvironmentTaskDetails


class DstackRunStatus(BaseModel):
    """Dstack run status response model"""
    status: str = Field(..., description="Run status: submitted, provisioning, running, done, failed, aborted, terminated")
    latest_job_submission: dict | None = None
    
    def get_status(self) -> str:
        """Get the status string, handling both string and dict formats"""
        if isinstance(self.status, dict):
            return self.status.get("status", "Unknown")
        return str(self.status)
    
    def is_provisioning(self) -> bool:
        """Check if run is in provisioning state"""
        status = self.get_status().lower()
        return "provisioning" in status or "submitted" in status
    
    def is_running(self) -> bool:
        """Check if run is in running state"""
        status = self.get_status().lower()
        return "running" in status
    
    def is_done(self) -> bool:
        """Check if run is done (successfully completed)"""
        status = self.get_status().lower()
        return status == "done"
    
    def is_failed(self) -> bool:
        """Check if run has failed"""
        status = self.get_status().lower()
        return status in ["failed", "aborted", "terminated"]

    def got_no_offers(self) -> bool:
        """Check if run got no offers"""
        if self.latest_job_submission is None:
            return False
        return self.latest_job_submission.get("status_message", "Unknown") == "no offers"
