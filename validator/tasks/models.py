import hashlib
from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator
from pydantic import model_validator

from core.constants.environments import EnvironmentName
from core.constants.environments import TrainingStartPoint
from core.constants.training import YARN_VALID_FACTORS
from core.models.dataset_models import FileFormat
from core.models.dataset_models import ImageTextPair
from core.models.image_models import ImageModelType
from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import BaselineStats
from core.models.reward_models import RewardFunction
from core.models.task_models import TaskType
from validator.scoring.models import EnvironmentWeight


class Backend(str, Enum):
    OBLIVUS = "oblivus"
    RUNPOD = "runpod"


class RawTask(BaseModel):
    """
    Task data as stored in the base Task table.
    """

    is_organic: bool
    task_id: UUID | None = None
    status: str
    model_id: str
    ds: str
    account_id: UUID
    times_delayed: int = 0
    hours_to_complete: float
    test_data: str | None = None
    training_data: str | None = None
    assigned_miners: list[int] | None = None
    miner_scores: list[float] | None = None
    training_repo_backup: str | None = None
    result_model_name: str | None = None

    created_at: datetime
    next_delay_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    termination_at: datetime | None = None
    completed_at: datetime | None = None
    n_eval_attempts: int = 0
    task_type: TaskType
    model_params_count: int = 0
    backend: Backend | None = None
    yarn_factor: int | None = None
    augmentation_config: AugmentationConfig | None = None
    augmented_model_id: str | None = None
    baseline_stats: BaselineStats | None = None
    training_start_point: TrainingStartPoint = TrainingStartPoint.DEFAULT

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())

    @field_validator("yarn_factor")
    @classmethod
    def validate_yarn_factor(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if not isinstance(v, int):
            raise ValueError("yarn_factor must be an integer")
        if v not in YARN_VALID_FACTORS:
            raise ValueError(f"yarn_factor must be a power of 2: {YARN_VALID_FACTORS}")
        return v


class DpoRawTask(RawTask):
    """
    DPO task data as stored in the database. It expand the RawTask with fields from the DpoTask table.
    """

    field_prompt: str
    field_system: str | None = None
    field_chosen: str
    field_rejected: str
    prompt_format: str | None = None
    chosen_format: str | None = None
    rejected_format: str | None = None
    file_format: FileFormat = FileFormat.HF
    task_type: TaskType = TaskType.DPOTASK
    synthetic_data: str | None = None


class GrpoRawTask(RawTask):
    """
    GRPO task data as stored in the database. It expand the RawTask with fields from the GrpoTask table.
    """

    field_prompt: str
    reward_functions: list[RewardFunction]
    file_format: FileFormat = FileFormat.HF
    task_type: TaskType = TaskType.GRPOTASK
    extra_column: str | None = None
    synthetic_data: str | None = None

    @model_validator(mode="after")
    def validate_reward_functions(self) -> "GrpoRawTask":
        for reward_function in self.reward_functions:
            if reward_function.func_hash is None:
                reward_function.func_hash = hashlib.sha256(reward_function.reward_func.encode()).hexdigest()
        return self


class EnvRawTask(RawTask):
    """
    Environment task data as stored in the database. It expand the RawTask with fields from the EnvTask table.
    """

    environment_names: list[EnvironmentName] = []
    environment_weights: list[EnvironmentWeight] = []
    eval_seed: int | None = None
    task_type: TaskType = TaskType.ENVIRONMENTTASK
    synthetic_data: str | None = None


class InstructTextRawTask(RawTask):
    """
    Instruct Text task data as stored in the database. It expand the RawTask with fields from the instruct_text_tasks table.
    """

    field_system: str | None = None
    field_instruction: str
    field_input: str | None = None
    field_output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    system_format: None = None  # NOTE: Needs updating to be optional once we accept it
    file_format: FileFormat = FileFormat.HF
    task_type: TaskType = TaskType.INSTRUCTTEXTTASK
    synthetic_data: str | None = None
    use_kl: bool = False
    kl_coef: float | None = None


class ChatRawTask(RawTask):
    """
    Chat task data as stored in the database. It expand the RawTask with fields from the chat_tasks table.
    """

    chat_template: str | None = "chatml"
    chat_column: str | None = "conversations"
    chat_role_field: str | None = "from"
    chat_content_field: str | None = "value"
    chat_user_reference: str | None = "user"
    chat_assistant_reference: str | None = "assistant"
    file_format: FileFormat = FileFormat.HF
    task_type: TaskType = TaskType.CHATTASK
    synthetic_data: str | None = None


class ImageRawTask(RawTask):
    """
    Image task data as stored in the database. It expand the RawTask with fields from the ImageTask table.
    """

    image_text_pairs: list[ImageTextPair] | None = None
    task_type: TaskType = TaskType.IMAGETASK
    model_type: ImageModelType = ImageModelType.SDXL
    trigger_word: str | None = None


# NOTE: As time goes on we will expand this class to be more of a 'submitted task'?
# Might wanna rename this some more
class Task(RawTask):
    trained_model_repository: str | None = None


class InstructTextTask(InstructTextRawTask):
    """
    Expands on the InstructTextRawTask with the trained_model_repository field.
    This field is not stored in the db directly, but is computed from the submissions table.

    """

    trained_model_repository: str | None = None


class ImageTask(ImageRawTask):
    trained_model_repository: str | None = None


class DpoTask(DpoRawTask):
    trained_model_repository: str | None = None


class GrpoTask(GrpoRawTask):
    trained_model_repository: str | None = None


class EnvTask(EnvRawTask):
    trained_model_repository: str | None = None


class ChatTask(ChatRawTask):
    trained_model_repository: str | None = None


class NetworkStats(BaseModel):
    number_of_jobs_training: int
    number_of_jobs_preevaluation: int
    number_of_jobs_evaluating: int
    number_of_jobs_success: int
    next_training_end: datetime | None
    job_can_be_made: bool = True


class DetailedNetworkStats(NetworkStats):
    instruct_training: int = 0
    instruct_preevaluation: int = 0
    instruct_evaluating: int = 0
    instruct_success: int = 0

    dpo_training: int = 0
    dpo_preevaluation: int = 0
    dpo_evaluating: int = 0
    dpo_success: int = 0

    grpo_training: int = 0
    grpo_preevaluation: int = 0
    grpo_evaluating: int = 0
    grpo_success: int = 0

    image_training: int = 0
    image_preevaluation: int = 0
    image_evaluating: int = 0
    image_success: int = 0


class HotkeyDetails(BaseModel):
    hotkey: str
    submission_id: UUID | None = None
    quality_score: float | None
    test_loss: float | None
    synth_loss: float | None
    repo: str | None
    rank: int | None
    score_reason: str | None = None
    offer_response: dict | None = None


class InstructTextTaskWithHotkeyDetails(InstructTextTask):
    hotkey_details: list[HotkeyDetails]


class ImageTaskWithHotkeyDetails(ImageTask):
    hotkey_details: list[HotkeyDetails]


class DpoTaskWithHotkeyDetails(DpoTask):
    hotkey_details: list[HotkeyDetails]


class GrpoTaskWithHotkeyDetails(GrpoTask):
    hotkey_details: list[HotkeyDetails]


class EnvTaskWithHotkeyDetails(EnvTask):
    hotkey_details: list[HotkeyDetails]


class ChatTaskWithHotkeyDetails(ChatTask):
    hotkey_details: list[HotkeyDetails]


# Type aliases for common task type groupings
AnyTextTypeRawTask = InstructTextRawTask | DpoRawTask | GrpoRawTask | ChatRawTask | EnvRawTask
AnyTypeRawTask = AnyTextTypeRawTask | ImageRawTask
AnyTypeTask = InstructTextTask | DpoTask | ImageTask | GrpoTask | ChatTask | EnvTask
AnyTypeTaskWithHotkeyDetails = (
    InstructTextTaskWithHotkeyDetails
    | ImageTaskWithHotkeyDetails
    | DpoTaskWithHotkeyDetails
    | GrpoTaskWithHotkeyDetails
    | ChatTaskWithHotkeyDetails
    | EnvTaskWithHotkeyDetails
)
