import hashlib
import math
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
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import ImageTextPair
from core.models.dataset_models import InstructTextDatasetType
from core.models.image_models import ImageModelType
from core.models.image_models import ImageTrainingMode
from core.models.image_models import supports_image_training_mode
from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import BaselineStats
from core.models.quant_artifact_models import QuantDecodedFile
from core.models.quant_models import QuantFixedSpec
from core.models.quant_models import QuantFreeformPenalty
from core.models.reward_models import RewardFunction
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.scoring.constants import QUANT_BAR_CAP
from validator.scoring.constants import QUANT_PENALTY_CURVATURE
from validator.scoring.constants import QUANT_PENALTY_KNEE
from validator.scoring.models import EnvironmentWeight
from validator.scoring.models import HubArtifactFile


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
    # Rows carved from the train side for a compression (quant) miner to measure on, never scored. On an instruct task
    # calibration_data is the raw JSON url; on a quant task it is the raw rows carried from the upstream or the given split
    # until prep, then the tokenised JSONL (with calibration_fingerprint / calibration_n_examples) that miners receive.
    calibration_data: str | None = None
    calibration_fingerprint: str | None = None
    calibration_n_examples: int | None = None
    # Instruct tasks: prep carves calibration_data from this task's own split (a tournament round-1 task). Quant tasks are
    # created flagged and always get a calibration set, carved unless they carry calibration_data from a given split.
    carve_calibration: bool = False
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
    model_type: ImageModelType = ImageModelType.FLUX
    training_mode: ImageTrainingMode = ImageTrainingMode.TEXT_TO_IMAGE
    trigger_word: str | None = None

    @model_validator(mode="after")
    def validate_image_training_mode(self) -> "ImageRawTask":
        if not supports_image_training_mode(self.model_type, self.training_mode):
            raise ValueError(f"{self.model_type.value} does not support {self.training_mode.value} training")
        if (
            self.training_mode == ImageTrainingMode.EDIT
            and self.image_text_pairs
            and any(pair.source_image_url is None for pair in self.image_text_pairs)
        ):
            raise ValueError("Every image-edit pair must include source_image_url")
        return self


class QuantRawTask(RawTask):
    """
    Fields shared by both quant tracks, stored in the quant_tasks table. Not itself a task type.
    """

    dataset_type: InstructTextDatasetType | ChatTemplateDatasetType
    file_format: FileFormat = FileFormat.HF
    # The text seed task whose boss output this chained task compresses (set at release).
    upstream_task_id: UUID | None = None
    synthetic_data: str | None = None


class ContinuousChunk(BaseModel):
    """One continuous-SFT chunk as the content service signs it: presigned train and test urls and the ds label."""

    train_url: str
    test_url: str
    label: str


class QuantPreppedSplit(BaseModel):
    """An already-prepped instruct task's uploaded train and test JSON, in the standard column names.

    A quant task given one skips the raw-dataset split and preps from these files, so it is scored on
    exactly the held-out rows the source task was.
    """

    training_data: str
    test_data: str
    calibration_data: str | None = None  # the source task's raw calibration JSON; None leaves the carve to the task's own prep


class QuantFixedRawTask(QuantRawTask):
    quant_spec: QuantFixedSpec
    task_type: TaskType = TaskType.QUANTFIXEDTASK


class QuantFreeformRawTask(QuantRawTask):
    # The penalty's three constants, as the excess over the reference (0.04, 0.08), pinned per task at
    # creation so re-tuning the constants never moves an old task. bar_cap is the hard cap. NULL penalty_knee /
    # penalty_curvature mean the cliff at bar_cap.
    bar_cap: float = QUANT_BAR_CAP
    penalty_knee: float | None = QUANT_PENALTY_KNEE
    penalty_curvature: float | None = QUANT_PENALTY_CURVATURE
    task_type: TaskType = TaskType.QUANTFREEFORMTASK


def freeform_penalty_policy(task: QuantFreeformRawTask) -> QuantFreeformPenalty:
    """The task's pinned penalty in ratio units; the cliff at bar_cap when the task carries no curvature."""
    cap = 1 + task.bar_cap
    if task.penalty_curvature is None or task.penalty_knee is None:
        return QuantFreeformPenalty.cliff(cap)
    return QuantFreeformPenalty(knee=1 + task.penalty_knee, curvature=task.penalty_curvature, cap=cap)


# The statuses of a task that has left data prep and can still be handed to miners. A compression task in one of them always has
# its published calibration set (the tasks_quant_calibration_required CHECK holds the same list).
QUANT_CALIBRATION_REQUIRED_STATUSES = frozenset(
    {
        TaskStatus.AWAITING_MODEL_PREP,
        TaskStatus.LOOKING_FOR_NODES,
        TaskStatus.DELAYED,
        TaskStatus.READY,
        TaskStatus.TRAINING,
    }
)


def has_published_calibration(task: RawTask) -> bool:
    """Whether the task holds the tokenised calibration set quant prep publishes: its file, fingerprint and example count.

    Before prep a compression task may hold only the raw rows copied from its source task (calibration_data alone)."""
    return bool(task.calibration_data and task.calibration_fingerprint and (task.calibration_n_examples or 0) > 0)


def quant_task_missing_calibration(task: RawTask) -> bool:
    """True for a compression task past data prep with no published calibration set: a state that must never be stored."""
    return (
        task.task_type in QUANT_TASK_TYPES
        and task.status in QUANT_CALIBRATION_REQUIRED_STATUSES
        and not has_published_calibration(task)
    )


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


class QuantFixedTask(QuantFixedRawTask):
    trained_model_repository: str | None = None


class QuantFreeformTask(QuantFreeformRawTask):
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


def _finite_or_none(value: float | None) -> float | None:
    """NaN and +/-inf are not valid JSON: the public scorecard reports them as null (not 0.0 and not 1e100, which would
    read as a real loss)."""
    return value if value is None or math.isfinite(value) else None


class ScorecardBytes(BaseModel):
    """Counted bytes: effective_bytes (S) = total_bytes x exp(ln_penalty). Non-finite floats are null."""

    artifact_bytes: int | None = None
    decoder_xz_bytes: int | None = None
    total_bytes: int | None = None
    ln_penalty: float | None = None
    effective_bytes: float | None = None

    @field_validator("ln_penalty", "effective_bytes", mode="after")
    @classmethod
    def _non_finite_is_null(cls, value: float | None) -> float | None:
        return _finite_or_none(value)


class ScorecardQuality(BaseModel):
    """Non-finite floats are null; a scored fail carries test_loss null and quant_failure set."""

    test_loss: float | None = None
    reference_loss: float | None = None
    ratio: float | None = None
    passed_bar: bool | None = None
    quant_failure: str | None = None

    @field_validator("test_loss", "reference_loss", "ratio", mode="after")
    @classmethod
    def _non_finite_is_null(cls, value: float | None) -> float | None:
        return _finite_or_none(value)


class ScorecardArtifact(BaseModel):
    """The public HF artifact as the hub listed it at scoring. decode_matches_hub: the trainer decoded exactly these
    files (name and bytes, and sha256 for LFS files); None when either list is unknown."""

    repo: str | None = None
    revision: str | None = None
    files: list[HubArtifactFile] = []
    decode_matches_hub: bool | None = None


class ScorecardDecoder(BaseModel):
    """Hashes of the registered decoder snapshot. The snapshot itself is private until the champion handover."""

    xz_bytes: int | None = None
    xz_sha256: str | None = None
    tar_sha256: str | None = None
    tree_sha: str | None = None
    xz_version: str | None = None


class ScorecardDecoded(BaseModel):
    """The decoded bf16 model. urls (re-signed per read) only once the tournament is completed; hf_repo is the public
    copy of a boss-round submission."""

    files: list[QuantDecodedFile] = []
    urls: dict[str, str] | None = None
    hf_repo: str | None = None


class ScorecardRelease(BaseModel):
    """Present only after the champion handover released this hotkey's decoder; training_repo is the public copy."""

    released_at: datetime
    decoder_snapshot_url: str | None = None
    training_repo: str | None = None
    training_commit_hash: str | None = None


class QuantFreeformScorecard(BaseModel):
    """What makes a freeform result verifiable without the miner's decoder."""

    bytes: ScorecardBytes
    quality: ScorecardQuality
    artifact: ScorecardArtifact
    decoder: ScorecardDecoder
    decoded: ScorecardDecoded
    release: ScorecardRelease | None = None


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
    quant_scorecard: QuantFreeformScorecard | None = None


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


class QuantFixedTaskWithHotkeyDetails(QuantFixedTask):
    hotkey_details: list[HotkeyDetails]


class QuantFreeformTaskWithHotkeyDetails(QuantFreeformTask):
    hotkey_details: list[HotkeyDetails]


# Type aliases for common task type groupings
AnyTextTypeRawTask = InstructTextRawTask | DpoRawTask | GrpoRawTask | ChatRawTask | EnvRawTask
AnyQuantRawTask = QuantFixedRawTask | QuantFreeformRawTask
AnyTypeRawTask = AnyTextTypeRawTask | ImageRawTask | AnyQuantRawTask
AnyTypeTask = InstructTextTask | DpoTask | ImageTask | GrpoTask | ChatTask | EnvTask | QuantFixedTask | QuantFreeformTask
AnyTypeTaskWithHotkeyDetails = (
    InstructTextTaskWithHotkeyDetails
    | ImageTaskWithHotkeyDetails
    | DpoTaskWithHotkeyDetails
    | GrpoTaskWithHotkeyDetails
    | ChatTaskWithHotkeyDetails
    | EnvTaskWithHotkeyDetails
    | QuantFixedTaskWithHotkeyDetails
    | QuantFreeformTaskWithHotkeyDetails
)
