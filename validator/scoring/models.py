"""Scoring models for task and tournament evaluation."""

from datetime import datetime
from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from core.constants.environments import EnvironmentName
from core.models.dataset_models import FileFormat
from core.models.task_models import TaskType


class TournamentScore(BaseModel):
    hotkey: str
    score: float


class EnvironmentWeight(BaseModel):
    """Weight for a single environment in tournament scoring."""

    environment: EnvironmentName
    weight: float = Field(default=1.0, ge=0.0, description="Scoring multiplier for this environment")


class PeriodScore(BaseModel):
    quality_score: float
    summed_task_score: float
    average_score: float
    std_score: float | None = 0.0
    hotkey: str
    weight_multiplier: float
    normalised_score: float | None = 0.0


class TaskNode(BaseModel):
    task_id: str
    hotkey: str
    quality_score: float


class MiniTaskWithScoringOnly(BaseModel):
    is_organic: bool
    task_id: UUID | None = None
    model_id: str
    ds: str
    file_format: FileFormat = FileFormat.HF
    status: str
    account_id: UUID
    times_delayed: int = 0
    hours_to_complete: float
    assigned_miners: list[int] | None = None
    miner_scores: list[float] | None = None
    task_type: TaskType
    created_at: datetime
    next_delay_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    termination_at: datetime | None = None
    completed_at: datetime | None = None
    model_params_count: int | None = 0

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())


class TaskResults(BaseModel):
    task: MiniTaskWithScoringOnly
    node_scores: list[TaskNode]


class NodeAggregationResult(BaseModel):
    task_work_scores: list[float] = Field(default_factory=list)
    average_raw_score: float | None = Field(default=0.0)
    summed_adjusted_task_scores: float = Field(default=0.0)
    quality_score: float | None = Field(default=0.0)
    emission: float | None = Field(default=0.0)
    task_raw_scores: list[float] = Field(default_factory=list)
    hotkey: str

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True


class Submission(BaseModel):
    submission_id: UUID = Field(default_factory=uuid4)
    score: float | None = None
    task_id: UUID
    hotkey: str
    repo: str
    model_hash: str | None = None
    created_on: datetime | None = None
    updated_on: datetime | None = None


class MinerResults(BaseModel):
    hotkey: str
    test_loss: float
    synth_loss: float
    is_finetune: bool
    score: float | None = 0.0
    submission: Submission | None = None
    score_reason: str | None = None
    adjusted_loss: float | None = None


class MinerResultsText(MinerResults):
    task_type: TaskType

    @field_validator("task_type")
    def validate_task_type(cls, v):
        if v not in {TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK, TaskType.ENVIRONMENTTASK}:
            raise ValueError("Must be INSTRUCTTEXTTASK, CHATTASK, DPOTASK, GRPOTASK or ENVIRONMENTTASK")
        return v


class MinerResultsImage(MinerResults):
    task_type: TaskType = TaskType.IMAGETASK


class PairwiseOutcome(BaseModel):
    """Universal outcome of a single pair comparison on a single environment.

    Produced by any eval type (PvP, MCTS, etc.) and fed into the universal
    points accumulator. The winner field is the hotkey of the winner, or
    None for a draw.
    """

    hotkey_a: str
    hotkey_b: str
    environment: EnvironmentName
    winner: str | None = Field(description="Hotkey of winner, or None for draw")


class GroupStagePoints(BaseModel):
    """Per-hotkey points from group stage evaluation (any eval type)."""

    hotkey: str
    points: float


class TournamentTypeResult(BaseModel):
    scores: list[TournamentScore]
    prev_winner_hotkey: str | None
    prev_winner_won_final: bool


class MinerRepos(BaseModel):
    """Miner hotkey → HuggingFace model repo mapping for tournament evaluation."""

    by_hotkey: dict[str, str] = Field(description="Mapping of hotkey → repo_id")

    @property
    def hotkeys(self) -> list[str]:
        return list(self.by_hotkey.keys())

    @property
    def repos(self) -> list[str]:
        return list(self.by_hotkey.values())

    def __len__(self) -> int:
        return len(self.by_hotkey)

    def subset(self, hotkeys: list[str]) -> "MinerRepos":
        """Return a new MinerRepos containing only the given hotkeys."""
        return MinerRepos(by_hotkey={hk: self.by_hotkey[hk] for hk in hotkeys if hk in self.by_hotkey})


class IndividualEvalResult(BaseModel):
    """Scores from individual eval containers for one environment."""

    environment_name: EnvironmentName
    scores_by_hotkey: dict[str, float]


class IndividualScoresByEnv(BaseModel):
    """Collected individual scores across multiple environments."""

    results: dict[EnvironmentName, IndividualEvalResult] = Field(default_factory=dict)

    def is_complete(self, envs: list[EnvironmentName], hotkeys: list[str]) -> bool:
        for env in envs:
            result = self.results.get(env)
            if not result or any(hk not in result.scores_by_hotkey for hk in hotkeys):
                return False
        return True

    def missing(self, envs: list[EnvironmentName], hotkeys: list[str]) -> list[tuple[EnvironmentName, list[str]]]:
        incomplete = []
        for env in envs:
            result = self.results.get(env)
            missing_hks = [hk for hk in hotkeys if hk not in (result.scores_by_hotkey if result else {})]
            if missing_hks:
                incomplete.append((env, missing_hks))
        return incomplete


class EnvMinerScores(BaseModel):
    """Continuous per-miner scores for a single environment, before rank normalization.

    Holds either a PvP env's win-rates or an INDIVIDUAL env's raw scores. Both are
    higher-is-better numbers per hotkey and are combined downstream by rank.
    """

    environment: EnvironmentName
    scores_by_hotkey: dict[str, float]


class EvalHotkeyResults(BaseModel):
    """Outcome of evaluating a batch of hotkeys."""

    evaluated: list[str] = Field(description="Hotkeys that were successfully evaluated")
    failed: list[str] = Field(default_factory=list, description="Hotkeys that failed evaluation")
