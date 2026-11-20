"""Scoring models for task and tournament evaluation."""

import math
from datetime import datetime
from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from core.constants.environments import EnvironmentName
from core.models.dataset_models import FileFormat
from core.models.quant_artifact_models import QuantDecodedFile
from core.models.quant_models import QuantEvalGpu
from core.models.task_models import TEXT_TASK_TYPES
from core.models.task_models import TaskType


class HubArtifactFile(BaseModel):
    """One file of a miner's HF artifact repo as the hub lists it.

    sha256 is the content sha256 for an LFS file (sha256_kind "sha256"); the hub does not expose one for a plain
    git file, whose git blob id (a sha1 over a header and the content) stands in (sha256_kind "git-blob-sha1").
    """

    name: str
    bytes: int
    sha256: str
    sha256_kind: str


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


class QuantSkuReference(BaseModel):
    """A bf16 reference as measured on one SKU (quant_sku_references); keyed by (hotkey, QuantEvalGpu.reference_key).

    hotkey is the miner whose own model was scored on an own-model task; "" is the task-common reference.
    """

    task_id: UUID
    hotkey: str = ""
    sku: str
    model: str
    loss: float
    per_example_losses: list[float]
    eval_set_fingerprint: str
    eval_gpu: QuantEvalGpu
    measured_at: datetime | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.hotkey, self.sku)

    def matches(self, other: "QuantSkuReference") -> bool:
        """Per-SKU kernels are deterministic, so a re-measurement must agree exactly.

        An example with no supervised token is NaN on both sides (null in the cache), and two NaNs count as
        the same entry: NaN != NaN would otherwise alert on every identical re-measurement.
        """
        return (
            self.model == other.model
            and self.loss == other.loss
            and self.eval_set_fingerprint == other.eval_set_fingerprint
            and len(self.per_example_losses) == len(other.per_example_losses)
            and all(
                mine == theirs or (math.isnan(mine) and math.isnan(theirs))
                for mine, theirs in zip(self.per_example_losses, other.per_example_losses)
            )
        )


QuantSkuReferences = dict[tuple[str, str], QuantSkuReference]  # (hotkey, reference_key) -> that reference


class QuantCalibrationReferenceRecord(BaseModel):
    """A miner's own bf16 model scored on the calibration rows (quant_calibration_references), one row per (task, hotkey).

    Published to the miner with the rows, never used to score. per_example_losses is NaN where an example has no
    supervised token (null in the table), like the SKU reference vectors.
    """

    task_id: UUID
    hotkey: str = ""
    model: str
    loss: float
    per_example_losses: list[float]
    eval_set_fingerprint: str
    eval_gpu: QuantEvalGpu
    measured_at: datetime | None = None

    @property
    def sku(self) -> str:
        return self.eval_gpu.reference_key


class QuantScorecardRow(BaseModel):
    """The task_nodes columns a freeform scorecard is built from. Lists are None when the row stores none."""

    hotkey: str
    test_loss: float | None = None
    quant_failure: str | None = None
    artifact_bytes: int | None = None
    decoder_bytes: int | None = None
    passed_bar: bool | None = None
    freeform_penalty: float | None = None
    eval_gpu: QuantEvalGpu | None = None
    artifact_files: list[HubArtifactFile] | None = None
    artifact_revision: str | None = None
    decode_artifact_files: list[QuantDecodedFile] | None = None
    decoded_files: list[QuantDecodedFile] | None = None
    decoded_hf_repo: str | None = None
    repo: str | None = None


class MinerResults(BaseModel):
    hotkey: str
    test_loss: float
    synth_loss: float
    is_finetune: bool
    score: float | None = 0.0
    submission: Submission | None = None
    score_reason: str | None = None
    adjusted_loss: float | None = None
    # Held-out loss per example, in eval-set order, for the paired boss-round comparison. Only
    # carried for instruct/DPO; None everywhere else. Never used for ranking - test_loss is.
    per_example_losses: list[float] | None = None
    eval_set_fingerprint: str | None = None
    # True when the evaluator attempted the vector pass and failed; False/default when absent or unused.
    per_example_losses_failed: bool = False
    # 1-based position among ranked valid results; None when not ranked.
    rank: int | None = None
    # The value the rank key ordered on: test_loss, or effective bytes for a freeform quant entry under the cap.
    rank_metric: float | None = None
    # Quant results only; None for every other task type.
    artifact_bytes: int | None = None
    decoder_bytes: int | None = None
    # Freeform: the hub listing the bytes were counted from (files with hashes, repo revision); persisted to
    # task_nodes.artifact_files / artifact_revision so the scorecard is verifiable without the decoder.
    artifact_files: list[HubArtifactFile] | None = None
    artifact_revision: str | None = None
    # Freeform: under the cap (the bar before the penalty); persisted to task_nodes.passed_bar.
    passed_bar: bool | None = None
    # Freeform: ln P as scored, curvature x max(0, r - knee)^2 (0 for a cliff task); persisted to
    # task_nodes.freeform_penalty so a re-rank from the rows alone orders on the same effective bytes.
    freeform_penalty: float | None = None
    # Set for a quant scored fail: ranked last on the task, not an evaluation error.
    quant_failure: str | None = None
    # The GPU and SKU a quant eval ran on; persisted to task_nodes.eval_gpu.
    eval_gpu: QuantEvalGpu | None = None
    # Quant only: the task's bf16 reference loss on eval_gpu's SKU. Ranking, the bar and adjusted_loss use
    # test_loss / reference_loss, so losses from different SKUs compare to first order.
    reference_loss: float | None = None

    @property
    def total_bytes(self) -> int | None:
        if self.artifact_bytes is None or self.decoder_bytes is None:
            return None
        return self.artifact_bytes + self.decoder_bytes

    @property
    def effective_bytes(self) -> float | None:
        """total_bytes x exp(freeform_penalty) from the persisted penalty; plain bytes when none was scored."""
        if self.total_bytes is None:
            return None
        return float(self.total_bytes) * math.exp(self.freeform_penalty or 0.0)


class MinerResultsText(MinerResults):
    task_type: TaskType

    @field_validator("task_type")
    def validate_task_type(cls, v):
        if v not in TEXT_TASK_TYPES:
            raise ValueError(f"Must be one of {sorted(t.value for t in TEXT_TASK_TYPES)}")
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
    deferred_hotkeys: list[str] = Field(default_factory=list)


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
    deferred: list[str] = Field(
        default_factory=list,
        description="Hotkeys that could not get eval capacity and should retry next cycle",
    )
