import secrets
from datetime import datetime
from enum import Enum
from typing import Any
from typing import NamedTuple
from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from core.constants.environments import EnvironmentName as EnvironmentName
from core.models.image_models import ImageModelType
from core.models.image_models import ImageTrainingMode
from core.models.payload_models import TrainingRepoResponse
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskType
from core.models.tournament_models import TournamentType
from core.models.trainer_contract_models import GPUInfo
from validator.scoring.models import EnvironmentWeight as EnvironmentWeight
from validator.scoring.models import EnvMinerScores as EnvMinerScores
from validator.scoring.models import EvalHotkeyResults as EvalHotkeyResults
from validator.scoring.models import GroupStagePoints as GroupStagePoints
from validator.scoring.models import PairwiseOutcome as PairwiseOutcome
from validator.scoring.models import TournamentScore
from validator.scoring.models import TournamentTypeResult as TournamentTypeResult
from validator.tasks.models import AnyTypeRawTask
from validator.tournament.decoder_snapshot import DecoderSnapshot


class TrainingRepoInfo(NamedTuple):
    training_repo: str | None
    training_commit_hash: str | None
    github_token: str | None
    requested_datasets: list[str] | None

    @classmethod
    def empty(cls) -> "TrainingRepoInfo":
        return cls(None, None, None, None)


class TournamentStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TrainingStatus(str, Enum):
    PENDING = "pending"
    TRAINING = "training"
    SUCCESS = "success"
    FAILURE = "failure"


class RoundStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"


class RoundType(str, Enum):
    GROUP = "group"
    KNOCKOUT = "knockout"


class GpuRequirement(str, Enum):
    A100 = "A100"
    H100_1X = "1xH100"
    H100_2X = "2xH100"
    H100_4X = "4xH100"
    H100_8X = "8xH100"

    @property
    def gpu_count(self) -> int:
        return {
            GpuRequirement.A100: 1,
            GpuRequirement.H100_1X: 1,
            GpuRequirement.H100_2X: 2,
            GpuRequirement.H100_4X: 4,
            GpuRequirement.H100_8X: 8,
        }[self]


class DedupTier(str, Enum):
    """How duplicate submissions were established."""

    T0 = "T0"
    T1 = "T1"
    T2 = "T2"


class DupRelationship(str, Enum):
    DUPLICATE = "duplicate"
    DISTINCT = "distinct"
    DROP_EVASION = "drop_evasion"


class TrainerInfo(BaseModel):
    trainer_ip: str = Field(..., description="Trainer IP address")
    gpus: list[GPUInfo] = Field(..., description="List of GPUs available on this trainer")


def generate_tournament_id() -> str:
    hash_part = secrets.token_hex(8)
    date_part = datetime.now().strftime("%Y%m%d")
    return f"tourn_{hash_part}_{date_part}"


def generate_round_id(tournament_id: str, round_number: int) -> str:
    return f"{tournament_id}_round_{round_number:03d}"


class TournamentData(BaseModel):
    tournament_id: str
    tournament_type: TournamentType
    status: TournamentStatus = TournamentStatus.PENDING
    base_winner_hotkey: str | None = Field(
        default=None, description="The defending champion's real hotkey at the START of this tournament (snapshot)."
    )
    winner_hotkey: str | None = Field(
        default=None,
        description="The tournament winner's hotkey at the END of this tournament. "
        "May be EMISSION_BURN_HOTKEY if the defending champion successfully defended.",
    )
    code_review: str | None = Field(
        default=None,
        description="Stored boss-round review. The live cheat check is per participant and runs before round 2.",
    )
    diff_report: str | None = Field(
        default=None, description="Optional S3 URL for the winner-vs-previous-boss diff report."
    )
    winner_model_repo: str | None = Field(
        default=None, description="HF repo of the winning trained model (for next tournament's final round)"
    )
    winner_model_base: str | None = Field(
        default=None, description="Base model the winner was trained from (for compatibility check)"
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Timestamp when the tournament was last updated (typically when it completed). "
        "Used for time-based decay calculations - represents when the champion won/defended.",
    )


class TournamentRoundData(BaseModel):
    round_id: str
    tournament_id: str
    round_number: int
    round_type: RoundType
    is_final_round: bool = False
    status: RoundStatus = RoundStatus.PENDING


class TournamentGroupData(BaseModel):
    group_id: str
    round_id: str


class TournamentPairData(BaseModel):
    pair_id: str
    round_id: str
    hotkey1: str
    hotkey2: str
    winner_hotkey: str | None = None


class TournamentParticipant(BaseModel):
    tournament_id: str
    hotkey: str
    eliminated_in_round_id: str | None = None
    final_position: int | None = None
    training_repo: str | None = None
    training_commit_hash: str | None = None
    github_token: str | None = None
    backup_repo: str | None = None
    requested_datasets: list[str] | None = None
    code_review: str | None = Field(
        default=None,
        description="Round-2 cheat review: clean, pending, accepted, rejected, or error.",
    )
    decoder_object_name: str | None = Field(
        default=None, description="MinIO object of the decoder/ snapshot taken at the registered commit."
    )
    decoder_xz_bytes: int | None = Field(default=None, description="Counted decoder bytes: size of the snapshot .tar.xz.")
    decoder_sha256: str | None = Field(
        default=None, description="sha256 of the snapshot .tar.xz; the trainer verifies it before decoding."
    )

    decoder_tar_sha256: str | None = Field(default=None, description="sha256 of the uncompressed canonical decoder tar.")
    decoder_tree_sha: str | None = Field(
        default=None, description="Git tree id of decoder/ at the registered commit; survives the handover squash."
    )
    decoder_xz_version: str | None = Field(default=None, description="xz version that compressed the snapshot.")
    decoder_released_url: str | None = Field(default=None, description="Public URL of the released snapshot (champion handover).")
    decoder_released_at: datetime | None = Field(default=None, description="When the decoder snapshot was released publicly.")

    @property
    def has_decoder_snapshot(self) -> bool:
        return None not in (self.decoder_object_name, self.decoder_xz_bytes, self.decoder_sha256)


class PendingDecoderRelease(BaseModel):
    """A tournament winner whose decoder snapshot is not yet public."""

    tournament_id: str
    hotkey: str


class FreeformReconcileReport(BaseModel):
    """What the freeform publication reconcile found and, when applied, did."""

    applied: bool = False
    pending_releases: list[PendingDecoderRelease] = Field(
        default_factory=list, description="Winners with a decoder snapshot and a pushed backup repo, not yet released."
    )
    blocked_releases: list[PendingDecoderRelease] = Field(
        default_factory=list, description="Winners with a decoder snapshot but no backup_repo; the code push must happen first."
    )
    pending_decoded_tournaments: list[str] = Field(
        default_factory=list, description="Tournaments with a winner and unpublished boss-round decoded models."
    )
    released: list[PendingDecoderRelease] = Field(default_factory=list, description="Decoder releases that now stand.")
    failed_releases: list[PendingDecoderRelease] = Field(
        default_factory=list, description="Releases refused or failed this pass."
    )
    published_models: list[str] = Field(default_factory=list, description="Decoded model repos created this pass.")


class TournamentTask(BaseModel):
    tournament_id: str
    round_id: str
    task_id: str
    group_id: str | None = None
    pair_id: str | None = None
    gpu_requirement: GpuRequirement | None = None

    @field_validator("task_id", mode="before")
    @classmethod
    def ensure_str(cls, v):
        if isinstance(v, UUID):
            return str(v)
        return v


class Group(BaseModel):
    member_ids: list[str]
    task_ids: list[str] | None = None


class BaseRound(BaseModel):
    round_id: str
    round_number: int = 1


class GroupRound(BaseRound):
    groups: list[Group]


class KnockoutRound(BaseRound):
    # pairs of hotkeys
    pairs: list[tuple[str, str]]
    tasks: list[str] | None = None


Round = GroupRound | KnockoutRound


class MatchRanking(BaseModel):
    """Competitors ranked best-first for one small-tournament match."""

    task_id: str
    ranked_hotkeys: list[str]


class EmptyScoreGroup(BaseModel):
    """A group whose task produced no usable scores — evaluation data is missing, not merely bad."""

    group_id: str
    task_id: str
    participants: int


class GroupMatchStanding(BaseModel):
    """A competitor's standing across all matches in a small tournament group."""

    hotkey: str
    total_rank: float
    matches_attended: int
    total_matches: int
    summed_loss: float
    # Sum over matches of the distance from that match's best metric, normalised by it; the tiebreak.
    summed_gap: float = 0.0
    # Average rank on the most recent round's task(s) within the cohort; the tiebreak after average rank. 0 when not tracked.
    recent_round_rank: float = 0.0

    @property
    def average_rank(self) -> float:
        return self.total_rank / self.total_matches if self.total_matches else float("inf")

    @property
    def has_error(self) -> bool:
        return self.matches_attended < self.total_matches


class FirstTwoRoundRanks(BaseModel):
    """A cohort member's rank on the R1 and R2 tasks, ranked within the cohort."""

    round1: int
    round2: int


class Round3ViabilityVerdict(BaseModel):
    """Whether a cohort member can still finish top or joint top over R1+R2+R3."""

    hotkey: str
    round1_rank: int
    round2_rank: int
    viable: bool
    # Smallest achievable (own rank total - best rival total) over every R3 rank assignment; <= 0 is viable.
    best_case_margin: int


class TournamentRound(BaseModel):
    round_structure: Round
    tasks: list[str] = Field(default_factory=list)
    is_final_round: bool = False


class TaskTrainingAssignment(BaseModel):
    """Data for assigning a task-hotkey pair for training with repo information."""

    task_id: str
    hotkey: str
    created_at: datetime
    priority: int = Field(
        default=1, description="Training priority: 1=organic (non-tournament/non-benchmark), 2=tournament, 3=benchmark"
    )
    training_repo: str | None = None
    training_commit_hash: str | None = None
    github_token: str | None = None
    requested_datasets: list[str] | None = None


class TournamentTaskTraining(BaseModel):
    task: AnyTypeRawTask
    hotkey: str
    training_status: TrainingStatus
    n_training_attempts: int
    created_at: datetime
    updated_at: datetime
    training_repo: str | None = None
    training_commit_hash: str | None = None
    github_token: str | None = None
    requested_datasets: list[str] | None = None
    priority: int = 1  # Training priority: 1=organic, 2=tournament, 3=benchmark
    trainer_ip: str | None = None


class TournamentTaskScore(BaseModel):
    task_id: str
    group_id: str | None
    pair_id: str | None
    winner: str | None
    participant_scores: list[dict]


class PvPPairEnvResult(BaseModel):
    """Head-to-head game results for one pair in one environment."""

    hotkey_a: str
    hotkey_b: str
    environment_name: str
    hotkey_a_wins: int
    hotkey_b_wins: int
    draws: int
    total_games: int


class PvPIndividualEnvScore(BaseModel):
    """Per-environment aggregate score for one environment participant."""

    hotkey: str
    environment_name: str
    score: float


class PairedExampleWins(BaseModel):
    """How a two-way task split example by example, for the two competitors that were scored on it.

    Reporting only - the winner is decided in round_results.py. Null on tasks with no stored
    per-example vectors, which is every task before the paired rollout and every task type outside
    PAIRED_BOSS_ROUND_TASK_TYPES, so consumers must keep their mean-loss view for those.
    """

    hotkey_a: str
    hotkey_b: str
    hotkey_a_wins: int
    hotkey_b_wins: int
    ties: int  # examples inside the dead zone, won by neither
    n_examples: int
    deadzone_nats: float


class DetailedTournamentTaskScore(TournamentTaskScore):
    task_type: TaskType | None = None
    environment_names: list[str] | None = None
    pvp_pair_results: list[PvPPairEnvResult] | None = None
    pvp_individual_scores: list[PvPIndividualEnvScore] | None = None
    paired_example_wins: PairedExampleWins | None = None


class TournamentRoundResult(BaseModel):
    round_id: str
    round_number: int
    round_type: str
    is_final_round: bool
    tasks: list[TournamentTaskScore]


class DetailedTournamentRoundResult(TournamentRoundResult):
    status: str
    participants: list[str]
    tasks: list[DetailedTournamentTaskScore]


class TournamentResults(BaseModel):
    tournament_id: str
    rounds: list[TournamentRoundResult]


class BossRankComparison(BaseModel):
    """A contender in a paid place (2nd or 3rd) of a tournament the boss kept, compared with the boss on average rank.

    The averages are over ``compared_round_numbers``: the rounds 1-3 in which both the contender and the boss have a rank
    (RoundRanks). ``share_to_boss`` is true when the boss's average is strictly lower (better) than the contender's: the
    contender's share of the pool is then paid like the boss's own share. With no compared round both averages are None
    and the contender is paid.
    """

    hotkey: str
    final_position: int
    contender_average_rank: float | None = None
    boss_average_rank: float | None = None
    compared_round_numbers: list[int] = Field(default_factory=list)
    share_to_boss: bool = False

    def describe(self) -> str:
        if self.contender_average_rank is None or self.boss_average_rank is None:
            return f"place {self.final_position} {self.hotkey}: no round ranks both it and the boss, paid"
        outcome = "share to the boss" if self.share_to_boss else "paid"
        return (
            f"place {self.final_position} {self.hotkey}: boss average rank {self.boss_average_rank:.3f} vs contender "
            f"{self.contender_average_rank:.3f} over rounds {self.compared_round_numbers}, {outcome}"
        )


class TournamentResultsWithWinners(BaseModel):
    tournament_id: str
    rounds: list[TournamentRoundResult]
    base_winner_hotkey: str | None = None
    winner_hotkey: str | None = None
    final_positions: dict[str, int] = Field(default_factory=dict)
    # Stored when the tournament completed with the boss kept (a tournament that plays the round plan); empty otherwise.
    boss_rank_comparisons: list[BossRankComparison] = Field(default_factory=list)


class PairedLossComparison(BaseModel):
    """Outcome of comparing boss and challenger example by example on the same held-out set.

    Recorded on every boss-round task whether the challenger won or lost — the observed win rate
    is the data needed to calibrate BOSS_ROUND_MIN_WIN_RATE against real matchups.
    """

    n_examples: int
    n_decided: int  # examples where the gap exceeded the tie dead zone
    challenger_example_wins: int
    boss_example_wins: int
    win_rate: float  # challenger_example_wins / n_decided, 0.0 when nothing was decided
    win_rate_lower_bound: float  # one-sided bootstrap bound
    mean_gap_nats: float  # boss - challenger over all examples; positive = challenger better
    mean_gap_lower_bound: float
    challenger_won: bool
    # Every comparable example fell inside the tie dead zone: the two models are indistinguishable
    # on this task, which is a different statement from the challenger having lost it. The task
    # still counts to the defender for the dethrone tally - somebody has to hold it - but recording
    # it as a boss win misdescribes what happened.
    is_draw: bool = False
    reason: str
    # The bars this verdict was judged against, carried with it so a reader does not have to guess
    # which constants were in force when the round ran.
    required_win_rate: float
    required_mean_gap_nats: float


class BossRoundTaskVerdict(BaseModel):
    """How one boss-round task resolved.

    ``is_draw`` is carried separately from ``winner_hotkey`` because a draw is not a result: nothing
    separated the two models, so the task tells the dethrone tally nothing and is excluded from it.
    ``winner_hotkey`` still names the defender on a draw, since the persisted score row needs
    somebody to hold the task.
    """

    # A mistyped field name on this model silently dropped a value that only stayed harmless because
    # the one consumer short-circuited before reading it. These carry crowning decisions; a typo
    # should fail loudly at construction.
    model_config = ConfigDict(extra="forbid")

    winner_hotkey: str
    # What decided it, for the persisted score_reason. None when the relative margin decided it.
    decided_by: str | None = None
    is_draw: bool = False
    # Quant only, for analytics (crowning never reads it): whether the challenger lost narrowly. False on every
    # path that cannot show it (no vectors, unpairable, no same-SKU reference, a failed score).
    challenger_within_allowed_loss: bool = False


class ContinuousSftLineageOutcome(BaseModel):
    """How one continuous-SFT lineage went in a boss round, for the dethrone gate.

    The gate asks the challenger to win every lineage, with one deliberate exception: a draw counts
    as satisfying it. Nothing separated the two models on that task, which is not the challenger
    failing to beat the boss - it is the task failing to tell them apart, and the challenger should
    not lose the crown to it. A loss is different: the boss demonstrably held that lineage.

    ``is_draw`` and a ``winner_hotkey`` of None are also different. A draw means both sides were
    scored and came out level; None means the task produced no comparison at all (failed, skipped,
    unpairable vectors), which is absence of evidence and keeps blocking as it always has.
    """

    model_config = ConfigDict(extra="forbid")

    lineage: str
    winner_hotkey: str | None = None
    is_draw: bool = False

    def satisfied_by(self, challenger_hotkey: str | None) -> bool:
        return self.is_draw or (challenger_hotkey is not None and self.winner_hotkey == challenger_hotkey)


class BossRoundDrawResolution(BaseModel):
    """What the draws in a completed boss round mean for advancing it.

    A boss-round task draws when every held-out example landed inside the tie dead zone (see
    compare_paired_losses), which happens when the randomly drawn dataset simply does not
    discriminate between the two models. That is a property of the task, not of the challenger, so
    it must not consume the one task the challenger is allowed to drop. Instead each drawn task is
    excluded from the tally and one additional task is added to the round to take its place, so the
    round still resolves on the full number of decided results.
    """

    model_config = ConfigDict(extra="forbid")

    drawn_task_ids: list[str] = Field(default_factory=list)
    # Task types to add, one per drawn task, mirroring what drew - except continuous-SFT, which is
    # stood in for by a plain instruct task on a fresh dataset pull. A drawn lineage has already
    # cleared its own dethrone gate (a draw satisfies it), so its decider exists only to top the
    # tally back up; adding a second continuous-SFT task on that lineage would instead put two
    # tasks on one sequential train_index and leave the chain's carry-forward ambiguous.
    decider_task_types: list[TaskType] = Field(default_factory=list)
    # False once deciders have already been added for this round, which caps them at one per drawn
    # task. A decider that itself draws is simply excluded and the round resolves on what is left.
    can_add_deciders: bool = True
    reason: str = ""

    @property
    def needs_deciders(self) -> bool:
        return bool(self.drawn_task_ids) and self.can_add_deciders


class BossTaskAnalytics(BaseModel):
    """Analytics' mirror of one quant boss-round task verdict: who won, the paired evidence, and the allowed loss.

    within_allowed_loss is None only when the challenger won or the task is not quant (a drawn task reports the
    answer); otherwise the same answer crowning used (the reversed paired bar or the freeform byte margin).
    """

    challenger_won: bool
    paired_comparison: PairedLossComparison | None = None
    reason: str | None = None
    within_allowed_loss: bool | None = None


class RoundOutcome(BaseModel):
    """Result of resolving a completed round.

    ``deferred_reason`` is set when the round cannot be resolved yet, either because tasks are about
    to be added to it (a boss-round draw decider), because tasks are already on it that never got
    nodes, or because crown tasks have an evaluation fault. The caller must not advance on a deferred
    outcome: ``winners`` is empty then, and an empty winner list otherwise means the boss retained by
    default.
    """

    model_config = ConfigDict(extra="forbid")

    winners: list[str] = Field(default_factory=list)
    deferred_reason: str | None = None
    draw_resolution: BossRoundDrawResolution | None = None
    # Round tasks with no assigned nodes, so they cannot have run. Resolving around them would score
    # a task that never happened against whoever the tally charges it to; the round goes back to
    # PENDING instead so node assignment picks them up. Populated when a decider was created but the
    # round never made it back to PENDING - a crash, or a failure partway through creating a batch.
    unassigned_task_ids: list[str] = Field(default_factory=list)
    # Seeded text boss round only: stage 1 is scored and the challenger can still be crowned, so the round needs its big
    # task before it can be resolved. The caller creates it and returns the round to PENDING.
    needs_text_big_task: bool = False
    # Planned image boss round only: crown tasks on which both sides trained and neither has a valid result. The round is held
    # and an alert is sent; it resolves on its own once the evaluation results exist.
    evaluation_fault_task_ids: list[str] = Field(default_factory=list)

    @property
    def is_deferred(self) -> bool:
        return self.deferred_reason is not None


class TaskPerformanceDifference(BaseModel):
    """Performance difference data for a single task"""

    task_id: str
    task_type: str
    boss_score: float | None
    challenger_score: float | None
    threshold_used: float  # boss-round win margin (BOSS_ROUND_WIN_MARGIN)
    performance_difference: float | None  # Percentage difference (positive = challenger better)
    challenger_won: bool
    # Set when the task was decided on per-example win rate rather than the margin above. Null
    # means challenger_won came from threshold_used, as it did for every pre-rollout tournament.
    paired_comparison: PairedLossComparison | None = None
    # Why the task went the way it did, when the paired path decided it. Not the same as
    # paired_comparison.reason: the KL scalar gate can overturn a won paired comparison, so
    # challenger_won may be False while the comparison inside it says the challenger won.
    decision_reason: str | None = None
    # Quant only: None when the challenger won (or the task is not quant); otherwise whether the challenger's loss
    # was within the allowed loss, by the same function crowning uses.
    within_allowed_loss: bool | None = None


class TournamentPerformanceData(BaseModel):
    """Performance data for tournament vs sync comparison"""

    tournament_task_id: str
    synthetic_task_id: str
    task_type: str
    tournament_winner_score: float
    best_synthetic_score: float
    performance_difference: float  # Percentage difference (positive = tournament better)


class TournamentDetailsResponse(BaseModel):
    tournament_id: str
    tournament_type: TournamentType
    status: TournamentStatus
    base_winner_hotkey: str | None
    winner_hotkey: str | None
    diff_report: str | None = None
    participants: list[TournamentParticipant]
    rounds: list[DetailedTournamentRoundResult]
    final_scores: list[TournamentScore]
    text_tournament_weight: float
    image_tournament_weight: float
    quant_tournament_weight: float = 0.0
    boss_round_performance: list[TaskPerformanceDifference] | None = None
    sync_performance: list[TournamentPerformanceData] | None = None


class TournamentAuditData(BaseModel):
    text_tournament_data: TournamentResultsWithWinners | None = None
    image_tournament_data: TournamentResultsWithWinners | None = None
    quant_tournament_data: TournamentResultsWithWinners | None = None
    participants: list[str] = []
    text_tournament_weight: float = 0.0
    image_tournament_weight: float = 0.0
    quant_tournament_weight: float = 0.0
    burn_weight: float = 0.0
    weekly_participation: list["HotkeyTaskParticipation"] = []


class BossRoundTaskCompletion(BaseModel):
    total_synth_tasks: int
    completed_synth_tasks: int


class BossRoundTaskPair(BaseModel):
    tournament_task_id: str
    synthetic_task_id: str
    winner_hotkey: str
    task_type: str


class TaskScore(BaseModel):
    hotkey: str
    test_loss: float
    synth_loss: float
    quality_score: float


class GitHubOwnerRepo(BaseModel):
    owner: str
    repo: str


class RespondingNode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    node: Any
    training_repo_response: TrainingRepoResponse
    decoder_snapshot: DecoderSnapshot | None = None


class QuantDecodeRow(BaseModel):
    """A task_nodes row's freeform decode outcome, as persisted from the trainer's decode result."""

    hotkey: str
    artifact_bytes: int | None
    decoder_bytes: int | None
    decoder_sha256: str | None
    quant_failure: str | None

    @property
    def settled(self) -> bool:
        # decoder_sha256 is written only by the decode result (never by evaluation), so it marks a settled decode.
        return self.quant_failure is not None or self.decoder_sha256 is not None


class NextTournamentInfo(BaseModel):
    tournament_type: TournamentType
    next_start_date: datetime | None = None
    next_end_date: datetime | None = None
    current_round_number: int | None = None
    tournament_status: str | None = None
    # Legacy fields for frontend compatibility
    interval_hours: int | None = None
    # New scheduling fields
    scheduled_day_of_week: int | None = None  # 0=Monday, 6=Sunday
    scheduled_hour: int | None = None  # 0-23 UTC
    scheduled_minute: int | None = None  # Always 0


class NextTournamentDates(BaseModel):
    text: NextTournamentInfo
    image: NextTournamentInfo
    quant: NextTournamentInfo | None = None


class ActiveTournamentParticipant(BaseModel):
    hotkey: str


class ActiveTournamentInfo(BaseModel):
    tournament_id: str
    tournament_type: TournamentType
    status: TournamentStatus
    participants: list[ActiveTournamentParticipant]
    created_at: datetime


class ActiveTournamentsResponse(BaseModel):
    text: ActiveTournamentInfo | None
    image: ActiveTournamentInfo | None
    quant: ActiveTournamentInfo | None = None


class TournamentBurnData(BaseModel):
    """Separated burn data by tournament type"""

    text_burn_proportion: float
    image_burn_proportion: float
    text_tournament_weight: float
    image_tournament_weight: float
    burn_weight: float
    quant_burn_proportion: float = 0.0
    quant_tournament_weight: float = 0.0


class LatestTournamentsDetailsResponse(BaseModel):
    """Response for latest tournaments with burn data"""

    text: TournamentDetailsResponse | None
    image: TournamentDetailsResponse | None
    burn_data: TournamentBurnData
    quant: TournamentDetailsResponse | None = None


class TournamentHistoryEntry(BaseModel):
    """Individual tournament entry for history response"""

    tournament_id: str
    tournament_type: TournamentType
    status: TournamentStatus
    winner_hotkey: str | None = None
    base_winner_hotkey: str | None = None
    created_at: datetime | None = None


class TournamentHistoryResponse(BaseModel):
    """Response for tournament history endpoint"""

    tournaments: list[TournamentHistoryEntry]


class BenchmarkTaskCopy(BaseModel):
    """Raw benchmark task copy data from database"""

    copy_task_id: str
    root_task_id: str
    participant_hotkey: str
    tournament_id: str | None = None
    created_at: datetime
    task_type: TaskType
    model_id: str
    dataset: str
    hours_to_complete: float
    model_params_count: int
    is_organic: bool
    task_created_at: datetime | None = None


class BenchmarkInstance(BaseModel):
    """A single benchmark instance (copy task) with its results"""

    copy_task_id: str
    participant_hotkey: str
    tournament_id: str
    created_at: datetime
    test_loss: float | None = None


class BenchmarkTimeline(BaseModel):
    """Timeline of benchmark results for a single root task"""

    root_task_id: str
    task_type: TaskType
    model_id: str
    dataset: str
    hours_to_complete: float
    model_params_count: int
    is_organic: bool
    task_created_at: datetime | None = None
    benchmarks: list[BenchmarkInstance] = Field(default_factory=list)


class BenchmarkTimelineResponse(BaseModel):
    """Response containing benchmark timelines for all tasks"""

    timelines: list[BenchmarkTimeline]


class HotkeyTournamentParticipation(BaseModel):
    """Tournament participation data for a specific hotkey"""

    hotkey: str
    participated_in_text: bool  # participated in the most recent text tournament
    participated_in_image: bool  # participated in the most recent image tournament
    text_proportion: float  # 0.0, 0.6, or 1.0 based on participation
    image_proportion: float  # 0.0, 0.4, or 1.0 based on participation


class HotkeyTaskParticipation(BaseModel):
    """Weekly task participation data for a specific hotkey"""

    hotkey: str
    text_task_proportion: float  # proportion of text tasks (0.0 to 1.0)
    image_task_proportion: float  # proportion of image tasks (0.0 to 1.0)
    total_tasks: int  # total number of tasks in the period


class NodeWeightsResult(BaseModel):
    """Result of node weight calculations"""

    node_ids: list[int]
    node_weights: list[float]

    def to_tuple(self) -> tuple[list[int], list[float]]:
        """Convert to tuple format for compatibility with existing code"""
        return self.node_ids, self.node_weights


class MinerEmissionWeight(BaseModel):
    hotkey: str
    rank: int
    weight: float


class TournamentWeightsResponse(BaseModel):
    burn_data: TournamentBurnData
    text_top_miners: list[MinerEmissionWeight]
    image_top_miners: list[MinerEmissionWeight]
    quant_top_miners: list[MinerEmissionWeight] = []


class WeightProjection(BaseModel):
    days: int
    weight: float
    total_alpha: float


class TournamentProjection(BaseModel):
    tournament_type: str
    current_champion_decay: float
    initial_weight: float
    projections: list[WeightProjection]
    # "champion" if the projected performance dethrones the boss, else "runner_up"
    placement: str = "champion"
    # Performance margin (env: win rate) the challenger must exceed to take the crown
    dethrone_threshold: float = 0.0


class WeightProjectionResponse(BaseModel):
    percentage_improvement: float
    text_projection: TournamentProjection
    image_projection: TournamentProjection
    quant_projection: TournamentProjection | None = None


class MultiWeightProjectionResponse(BaseModel):
    projections: list[WeightProjectionResponse]


class BossBattleResponse(BaseModel):
    """Response for boss battle performance differences"""

    text_tournament_id: str | None
    text_performance_differences: list[TaskPerformanceDifference]
    image_tournament_id: str | None
    image_performance_differences: list[TaskPerformanceDifference]
    quant_tournament_id: str | None = None
    quant_performance_differences: list[TaskPerformanceDifference] = []


class DedupReviewStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    SKIPPED = "skipped"


class TaskFailureReviewStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"


class TournamentTaskFailureReview(BaseModel):
    """Manual gate for a task whose trainings mostly failed.

    Held at PENDING_REVIEW until a human confirms the failures are the miners' own and not
    an infrastructure fault, at which point the task is allowed to complete.
    """

    task_id: str
    tournament_id: str
    round_id: str
    status: TaskFailureReviewStatus = TaskFailureReviewStatus.PENDING_REVIEW
    failed_hotkeys: list[str] = []
    total_trainings: int = 0
    notes: str | None = None
    created_at: datetime | None = None
    reviewed_at: datetime | None = None


class IntegrityVerdict(BaseModel):
    flagged: bool
    needs_review: bool = False
    confidence: float = 0.0
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)
    unresolved_dependencies: list[str] = Field(default_factory=list)


class DedupPairVerdict(BaseModel):
    hotkey_a: str
    hotkey_b: str
    tier: DedupTier
    relationship: DupRelationship
    confidence: float
    reason: str


class DedupClusterRecord(BaseModel):
    members: list[str]
    basis: DedupTier
    reason: str


class PublishedRepo(BaseModel):
    hotkey: str
    public_repo_url: str
    commit_hash: str | None = None


class TournamentDedupReview(BaseModel):
    round_id: str
    tournament_id: str
    tournament_type: str
    status: DedupReviewStatus = DedupReviewStatus.PENDING_REVIEW
    cohort: list[str] = []
    clusters: list[DedupClusterRecord] = []
    pair_verdicts: list[DedupPairVerdict] = []
    flagged_hotkeys: list[str] = []
    approved_eliminations: list[str] = []
    published_repos: list[PublishedRepo] = []
    report_url: str | None = None
    notes: str | None = None
    created_at: datetime | None = None
    reviewed_at: datetime | None = None
    resolved_at: datetime | None = None


class GateDecision(BaseModel):
    halt: bool
    eliminate: set[str] = Field(default_factory=set)


class RepoRef(BaseModel):
    hotkey: str
    repo_url: str
    commit_hash: str | None = None
    github_token: str | None = None


class PreparedRepo(BaseModel):
    hotkey: str
    repo_url: str
    head_commit: str | None = None
    normalized_digest: str | None = None
    content_chars: int = 0
    path: str | None = None
    clone_ok: bool = False


class PairVerdict(BaseModel):
    hotkey_a: str
    hotkey_b: str
    tier: DedupTier
    relationship: DupRelationship
    confidence: float
    reason: str


class DedupCluster(BaseModel):
    members: list[str]
    basis: DedupTier
    reason: str


class DedupResult(BaseModel):
    cohort: list[str]
    clusters: list[DedupCluster] = []
    pair_verdicts: list[PairVerdict] = []
    flagged_hotkeys: list[str] = []
    evasion_hotkeys: list[str] = []
    unclonable_hotkeys: list[str] = []
    unresolved_pairs: list[tuple[str, str]] = []


class ChunkScheme(str, Enum):
    """How the content service cuts the continuous-SFT data: the unit a train_index counts in."""

    V1 = "v1"
    V2 = "v2"  # chunk k of V1 is chunks 2k and 2k+1 of this scheme


class ContinuousSftState(BaseModel):
    """Per-lineage state for the continuous-SFT boss task (one row per lineage slug).

    train_index: monotonic cursor passed to the stateless content service (advanced by one chunk per task),
        counted in chunk_scheme units. A lineage with no row starts at index 0 in the current scheme.
    last_winner_repo: previous lowest-eval-loss winner, carried forward as the next base; None first run.
    """

    lineage: str
    train_index: int = 0
    chunk_scheme: ChunkScheme = ChunkScheme.V2
    last_winner_repo: str | None = None
    updated_at: datetime | None = None


class ContinuousSftTaskChunk(BaseModel):
    """The cursor value and chunk scheme one continuous-SFT task was created with."""

    task_id: UUID
    lineage: str
    train_index: int
    chunk_scheme: ChunkScheme


class GpuCostCategoryBreakdown(BaseModel):
    wall_hours: float = 0.0
    gpu_hours: float = 0.0
    cost_usd: float = 0.0
    success_count: int = 0
    failure_count: int = 0


class TaskGpuCostBreakdown(BaseModel):
    task_id: UUID
    tournament_id: str
    task_type: str | None = None
    base_model: str | None = None
    round_number: int | None = None
    round_type: str | None = None
    hotkey_count: int = 0
    training: GpuCostCategoryBreakdown = Field(default_factory=GpuCostCategoryBreakdown)
    prep: GpuCostCategoryBreakdown = Field(default_factory=GpuCostCategoryBreakdown)
    evaluation: GpuCostCategoryBreakdown = Field(default_factory=GpuCostCategoryBreakdown)
    total_cost_usd: float = 0.0


class TournamentGpuCostBreakdown(BaseModel):
    tournament_id: str
    tournament_type: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    participant_count: int = 0
    fee_collected_tao: float = 0.0
    fee_collected_usd: float = 0.0
    training: GpuCostCategoryBreakdown = Field(default_factory=GpuCostCategoryBreakdown)
    prep: GpuCostCategoryBreakdown = Field(default_factory=GpuCostCategoryBreakdown)
    evaluation: GpuCostCategoryBreakdown = Field(default_factory=GpuCostCategoryBreakdown)
    total_attributed_cost_usd: float = 0.0


class TournamentGpuCostTotals(BaseModel):
    training_wall_hours: float = 0.0
    training_gpu_hours: float = 0.0
    training_cost_usd: float = 0.0
    prep_wall_hours: float = 0.0
    prep_gpu_hours: float = 0.0
    prep_cost_usd: float = 0.0
    evaluation_wall_hours: float = 0.0
    provisioned_h100_gpu_hours: float = 0.0
    provisioned_h100_cost_usd: float = 0.0
    attributed_h100_gpu_hours: float = 0.0
    attributed_h100_cost_usd: float = 0.0
    idle_h100_gpu_hours: float = 0.0
    idle_h100_cost_usd: float = 0.0
    evaluation_a100_gpu_hours: float = 0.0
    evaluation_a100_cost_usd: float = 0.0
    total_bill_usd: float = 0.0
    total_fees_collected_tao: float = 0.0
    total_fees_collected_usd: float = 0.0


class WeeklyTournamentGpuCostsResponse(BaseModel):
    week_offset: int
    window_start: datetime
    window_end: datetime
    is_current_window: bool
    first_tournament_started_at: datetime | None = None
    last_tournament_completed_at: datetime | None = None
    h100_8x_hourly_rate_usd: float
    h100_sxm_8x_hourly_rate_usd: float = 34.4
    h100_nvl_8x_hourly_rate_usd: float = 32.0
    h100_other_8x_hourly_rate_usd: float = 29.0
    a100_hourly_rate_usd: float
    tao_price_usd: float | None = None
    totals: TournamentGpuCostTotals
    tournaments: list[TournamentGpuCostBreakdown]
    tasks: list[TaskGpuCostBreakdown]


class SeedRole(str, Enum):
    """What a tournament seed task is for (validator/tournament/seed_tasks.py).

    The two input seeds are the instruct models R1-R3 compress. BOSS_INSTRUCT is the boss round's input: an instruct seed
    only the boss trains. BOSS_CHAINED_FREEFORM and BOSS_CHAINED_FIXED are the boss round's two tasks, compressions of
    that input, attached to the boss round later. The three BOSS_CONTINUOUS_* values are not created by this code; the DB
    CHECK constraint lists them.
    """

    INPUT_SMALL = "input_small"
    INPUT_LARGE = "input_large"
    BOSS_INSTRUCT = "boss_instruct"
    BOSS_CHAINED_FREEFORM = "boss_chained_freeform"
    BOSS_CHAINED_FIXED = "boss_chained_fixed"
    BOSS_CONTINUOUS_INSTRUCT = "boss_continuous_instruct"
    BOSS_CONTINUOUS_FREEFORM = "boss_continuous_freeform"
    BOSS_CONTINUOUS_FIXED = "boss_continuous_fixed"
    TEXT_R1 = "text_r1"
    TEXT_R2 = "text_r2"
    TEXT_R3 = "text_r3"
    TEXT_CONTINUOUS = "text_continuous"
    BOSS_R2_FREEFORM = "boss_r2_freeform"
    BOSS_R2_FIXED = "boss_r2_fixed"
    BOSS_R3_FREEFORM = "boss_r3_freeform"
    BOSS_R3_FIXED = "boss_r3_fixed"
    QUANT_R1 = "quant_r1"
    QUANT_R2 = "quant_r2"
    QUANT_R3 = "quant_r3"

    @property
    def label(self) -> str:
        """The name used in docs, logs and alerts (the values stay the stable identifiers)."""
        return {
            SeedRole.INPUT_SMALL: "Small input model",
            SeedRole.INPUT_LARGE: "Large input model",
            SeedRole.BOSS_INSTRUCT: "Boss instruct model",
            SeedRole.BOSS_CHAINED_FREEFORM: "Boss instruct model freeform",
            SeedRole.BOSS_CHAINED_FIXED: "Boss instruct model fixed",
            SeedRole.BOSS_CONTINUOUS_INSTRUCT: "Boss continuous instruct model",
            SeedRole.BOSS_CONTINUOUS_FREEFORM: "Continuous model freeform",
            SeedRole.BOSS_CONTINUOUS_FIXED: "Continuous model fixed",
            SeedRole.TEXT_R1: "Text R1",
            SeedRole.TEXT_R2: "Text R2",
            SeedRole.TEXT_R3: "Text R3",
            SeedRole.TEXT_CONTINUOUS: "Text continuous",
            SeedRole.BOSS_R2_FREEFORM: "R2 model freeform",
            SeedRole.BOSS_R2_FIXED: "R2 model fixed",
            SeedRole.BOSS_R3_FREEFORM: "R3 model freeform",
            SeedRole.BOSS_R3_FIXED: "R3 model fixed",
            SeedRole.QUANT_R1: "Quant R1",
            SeedRole.QUANT_R2: "Quant R2",
            SeedRole.QUANT_R3: "Quant R3",
        }[self]


INPUT_SEED_ROLES = frozenset({SeedRole.INPUT_SMALL, SeedRole.INPUT_LARGE})


class ContinuousRowsPart(str, Enum):
    """Which disjoint share of a continuous chunk's train rows a task trains on (see select_continuous_rows)."""

    INSTRUCT = "instruct"  # the continuous instruct task: the first rows in the tournament-seeded hash order
    COMPRESSION = "compression"  # its freeform / fixed compression tasks: the next rows, the same for both tasks


class QuantSourceSlot(str, Enum):
    """The place a quant task takes in a quant tournament, and so which text seed task's boss model it compresses.

    r1, r2 and r3 are the miner rounds (one quant task each, trained by the quant boss as a seed and by the round's miners);
    the boss_* slots are the compressions of the quant boss round: the R2 model freeform, the R3 model fixed and the
    continuous model freeform and fixed. Nothing creates boss_r2_fixed or boss_r3_freeform; rows of them load.
    """

    R1 = "r1"
    R2 = "r2"
    R3 = "r3"
    BOSS_R2_FREEFORM = "boss_r2_freeform"
    BOSS_R2_FIXED = "boss_r2_fixed"
    BOSS_R3_FREEFORM = "boss_r3_freeform"
    BOSS_R3_FIXED = "boss_r3_fixed"
    BOSS_CONTINUOUS_FREEFORM = "boss_continuous_freeform"
    BOSS_CONTINUOUS_FIXED = "boss_continuous_fixed"


class QuantSlotSpec(BaseModel):
    """What a slot is made of: the text seed role whose boss model it compresses, its quant task type and the quant seed role
    that holds its task (the boss trains every slot's task as a seed)."""

    text_role: SeedRole
    task_type: TaskType
    quant_seed_role: SeedRole | None = None


QUANT_SLOT_SPECS: dict[QuantSourceSlot, QuantSlotSpec] = {
    QuantSourceSlot.R1: QuantSlotSpec(
        text_role=SeedRole.TEXT_R1, task_type=TaskType.QUANTFREEFORMTASK, quant_seed_role=SeedRole.QUANT_R1
    ),
    QuantSourceSlot.R2: QuantSlotSpec(
        text_role=SeedRole.TEXT_R2, task_type=TaskType.QUANTFIXEDTASK, quant_seed_role=SeedRole.QUANT_R2
    ),
    QuantSourceSlot.R3: QuantSlotSpec(
        text_role=SeedRole.TEXT_R3, task_type=TaskType.QUANTFREEFORMTASK, quant_seed_role=SeedRole.QUANT_R3
    ),
    QuantSourceSlot.BOSS_R2_FREEFORM: QuantSlotSpec(
        text_role=SeedRole.TEXT_R2, task_type=TaskType.QUANTFREEFORMTASK, quant_seed_role=SeedRole.BOSS_R2_FREEFORM
    ),
    QuantSourceSlot.BOSS_R2_FIXED: QuantSlotSpec(
        text_role=SeedRole.TEXT_R2, task_type=TaskType.QUANTFIXEDTASK, quant_seed_role=SeedRole.BOSS_R2_FIXED
    ),
    QuantSourceSlot.BOSS_R3_FREEFORM: QuantSlotSpec(
        text_role=SeedRole.TEXT_R3, task_type=TaskType.QUANTFREEFORMTASK, quant_seed_role=SeedRole.BOSS_R3_FREEFORM
    ),
    QuantSourceSlot.BOSS_R3_FIXED: QuantSlotSpec(
        text_role=SeedRole.TEXT_R3, task_type=TaskType.QUANTFIXEDTASK, quant_seed_role=SeedRole.BOSS_R3_FIXED
    ),
    QuantSourceSlot.BOSS_CONTINUOUS_FREEFORM: QuantSlotSpec(
        text_role=SeedRole.TEXT_CONTINUOUS,
        task_type=TaskType.QUANTFREEFORMTASK,
        quant_seed_role=SeedRole.BOSS_CONTINUOUS_FREEFORM,
    ),
    QuantSourceSlot.BOSS_CONTINUOUS_FIXED: QuantSlotSpec(
        text_role=SeedRole.TEXT_CONTINUOUS,
        task_type=TaskType.QUANTFIXEDTASK,
        quant_seed_role=SeedRole.BOSS_CONTINUOUS_FIXED,
    ),
}
# The slot of each quant seed role (rounds 1-3 and the boss round).
QUANT_SLOT_BY_SEED_ROLE: dict[SeedRole, QuantSourceSlot] = {
    spec.quant_seed_role: slot for slot, spec in QUANT_SLOT_SPECS.items() if spec.quant_seed_role is not None
}


class QuantTournamentSource(BaseModel):
    """The text tournament a quant tournament takes its tasks from (table quant_tournament_sources). One text tournament
    feeds at most one quant tournament."""

    quant_tournament_id: str
    source_tournament_id: str
    created_at: datetime | None = None


class QuantTaskSource(BaseModel):
    """Where one quant task came from (table quant_task_sources): the slot it fills in its quant tournament and the text
    seed task whose boss model it compresses.

    source_task_id is the text seed task (it follows a replaced text seed); source_model_repo is the exact hub repo handed
    to the quant task as its model when it was released (None while it is held); source_is_fallback is True when that repo is
    the text task's base model because the text boss produced no usable model; superseded is True for a quant task that was
    replaced by another task for the same slot.
    """

    quant_task_id: UUID
    quant_tournament_id: str
    slot: QuantSourceSlot
    source_tournament_id: str
    source_role: SeedRole
    source_task_id: UUID
    source_model_repo: str | None = None
    source_is_fallback: bool = False
    superseded: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class QuantRoundSpec(BaseModel):
    """One miner round of a quant tournament: the quant task every entrant plays on the text boss's model of a text seed
    role, and the cut after it."""

    round_number: int
    slot: QuantSourceSlot
    task_type: TaskType
    input_role: SeedRole  # the TEXT seed role of the bound text tournament whose boss model the round compresses
    advance: int
    quant_format: QuantFormat | None = None  # a fixed round's pinned format; None draws one from the round id


# The quant boss round's tasks: four compressions of the text boss's models, the R2 model freeform, the R3 model fixed, and
# the continuous model freeform and fixed.
BOSS_ROUND_SEED_ROLES = frozenset(
    {
        SeedRole.BOSS_R2_FREEFORM,
        SeedRole.BOSS_R3_FIXED,
        SeedRole.BOSS_CONTINUOUS_FREEFORM,
        SeedRole.BOSS_CONTINUOUS_FIXED,
    }
)
# Boss-round roles nothing creates (the R2 model fixed and the R3 model freeform). Rows of them are read like a boss-round
# seed, and a boss round whose tournament holds them attaches them too (the crown rule then never crowns it).
UNCREATED_BOSS_ROUND_SEED_ROLES = frozenset({SeedRole.BOSS_R2_FIXED, SeedRole.BOSS_R3_FREEFORM})
# Retired boss round roles of the earlier layout: two compressions chained on a BOSS_INSTRUCT seed. Nothing creates these
# roles; rows of them are read like a boss-round seed.
RETIRED_BOSS_ROUND_SEED_ROLES = frozenset({SeedRole.BOSS_CHAINED_FREEFORM, SeedRole.BOSS_CHAINED_FIXED})
# Every role a row may carry that is a boss-round seed: where rows are READ (hold, attach, completion, ranks), not where
# roles are created or counted for a new boss round.
ANY_BOSS_ROUND_SEED_ROLES = BOSS_ROUND_SEED_ROLES | UNCREATED_BOSS_ROUND_SEED_ROLES | RETIRED_BOSS_ROUND_SEED_ROLES
# The quant tournament's round seeds: the one quant task of each of rounds 1-3, trained by the quant boss as soon as its
# source model lands and by the round's miners once the round attaches it.
QUANT_ROUND_SEED_ROLES = frozenset({SeedRole.QUANT_R1, SeedRole.QUANT_R2, SeedRole.QUANT_R3})
# Every quant seed role a row may carry: the boss's compression entries whose result is stored at seed time.
ANY_QUANT_SEED_ROLES = ANY_BOSS_ROUND_SEED_ROLES | QUANT_ROUND_SEED_ROLES
# The instruct seeds only the boss trains: upstreams of the chained compression seeds.
BOSS_INSTRUCT_SEED_ROLES = frozenset({SeedRole.BOSS_INSTRUCT, SeedRole.BOSS_CONTINUOUS_INSTRUCT})
# The boss's compression entries. A boss-side training failure of one is the boss's failed entry, never redrawn: a redraw
# cannot change the boss's code.
BOSS_COMPRESSION_SEED_ROLES = BOSS_ROUND_SEED_ROLES
ANY_BOSS_COMPRESSION_SEED_ROLES = ANY_BOSS_ROUND_SEED_ROLES - BOSS_INSTRUCT_SEED_ROLES
# The text tournament's seeds: the instruct task of each of rounds 1-3 and the continuous-SFT task of the boss round. All are
# trained by the boss at tournament start and attached to their round later.
TEXT_ROUND_SEED_ROLES = frozenset({SeedRole.TEXT_R1, SeedRole.TEXT_R2, SeedRole.TEXT_R3})
TEXT_SEED_ROLES = TEXT_ROUND_SEED_ROLES | {SeedRole.TEXT_CONTINUOUS}
# Seeds a round attaches by registering the seed task itself in the round's group or pair: a prep failure after attach is
# redrawn and the replacement takes the attached task's place (replace_seed).
SWAPPED_SEED_ROLES = TEXT_SEED_ROLES | QUANT_ROUND_SEED_ROLES
# Every role whose task is attached to a round after it was trained: its finalisation is held until it is attached.
ATTACHED_SEED_ROLES = ANY_BOSS_ROUND_SEED_ROLES | TEXT_SEED_ROLES | QUANT_ROUND_SEED_ROLES


class RoundRanks(BaseModel):
    """The 1-based rank of every entry in one of rounds 1-3, the boss (EMISSION_BURN_HOTKEY) included, ranked together."""

    round_number: int
    round_id: str
    task_ids: list[str]
    ranks: dict[str, int]


class TournamentRoundRanks(BaseModel):
    """Per-round ranks of the boss and the challengers over rounds 1-3 of a tournament (round_results.round_ranks_with_boss)."""

    tournament_id: str
    rounds: list[RoundRanks]

    def average_rank(self, hotkey: str) -> float | None:
        """Mean rank over the rounds the hotkey was ranked in; None when it was ranked in none."""
        ranks = [round_ranks.ranks[hotkey] for round_ranks in self.rounds if hotkey in round_ranks.ranks]
        return sum(ranks) / len(ranks) if ranks else None


class TextRoundSpec(BaseModel):
    """One miner round of a seeded text tournament: the seed task every entrant plays, its size band and the cut after it."""

    round_number: int
    seed_role: SeedRole
    model_size_b: tuple[float, float]
    advance: int


ImageTaskSpec = tuple[ImageModelType, ImageTrainingMode]


class ImageRoundSpec(BaseModel):
    """One miner round of an image tournament: the (model type, training mode) slots its one task is drawn from, the cap on
    the task's image pairs (None: the synthetic default range) and the cut after it."""

    round_number: int
    task_specs: tuple[ImageTaskSpec, ...]
    max_num_prompts: int | None = None
    advance: int


class SeedStatus(str, Enum):
    TRAINING = "training"  # the seed task exists; the boss's run (or an input seed's evaluation) is not finished
    TRAINED = "trained"  # boss-round seeds: the boss's run is over (a compression entry may have failed), held for the boss round
    READY = "ready"  # usable: an input seed scored SUCCESS with a full-model boss output; a boss-round seed SUCCESS
    REPLACED = "replaced"  # superseded by a later attempt of the same role
    FAILED = "failed"  # out of replacements: final; halts the tournament only for an input-seed role


class TournamentSeedTask(BaseModel):
    """One attempt at one seed role of a tournament: the task the boss trains upfront and the code it trained with."""

    tournament_id: str
    role: SeedRole
    attempt: int = 1
    task_id: UUID
    status: SeedStatus = SeedStatus.TRAINING
    boss_repo: str | None = None
    boss_commit: str | None = None
    attached_round_id: str | None = None  # set once the boss round has attached this seed; never cleared (even if detached)
    # Only meaningful on a replaced row: it was replaced without charging the role's SEED_MAX_REPLACEMENTS, either by its own
    # prep-stage dataset rejection (dataset-drawn roles) or because its upstream instruct seed was replaced.
    prep_rejected: bool = False
    merged_output_repo: str | None = None  # the full model the boss's adapter output was merged into
    created_at: datetime | None = None
    updated_at: datetime | None = None
