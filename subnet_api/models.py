"""Public response models.

These are allowlists: any field not declared here is dropped before a response leaves, so a
private field the validator adds later (training repos, tokens) is never relayed by accident.
Types are deliberately loose (str instead of enums, dicts for nested stats) so a new value on
the validator cannot turn a public endpoint into a 500.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())


# Auditing


class HotkeyDetails(PublicModel):
    hotkey: str
    submission_id: str | None = None
    quality_score: float | None = None
    test_loss: float | None = None
    synth_loss: float | None = None
    repo: str | None = None
    rank: int | None = None
    score_reason: str | None = None
    offer_response: dict[str, Any] | None = None


class AuditTask(PublicModel):
    is_organic: bool
    task_id: str | None = None
    status: str
    model_id: str
    ds: str
    account_id: str | None = None
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
    model_params_count: int = 0
    backend: str | None = None
    yarn_factor: int | None = None
    augmentation_config: dict[str, Any] | None = None
    augmented_model_id: str | None = None
    baseline_stats: dict[str, Any] | None = None
    training_start_point: str | None = None
    task_type: str
    trained_model_repository: str | None = None
    synthetic_data: str | None = None


class InstructTextAuditTask(AuditTask):
    field_instruction: str | None = None
    field_system: str | None = None
    field_input: str | None = None
    field_output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    system_format: str | None = None
    file_format: str | None = None
    use_kl: bool = False
    kl_coef: float | None = None


class ChatAuditTask(AuditTask):
    chat_template: str | None = None
    chat_column: str | None = None
    chat_role_field: str | None = None
    chat_content_field: str | None = None
    chat_user_reference: str | None = None
    chat_assistant_reference: str | None = None
    file_format: str | None = None


class DpoAuditTask(AuditTask):
    field_prompt: str | None = None
    field_chosen: str | None = None
    field_rejected: str | None = None
    field_system: str | None = None
    prompt_format: str | None = None
    chosen_format: str | None = None
    rejected_format: str | None = None
    file_format: str | None = None


class GrpoAuditTask(AuditTask):
    field_prompt: str | None = None
    reward_functions: list[dict[str, Any]] = Field(default_factory=list)
    extra_column: str | None = None
    file_format: str | None = None


class ImageAuditTask(AuditTask):
    image_text_pairs: list[dict[str, Any]] | None = None
    model_type: str | None = None
    trigger_word: str | None = None


class EnvAuditTask(AuditTask):
    environment_names: list[str] = Field(default_factory=list)
    environment_weights: list[dict[str, Any]] = Field(default_factory=list)
    eval_seed: int | None = None


class AuditTaskWithHotkeyDetails(AuditTask):
    hotkey_details: list[HotkeyDetails] = Field(default_factory=list)


class InstructTextAuditTaskWithHotkeyDetails(InstructTextAuditTask):
    hotkey_details: list[HotkeyDetails] = Field(default_factory=list)


class ChatAuditTaskWithHotkeyDetails(ChatAuditTask):
    hotkey_details: list[HotkeyDetails] = Field(default_factory=list)


class DpoAuditTaskWithHotkeyDetails(DpoAuditTask):
    hotkey_details: list[HotkeyDetails] = Field(default_factory=list)


class GrpoAuditTaskWithHotkeyDetails(GrpoAuditTask):
    hotkey_details: list[HotkeyDetails] = Field(default_factory=list)


class ImageAuditTaskWithHotkeyDetails(ImageAuditTask):
    hotkey_details: list[HotkeyDetails] = Field(default_factory=list)


class EnvAuditTaskWithHotkeyDetails(EnvAuditTask):
    hotkey_details: list[HotkeyDetails] = Field(default_factory=list)


_TASKS: dict[str, type[AuditTask]] = {
    "InstructTextTask": InstructTextAuditTask,
    "TextTask": InstructTextAuditTask,
    "ChatTask": ChatAuditTask,
    "DpoTask": DpoAuditTask,
    "GrpoTask": GrpoAuditTask,
    "ImageTask": ImageAuditTask,
    "EnvTask": EnvAuditTask,
}

_TASKS_WITH_HOTKEYS: dict[str, type[AuditTask]] = {
    "InstructTextTask": InstructTextAuditTaskWithHotkeyDetails,
    "TextTask": InstructTextAuditTaskWithHotkeyDetails,
    "ChatTask": ChatAuditTaskWithHotkeyDetails,
    "DpoTask": DpoAuditTaskWithHotkeyDetails,
    "GrpoTask": GrpoAuditTaskWithHotkeyDetails,
    "ImageTask": ImageAuditTaskWithHotkeyDetails,
    "EnvTask": EnvAuditTaskWithHotkeyDetails,
}


def audit_task(raw: dict[str, Any], *, with_hotkeys: bool = False) -> dict[str, Any]:
    """Filter one validator task dict to its public fields, chosen by task type."""
    if with_hotkeys:
        model = _TASKS_WITH_HOTKEYS.get(str(raw.get("task_type")), AuditTaskWithHotkeyDetails)
    else:
        model = _TASKS.get(str(raw.get("task_type")), AuditTask)
    return model.model_validate(raw).model_dump(mode="json")


class ScoresUrlResponse(PublicModel):
    url: str


# Tournaments and transfers


class TournamentBalanceResponse(PublicModel):
    coldkey: str
    balance_rao: int
    total_sent_rao: int
    transfer_count: int
    last_transfer_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TournamentFeesResponse(PublicModel):
    text_tournament_fee_rao: int
    image_tournament_fee_rao: int
    environment_tournament_fee_rao: int | None = None


class TournamentParticipantPublic(PublicModel):
    hotkey: str
    eliminated_in_round_id: str | None = None
    final_position: int | None = None


class TournamentPvPPairResult(PublicModel):
    hotkey_a: str
    hotkey_b: str
    environment_name: str
    hotkey_a_wins: int
    hotkey_b_wins: int
    draws: int
    total_games: int


class TournamentPvPIndividualScore(PublicModel):
    hotkey: str
    environment_name: str
    score: float


class TournamentTaskScorePublic(PublicModel):
    task_id: str
    group_id: str | None = None
    pair_id: str | None = None
    winner: str | None = None
    participant_scores: list[dict[str, Any]] = Field(default_factory=list)
    task_type: str | None = None
    environment_names: list[str] | None = None
    pvp_pair_results: list[TournamentPvPPairResult] | None = None
    pvp_individual_scores: list[TournamentPvPIndividualScore] | None = None


class TournamentRoundPublic(PublicModel):
    round_id: str
    round_number: int
    round_type: str
    is_final_round: bool
    status: str
    participants: list[str] = Field(default_factory=list)
    tasks: list[TournamentTaskScorePublic] = Field(default_factory=list)


class TournamentFinalScorePublic(PublicModel):
    hotkey: str
    score: float | None = None


class TournamentDetailsPublicResponse(PublicModel):
    tournament_id: str
    tournament_type: str
    status: str
    base_winner_hotkey: str | None = None
    winner_hotkey: str | None = None
    participants: list[TournamentParticipantPublic] = Field(default_factory=list)
    rounds: list[TournamentRoundPublic] = Field(default_factory=list)
    final_scores: list[TournamentFinalScorePublic] = Field(default_factory=list)
    text_tournament_weight: float | None = None
    image_tournament_weight: float | None = None
    environment_tournament_weight: float | None = None
    boss_round_performance: list[dict[str, Any]] | None = None
    sync_performance: list[dict[str, Any]] | None = None


class LatestTournamentsDetailsPublicResponse(PublicModel):
    text: TournamentDetailsPublicResponse | None = None
    image: TournamentDetailsPublicResponse | None = None
    environment: TournamentDetailsPublicResponse | None = None


# Performance


class MinerEmissionWeight(PublicModel):
    hotkey: str
    rank: int
    weight: float


class TournamentBurnData(PublicModel):
    text_performance_diff: float | None = None
    image_performance_diff: float | None = None
    environment_performance_diff: float | None = None
    text_burn_proportion: float
    image_burn_proportion: float
    environment_burn_proportion: float | None = None
    text_tournament_weight: float
    image_tournament_weight: float
    environment_tournament_weight: float | None = None
    text_base_weight: float | None = None
    image_base_weight: float | None = None
    burn_weight: float


class TournamentWeightsResponse(PublicModel):
    burn_data: TournamentBurnData
    text_top_miners: list[MinerEmissionWeight] = Field(default_factory=list)
    image_top_miners: list[MinerEmissionWeight] = Field(default_factory=list)
    environment_top_miners: list[MinerEmissionWeight] | None = None


class WeightProjection(PublicModel):
    days: int
    weight: float
    total_alpha: float


class TournamentProjection(PublicModel):
    tournament_type: str
    current_champion_decay: float
    initial_weight: float
    projections: list[WeightProjection]
    placement: str = "champion"
    dethrone_threshold: float = 0.0
    emission_boost: float = 0.0


class WeightProjectionResponse(PublicModel):
    percentage_improvement: float
    text_projection: TournamentProjection
    image_projection: TournamentProjection


class MultiWeightProjectionResponse(PublicModel):
    projections: list[WeightProjectionResponse]


class TaskPerformanceDifference(PublicModel):
    task_id: str
    task_type: str
    boss_score: float | None = None
    challenger_score: float | None = None
    threshold_used: float
    performance_difference: float | None = None
    challenger_won: bool
    paired_comparison: dict[str, Any] | None = None
    decision_reason: str | None = None


class BossBattleResponse(PublicModel):
    text_tournament_id: str | None = None
    text_performance_differences: list[TaskPerformanceDifference] = Field(default_factory=list)
    image_tournament_id: str | None = None
    image_performance_differences: list[TaskPerformanceDifference] = Field(default_factory=list)
    environment_tournament_id: str | None = None
    environment_performance_differences: list[TaskPerformanceDifference] | None = None
