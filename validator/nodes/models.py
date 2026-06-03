from datetime import datetime
from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from core.models.task_models import TaskType


class OnChainIncentive(BaseModel):
    raw_value: int
    normalized: float
    network_share_percent: float


class CalculatedPerformanceWeight(BaseModel):
    weight_value: float
    network_share_percent: float


class TaskSourcePerformance(BaseModel):
    task_count: int = 0
    average_score: float = 0.0
    normalized_score: float = 0.0


class QualityMetrics(BaseModel):
    total_score: float
    total_count: int
    total_success: int
    total_quality: int
    avg_quality_score: float
    success_rate: float
    quality_rate: float


class WorkloadMetrics(BaseModel):
    competition_hours: int = Field(ge=0)
    total_params_billions: float = Field(ge=0.0)


class ModelMetrics(BaseModel):
    modal_model: str
    unique_models: int = Field(ge=0)
    unique_datasets: int = Field(ge=0)


class NodeStats(BaseModel):
    quality_metrics: QualityMetrics
    workload_metrics: WorkloadMetrics
    model_metrics: ModelMetrics

    model_config = ConfigDict(protected_namespaces=())


class AllNodeStats(BaseModel):
    daily: NodeStats
    three_day: NodeStats
    weekly: NodeStats
    monthly: NodeStats
    all_time: NodeStats

    @classmethod
    def get_periods_sql_mapping(cls) -> dict[str, str]:
        return {"daily": "24 hours", "three_day": "3 days", "weekly": "7 days", "monthly": "30 days", "all_time": "all"}


class LeaderboardRow(BaseModel):
    hotkey: str
    stats: AllNodeStats


class PeriodScore(BaseModel):
    average_score: float = 0.0
    normalized_score: float = 0.0
    weight_multiplier: float = 0.0
    weighted_contribution: float = 0.0


class TaskTypePerformance(BaseModel):
    one_day: PeriodScore = Field(default_factory=PeriodScore)
    three_day: PeriodScore = Field(default_factory=PeriodScore)
    seven_day: PeriodScore = Field(default_factory=PeriodScore)
    
    organic_performance: TaskSourcePerformance = Field(default_factory=TaskSourcePerformance)
    synthetic_performance: TaskSourcePerformance = Field(default_factory=TaskSourcePerformance)
    
    total_submissions: int = 0
    average_score: float = 0.0
    weight_contribution: float = 0.0
    
    average_work_score: float = 0.0
    total_work_score: float = 0.0
    average_adjusted_score: float = 0.0
    total_adjusted_score: float = 0.0


class TaskTypeBreakdown(BaseModel):
    instruct_text: TaskTypePerformance = Field(default_factory=TaskTypePerformance)
    dpo: TaskTypePerformance = Field(default_factory=TaskTypePerformance)
    image: TaskTypePerformance = Field(default_factory=TaskTypePerformance)
    grpo: TaskTypePerformance = Field(default_factory=TaskTypePerformance)


class PeriodTotals(BaseModel):
    one_day_total: float = 0.0
    three_day_total: float = 0.0
    seven_day_total: float = 0.0


class TaskSubmissionResult(BaseModel):
    task_id: UUID
    task_type: TaskType
    is_organic: bool
    created_at: datetime
    score: float
    rank: int
    total_participants: int
    percentile: float
    work_score: float = 0.0
    adjusted_score: float = 0.0
    hours_to_complete: float = 0
    model_size_billions: float = 0.0


class MinerPerformanceMetrics(BaseModel):
    total_tasks_participated: int = 0
    tasks_last_24h: int = 0
    tasks_last_7d: int = 0
    
    positive_score_rate: float = 0.0
    average_percentile_rank: float = 0.0
    
    average_work_score: float = 0.0
    total_work_score: float = 0.0
    average_adjusted_score: float = 0.0
    total_adjusted_score: float = 0.0
    
    task_type_distribution: dict[str, float] = Field(default_factory=dict)


class WeightingDetails(BaseModel):
    one_day_weight: float
    three_day_weight: float
    seven_day_weight: float
    instruct_text_weight: float
    dpo_weight: float
    image_weight: float
    grpo_weight: float


class MinerDetailsResponse(BaseModel):
    hotkey: str
    node_id: int | None = None
    
    current_incentive: OnChainIncentive
    
    weighting_details: WeightingDetails
    
    task_type_breakdown: TaskTypeBreakdown
    
    period_totals: PeriodTotals
    
    recent_submissions: list[TaskSubmissionResult] = Field(default_factory=list)
    
    performance_metrics: MinerPerformanceMetrics
