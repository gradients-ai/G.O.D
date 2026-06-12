"""
Pydantic models for model prep: augmentation config and baseline stats.
Per-type stats models for instruct, DPO, GRPO tasks.
"""

import json
from enum import Enum
from typing import Annotated
from typing import Literal
from typing import Union

from pydantic import BaseModel
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag
from pydantic import model_validator

from core.constants import EnvironmentName


# --- Augmentation models ---

class ModelPrepResult(BaseModel):
    """Result of LoRA detection and optional merge."""

    effective_model_path: str = Field(description="Path or repo ID to use for loading (merged if LoRA)")
    base_model_id: str | None = Field(default=None, description="Original base model if LoRA was merged")
    was_lora: bool = Field(default=False, description="Whether the input was a LoRA adapter")



class AugmentationType(str, Enum):
    GAUSSIAN_NOISE = "gaussian_noise"
    WEIGHT_SCALING = "weight_scaling"
    MAGNITUDE_PRUNING = "magnitude_pruning"
    LAYER_REINIT = "layer_reinit"


class AugmentationScope(str, Enum):
    SINGLE_LAYER = "single_layer"
    LAYER_TYPE_GROUP = "layer_type_group"
    MULTI_LAYER = "multi_layer"
    ALL_LAYERS = "all_layers"


class AugmentationConfig(BaseModel):
    aug_type: AugmentationType
    scope: AugmentationScope
    seed: int
    intensity: float

    @model_validator(mode="before")
    @classmethod
    def parse_json_string(cls, data):
        if isinstance(data, str):
            return json.loads(data)
        return data


# --- Shared stats building blocks ---

class SeqLengthDistribution(BaseModel):
    mean: float
    p50: int
    p95: int
    p99: int
    max: int


class LayerGroupWeightStats(BaseModel):
    weight_rms: float
    weight_norm: float
    max_abs: float


class WeightStats(BaseModel):
    by_group: dict[str, LayerGroupWeightStats]


# --- Per-type dataset stats ---

class DatasetStatsBase(BaseModel):
    # Full-dataset token estimate (mean tokens/record from a sample * num_records).
    # num_records == 0 marks legacy rows where total_tokens covered only a
    # 100-record sample — don't use those for time budgeting.
    total_tokens: int
    num_records: int = 0
    seq_length_distribution: SeqLengthDistribution
    near_duplicate_rate: float
    bits_per_byte: float | None = None
    vocab_size: int
    unique_tokens_in_data: int = 0
    vocab_coverage_ratio: float = 0.0


class InstructDatasetStats(DatasetStatsBase):
    prompt_tokens: int
    completion_tokens: int
    completion_length_distribution: SeqLengthDistribution


class DpoDatasetStats(DatasetStatsBase):
    prompt_tokens: int
    chosen_tokens: int
    rejected_tokens: int
    chosen_length_distribution: SeqLengthDistribution
    rejected_length_distribution: SeqLengthDistribution
    chosen_rejected_length_ratio: float


class GrpoDatasetStats(DatasetStatsBase):
    prompt_tokens: int
    prompt_length_distribution: SeqLengthDistribution


# --- Per-type training dynamics ---

class TrainingDynamicsBase(BaseModel):
    init_loss: float
    init_loss_std: float = 0.0
    activation_rms: dict[str, float]
    output_entropy: float
    output_entropy_std: float = 0.0


class InstructTrainingDynamics(TrainingDynamicsBase):
    masked_completion_loss: float


class DpoTrainingDynamics(TrainingDynamicsBase):
    ref_log_prob_chosen: float
    ref_log_prob_rejected: float
    implicit_reward_gap: float


class GrpoTrainingDynamics(TrainingDynamicsBase):
    baseline_reward_scores: dict[str, float]


# --- Throughput probe ---

class ThroughputStats(BaseModel):
    """Timed fwd+bwd on the loaded model in the prep container."""
    tokens_per_sec: float
    seq_len: int
    micro_batch_size: int
    n_gpus: int
    gpu_name: str = ""


# --- Per-type baseline stats ---

class InstructBaselineStats(BaseModel):
    task_type: Literal["instruct"] = "instruct"
    dataset: InstructDatasetStats
    weights: WeightStats
    training: InstructTrainingDynamics
    throughput: ThroughputStats | None = None


class DpoBaselineStats(BaseModel):
    task_type: Literal["dpo"] = "dpo"
    dataset: DpoDatasetStats
    weights: WeightStats
    training: DpoTrainingDynamics
    throughput: ThroughputStats | None = None


class GrpoBaselineStats(BaseModel):
    task_type: Literal["grpo"] = "grpo"
    dataset: GrpoDatasetStats
    weights: WeightStats
    training: GrpoTrainingDynamics
    throughput: ThroughputStats | None = None


# --- Environment stats ---

class EnvStats(BaseModel):
    num_episodes: int
    mean_score: float = 0.0
    std_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    median_score: float = 0.0


class EnvBaselineStats(BaseModel):
    task_type: Literal["env"] = "env"
    weights: WeightStats
    env_stats: dict[EnvironmentName, EnvStats]


def _baseline_stats_discriminator(v) -> str:
    if isinstance(v, dict):
        return v.get("task_type", "instruct")
    return getattr(v, "task_type", "instruct")


BaselineStats = Annotated[
    Union[
        Annotated[InstructBaselineStats, Tag("instruct")],
        Annotated[DpoBaselineStats, Tag("dpo")],
        Annotated[GrpoBaselineStats, Tag("grpo")],
        Annotated[EnvBaselineStats, Tag("env")],
    ],
    Discriminator(_baseline_stats_discriminator),
]
