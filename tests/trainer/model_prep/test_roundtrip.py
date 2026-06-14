"""Tests for model prep data surviving serialization round-trips.

Augmentation configs and baseline stats cross the boundary between
model prep container output → DB → task objects. If the discriminated
union or enum serialization breaks, training gets wrong parameters.
"""

from pydantic import TypeAdapter

from core.constants.environments import EnvironmentName
from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from core.models.model_prep_models import BaselineStats
from core.models.model_prep_models import DpoBaselineStats
from core.models.model_prep_models import DpoDatasetStats
from core.models.model_prep_models import DpoTrainingDynamics
from core.models.model_prep_models import EnvBaselineStats
from core.models.model_prep_models import EnvStats
from core.models.model_prep_models import GrpoBaselineStats
from core.models.model_prep_models import GrpoDatasetStats
from core.models.model_prep_models import GrpoTrainingDynamics
from core.models.model_prep_models import InstructBaselineStats
from core.models.model_prep_models import InstructDatasetStats
from core.models.model_prep_models import InstructTrainingDynamics
from core.models.model_prep_models import LayerGroupWeightStats
from core.models.model_prep_models import SeqLengthDistribution
from core.models.model_prep_models import WeightStats
from core.models.payload_models import ModelPrepResponse


def _seq_dist():
    return SeqLengthDistribution(mean=100.0, p50=95, p95=200, p99=250, max=300)


def _weight_stats():
    return WeightStats(by_group={"ffn_up": LayerGroupWeightStats(weight_rms=0.01, weight_norm=1.0, max_abs=0.05)})


def _base_training_kwargs():
    return dict(
        init_loss=2.5,
        activation_rms={"layer_0": 1.0},
        output_entropy=3.0,
    )


# --- 6a: AugmentationConfig round-trip ---


class TestAugmentationConfigRoundtrip:
    def test_all_enum_values_survive_json(self):
        for aug_type in AugmentationType:
            for scope in AugmentationScope:
                config = AugmentationConfig(aug_type=aug_type, scope=scope, seed=42, intensity=0.01)
                json_str = config.model_dump_json()
                restored = AugmentationConfig.model_validate_json(json_str)
                assert restored.aug_type == aug_type
                assert restored.scope == scope
                assert restored.seed == 42
                assert restored.intensity == 0.01


# --- 6b: InstructBaselineStats discriminated union ---


class TestInstructBaselineStatsRoundtrip:
    def test_roundtrip(self):
        stats = InstructBaselineStats(
            dataset=InstructDatasetStats(
                total_tokens=1000, seq_length_distribution=_seq_dist(),
                near_duplicate_rate=0.1, bits_per_byte=1.5, vocab_size=32000,
                prompt_tokens=500, completion_tokens=500,
                completion_length_distribution=_seq_dist(),
            ),
            weights=_weight_stats(),
            training=InstructTrainingDynamics(masked_completion_loss=1.8, **_base_training_kwargs()),
        )
        json_str = stats.model_dump_json()

        adapter = TypeAdapter(BaselineStats)
        restored = adapter.validate_json(json_str)
        assert isinstance(restored, InstructBaselineStats)
        assert restored.task_type == "instruct"
        assert restored.training.masked_completion_loss == 1.8


# --- 6c: EnvBaselineStats with EnvironmentName keys ---


class TestEnvBaselineStatsRoundtrip:
    def test_roundtrip_preserves_env_keys(self):
        stats = EnvBaselineStats(
            weights=_weight_stats(),
            env_stats={
                EnvironmentName.LIARS_DICE: EnvStats(num_episodes=50, mean_score=0.6),
                EnvironmentName.GIN_RUMMY: EnvStats(num_episodes=25, mean_score=0.3),
            },
        )
        json_str = stats.model_dump_json()

        adapter = TypeAdapter(BaselineStats)
        restored = adapter.validate_json(json_str)
        assert isinstance(restored, EnvBaselineStats)
        assert restored.task_type == "env"
        assert EnvironmentName.LIARS_DICE in restored.env_stats
        assert restored.env_stats[EnvironmentName.LIARS_DICE].mean_score == 0.6


# --- 6d: DpoBaselineStats and GrpoBaselineStats ---


class TestDpoGrpoRoundtrip:
    def test_dpo_roundtrip(self):
        stats = DpoBaselineStats(
            dataset=DpoDatasetStats(
                total_tokens=2000, seq_length_distribution=_seq_dist(),
                near_duplicate_rate=0.05, bits_per_byte=1.2, vocab_size=32000,
                prompt_tokens=500, chosen_tokens=700, rejected_tokens=800,
                chosen_length_distribution=_seq_dist(),
                rejected_length_distribution=_seq_dist(),
                chosen_rejected_length_ratio=0.875,
            ),
            weights=_weight_stats(),
            training=DpoTrainingDynamics(
                ref_log_prob_chosen=-1.0, ref_log_prob_rejected=-2.0,
                implicit_reward_gap=1.0, **_base_training_kwargs(),
            ),
        )
        json_str = stats.model_dump_json()
        adapter = TypeAdapter(BaselineStats)
        restored = adapter.validate_json(json_str)
        assert isinstance(restored, DpoBaselineStats)
        assert restored.task_type == "dpo"

    def test_grpo_roundtrip(self):
        stats = GrpoBaselineStats(
            dataset=GrpoDatasetStats(
                total_tokens=1500, seq_length_distribution=_seq_dist(),
                near_duplicate_rate=0.02, bits_per_byte=1.3, vocab_size=32000,
                prompt_tokens=1500, prompt_length_distribution=_seq_dist(),
            ),
            weights=_weight_stats(),
            training=GrpoTrainingDynamics(
                baseline_reward_scores={"reward_fn_0": 0.7},
                **_base_training_kwargs(),
            ),
        )
        json_str = stats.model_dump_json()
        adapter = TypeAdapter(BaselineStats)
        restored = adapter.validate_json(json_str)
        assert isinstance(restored, GrpoBaselineStats)
        assert restored.task_type == "grpo"


# --- 6e: ModelPrepResponse with both fields ---


class TestModelPrepResponseRoundtrip:
    def test_full_response(self):
        instruct_stats = InstructBaselineStats(
            dataset=InstructDatasetStats(
                total_tokens=1000, seq_length_distribution=_seq_dist(),
                near_duplicate_rate=0.1, bits_per_byte=1.5, vocab_size=32000,
                prompt_tokens=500, completion_tokens=500,
                completion_length_distribution=_seq_dist(),
            ),
            weights=_weight_stats(),
            training=InstructTrainingDynamics(masked_completion_loss=1.8, **_base_training_kwargs()),
        )
        response = ModelPrepResponse(
            augmented_model_id="gradients-io/augmented-abc123",
            baseline_stats=instruct_stats,
        )
        json_str = response.model_dump_json()
        restored = ModelPrepResponse.model_validate_json(json_str)
        assert restored.augmented_model_id == "gradients-io/augmented-abc123"
        assert isinstance(restored.baseline_stats, InstructBaselineStats)

    def test_none_fields(self):
        response = ModelPrepResponse()
        json_str = response.model_dump_json()
        restored = ModelPrepResponse.model_validate_json(json_str)
        assert restored.augmented_model_id is None
        assert restored.baseline_stats is None
