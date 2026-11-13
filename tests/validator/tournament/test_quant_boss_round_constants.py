"""Quant boss-round composition and the continuous-quant ds encoding (pure functions)."""

from types import SimpleNamespace

from core.models.task_models import TaskType
from core.models.tournament_models import TournamentType
from validator.tournament import constants as t_cst


class TestQuantFinalRoundComposition:
    def test_distribution_is_two_freeform_and_two_fixed(self):
        assert t_cst.FINAL_ROUND_QUANT_TASK_DISTRIBUTION == {
            TaskType.QUANTFREEFORMTASK: 2,
            TaskType.QUANTFIXEDTASK: 2,
        }

    def test_final_round_total_is_the_sum_of_the_distribution(self):
        assert t_cst.FINAL_ROUND_QUANT_TASKS == sum(t_cst.FINAL_ROUND_QUANT_TASK_DISTRIBUTION.values()) == 4

    def test_expected_boss_round_task_count_per_type(self):
        assert t_cst.expected_boss_round_task_count(TournamentType.QUANT) == 4
        assert t_cst.expected_boss_round_task_count(TournamentType.TEXT) == t_cst.FINAL_ROUND_TEXT_TASKS

    def test_the_crown_is_seven_head_to_heads_with_the_per_task_margins(self):
        assert t_cst.CROWN_TASKS == len(t_cst.QUANT_ROUND_PLAN) + t_cst.FINAL_ROUND_QUANT_TASKS == 7
        assert t_cst.QUANT_FREEFORM_BOSS_BYTE_MARGIN == 0.03
        assert t_cst.QUANT_FIXED_BOSS_MIN_IMPROVEMENT == 0.01

    def test_no_instruct_chat_or_dpo_in_the_quant_boss_round(self):
        for task_type in (TaskType.INSTRUCTTEXTTASK, TaskType.CHATTASK, TaskType.DPOTASK, TaskType.GRPOTASK):
            assert task_type not in t_cst.FINAL_ROUND_QUANT_TASK_DISTRIBUTION

    def test_only_fixed_quant_is_paired(self):
        assert t_cst.QUANT_PAIRED_TASK_TYPES == (TaskType.QUANTFIXEDTASK,)
        assert TaskType.QUANTFIXEDTASK not in t_cst.PAIRED_BOSS_ROUND_TASK_TYPES


class TestContinuousQuantDs:
    def test_round_trip_recovers_the_train_index(self):
        ds = t_cst.continuous_quant_ds(7, "t1", "chunk-00007")
        assert ds == "continuous-quant:7:t1:chunk-00007"
        assert t_cst.continuous_quant_train_index(ds) == 7

    def test_the_tournament_id_is_recovered(self):
        assert t_cst.continuous_quant_tournament(t_cst.continuous_quant_ds(3, "t1", "a:b")) == "t1"
        assert t_cst.continuous_quant_tournament("continuous-quant:3:chunk") is None  # no tournament carried

    def test_label_with_colons_still_recovers_the_index(self):
        assert t_cst.continuous_quant_train_index(t_cst.continuous_quant_ds(3, "t1", "a:b")) == 3

    def test_other_ds_values_return_none(self):
        for ds in (None, "", "tatsu-lab/alpaca", "continuous-quant", "continuous-quant:x:y", "continuous-sft:qwen3-8b:1"):
            assert t_cst.continuous_quant_train_index(ds) is None

    def test_does_not_parse_as_a_continuous_sft_lineage(self):
        # Otherwise the evaluator would pin a seed tokenizer and hours would be sized off a seed.
        ds = t_cst.continuous_quant_ds(1, "t1", "chunk")
        assert t_cst.continuous_sft_lineage_from_ds(ds) is None
        assert t_cst.continuous_sft_seed_repo_for_ds(ds) is None

    def test_lineage_cursor_is_not_a_continuous_sft_lineage(self):
        assert t_cst.QUANT_CONTINUOUS_LINEAGE not in t_cst.CONTINUOUS_SFT_LINEAGES

    def test_is_continuous_quant_task_needs_a_quant_type_and_the_prefix(self):
        ds = t_cst.continuous_quant_ds(1, "t1", "chunk")
        assert t_cst.is_continuous_quant_task(SimpleNamespace(task_type=TaskType.QUANTFIXEDTASK, ds=ds))
        assert t_cst.is_continuous_quant_task(SimpleNamespace(task_type=TaskType.QUANTFREEFORMTASK, ds=ds))
        assert not t_cst.is_continuous_quant_task(SimpleNamespace(task_type=TaskType.QUANTFIXEDTASK, ds="org/data"))
        assert not t_cst.is_continuous_quant_task(SimpleNamespace(task_type=TaskType.INSTRUCTTEXTTASK, ds=ds))
