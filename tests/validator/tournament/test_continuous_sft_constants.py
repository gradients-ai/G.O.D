"""Continuous-SFT constants + ds/lineage/routing helpers (pure functions).

Guards the encodings the whole feature routes on: the boss-round task mix, the lineage<->ds
round-trip (carry-forward routes winners by parsing ds), and the seed resolution the evaluator pins
the tokenizer to. A silent drift in any of these corrupts the competition without an error, so
these are cheap, high-value regression guards.
"""

from types import SimpleNamespace

from core.constants.environments import TrainingStartPoint
from core.models.task_models import TaskType
from validator.tournament import constants as t_cst


class TestFinalRoundComposition:
    def test_distribution_is_2_instruct_1_dpo_1_grpo(self):
        assert t_cst.FINAL_ROUND_TEXT_TASK_DISTRIBUTION == {
            TaskType.INSTRUCTTEXTTASK: 2,
            TaskType.DPOTASK: 1,
            TaskType.GRPOTASK: 1,
        }

    def test_continuous_task_count_equals_lineage_count(self):
        assert t_cst.FINAL_ROUND_CONTINUOUS_SFT_TASKS == len(t_cst.CONTINUOUS_SFT_LINEAGES)

    def test_final_round_total_is_derived_sum_not_stale_literal(self):
        # The completeness gate compares against this; it must stay = standard mix + continuous.
        # 5 tasks means the challenger may drop one and still dethrone (4/5).
        expected = sum(t_cst.FINAL_ROUND_TEXT_TASK_DISTRIBUTION.values()) + t_cst.FINAL_ROUND_CONTINUOUS_SFT_TASKS
        assert t_cst.FINAL_ROUND_TEXT_TASKS == expected == 5


class TestLineages:
    def test_lineage_is_qwen_with_expected_seed(self):
        # Seed-repo typo would silently train the wrong base every week.
        assert t_cst.CONTINUOUS_SFT_LINEAGES == {"qwen": "Qwen/Qwen3-8B-Base"}

    def test_training_hours_fallback_is_four(self):
        # Initial/fallback only — post-prep the throughput pipeline resizes the budget.
        assert t_cst.CONTINUOUS_SFT_TRAINING_HOURS == 4.0


class TestDsRoundTrip:
    def test_encode_then_decode_recovers_lineage(self):
        ds = t_cst.continuous_sft_ds("qwen", "chunk-00003")
        assert ds == "continuous-sft:qwen:chunk-00003"
        assert t_cst.continuous_sft_lineage_from_ds(ds) == "qwen"

    def test_label_containing_colons_still_recovers_lineage(self):
        # split(":", 2) => label keeps its colons, lineage still parses.
        ds = t_cst.continuous_sft_ds("qwen", "a:b:c")
        assert ds == "continuous-sft:qwen:a:b:c"
        assert t_cst.continuous_sft_lineage_from_ds(ds) == "qwen"

    def test_non_continuous_ds_returns_none(self):
        # A real GRPO/DPO ds must never be misclassified as continuous (would corrupt the
        # boss-round type counting and mis-route carry-forward).
        for ds in (None, "", "tatsu-lab/alpaca", "continuous-sft", "something:qwen:x"):
            assert t_cst.continuous_sft_lineage_from_ds(ds) is None


class TestIsContinuousSftTask:
    def _task(self, task_type, start_point):
        return SimpleNamespace(task_type=task_type, training_start_point=start_point)

    def test_true_only_for_chattask_and_continuous_start_point(self):
        assert t_cst.is_continuous_sft_task(
            self._task(TaskType.CHATTASK, TrainingStartPoint.CONTINUOUS_SFT)
        )

    def test_chattask_with_other_start_point_is_false(self):
        assert not t_cst.is_continuous_sft_task(self._task(TaskType.CHATTASK, TrainingStartPoint.DEFAULT))

    def test_continuous_start_point_but_non_chat_is_false(self):
        assert not t_cst.is_continuous_sft_task(
            self._task(TaskType.INSTRUCTTEXTTASK, TrainingStartPoint.CONTINUOUS_SFT)
        )


class TestSeedRouting:
    def test_seed_repo_for_ds_pins_the_lineage_seed(self):
        assert t_cst.continuous_sft_seed_repo_for_ds(t_cst.continuous_sft_ds("qwen", "x")) == "Qwen/Qwen3-8B-Base"

    def test_seed_repo_none_for_non_continuous_ds(self):
        assert t_cst.continuous_sft_seed_repo_for_ds("tatsu-lab/alpaca") is None
        assert t_cst.continuous_sft_seed_repo(None) is None


class TestPreBossRouting:
    def test_pre_boss_model_is_qwen3_32b(self):
        assert t_cst.PRE_BOSS_MODEL == "Qwen/Qwen3-32B"

    def test_is_pre_boss_task_requires_instruct_type_and_forced_model(self):
        def _task(task_type, model_id):
            return SimpleNamespace(task_type=task_type, model_id=model_id)

        assert t_cst.is_pre_boss_task(_task(TaskType.INSTRUCTTEXTTASK, t_cst.PRE_BOSS_MODEL))
        # The continuous-SFT boss task is CHATTASK — must not match, its replacement/routing differs.
        assert not t_cst.is_pre_boss_task(_task(TaskType.CHATTASK, t_cst.PRE_BOSS_MODEL))
        assert not t_cst.is_pre_boss_task(_task(TaskType.INSTRUCTTEXTTASK, "unsloth/Llama-3.2-3B"))
