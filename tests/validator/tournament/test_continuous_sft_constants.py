"""Continuous-SFT constants + ds/lineage/routing helpers (pure functions).

Guards the encodings the whole feature routes on: the boss-round task mix, the lineage<->ds
round-trip (carry-forward routes winners by parsing ds), and the seed/remote-code resolution the
evaluator pins the tokenizer to. A silent drift in any of these corrupts the competition without
an error, so these are cheap, high-value regression guards.
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
        expected = sum(t_cst.FINAL_ROUND_TEXT_TASK_DISTRIBUTION.values()) + t_cst.FINAL_ROUND_CONTINUOUS_SFT_TASKS
        assert t_cst.FINAL_ROUND_TEXT_TASKS == expected == 6


class TestLineages:
    def test_lineages_are_quasar_and_qwen_with_expected_seeds(self):
        # Seed-repo typo would silently train the wrong base every week.
        assert t_cst.CONTINUOUS_SFT_LINEAGES == {
            "quasar": "gradients-io-tournaments/continuous-sft-seed-quasar-king",
            "qwen": "Qwen/Qwen3-8B-Base",
        }

    def test_only_quasar_needs_remote_code(self):
        assert t_cst.CONTINUOUS_SFT_REMOTE_CODE_LINEAGES == {"quasar"}

    def test_training_hours_fallback_is_four(self):
        # Initial/fallback only — post-prep the throughput pipeline resizes the budget.
        assert t_cst.CONTINUOUS_SFT_TRAINING_HOURS == 4.0


class TestDsRoundTrip:
    def test_encode_then_decode_recovers_lineage(self):
        for lineage in ("quasar", "qwen"):
            ds = t_cst.continuous_sft_ds(lineage, "chunk-00003")
            assert ds == f"continuous-sft:{lineage}:chunk-00003"
            assert t_cst.continuous_sft_lineage_from_ds(ds) == lineage

    def test_label_containing_colons_still_recovers_lineage(self):
        # split(":", 2) => label keeps its colons, lineage still parses.
        ds = t_cst.continuous_sft_ds("qwen", "a:b:c")
        assert ds == "continuous-sft:qwen:a:b:c"
        assert t_cst.continuous_sft_lineage_from_ds(ds) == "qwen"

    def test_non_continuous_ds_returns_none(self):
        # A real GRPO/DPO ds must never be misclassified as continuous (would corrupt the
        # boss-round type counting and mis-route carry-forward).
        for ds in (None, "", "tatsu-lab/alpaca", "continuous-sft", "something:quasar:x"):
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


class TestSeedAndRemoteCodeRouting:
    def test_seed_repo_for_ds_pins_the_lineage_seed(self):
        assert (
            t_cst.continuous_sft_seed_repo_for_ds(t_cst.continuous_sft_ds("quasar", "x"))
            == "gradients-io-tournaments/continuous-sft-seed-quasar-king"
        )
        assert t_cst.continuous_sft_seed_repo_for_ds(t_cst.continuous_sft_ds("qwen", "x")) == "Qwen/Qwen3-8B-Base"

    def test_seed_repo_none_for_non_continuous_ds(self):
        assert t_cst.continuous_sft_seed_repo_for_ds("tatsu-lab/alpaca") is None
        assert t_cst.continuous_sft_seed_repo(None) is None

    def test_remote_code_repo_only_for_quasar(self):
        assert (
            t_cst.continuous_sft_remote_code_repo_for_ds(t_cst.continuous_sft_ds("quasar", "x"))
            == "gradients-io-tournaments/continuous-sft-seed-quasar-king"
        )
        # qwen is standard-arch: must NOT load remote code.
        assert t_cst.continuous_sft_remote_code_repo_for_ds(t_cst.continuous_sft_ds("qwen", "x")) is None
        assert t_cst.continuous_sft_remote_code_repo_for_ds("tatsu-lab/alpaca") is None


QUASAR_SEED = "gradients-io-tournaments/continuous-sft-seed-quasar-king"


class TestPreBossQuasarRouting:
    def test_pre_boss_model_is_the_quasar_seed(self):
        # The pre-boss task must share the lineage seed: remote-code pinning, GPU forcing and the
        # audited-mirror trust chain all key off this exact repo.
        assert t_cst.PRE_BOSS_QUASAR_MODEL == QUASAR_SEED

    def test_custom_arch_seed_model_only_for_remote_code_lineage_seeds(self):
        assert t_cst.is_custom_arch_seed_model(QUASAR_SEED)
        # qwen is standard-arch: its seed as a base model needs no pinning.
        assert not t_cst.is_custom_arch_seed_model("Qwen/Qwen3-8B-Base")
        assert not t_cst.is_custom_arch_seed_model("tatsu-lab/whatever")
        assert not t_cst.is_custom_arch_seed_model(None)

    def test_is_pre_boss_quasar_task_requires_instruct_type_and_seed_model(self):
        def _task(task_type, model_id):
            return SimpleNamespace(task_type=task_type, model_id=model_id)

        assert t_cst.is_pre_boss_quasar_task(_task(TaskType.INSTRUCTTEXTTASK, QUASAR_SEED))
        # The continuous-SFT boss task is CHATTASK — must not match, its replacement/routing differs.
        assert not t_cst.is_pre_boss_quasar_task(_task(TaskType.CHATTASK, QUASAR_SEED))
        assert not t_cst.is_pre_boss_quasar_task(_task(TaskType.INSTRUCTTEXTTASK, "unsloth/Llama-3.2-3B"))

    def test_remote_code_repo_for_task_keys_by_ds_then_model(self):
        # continuous-SFT ds keying is unchanged (base is the carried winner, not the seed).
        assert (
            t_cst.remote_code_repo_for_task("org/carried-winner", t_cst.continuous_sft_ds("quasar", "x"))
            == QUASAR_SEED
        )
        # pre-boss task: standard ds, but the base IS the seed mirror -> pin to it.
        assert t_cst.remote_code_repo_for_task(QUASAR_SEED, "tatsu-lab/alpaca") == QUASAR_SEED
        # standard-arch tasks get no remote code from either key.
        assert t_cst.remote_code_repo_for_task("Qwen/Qwen3-8B-Base", "tatsu-lab/alpaca") is None
        assert t_cst.remote_code_repo_for_task("Qwen/Qwen3-8B-Base", t_cst.continuous_sft_ds("qwen", "x")) is None
        assert t_cst.remote_code_repo_for_task(None, None) is None
