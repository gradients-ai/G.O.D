"""Quant compute sizing: GPUs exactly like instruct, hours from prep throughput (flat fallback without stats)."""

import uuid
from datetime import datetime

import pytest

from core.constants.environments import TrainingStartPoint
from core.models.dataset_models import InstructTextDatasetType
from core.models.model_prep_models import DeviceProbeStats
from core.models.model_prep_models import EnvBaselineStats
from core.models.model_prep_models import InstructBaselineStats
from core.models.model_prep_models import InstructDatasetStats
from core.models.model_prep_models import InstructTrainingDynamics
from core.models.model_prep_models import SeqLengthDistribution
from core.models.model_prep_models import ThroughputStats
from core.models.model_prep_models import WeightStats
from core.models.quant_models import QuantFormat
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.tasks.datasets import constants as data_cst
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tasks.synthetics.scheduler import _analytic_tokens_per_sec_per_gpu
from validator.tasks.synthetics.scheduler import compute_hours_from_baseline_stats
from validator.tasks.synthetics.scheduler import compute_training_hours
from validator.tasks.synthetics.scheduler import quant_fixed_hours
from validator.tasks.synthetics.scheduler import quant_flat_hours
from validator.tasks.synthetics.scheduler import quant_hours_after_prep
from validator.tasks.synthetics.scheduler import quant_hours_breakdown
from validator.tasks.synthetics.scheduler import quant_measured_tokens_per_sec
from validator.tasks.synthetics.scheduler import quant_training_hours
from validator.tournament.gpu_requirements import get_tournament_gpu_requirement
from validator.tournament.models import GpuRequirement


PARAMS_3B = 3e9
PARAMS_FA = 3.82e9
PROBE_TPS_FA = 8_700.0
FREEFORM = TaskType.QUANTFREEFORMTASK
FIXED = TaskType.QUANTFIXEDTASK
PARAMS_B_GRID = [0.35, 1.5, 3.82, 8.0, 12.0]


class TestQuantGpuRequirement:
    @pytest.mark.parametrize("params", [2e9, 4e9, 6e9, 8e9, 12e9])
    @pytest.mark.parametrize("task_type", sorted(QUANT_TASK_TYPES))
    def test_quant_is_sized_like_instruct(self, task_type, params):
        expected = get_tournament_gpu_requirement(TaskType.INSTRUCTTEXTTASK, int(params))
        assert get_tournament_gpu_requirement(task_type, int(params)) == expected

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            (2e9, GpuRequirement.H100_1X),
            (4e9, GpuRequirement.H100_1X),
            (6e9, GpuRequirement.H100_2X),
            (8e9, GpuRequirement.H100_2X),
            (12e9, GpuRequirement.H100_2X),
        ],
    )
    def test_quant_boundaries(self, params, expected):
        assert get_tournament_gpu_requirement(TaskType.QUANTFIXEDTASK, int(params)) == expected


class TestQuantFlatHours:
    @pytest.mark.parametrize(("boss", "expected"), [(False, 2.0), (True, 2.0)])
    @pytest.mark.parametrize("params", [1e9, 8e9, 30e9])
    def test_freeform_is_flat_per_round(self, params, boss, expected):
        assert quant_flat_hours(TaskType.QUANTFREEFORMTASK, int(params), boss) == expected

    @pytest.mark.parametrize(
        ("params_b", "expected"),
        [(1, 1.25), (4, 1.25), (6, 1.625), (8, 2.0), (12, 2.0)],
    )
    def test_fixed_ramps_linearly_with_size(self, params_b, expected):
        assert quant_fixed_hours(params_b) == pytest.approx(expected)
        for boss in (False, True):
            assert quant_flat_hours(TaskType.QUANTFIXEDTASK, int(params_b * 1e9), boss) == pytest.approx(expected)

    def test_unknown_params_fall_back_to_the_default_size(self):
        assert quant_flat_hours(TaskType.QUANTFIXEDTASK, 0, boss=False) == 2.0

    def test_only_quant_types_are_accepted(self):
        with pytest.raises(ValueError):
            quant_flat_hours(TaskType.INSTRUCTTEXTTASK, int(3e9), boss=False)

    def test_quant_types_have_no_hours_multiplier(self):
        assert not QUANT_TASK_TYPES & data_cst.TASK_TYPE_HOURS_MULTIPLIER.keys()


def _baseline_stats(
    total_tokens: int = int(40e6), num_records: int = 100_000, tokens_per_sec: float | None = None, mean_seq: float = 400.0
) -> InstructBaselineStats:
    seq_dist = SeqLengthDistribution(mean=mean_seq, p50=350, p95=900, p99=1200, max=2000)
    throughput = None
    if tokens_per_sec is not None:
        throughput = ThroughputStats(tokens_per_sec=tokens_per_sec, seq_len=1024, micro_batch_size=8, n_gpus=1)
    return InstructBaselineStats(
        dataset=InstructDatasetStats(
            total_tokens=total_tokens,
            num_records=num_records,
            seq_length_distribution=seq_dist,
            near_duplicate_rate=0.0,
            vocab_size=150_000,
            prompt_tokens=total_tokens // 2,
            completion_tokens=total_tokens // 2,
            completion_length_distribution=seq_dist,
        ),
        weights=WeightStats(by_group={}),
        training=InstructTrainingDynamics(init_loss=2.0, activation_rms={}, output_entropy=3.0, masked_completion_loss=2.5),
        throughput=throughput,
    )


def _fa_stats() -> InstructBaselineStats:
    """Prep stats: probe 8,700 tok/s, mean sequence 1,041.1."""
    return _baseline_stats(tokens_per_sec=PROBE_TPS_FA, mean_seq=1041.1)


def _scaled_stats(params_b: float) -> InstructBaselineStats:
    """Probe throughput scaled 1/params from the 3.82B point."""
    return _baseline_stats(tokens_per_sec=PROBE_TPS_FA * 3.82 / params_b, mean_seq=1041.1)


class TestQuantHoursCalibration:
    @pytest.mark.parametrize(("boss", "expected"), [(False, 2.0), (True, 2.0)])
    def test_freeform_pin(self, boss, expected):
        assert quant_training_hours(FREEFORM, int(PARAMS_FA), boss, _fa_stats()) == expected

    @pytest.mark.parametrize("boss", [False, True])
    def test_fixed_pin(self, boss):
        assert quant_training_hours(FIXED, int(PARAMS_FA), boss, _fa_stats()) == 1.25

    @pytest.mark.parametrize(
        ("task_type", "token", "model", "host"),
        [(FREEFORM, 3524, 1910, 913), (FIXED, 2903, 306, 829)],
    )
    def test_breakdown_terms_at_the_reference_point(self, task_type, token, model, host):
        breakdown = quant_hours_breakdown(task_type, PARAMS_FA, _fa_stats(), boss=False)

        assert breakdown.token_seconds == pytest.approx(token, abs=1)
        assert breakdown.model_seconds == pytest.approx(model, abs=1)
        assert breakdown.host_seconds == pytest.approx(host, abs=1)
        assert breakdown.gpus == 1

    @pytest.mark.parametrize(
        ("task_type", "expected"),
        [
            (FREEFORM, [0.5, 1.0, 2.0, 2.25, 3.0]),
            (FIXED, [0.5, 0.75, 1.25, 1.25, 2.0]),
        ],
    )
    def test_hours_table(self, task_type, expected):
        hours = [quant_training_hours(task_type, int(b * 1e9), False, _scaled_stats(b)) for b in PARAMS_B_GRID]

        assert hours == expected


def _task_stats(tokens_per_sec: float, mean_seq: float, total_tokens: int, num_records: int) -> InstructBaselineStats:
    return _baseline_stats(total_tokens=total_tokens, num_records=num_records, tokens_per_sec=tokens_per_sec, mean_seq=mean_seq)


_SPLIT_A = (1041.1, 26_000_000, 24_183)
TASK_A = (3.82e9, _task_stats(8_700.0, *_SPLIT_A))
TASK_B = (3.4e9, _task_stats(10_000.0, 80.0, 720_000, 9_000))
TASK_C = (1.7e9, _task_stats(18_000.0, 500.0, 8_200_000, 16_400))
TASK_D = (1.2e9, _task_stats(26_000.0, 200.0, 1_500_000, 7_500))
TASK_E = (1.5e9, _task_stats(30_000.0, 500.0, 21_000_000, 42_000))
TASK_F = (2.5e9, _task_stats(20_000.0, 200.0, 17_000_000, 85_000))
TASK_G = (3.9e9, _task_stats(16_000.0, 50.0, 380_000, 7_600))
B8 = (8e9, _task_stats(8_700 * 3.82 / 8, *_SPLIT_A))
B12 = (12e9, _task_stats(8_700 * 3.82 / 12, *_SPLIT_A))


class TestQuantTrainTokens:
    @pytest.mark.parametrize(
        ("stats", "expected"),
        [
            (_task_stats(10_000.0, 80.0, 720_000, 9_000), 720_000),
            (_task_stats(8_700.0, *_SPLIT_A), 8_000_000),
            (_task_stats(8_700.0, 1041.1, 500_000_000, 500_000), 8_000_000),
            (_task_stats(8_700.0, 1041.1, 26_000_000, 0), 8_000_000),
            (_task_stats(8_700.0, 10.0, 10_000, 1_000), 64_000),
        ],
        ids=["small-split", "capped", "huge-split", "legacy-no-rows", "short-rows-floor"],
    )
    def test_freeform_is_capped_by_the_train_split(self, stats, expected):
        assert quant_hours_breakdown(FREEFORM, PARAMS_FA, stats, boss=False).train_tokens == expected

    def test_fixed_is_uncapped(self):
        breakdown = quant_hours_breakdown(FIXED, TASK_B[0], TASK_B[1], boss=False)

        assert breakdown.train_tokens == 7_000_000


class TestQuantHoursPerTask:
    @pytest.mark.parametrize(
        ("task", "train", "token", "model", "host", "hours", "boss_hours"),
        [
            (TASK_A, 8_000_000, 3524, 1910, 913, 2.0, 2.0),
            (TASK_B, 720_000, 291, 1700, 896, 1.0, 1.0),
            (TASK_C, 8_000_000, 1519, 850, 828, 1.0, 1.0),
            (TASK_D, 1_500_000, 230, 600, 808, 0.75, 0.75),
            (TASK_E, 8_000_000, 911, 750, 820, 1.0, 1.0),
            (TASK_F, 8_000_000, 1275, 1250, 860, 1.25, 1.25),
            (TASK_G, 486_400, 127, 1950, 916, 1.0, 1.0),
            (B8, 8_000_000, 3690, 2000, 1080, 2.25, 2.25),
            (B12, 8_000_000, 5535, 3000, 1240, 3.0, 3.0),
        ],
        ids=["a", "b", "c", "d", "e", "f", "g", "8b", "12b"],
    )
    def test_freeform_terms_and_hours(self, task, train, token, model, host, hours, boss_hours):
        params, stats = task
        breakdown = quant_hours_breakdown(FREEFORM, params, stats, boss=False)

        assert breakdown.train_tokens == train
        assert breakdown.token_seconds == pytest.approx(token, abs=1)
        assert breakdown.model_seconds == pytest.approx(model, abs=1)
        assert breakdown.host_seconds == pytest.approx(host, abs=1)
        assert breakdown.slack == 1.1
        assert breakdown.hours == hours
        assert quant_training_hours(FREEFORM, int(params), True, stats) == boss_hours

    @pytest.mark.parametrize(
        ("task", "hours"),
        [
            (TASK_A, 1.25),
            (TASK_B, 1.0),
            (TASK_C, 0.75),
            (TASK_D, 0.5),
            (TASK_E, 0.5),
            (TASK_F, 0.75),
            (TASK_G, 0.75),
            (B8, 1.25),
            (B12, 2.0),
        ],
        ids=["a", "b", "c", "d", "e", "f", "g", "8b", "12b"],
    )
    def test_fixed_is_unchanged(self, task, hours):
        params, stats = task
        breakdown = quant_hours_breakdown(FIXED, params, stats, boss=False)

        assert breakdown.hours == hours
        assert breakdown.slack == 1.0
        assert breakdown.train_tokens == 7_000_000


class TestQuantFreeformWorkloadVariants:
    def test_extended_differs_only_in_eval_passes_epochs_and_host_fixed_seconds(self):
        base = data_cst.QUANT_FREEFORM_WORKLOAD.model_dump()
        extended = data_cst.QUANT_FREEFORM_EXTENDED_WORKLOAD.model_dump()
        differing = {key for key in base if base[key] != extended[key]}

        assert differing == {"eval_passes", "train_max_epochs", "host_fixed_seconds"}
        assert (extended["eval_passes"], extended["train_max_epochs"], extended["host_fixed_seconds"]) == (40, 2.0, 1_400.0)

    def test_workload_constants(self):
        freeform = data_cst.QUANT_FREEFORM_WORKLOAD
        fixed = data_cst.QUANT_FIXED_WORKLOAD
        extended = data_cst.QUANT_FREEFORM_EXTENDED_WORKLOAD

        assert freeform.host_fixed_seconds == 760
        assert fixed.host_fixed_seconds == 600
        assert extended.host_fixed_seconds == 1_400
        assert freeform.forward_tokens == fixed.forward_tokens == extended.forward_tokens == 1_048_576
        assert (freeform.eval_passes, fixed.eval_passes, extended.eval_passes) == (32, 20, 40)

    @pytest.fixture
    def extended(self, monkeypatch):
        monkeypatch.setattr(data_cst, "QUANT_FREEFORM_WORKLOAD", data_cst.QUANT_FREEFORM_EXTENDED_WORKLOAD)

    @pytest.mark.usefixtures("extended")
    @pytest.mark.parametrize(
        ("task", "hours"),
        [
            (TASK_A, 2.25),
            (TASK_B, 1.25),
            (TASK_C, 1.25),
            (TASK_D, 1.0),
            (TASK_E, 1.0),
            (TASK_F, 1.25),
            (TASK_G, 1.25),
            (B8, 2.5),
            (B12, 3.5),
        ],
        ids=["a", "b", "c", "d", "e", "f", "g", "8b", "12b"],
    )
    def test_extended_real_task_hours(self, task, hours):
        params, stats = task

        assert quant_training_hours(FREEFORM, int(params), False, stats) == hours

    @pytest.mark.usefixtures("extended")
    def test_extended_param_grid_and_reference_terms(self):
        hours = [quant_training_hours(FREEFORM, int(b * 1e9), False, _scaled_stats(b)) for b in PARAMS_B_GRID]
        breakdown = quant_hours_breakdown(FREEFORM, PARAMS_FA, _fa_stats(), boss=False)

        assert hours == [0.75, 1.25, 2.25, 2.5, 3.5]
        assert breakdown.token_seconds == pytest.approx(3708, abs=1)
        assert breakdown.model_seconds == pytest.approx(1910, abs=1)
        assert breakdown.host_seconds == pytest.approx(1553, abs=1)


class TestQuantHoursRate:
    @staticmethod
    def _with_device(stats, train_tokens_per_sec):
        stats.device = DeviceProbeStats(train_tokens_per_sec=train_tokens_per_sec)
        return stats

    def test_hardened_rate_is_used_when_present(self):
        stats = self._with_device(_baseline_stats(tokens_per_sec=8_700.0, mean_seq=1041.1), 12_000.0)

        assert quant_measured_tokens_per_sec(stats) == 12_000.0
        hardened = quant_hours_breakdown(FREEFORM, PARAMS_FA, stats, boss=False)
        plain = quant_hours_breakdown(FREEFORM, PARAMS_FA, _fa_stats(), boss=False)
        assert hardened.token_seconds == pytest.approx(plain.token_seconds * 8_700.0 / 12_000.0)

    @pytest.mark.parametrize("probe", [None, 0.0, DeviceProbeStats()], ids=["no-device", "zero", "empty-probe"])
    def test_falls_back_to_the_three_step_reading(self, probe):
        stats = _fa_stats()
        if isinstance(probe, DeviceProbeStats):
            stats.device = probe
        elif probe is not None:
            self._with_device(stats, probe)

        assert quant_measured_tokens_per_sec(stats) == PROBE_TPS_FA
        assert quant_hours_breakdown(FREEFORM, PARAMS_FA, stats, boss=False).hours == 2.0

    def test_a_much_lower_hardened_rate_sets_more_hours(self):
        slow = self._with_device(_fa_stats(), 4_000.0)

        without = quant_hours_breakdown(FREEFORM, PARAMS_FA, _fa_stats(), boss=False)
        with_probe = quant_hours_breakdown(FREEFORM, PARAMS_FA, slow, boss=False)
        assert with_probe.hours > without.hours


class TestQuantHoursSlack:
    def test_slack_values(self):
        assert quant_hours_breakdown(FREEFORM, PARAMS_FA, _fa_stats(), boss=False).slack == data_cst.QUANT_HOURS_SLACK == 1.1
        assert quant_hours_breakdown(FIXED, PARAMS_FA, _fa_stats(), boss=False).slack == 1.0

    def test_slack_is_applied_before_the_ceil(self):
        # Scale the reference throughput down until the unslacked sum is ~6,700 s: 2.0 h without slack, 2.25 h with it.
        base = quant_hours_breakdown(FREEFORM, PARAMS_FA, _fa_stats(), boss=False)
        fixed_seconds = base.model_seconds + base.host_seconds
        tps = PROBE_TPS_FA * base.token_seconds / (6_700 - fixed_seconds)
        breakdown = quant_hours_breakdown(FREEFORM, PARAMS_FA, _baseline_stats(tokens_per_sec=tps, mean_seq=1041.1), boss=False)
        unslacked = breakdown.token_seconds + breakdown.model_seconds + breakdown.host_seconds

        assert unslacked < 7200 < unslacked * 1.1
        assert unslacked / 3600 < 2.0
        assert breakdown.hours == 2.25


class TestQuantHoursShape:
    @pytest.mark.parametrize("task_type", [FREEFORM, FIXED])
    def test_non_increasing_in_throughput(self, task_type):
        hours = [
            quant_training_hours(task_type, int(PARAMS_FA), False, _baseline_stats(tokens_per_sec=tps, mean_seq=1041.1))
            for tps in (3_000.0, 6_000.0, 8_700.0, 15_000.0, 30_000.0)
        ]

        assert hours == sorted(hours, reverse=True)

    @pytest.mark.parametrize("boss", [False, True])
    @pytest.mark.parametrize("task_type", [FREEFORM, FIXED])
    def test_non_decreasing_in_params(self, task_type, boss):
        hours = [quant_training_hours(task_type, int(b * 1e9), boss, _scaled_stats(b)) for b in PARAMS_B_GRID]

        assert hours == sorted(hours)

    def test_gpu_split_halves_model_and_token_terms_not_host(self):
        # 8B uses 2 GPUs; compare against the same model priced on 1 GPU by hand.
        stats = _baseline_stats(tokens_per_sec=8_000.0, mean_seq=1041.1)
        two = quant_hours_breakdown(FREEFORM, 8e9, stats, boss=False)
        one = quant_hours_breakdown(FREEFORM, 4e9, stats, boss=False)  # 1 GPU, same tps/gpu if unclamped

        assert (two.gpus, one.gpus) == (2, 1)
        assert two.token_seconds == pytest.approx(one.token_seconds / 2)
        assert two.model_seconds == pytest.approx(one.model_seconds)  # 500*8/2 == 500*4/1
        workload = data_cst.QUANT_FREEFORM_WORKLOAD
        assert two.host_seconds == pytest.approx(workload.host_seconds_per_b_params * 8 + workload.host_fixed_seconds)

    def test_fixed_host_seconds_do_not_scale_with_the_model(self):
        # The fixed host / I/O seconds are the same on a 0.35B and a 12B.
        small = quant_hours_breakdown(FREEFORM, 0.35e9, _scaled_stats(0.35), boss=False)
        large = quant_hours_breakdown(FREEFORM, 12e9, _scaled_stats(12.0), boss=False)
        workload = data_cst.QUANT_FREEFORM_WORKLOAD

        assert small.host_seconds - 0.35 * workload.host_seconds_per_b_params == pytest.approx(760)
        assert large.host_seconds - 12 * workload.host_seconds_per_b_params == pytest.approx(760)

    def test_tiny_model_is_floored(self):
        assert quant_training_hours(FIXED, int(0.1e9), False, _scaled_stats(0.1)) == data_cst.QUANT_TRAINING_HOURS_MIN

    def test_tiny_freeform_model_is_floored_and_lifted_by_the_extended_workload(self, monkeypatch):
        # 0.35B models to under the floor.
        stats = _scaled_stats(0.35)
        breakdown = quant_hours_breakdown(FREEFORM, 0.35e9, stats, boss=False)

        assert breakdown.hours == data_cst.QUANT_TRAINING_HOURS_MIN == 0.5
        monkeypatch.setattr(data_cst, "QUANT_FREEFORM_WORKLOAD", data_cst.QUANT_FREEFORM_EXTENDED_WORKLOAD)
        assert quant_training_hours(FREEFORM, int(0.35e9), False, stats) == 0.75

    def test_huge_token_work_is_capped(self):
        slow = _baseline_stats(tokens_per_sec=1.0, mean_seq=1041.1)

        assert quant_training_hours(FREEFORM, int(12e9), False, slow) == data_cst.MAX_TRAINING_HOURS

    def test_boss_gets_no_extra_time(self):
        assert quant_training_hours(FREEFORM, int(PARAMS_FA), True, _fa_stats()) == 2.0
        assert quant_training_hours(FREEFORM, int(8e9), True, _scaled_stats(8.0)) == 2.25
        assert quant_training_hours(FREEFORM, int(12e9), True, _scaled_stats(12.0)) == 3.0
        assert data_cst.QUANT_BOSS_HOURS_MULTIPLIER == {TaskType.QUANTFREEFORMTASK: 1.0, TaskType.QUANTFIXEDTASK: 1.0}

    def test_absurd_throughput_is_clamped_to_three_times_analytic(self):
        absurd = quant_hours_breakdown(FREEFORM, PARAMS_FA, _baseline_stats(tokens_per_sec=1e9, mean_seq=1041.1), False)
        at_cap = _baseline_stats(tokens_per_sec=3.0 * _analytic_tokens_per_sec_per_gpu(PARAMS_FA), mean_seq=1041.1)
        capped = quant_hours_breakdown(FREEFORM, PARAMS_FA, at_cap, False)

        assert absurd.token_seconds == pytest.approx(capped.token_seconds)

    def test_only_quant_types_are_priced(self):
        with pytest.raises(ValueError):
            quant_hours_breakdown(TaskType.INSTRUCTTEXTTASK, PARAMS_FA, _fa_stats(), boss=False)


class TestQuantHoursFallbacks:
    @pytest.mark.parametrize(
        "stats",
        [
            None,
            EnvBaselineStats.model_construct(),
            _baseline_stats(tokens_per_sec=None),
            _baseline_stats(tokens_per_sec=0.0),
            _baseline_stats(tokens_per_sec=8_700.0, mean_seq=0.0),
        ],
        ids=["none", "env", "no-throughput", "zero-throughput", "zero-mean-seq"],
    )
    def test_unusable_stats_give_the_flat_values(self, stats):
        assert quant_hours_breakdown(FREEFORM, PARAMS_FA, stats, boss=False) is None
        assert quant_training_hours(FREEFORM, int(PARAMS_FA), False, stats) == 2.0
        assert quant_training_hours(FREEFORM, int(PARAMS_FA), True, stats) == 2.0
        assert quant_training_hours(FIXED, int(8e9), True, stats) == 2.0

    def test_default_is_flat(self):
        assert quant_training_hours(FREEFORM, int(PARAMS_FA), boss=True) == 2.0


class TestQuantHoursFromBaselineStats:
    @pytest.mark.parametrize("task_type", sorted(QUANT_TASK_TYPES))
    def test_post_prep_stats_never_resize_a_quant_type(self, task_type):
        stats = _baseline_stats(total_tokens=int(40e6), num_records=100_000)

        assert compute_hours_from_baseline_stats(1.75, stats, task_type, model_params_count=int(PARAMS_3B)) == 1.75

    def test_instruct_is_still_resized(self):
        stats = _baseline_stats(total_tokens=int(40e6), num_records=100_000)

        hours = compute_hours_from_baseline_stats(1.75, stats, TaskType.INSTRUCTTEXTTASK, model_params_count=int(PARAMS_3B))

        assert hours == compute_training_hours(stats.dataset.total_tokens, PARAMS_3B, TaskType.INSTRUCTTEXTTASK)


def _quant_task(task_type: TaskType, params: float, hours: float, **overrides):
    fields = dict(
        is_organic=False,
        task_id=uuid.uuid4(),
        status=TaskStatus.AWAITING_MODEL_PREP,
        model_id="org/model",
        ds="org/dataset",
        account_id=uuid.uuid4(),
        hours_to_complete=hours,
        created_at=datetime(2026, 10, 29),
        model_params_count=int(params),
        dataset_type=InstructTextDatasetType(field_instruction="instruction", field_output="output"),
    )
    fields.update(overrides)
    if task_type == TaskType.QUANTFIXEDTASK:
        return QuantFixedRawTask(**fields, quant_spec=QuantFormat.W4A16_SYM_G128.spec())
    return QuantFreeformRawTask(**fields)


def _own_model_fixed():
    return _quant_task(TaskType.QUANTFIXEDTASK, PARAMS_FA, 4.0, training_start_point=TrainingStartPoint.OWN_R1_MODEL)


class TestQuantHoursAfterPrep:
    def test_an_unchained_fixed_task_has_no_instruct_term(self):
        """A fixed task is priced like any quant task, boss or not (multiplier 1.0)."""
        task = _quant_task(TaskType.QUANTFIXEDTASK, PARAMS_FA, 4.0)
        stats = _fa_stats()

        assert quant_hours_after_prep(task, stats, boss=True) == quant_training_hours(FIXED, int(PARAMS_FA), True, stats)
        assert quant_hours_after_prep(task, stats, boss=True) == quant_hours_after_prep(task, stats, boss=False)
        assert quant_hours_after_prep(task, stats, boss=True) <= 1.25 + 1.0  # no training component on top

    def test_missing_stats_keep_the_creation_hours(self):
        task = _quant_task(TaskType.QUANTFIXEDTASK, 3e9, 3.5)

        assert quant_hours_after_prep(task, None, boss=True) == 3.5

    @pytest.mark.parametrize(
        ("task", "boss", "expected"),
        [
            (_quant_task(TaskType.QUANTFREEFORMTASK, PARAMS_FA, 2.0), False, 2.0),
            (_quant_task(TaskType.QUANTFREEFORMTASK, PARAMS_FA, 2.0), True, 2.0),
            (_quant_task(TaskType.QUANTFIXEDTASK, PARAMS_FA, 4.0, upstream_task_id=uuid.uuid4()), True, 1.25),
            (_own_model_fixed(), True, 1.25),
            (_quant_task(TaskType.QUANTFIXEDTASK, PARAMS_FA, 4.0), True, 1.25),
        ],
        ids=["freeform", "freeform-boss", "chained", "own-model", "unchained"],
    )
    def test_every_other_quant_task_is_resized_from_its_stats(self, task, boss, expected):
        assert quant_hours_after_prep(task, _fa_stats(), boss) == expected

    @pytest.mark.parametrize("stats", [None, _baseline_stats(tokens_per_sec=None)], ids=["none", "no-throughput"])
    def test_unusable_stats_keep_the_current_hours(self, stats):
        task = _quant_task(TaskType.QUANTFREEFORMTASK, PARAMS_FA, 3.0)

        assert quant_hours_after_prep(task, stats, boss=True) == 3.0
