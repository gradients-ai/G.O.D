"""Chained boss tasks in the orchestrator: the release of a held task once its upstream instruct task is
scored, model prep of a released task, and the near-duplicate gate.

Both players compress one shared input: the repo our trainer uploaded for the usable upstream run with the
best loss (chaining.shared_input_candidates), falling back to the next player when that repo is not a full
model. With no usable output every player is scored out with UPSTREAM_FAILURE_REASON and the task fails.
"""

import json
import uuid
from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from huggingface_hub.errors import RepositoryNotFoundError

import validator.tasks.datasets.constants as data_cst
import validator.tournament.constants as t_cst
from core.constants.environments import EnvironmentName
from core.constants.environments import TrainingStartPoint
from core.constants.quant import QUANT_ADAPTER_CONFIG_FILENAME
from core.constants.quant import QUANT_MODEL_CONFIG_FILENAME
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.model_prep_models import InstructBaselineStats
from core.models.model_prep_models import InstructDatasetStats
from core.models.model_prep_models import InstructTrainingDynamics
from core.models.model_prep_models import SeqLengthDistribution
from core.models.model_prep_models import ThroughputStats
from core.models.model_prep_models import WeightStats
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.tasks import quant_input_repo
from validator.tasks.models import EnvRawTask
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tasks.synthetics.scheduler import quant_training_hours
from validator.tournament import orchestrator
from validator.tournament.models import TournamentType


PLAYER = "5Player"
FAILED_PLAYER = "5FailedPlayer"
BOSS = "5Boss"
TOURNAMENT_ID = "tourn_boss"
UPSTREAM_DS = "org/instruct-dataset"
STALE_SHARDS = [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)]
_REAL_IS_FULL_MODEL_REPO = orchestrator._is_full_model_repo
CONFIG = SimpleNamespace(psql_db=None, discord_url=None)


def _base_fields(**overrides) -> dict:
    return {
        "is_organic": False,
        "task_id": uuid.uuid4(),
        "model_id": "org/base-model",
        "ds": UPSTREAM_DS,
        "account_id": uuid.uuid4(),
        "hours_to_complete": 3.0,
        "created_at": datetime(2026, 10, 19),
        **overrides,
    }


def _upstream(status: TaskStatus = TaskStatus.SUCCESS) -> InstructTextRawTask:
    return InstructTextRawTask(
        **_base_fields(status=status, training_data="s3://bucket/train.json", test_data="s3://bucket/test.json"),
        field_instruction="instruction",
        field_input="input",
        field_output="output",
        field_system="system",
        format="{instruction} {input}",
        no_input_format="{instruction}",
    )


def _chained_fields(upstream: InstructTextRawTask, **overrides) -> dict:
    chained = {
        "status": TaskStatus.AWAITING_UPSTREAM,
        "upstream_task_id": upstream.task_id,
        "hours_to_complete": 1.0,
        "dataset_type": InstructTextDatasetType(field_instruction="question"),
    }
    return _base_fields(**{**chained, **overrides})


def _chained_fixed(upstream: InstructTextRawTask, **overrides) -> QuantFixedRawTask:
    return QuantFixedRawTask(**_chained_fields(upstream, quant_spec=QuantFormat.W4A8_FP8_G128.spec(), **overrides))


def _chained_freeform(upstream: InstructTextRawTask, **overrides) -> QuantFreeformRawTask:
    return QuantFreeformRawTask(**_chained_fields(upstream, **overrides))


def _expected_repo(upstream: InstructTextRawTask, hotkey: str) -> str:
    return f"validator-org/{upstream.task_id}-{hotkey}"


@pytest.fixture
def release(monkeypatch):
    upstream = _upstream()
    mocks = SimpleNamespace(
        upstream=upstream,
        get_task=AsyncMock(return_value=upstream),
        nodes=AsyncMock(return_value=[SimpleNamespace(hotkey=BOSS), SimpleNamespace(hotkey=PLAYER)]),
        candidates=AsyncMock(return_value=[PLAYER, BOSS]),
        input_repo=AsyncMock(side_effect=lambda task_id, hotkey, _db: _expected_repo(upstream, hotkey)),
        full_model=AsyncMock(return_value=True),
        persist=AsyncMock(),
        update_task=AsyncMock(),
        non_boss=AsyncMock(return_value=False),
        input_seed=AsyncMock(return_value=False),
    )
    monkeypatch.setattr(orchestrator, "_is_non_boss_tournament_task", mocks.non_boss)
    monkeypatch.setattr(orchestrator, "_is_input_seed", mocks.input_seed)
    # The upstream is an instruct task that is not a seed (a text seed upstream is released by its own rule).
    monkeypatch.setattr(orchestrator.seeds_sql, "get_seed_task_by_task_id", AsyncMock(return_value=None))
    monkeypatch.setattr(orchestrator.task_sql, "get_task", mocks.get_task)
    monkeypatch.setattr(orchestrator.task_sql, "get_nodes_assigned_to_task", mocks.nodes)
    monkeypatch.setattr(orchestrator.chaining, "shared_input_candidates", mocks.candidates)
    monkeypatch.setattr(orchestrator.chaining, "chained_input_repo", mocks.input_repo)
    monkeypatch.setattr(orchestrator, "_is_full_model_repo", mocks.full_model)
    monkeypatch.setattr(orchestrator.submissions_and_scoring_sql, "set_quant_decode_result", mocks.persist)
    monkeypatch.setattr(orchestrator.task_sql, "update_task", mocks.update_task)
    return mocks


def _scored_out(mocks) -> list[str]:
    calls = mocks.persist.await_args_list
    assert all(call.kwargs["quant_failure"] == t_cst.UPSTREAM_FAILURE_REASON for call in calls)
    assert all(call.kwargs["artifact_bytes"] is None and call.kwargs["decoder_sha256"] is None for call in calls)
    return [call.args[1] for call in calls]


class TestReleaseChainedTask:
    @pytest.mark.parametrize("status", [TaskStatus.TRAINING, TaskStatus.PREEVALUATION, TaskStatus.FAILURE])
    async def test_held_while_upstream_is_not_success(self, release, status):
        release.upstream.status = status
        task = _chained_fixed(release.upstream)

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.status == TaskStatus.AWAITING_UPSTREAM
        release.nodes.assert_not_awaited()
        release.update_task.assert_not_awaited()

    async def test_held_when_upstream_is_missing(self, release):
        release.get_task.return_value = None
        task = _chained_fixed(release.upstream)

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.status == TaskStatus.AWAITING_UPSTREAM
        release.update_task.assert_not_awaited()

    async def test_held_without_assigned_nodes(self, release):
        release.nodes.return_value = []
        task = _chained_fixed(release.upstream)

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.status == TaskStatus.AWAITING_UPSTREAM
        release.candidates.assert_not_awaited()
        release.update_task.assert_not_awaited()

    @pytest.mark.parametrize("make_task", [_chained_fixed, _chained_freeform])
    async def test_best_upstream_output_is_the_shared_input(self, release, make_task):
        task = make_task(release.upstream)

        await orchestrator._release_chained_task(task, CONFIG)

        release.candidates.assert_awaited_once_with(release.upstream.task_id, [BOSS, PLAYER], None)
        release.full_model.assert_awaited_once_with(_expected_repo(release.upstream, PLAYER))
        assert task.model_id == _expected_repo(release.upstream, PLAYER)
        release.persist.assert_not_awaited()
        assert task.status == TaskStatus.PENDING
        release.update_task.assert_awaited_once_with(task, None)

    async def test_one_usable_output_is_compressed_by_both_players(self, release):
        release.nodes.return_value = [SimpleNamespace(hotkey=BOSS), SimpleNamespace(hotkey=FAILED_PLAYER)]
        release.candidates.return_value = [BOSS]
        task = _chained_fixed(release.upstream)

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.model_id == _expected_repo(release.upstream, BOSS)
        release.persist.assert_not_awaited()
        assert task.status == TaskStatus.PENDING

    async def test_non_full_model_falls_back_to_the_other_player(self, release):
        release.full_model.side_effect = lambda repo: repo != _expected_repo(release.upstream, PLAYER)
        task = _chained_fixed(release.upstream)

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.model_id == _expected_repo(release.upstream, BOSS)
        assert task.status == TaskStatus.PENDING

    async def test_stale_shard_index_falls_back_to_the_other_player(self, release, monkeypatch, tmp_path):
        """The best output ships one complete model.safetensors plus an index of shards that do not exist:
        vLLM cannot load it, so the shared input is the other player's consistent repo."""
        monkeypatch.setattr(orchestrator, "_is_full_model_repo", _REAL_IS_FULL_MODEL_REPO)
        stale = ["config.json", "model.safetensors", "model.safetensors.index.json"]
        sharded = ["config.json", "model.safetensors.index.json", *STALE_SHARDS]
        files = {_expected_repo(release.upstream, PLAYER): stale, _expected_repo(release.upstream, BOSS): sharded}
        index_path = tmp_path / "model.safetensors.index.json"
        index_path.write_text(json.dumps({"weight_map": {f"w{i}": shard for i, shard in enumerate(STALE_SHARDS)}}))
        monkeypatch.setattr(quant_input_repo, "list_repo_files", MagicMock(side_effect=lambda repo, token=None: files[repo]))
        monkeypatch.setattr(quant_input_repo, "hf_hub_download", MagicMock(return_value=str(index_path)))
        task = _chained_freeform(release.upstream)

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.model_id == _expected_repo(release.upstream, BOSS)
        assert task.status == TaskStatus.PENDING

    async def test_missing_uploaded_repo_falls_back_to_the_other_player(self, release):
        release.input_repo.side_effect = lambda task_id, hotkey, _db: None if hotkey == PLAYER else "org/boss-repo"
        task = _chained_freeform(release.upstream)

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.model_id == "org/boss-repo"

    @pytest.mark.parametrize("candidates", [[], [PLAYER, BOSS]])
    async def test_no_usable_full_model_fails_both_players(self, release, candidates):
        release.candidates.return_value = candidates
        release.full_model.return_value = False
        task = _chained_fixed(release.upstream)

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.status == TaskStatus.FAILURE
        assert task.model_id == "org/base-model"
        assert task.training_data is None
        assert _scored_out(release) == [BOSS, PLAYER]
        release.update_task.assert_awaited_once_with(task, None)

    async def test_hub_error_holds_the_task(self, release):
        release.full_model.side_effect = RuntimeError("hub down")
        task = _chained_fixed(release.upstream)

        with pytest.raises(RuntimeError):
            await orchestrator._release_chained_task(task, CONFIG)

        assert (task.status, task.model_id) == (TaskStatus.AWAITING_UPSTREAM, "org/base-model")
        release.persist.assert_not_awaited()
        release.update_task.assert_not_awaited()

    async def test_held_copy_of_a_released_task_takes_the_rule_again(self, release):
        # An identical held replacement carries the model_id its original was released on.
        release.candidates.return_value = [BOSS, PLAYER]
        task = _chained_fixed(release.upstream, model_id=_expected_repo(release.upstream, PLAYER))

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.model_id == _expected_repo(release.upstream, BOSS)

    async def test_pinned_task_copies_the_upstream_split(self, release):
        upstream = release.upstream
        task = _chained_fixed(upstream)

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.training_data == upstream.training_data
        assert task.test_data == upstream.test_data
        assert task.file_format == FileFormat.S3
        assert task.dataset_type == InstructTextDatasetType(
            field_instruction="instruction",
            field_input="input",
            field_output="output",
            field_system="system",
            format="{instruction} {input}",
            no_input_format="{instruction}",
        )
        assert task.hours_to_complete == 2.0  # flat fixed budget at the default 8B size
        assert task.termination_at is not None

    async def test_pinned_task_inherits_the_upstream_calibration_rows(self, release):
        release.upstream.calibration_data = "s3://bucket/upstream-calibration.json"
        task = _chained_fixed(release.upstream)

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.calibration_data == "s3://bucket/upstream-calibration.json"

    @pytest.mark.parametrize("make_task", [_chained_fixed, _chained_freeform])
    async def test_a_released_task_always_carves_when_it_has_no_raw_rows(self, release, make_task, monkeypatch):
        alert = AsyncMock()
        monkeypatch.setattr(orchestrator, "alert_seed_issue", alert)
        task = make_task(release.upstream)
        assert task.carve_calibration is False

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.carve_calibration is True and task.status == TaskStatus.PENDING

    async def test_a_copy_of_the_upstream_calibration_rows_does_not_alert(self, release, monkeypatch):
        alert = AsyncMock()
        monkeypatch.setattr(orchestrator, "alert_seed_issue", alert)
        release.upstream.calibration_data = "s3://bucket/upstream-calibration.json"
        task = _chained_fixed(release.upstream)

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.carve_calibration is True
        alert.assert_not_awaited()

    async def test_an_upstream_without_calibration_rows_alerts_and_the_task_carves_its_own(self, release, monkeypatch, caplog):
        alert = AsyncMock()
        monkeypatch.setattr(orchestrator, "alert_seed_issue", alert)
        task = _chained_freeform(release.upstream)

        with caplog.at_level("ERROR"):
            await orchestrator._release_chained_task(task, CONFIG)

        assert task.calibration_data is None and task.carve_calibration is True
        alert.assert_awaited_once()
        key, message, _ = alert.await_args.args
        assert key == f"{task.task_id}:quant-upstream-no-calibration"
        assert "published no calibration rows" in message and "carves its own" in message

    async def test_a_task_on_its_own_data_carves_and_does_not_alert(self, release, monkeypatch):
        alert = AsyncMock()
        monkeypatch.setattr(orchestrator, "alert_seed_issue", alert)
        task = _chained_freeform(release.upstream, ds="org/fresh-dataset")

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.carve_calibration is True
        alert.assert_not_awaited()

    async def test_a_non_boss_round_task_gets_the_non_boss_hours_from_the_upstream_stats(self, release):
        release.non_boss.return_value = True
        release.upstream.model_params_count = 3_820_000_000
        release.upstream.baseline_stats = _usable_stats()
        task = _chained_fixed(release.upstream)  # created with 1.0 h

        await orchestrator._release_chained_task(task, CONFIG)

        expected = quant_training_hours(task.task_type, 3_820_000_000, boss=False, baseline_stats=_usable_stats())
        assert task.hours_to_complete == expected != 1.0
        assert task.training_data == release.upstream.training_data  # the data is still copied

    async def test_fresh_data_decider_keeps_its_own_data_on_the_shared_input(self, release):
        task = _chained_freeform(release.upstream, ds="org/fresh-dataset")
        dataset_type = task.dataset_type

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.model_id == _expected_repo(release.upstream, PLAYER)
        assert task.ds == "org/fresh-dataset"
        assert task.training_data is None and task.test_data is None
        assert task.dataset_type is dataset_type
        assert task.hours_to_complete == 1.0
        assert task.status == TaskStatus.PENDING


class TestIsFullModelRepo:
    @pytest.mark.parametrize(
        ("files", "full"),
        [
            ([QUANT_MODEL_CONFIG_FILENAME, "model.safetensors"], True),
            ([QUANT_ADAPTER_CONFIG_FILENAME, "adapter_model.safetensors"], False),
            ([QUANT_MODEL_CONFIG_FILENAME, QUANT_ADAPTER_CONFIG_FILENAME], False),
            ([QUANT_MODEL_CONFIG_FILENAME, "tokenizer.json"], False),
        ],
    )
    async def test_listing_decides(self, monkeypatch, files, full):
        monkeypatch.setattr(quant_input_repo, "list_repo_files", MagicMock(return_value=files))

        assert await orchestrator._is_full_model_repo("org/repo") is full

    async def test_missing_repo_is_not_a_full_model(self, monkeypatch):
        monkeypatch.setattr(quant_input_repo, "list_repo_files", MagicMock(side_effect=RepositoryNotFoundError("missing")))

        assert await orchestrator._is_full_model_repo("org/repo") is False

    async def test_other_hub_errors_raise(self, monkeypatch):
        monkeypatch.setattr(quant_input_repo, "list_repo_files", MagicMock(side_effect=ConnectionError("hub down")))

        with pytest.raises(ConnectionError):
            await orchestrator._is_full_model_repo("org/repo")


class _StopCycle(Exception):
    """Raised from the patched sleep to end an orchestrator loop after one pass."""


def _one_pass_asyncio(created: list) -> SimpleNamespace:
    return SimpleNamespace(sleep=AsyncMock(side_effect=_StopCycle), create_task=created.append)


class TestReleaseCycle:
    async def test_an_error_in_one_task_does_not_stop_the_pass(self, monkeypatch, release):
        first, second = _chained_fixed(release.upstream), _chained_fixed(release.upstream)
        monkeypatch.setattr(orchestrator.task_sql, "get_tasks_with_status", AsyncMock(return_value=[first, second]))
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=TOURNAMENT_ID))
        monkeypatch.setattr(orchestrator, "asyncio", _one_pass_asyncio([]))
        release.get_task.side_effect = [RuntimeError("db hiccup"), release.upstream]

        with pytest.raises(_StopCycle):
            await orchestrator.process_awaiting_upstream_tasks(CONFIG)

        orchestrator.task_sql.get_tasks_with_status.assert_awaited_once_with(
            TaskStatus.AWAITING_UPSTREAM, None, tournament_filter="only"
        )
        assert first.status == TaskStatus.AWAITING_UPSTREAM
        assert second.status == TaskStatus.PENDING


@pytest.fixture
def model_prep(monkeypatch):
    created: list = []
    mocks = SimpleNamespace(
        created=created,
        tasks=AsyncMock(return_value=[]),
        has_starting=AsyncMock(return_value=True),
        needing=AsyncMock(return_value=[(PLAYER, "validator-org/continuation")]),
        dispatch=AsyncMock(),
        set_stats=AsyncMock(),
        update_task=AsyncMock(),
    )
    monkeypatch.setattr(orchestrator, "asyncio", _one_pass_asyncio(created))
    monkeypatch.setattr(orchestrator.task_sql, "get_tasks_with_status", mocks.tasks)
    monkeypatch.setattr(orchestrator.task_sql, "has_miners_with_starting_model", mocks.has_starting)
    monkeypatch.setattr(orchestrator.task_sql, "get_miners_needing_baseline_stats", mocks.needing)
    monkeypatch.setattr(orchestrator.task_sql, "set_miner_baseline_stats", mocks.set_stats)
    monkeypatch.setattr(orchestrator.task_sql, "update_task", mocks.update_task)
    monkeypatch.setattr(orchestrator, "_recover_miner_preps_from_trainers", AsyncMock(return_value=set()))
    monkeypatch.setattr(orchestrator, "get_task_gpu_requirement", MagicMock())
    monkeypatch.setattr(orchestrator, "_check_suitable_gpus", AsyncMock(return_value=("10.0.0.1", [0], "nvlink")))
    monkeypatch.setattr(orchestrator, "_start_prep_cost_run", AsyncMock())
    monkeypatch.setattr(orchestrator, "_finish_prep_cost_run", AsyncMock())
    monkeypatch.setattr(orchestrator.tournament_sql, "update_gpu_availability", AsyncMock())
    monkeypatch.setattr(orchestrator, "dispatch_augmentation_and_stats", mocks.dispatch)
    return mocks


async def _run_model_prep_pass(mocks, task) -> None:
    mocks.tasks.return_value = [task]
    with pytest.raises(_StopCycle):
        await orchestrator.process_awaiting_model_prep_tasks(CONFIG)
    for coroutine in mocks.created:
        await coroutine


def _stats() -> SimpleNamespace:
    return SimpleNamespace(dataset=None)


def _env_task() -> EnvRawTask:
    return EnvRawTask(
        **_base_fields(status=TaskStatus.AWAITING_MODEL_PREP, ds="env"),
        environment_names=[next(iter(EnvironmentName))],
    )


class TestModelPrep:
    async def test_released_chained_task_takes_the_task_level_path(self, model_prep, monkeypatch):
        task_level = AsyncMock(return_value=True)
        monkeypatch.setattr(orchestrator, "_recover_model_prep_from_trainer", task_level)
        task = _chained_fixed(_upstream(), status=TaskStatus.AWAITING_MODEL_PREP, model_id="validator-org/shared-input")

        await _run_model_prep_pass(model_prep, task)

        model_prep.has_starting.assert_not_awaited()
        model_prep.needing.assert_not_awaited()
        task_level.assert_awaited_once()

    async def test_own_model_quant_task_preps_on_its_base_model_not_a_miners_repo(self, model_prep, monkeypatch):
        """Every player has its own starting repo, but the task-level prep (eval set, params) is on task.model_id."""
        monkeypatch.setattr(orchestrator, "_recover_model_prep_from_trainer", AsyncMock(return_value=False))
        model_prep.dispatch.return_value = SimpleNamespace(augmented_model_id=None, baseline_stats=None)
        task = QuantFreeformRawTask(
            **_base_fields(status=TaskStatus.AWAITING_MODEL_PREP, model_id="org/r1-base"),
            dataset_type=InstructTextDatasetType(field_instruction="instruction"),
            training_start_point=TrainingStartPoint.OWN_R1_MODEL,
        )

        await _run_model_prep_pass(model_prep, task)

        model_prep.needing.assert_not_awaited()  # no per-miner baseline preps for quant
        model_prep.has_starting.assert_not_awaited()
        assert model_prep.dispatch.await_args.kwargs["model_id"] == "org/r1-base"
        assert task.status == TaskStatus.LOOKING_FOR_NODES

    async def test_env_per_miner_prep_is_unchanged(self, model_prep):
        model_prep.dispatch.return_value = SimpleNamespace(augmented_model_id="org/augmented", baseline_stats=_stats())
        task = _env_task()

        await _run_model_prep_pass(model_prep, task)

        kwargs = model_prep.dispatch.await_args.kwargs
        assert kwargs["is_env_task"] is True
        assert kwargs["environment_names"] == task.environment_names
        model_prep.set_stats.assert_awaited_once()


def _stats_with_near_duplicates(rate: float) -> SimpleNamespace:
    return SimpleNamespace(dataset=SimpleNamespace(near_duplicate_rate=rate))


class TestNearDuplicateExemption:
    def test_continuous_sft_task_is_allowed_above_the_cutoff(self):
        task = SimpleNamespace(
            task_id=uuid.uuid4(), task_type=TaskType.CHATTASK, training_start_point=TrainingStartPoint.CONTINUOUS_SFT
        )

        assert not orchestrator._exceeds_near_duplicate_threshold(
            task, _stats_with_near_duplicates(t_cst.MAX_NEAR_DUPLICATE_RATE)
        )

    def test_ordinary_quant_task_is_still_rejected(self):
        task = _chained_fixed(_upstream(), status=TaskStatus.AWAITING_MODEL_PREP)

        assert task.task_type == TaskType.QUANTFIXEDTASK
        assert orchestrator._exceeds_near_duplicate_threshold(task, _stats_with_near_duplicates(t_cst.MAX_NEAR_DUPLICATE_RATE))


def _usable_stats() -> InstructBaselineStats:
    """Usable prep stats."""
    seq_dist = SeqLengthDistribution(mean=1041.1, p50=900, p95=2000, p99=3000, max=4000)
    return InstructBaselineStats(
        dataset=InstructDatasetStats(
            total_tokens=40_000_000,
            num_records=100_000,
            seq_length_distribution=seq_dist,
            near_duplicate_rate=0.0,
            vocab_size=150_000,
            prompt_tokens=20_000_000,
            completion_tokens=20_000_000,
            completion_length_distribution=seq_dist,
        ),
        weights=WeightStats(by_group={}),
        training=InstructTrainingDynamics(init_loss=2.0, activation_rms={}, output_entropy=3.0, masked_completion_loss=2.5),
        throughput=ThroughputStats(tokens_per_sec=8_700.0, seq_len=1024, micro_batch_size=8, n_gpus=1),
    )


class TestHoursAfterPrep:
    def _patch_round(
        self, monkeypatch, *, in_tournament: bool, final: bool, text_tournament_id: str | None = None, seeded: bool = False
    ) -> None:
        """text_tournament_id: the tournament a non-quant task belongs to (None for an organic task); seeded: whether that
        text tournament has seed rows."""
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=text_tournament_id))
        monkeypatch.setattr(orchestrator.seeds_sql, "is_seeded_text_tournament", AsyncMock(return_value=seeded))
        tournament_type = TournamentType.QUANT if in_tournament else None
        monkeypatch.setattr(orchestrator.tournament_sql, "get_tournament_type_for_task", AsyncMock(return_value=tournament_type))
        monkeypatch.setattr(orchestrator, "is_final_round_task", AsyncMock(return_value=final))
        monkeypatch.setattr(orchestrator.seeds_sql, "get_seed_task_by_task_id", AsyncMock(return_value=None))

    async def test_quant_task_without_stats_keeps_its_hours(self, monkeypatch):
        self._patch_round(monkeypatch, in_tournament=True, final=True)
        task = QuantFreeformRawTask(
            **_base_fields(status=TaskStatus.AWAITING_MODEL_PREP),
            dataset_type=InstructTextDatasetType(field_instruction="instruction"),
        )

        assert await orchestrator._hours_after_prep(task, CONFIG) == task.hours_to_complete

    async def test_chained_task_without_stats_keeps_its_hours(self, monkeypatch):
        self._patch_round(monkeypatch, in_tournament=True, final=True)
        task = _chained_freeform(_upstream(), status=TaskStatus.AWAITING_MODEL_PREP)

        assert await orchestrator._hours_after_prep(task, CONFIG) == task.hours_to_complete

    @pytest.mark.parametrize(
        ("in_tournament", "final", "expected"),
        [(True, False, 2.0), (True, True, 2.0), (False, False, 2.0)],
        ids=["early-round", "boss-round", "organic"],
    )
    async def test_quant_task_with_stats_is_resized_and_the_boss_flag_follows_the_round(
        self, monkeypatch, in_tournament, final, expected
    ):
        self._patch_round(monkeypatch, in_tournament=in_tournament, final=final)
        task = _chained_freeform(
            _upstream(), status=TaskStatus.AWAITING_MODEL_PREP, model_params_count=3_820_000_000, baseline_stats=_usable_stats()
        )

        assert await orchestrator._hours_after_prep(task, CONFIG) == expected


class TestCopyUpstreamData:
    def test_hours_come_from_the_upstream_stats(self):
        upstream = _upstream()
        upstream.model_params_count = 3_820_000_000
        upstream.baseline_stats = _usable_stats()
        task = _chained_freeform(upstream)

        orchestrator._copy_upstream_data(task, upstream)

        assert task.hours_to_complete == 2.0

    def test_upstream_without_stats_gets_the_flat_freeform_hours(self):
        upstream = _upstream()
        task = _chained_freeform(upstream)

        orchestrator._copy_upstream_data(task, upstream)

        assert task.hours_to_complete == data_cst.QUANT_FREEFORM_EARLY_HOURS


    def test_non_boss_copy_recomputes_the_non_boss_hours_and_copies_data_and_calibration(self):
        upstream = _upstream()
        upstream.model_params_count = 3_820_000_000
        upstream.baseline_stats = _usable_stats()
        upstream.calibration_data = "s3://bucket/cal.json"
        task = _chained_fixed(upstream)
        before = datetime.utcnow()

        orchestrator._copy_upstream_data(task, upstream, boss=False)

        expected = quant_training_hours(task.task_type, 3_820_000_000, boss=False, baseline_stats=_usable_stats())
        assert task.hours_to_complete == expected
        assert task.termination_at >= before + timedelta(hours=expected)
        assert task.calibration_data == "s3://bucket/cal.json" and task.test_data == upstream.test_data

    def test_boss_is_the_default(self):
        upstream = _upstream()
        task = _chained_fixed(upstream)

        orchestrator._copy_upstream_data(task, upstream)

        assert task.hours_to_complete == 2.0


class TestNonBossInstructCap:
    @pytest.fixture
    def resized(self, monkeypatch):
        """The measured-stats estimate is 4.5 h, over the non-boss cap and under the boss cap."""
        monkeypatch.setattr(orchestrator, "compute_hours_from_baseline_stats", lambda *args, **kwargs: 4.5)

    def _patch_round(
        self, monkeypatch, *, in_tournament: bool, final: bool, text_tournament_id: str | None = None, seeded: bool = False
    ) -> None:
        """text_tournament_id: the tournament a non-quant task belongs to (None for an organic task); seeded: whether that
        text tournament has seed rows."""
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=text_tournament_id))
        monkeypatch.setattr(orchestrator.seeds_sql, "is_seeded_text_tournament", AsyncMock(return_value=seeded))
        tournament_type = TournamentType.QUANT if in_tournament else None
        monkeypatch.setattr(orchestrator.tournament_sql, "get_tournament_type_for_task", AsyncMock(return_value=tournament_type))
        monkeypatch.setattr(orchestrator, "is_final_round_task", AsyncMock(return_value=final))
        monkeypatch.setattr(orchestrator.seeds_sql, "get_seed_task_by_task_id", AsyncMock(return_value=None))

    async def test_non_boss_instruct_is_capped(self, resized, monkeypatch):
        self._patch_round(monkeypatch, in_tournament=True, final=False)

        assert await orchestrator._hours_after_prep(_upstream(), CONFIG) == data_cst.NON_BOSS_INSTRUCT_MAX_TRAINING_HOURS

    async def test_non_boss_estimate_under_the_cap_is_kept(self, monkeypatch):
        monkeypatch.setattr(orchestrator, "compute_hours_from_baseline_stats", lambda *args, **kwargs: 1.75)
        self._patch_round(monkeypatch, in_tournament=True, final=False)

        assert await orchestrator._hours_after_prep(_upstream(), CONFIG) == 1.75

    async def test_boss_round_instruct_is_not_capped(self, resized, monkeypatch):
        self._patch_round(monkeypatch, in_tournament=True, final=True)

        assert await orchestrator._hours_after_prep(_upstream(), CONFIG) == 4.5

    async def test_organic_instruct_is_not_capped(self, resized, monkeypatch):
        self._patch_round(monkeypatch, in_tournament=False, final=False)

        assert await orchestrator._hours_after_prep(_upstream(), CONFIG) == 4.5

    async def test_a_round_drawn_task_of_a_seeded_text_tournament_is_capped_like_its_seed(self, resized, monkeypatch):
        self._patch_round(monkeypatch, in_tournament=False, final=False, text_tournament_id="text-1", seeded=True)

        assert await orchestrator._hours_after_prep(_upstream(), CONFIG) == data_cst.NON_BOSS_INSTRUCT_MAX_TRAINING_HOURS

    async def test_a_task_of_an_unseeded_text_tournament_is_not_capped(self, resized, monkeypatch):
        self._patch_round(monkeypatch, in_tournament=False, final=False, text_tournament_id="text-1", seeded=False)

        assert await orchestrator._hours_after_prep(_upstream(), CONFIG) == 4.5

    async def test_env_task_is_not_capped(self, resized, monkeypatch):
        self._patch_round(monkeypatch, in_tournament=True, final=False)
        env_task = SimpleNamespace(task_type=TaskType.ENVIRONMENTTASK, task_id=uuid.uuid4(), hours_to_complete=3.0)
        env_task.baseline_stats = None
        env_task.model_id = env_task.model_params_count = env_task.training_start_point = env_task.ds = None

        assert await orchestrator._hours_after_prep(env_task, CONFIG) == 4.5
