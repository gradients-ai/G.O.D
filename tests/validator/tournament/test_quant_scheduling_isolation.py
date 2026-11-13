"""A quant row that raises while being scheduled must not hold the text and image rows behind it; a text or image row that
raises behaves as it always did (logged, retried in place)."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from core.models.quant_models import QuantEvalGpu
from core.models.task_models import TaskType
from validator.scoring.models import QuantCalibrationReferenceRecord
from validator.scoring.models import QuantSkuReference
from validator.tournament import orchestrator
from validator.tournament.seed_tasks import INPUT_SMALL_SEED_TRAINING_RANK
from validator.tournament.seed_tasks import NON_SEED_TRAINING_RANK


CONFIG = SimpleNamespace(psql_db=None)
REAL_CREATE_TRAINING_REQUEST = orchestrator._create_training_request


def _sku_reference(task) -> QuantSkuReference:
    return QuantSkuReference(
        task_id=task.task_id,
        hotkey="",
        sku="h100-sxm",
        model="org/bf16",
        loss=2.0,
        per_example_losses=[2.0],
        eval_set_fingerprint="fp",
        eval_gpu=QuantEvalGpu(name="NVIDIA H100", sm_count=132, sku="h100-sxm"),
    )


def _calibration_record(task, fingerprint="fp") -> QuantCalibrationReferenceRecord:
    return QuantCalibrationReferenceRecord(
        task_id=task.task_id,
        hotkey="",
        model="org/bf16",
        loss=2.0,
        per_example_losses=[1.0, 2.0, 3.0],
        eval_set_fingerprint=fingerprint,
        eval_gpu=QuantEvalGpu(name="NVIDIA H100", sm_count=132, sku="h100-sxm"),
    )


def _row(task_type: TaskType, hotkey: str = "5Miner", calibrated: bool = True) -> SimpleNamespace:
    task = SimpleNamespace(
        task_id=uuid.uuid4(),
        task_type=task_type,
        hours_to_complete=2.0,
        calibration_data="s3://cal.jsonl" if calibrated else None,
        calibration_fingerprint="fp" if calibrated else None,
        calibration_n_examples=3 if calibrated else None,
    )
    return SimpleNamespace(
        task=task,
        hotkey=hotkey,
        n_training_attempts=0,
        training_repo="https://github.com/x/repo",
        training_commit_hash="c" * 40,
        github_token=None,
        requested_datasets=None,
    )


class Scheduling:
    def __init__(self, monkeypatch) -> None:
        self.ranks: dict[str, int] = {}
        self.reference_raises: set[uuid.UUID] = set()
        self.build_raises: set[uuid.UUID] = set()
        self.build_exception: Exception = RuntimeError("presign failed")
        self.built: list[uuid.UUID] = []
        self.sleeps: list[float] = []
        self.on_sleep = None

        async def reference_ready(task, hotkey, config):
            if task.task_id in self.reference_raises:
                raise RuntimeError("reference lookup blew up")
            return True

        async def create_request(task, *_args):
            if task.task_id in self.build_raises:
                raise self.build_exception
            self.built.append(task.task_id)
            return SimpleNamespace()

        async def sleep(seconds):
            self.sleeps.append(seconds)
            if self.on_sleep is not None:
                self.on_sleep()

        async def training_ranks(task_ids, _db):
            return {task_id: self.ranks.get(task_id, NON_SEED_TRAINING_RANK) for task_id in task_ids}

        monkeypatch.setattr(orchestrator, "training_ranks", training_ranks)
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value="tourn"))
        monkeypatch.setattr(orchestrator, "_reference_ready", reference_ready)
        monkeypatch.setattr(orchestrator, "get_task_gpu_requirement", lambda task: SimpleNamespace(value="h100"))
        monkeypatch.setattr(orchestrator, "_update_all_trainers_gpu_availability", AsyncMock())
        monkeypatch.setattr(orchestrator, "_check_suitable_gpus", AsyncMock(return_value=("1.2.3.4", [0], "NVLINK")))
        monkeypatch.setattr(orchestrator, "_create_training_request", create_request)
        monkeypatch.setattr(orchestrator, "start_training_task", AsyncMock(return_value=True))
        monkeypatch.setattr(orchestrator.tournament_sql, "update_tournament_task_training_status", AsyncMock())
        monkeypatch.setattr(orchestrator.tournament_sql, "update_gpu_availability", AsyncMock())
        monkeypatch.setattr(orchestrator.gpu_cost_sql, "start_cost_run", AsyncMock())
        monkeypatch.setattr(orchestrator.asyncio, "sleep", sleep)


@pytest.fixture
def scheduling(monkeypatch) -> Scheduling:
    return Scheduling(monkeypatch)


class TestAQuantRowThatRaises:
    async def test_a_raising_reference_check_does_not_abort_the_cycle(self, scheduling):
        quant, text = _row(TaskType.QUANTFIXEDTASK), _row(TaskType.INSTRUCTTEXTTASK)
        scheduling.reference_raises.add(quant.task.task_id)
        pending = [text, quant]  # the last row is taken first

        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        assert scheduling.built == [text.task.task_id]
        assert pending == []

    async def test_a_raising_request_build_pops_the_row_and_the_next_is_scheduled(self, scheduling):
        quant, text = _row(TaskType.QUANTFIXEDTASK), _row(TaskType.INSTRUCTTEXTTASK)
        scheduling.build_raises.add(quant.task.task_id)
        pending = [text, quant]

        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        assert scheduling.built == [text.task.task_id]
        assert pending == []
        assert orchestrator.cst.TRAINING_START_RETRY_INTERVAL not in scheduling.sleeps

    async def test_an_instruct_seed_row_of_a_quant_tournament_is_treated_the_same(self, scheduling):
        seed, text = _row(TaskType.INSTRUCTTEXTTASK), _row(TaskType.INSTRUCTTEXTTASK)
        scheduling.ranks[str(seed.task.task_id)] = INPUT_SMALL_SEED_TRAINING_RANK
        scheduling.build_raises.add(seed.task.task_id)
        pending = [text, seed]

        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        assert scheduling.built == [text.task.task_id]
        assert pending == []


class TestAQuantRowWithoutItsInputs:
    @pytest.fixture(autouse=True)
    def alerts(self, monkeypatch):
        self.alert = AsyncMock()
        monkeypatch.setattr(orchestrator, "notify_evaluation_exception", self.alert)
        monkeypatch.setattr(orchestrator, "_QUANT_INPUTS_MISSING_HELD_ROWS", {})

    @pytest.mark.parametrize("task_type", [TaskType.QUANTFIXEDTASK, TaskType.QUANTFREEFORMTASK])
    @pytest.mark.parametrize("hotkey", ["5Miner", orchestrator.EMISSION_BURN_HOTKEY])
    async def test_it_is_held_before_any_gpu_lookup_and_nothing_is_dispatched_or_marked(self, scheduling, task_type, hotkey):
        quant, text = _row(task_type, hotkey, calibrated=False), _row(TaskType.INSTRUCTTEXTTASK)
        pending = [text, quant]

        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        assert scheduling.built == [text.task.task_id]  # the row behind it is scheduled; the held row never built a request
        assert pending == []
        assert orchestrator._check_suitable_gpus.await_count == 1  # the text row's lookup only
        orchestrator.start_training_task.assert_awaited_once()
        orchestrator.tournament_sql.update_tournament_task_training_status.assert_awaited_once()  # the text row: TRAINING
        assert orchestrator.tournament_sql.update_tournament_task_training_status.await_args.args[0] == text.task.task_id
        orchestrator.tournament_sql.update_gpu_availability.assert_awaited_once()
        self.alert.assert_awaited_once()

    async def test_the_loop_logs_the_held_pairs_at_error_level(self, scheduling, caplog):
        with caplog.at_level("ERROR"):
            await orchestrator.schedule_tasks_for_training([_row(TaskType.QUANTFIXEDTASK, calibrated=False)], CONFIG)

        assert "no published calibration set; none was dispatched" in caplog.text

    async def test_it_is_alerted_again_every_nth_cycle(self, scheduling):
        every = orchestrator.QUANT_INPUTS_MISSING_REALERT_HELD_ROWS
        row = _row(TaskType.QUANTFREEFORMTASK, calibrated=False)

        for _ in range(every):
            await orchestrator.schedule_tasks_for_training([row], CONFIG)

        assert self.alert.await_count == 2  # the first cycle and the `every`-th
        assert scheduling.built == []

    @pytest.fixture
    def real_request_build(self, monkeypatch):
        """The real _create_training_request over stored references; `store` holds what the database would."""
        store = SimpleNamespace(sku_references=True, calibration="fp", url="https://minio.example/cal.jsonl?sig=x")

        async def sku_references(task_id, psql_db):
            return {("", "h100-sxm"): _sku_reference(SimpleNamespace(task_id=task_id))} if store.sku_references else {}

        async def calibration_references(task_id, psql_db):
            task = SimpleNamespace(task_id=task_id)
            return {"": _calibration_record(task, store.calibration)} if store.calibration else {}

        monkeypatch.setattr(orchestrator, "_create_training_request", REAL_CREATE_TRAINING_REQUEST)
        monkeypatch.setattr(orchestrator.task_sql, "get_expected_repo_name", AsyncMock(return_value=None))
        monkeypatch.setattr(orchestrator.task_sql, "get_starting_model_repo", AsyncMock(return_value=None))
        monkeypatch.setattr(orchestrator.task_sql, "get_miner_baseline_stats", AsyncMock(return_value=None))
        monkeypatch.setattr(orchestrator, "get_quant_sku_references", sku_references)
        monkeypatch.setattr(orchestrator, "get_quant_calibration_references", calibration_references)
        monkeypatch.setattr(
            orchestrator.async_minio_client, "get_new_presigned_url", AsyncMock(side_effect=lambda *_: store.url)
        )
        return store

    def _quant_row(self, hotkey: str = "5Miner") -> SimpleNamespace:
        quant = _row(TaskType.QUANTFIXEDTASK, hotkey)
        quant.task.model_id = "org/model"
        quant.task.augmented_model_id = None
        quant.task.baseline_stats = None
        quant.task.training_start_point = None
        quant.task.quant_spec = object()
        return quant

    @pytest.mark.parametrize("hotkey", ["5Miner", orchestrator.EMISSION_BURN_HOTKEY])
    @pytest.mark.parametrize(
        "fault,reason",
        [
            ("no_target", "no target to publish"),
            ("no_calibration_reference", "not measured on the calibration rows"),
            ("stale_calibration_reference", "another example set"),
            ("unpresignable", "could not be published"),
        ],
    )
    async def test_the_dispatch_backstop_holds_the_row_without_touching_it(
        self, scheduling, real_request_build, fault, reason, hotkey
    ):
        if fault == "no_target":
            real_request_build.sku_references = False
        elif fault == "no_calibration_reference":
            real_request_build.calibration = None
        elif fault == "stale_calibration_reference":
            real_request_build.calibration = "fp-stale"
        else:
            real_request_build.url = None
        quant = self._quant_row(hotkey)
        pending = [quant]

        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        assert pending == []
        self.alert.assert_awaited_once()
        assert reason in self.alert.await_args.kwargs["error"]
        assert self.alert.await_args.kwargs["hotkeys"] == [hotkey]
        orchestrator.start_training_task.assert_not_awaited()
        orchestrator.tournament_sql.update_tournament_task_training_status.assert_not_awaited()
        orchestrator.tournament_sql.update_gpu_availability.assert_not_awaited()
        assert orchestrator.cst.TRAINING_START_RETRY_INTERVAL not in scheduling.sleeps

    async def test_a_row_with_every_input_is_dispatched(self, scheduling, real_request_build, monkeypatch):
        built = MagicMock(return_value=SimpleNamespace())
        monkeypatch.setattr(orchestrator, "_quant_training_data", built)
        monkeypatch.setattr(orchestrator, "TrainerProxyRequest", lambda **kwargs: SimpleNamespace(**kwargs))
        pending = [self._quant_row()]

        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        orchestrator.start_training_task.assert_awaited_once()
        kwargs = built.call_args.kwargs
        assert kwargs["quant_target"].reference_loss == 2.0
        assert kwargs["calibration"].reference.per_example_losses == [1.0, 2.0, 3.0]
        self.alert.assert_not_awaited()

    async def test_a_held_row_is_alerted_again_every_nth_check(self, scheduling, real_request_build):
        real_request_build.sku_references = False
        every = orchestrator.QUANT_INPUTS_MISSING_REALERT_HELD_ROWS
        row = self._quant_row()

        for _ in range(every):
            await orchestrator.schedule_tasks_for_training([row], CONFIG)

        assert self.alert.await_count == 2
        orchestrator.start_training_task.assert_not_awaited()


class TestATextRowThatRaises:
    async def test_a_raising_request_build_is_retried_in_place_with_the_same_sleep(self, scheduling):
        text, other = _row(TaskType.INSTRUCTTEXTTASK), _row(TaskType.IMAGETASK)
        scheduling.build_raises.add(text.task.task_id)
        pending = [other, text]
        scheduling.on_sleep = scheduling.build_raises.clear  # the failure clears after the first retry sleep
        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        assert scheduling.sleeps.count(orchestrator.cst.TRAINING_START_RETRY_INTERVAL) == 1
        assert scheduling.built == [text.task.task_id, other.task.task_id]

    async def test_a_raising_reference_check_on_a_text_row_still_propagates(self, scheduling):
        text = _row(TaskType.INSTRUCTTEXTTASK)
        scheduling.reference_raises.add(text.task.task_id)

        with pytest.raises(RuntimeError, match="reference lookup blew up"):
            await orchestrator.schedule_tasks_for_training([text], CONFIG)
