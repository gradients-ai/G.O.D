"""The own-model quant reference pre-pass: candidate selection, previous-round copy, measurement and failure handling."""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

import validator.db.constants as db_cst
import validator.tournament.constants as t_cst
from core.constants.environments import TrainingStartPoint
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import EvaluationResultText
from core.models.quant_models import QuantEvalGpu
from core.models.quant_models import QuantFormat
from core.models.quant_models import QuantReferenceScore
from core.models.task_models import TaskStatus
from validator.evaluation.errors import EvaluationCapacityUnavailable
from validator.scoring.models import QuantCalibrationReferenceRecord
from validator.scoring.models import QuantSkuReference
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tournament import quant_reference_prepass as prepass
from validator.tournament.models import TournamentTask


HOTKEY = "5Miner"
OTHER = "5Other"
REPO = "miner/own-t1"
FINGERPRINT = "fp-same"
SKU = "h100-sxm"
GPU = QuantEvalGpu(name="NVIDIA H100", sm_count=132, sku=SKU)
TOURNAMENT_ID = "tourn"


def _task(cls=QuantFreeformRawTask, **overrides):
    fields = {
        "is_organic": False,
        "task_id": uuid.uuid4(),
        "status": TaskStatus.TRAINING,
        "model_id": "org/model",
        "ds": "org/dataset",
        "account_id": uuid.uuid4(),
        "hours_to_complete": 2.0,
        "training_data": "s3://bucket/train.json",
        "test_data": "https://s3.example/test.json",
        "created_at": datetime(2026, 10, 19),
        "dataset_type": InstructTextDatasetType(field_instruction="instruction", field_output="output"),
        "training_start_point": TrainingStartPoint.OWN_R1_MODEL,
        **overrides,
    }
    if cls is QuantFixedRawTask:
        fields["quant_spec"] = QuantFormat.W4A8_FP8_G128.spec()
    return cls(**fields)


def _reference(task_id, hotkey=HOTKEY, fingerprint=FINGERPRINT, model=REPO, loss=2.0) -> QuantSkuReference:
    return QuantSkuReference(
        task_id=task_id,
        hotkey=hotkey,
        sku=SKU,
        model=model,
        loss=loss,
        per_example_losses=[loss],
        eval_set_fingerprint=fingerprint,
        eval_gpu=GPU,
        measured_at=datetime(2026, 10, 20),
    )


def _result(loss=2.0, quant_failure=None, with_reference=True) -> EvaluationResultText:
    reference = (
        QuantReferenceScore(
            model=REPO, eval_loss=loss, per_example_losses=[loss], eval_set_fingerprint="fp-measured", eval_gpu=GPU
        )
        if with_reference
        else None
    )
    return EvaluationResultText(is_finetune=True, eval_loss=loss, quant_failure=quant_failure, quant_reference=reference)


class FakeDb:
    """In-memory seams behind everything the pre-pass reads and writes."""

    def __init__(self, monkeypatch):
        self.tasks: dict[TaskStatus, list] = {}
        self.nodes: dict[str, list[str]] = {}
        self.starting: dict[tuple[str, str], str] = {}
        self.failed: dict[str, set[str]] = {}
        self.references: dict[uuid.UUID, dict] = {}
        self.rounds: list = []
        self.round_tasks: dict[str, list[str]] = {}
        self.added: list[QuantSkuReference] = []
        self.calibration: dict[uuid.UUID, dict[str, QuantCalibrationReferenceRecord]] = {}
        self.calibration_added: list[QuantCalibrationReferenceRecord] = []

        async def tasks_with_status(status, psql_db, tournament_filter="all"):
            assert tournament_filter == "only"
            return self.tasks.get(status, [])

        async def nodes_for(task_id, psql_db):
            return [SimpleNamespace(hotkey=h) for h in self.nodes.get(task_id, [])]

        async def starting_model(task_id, hotkey, psql_db):
            return self.starting.get((task_id, hotkey))

        async def node_losses(task_id, psql_db):
            return [{db_cst.HOTKEY: h, db_cst.QUANT_FAILURE: "no model"} for h in self.failed.get(str(task_id), set())]

        async def get_references(task_id, psql_db):
            return dict(self.references.get(task_id, {}))

        async def add_reference(reference, psql_db):
            self.added.append(reference)
            self.references.setdefault(reference.task_id, {})[reference.key] = reference
            return reference

        async def tournament_id(task_id, psql_db):
            return TOURNAMENT_ID

        async def rounds(tid, psql_db):
            return self.rounds

        async def round_tasks(round_id, psql_db):
            ids = self.round_tasks.get(round_id, [])
            return [TournamentTask(tournament_id=TOURNAMENT_ID, round_id=round_id, task_id=t) for t in ids]

        async def get_calibration(task_id, psql_db):
            return dict(self.calibration.get(task_id, {}))

        async def add_calibration(record, psql_db):
            self.calibration_added.append(record)
            self.calibration.setdefault(record.task_id, {})[record.hotkey] = record
            return record

        monkeypatch.setattr(prepass, "get_quant_calibration_references", get_calibration)
        monkeypatch.setattr(prepass, "add_quant_calibration_reference", add_calibration)
        monkeypatch.setattr(prepass.task_sql, "get_tasks_with_status", tasks_with_status)
        monkeypatch.setattr(prepass.task_sql, "get_nodes_assigned_to_task", nodes_for)
        monkeypatch.setattr(prepass.task_sql, "get_starting_model_repo", starting_model)
        monkeypatch.setattr(prepass, "get_task_node_losses", node_losses)
        monkeypatch.setattr(prepass, "get_quant_sku_references", get_references)
        monkeypatch.setattr(prepass, "add_quant_sku_reference", add_reference)
        monkeypatch.setattr(prepass.tournament_sql, "get_tournament_id_by_task_id", tournament_id)
        monkeypatch.setattr(prepass.tournament_sql, "get_tournament_rounds", rounds)
        monkeypatch.setattr(prepass.tournament_sql, "get_tournament_tasks", round_tasks)

    def put(self, task, hotkeys=(HOTKEY,), status=TaskStatus.TRAINING):
        self.tasks.setdefault(status, []).append(task)
        self.nodes[str(task.task_id)] = list(hotkeys)
        for hotkey in hotkeys:
            self.starting[(str(task.task_id), hotkey)] = REPO

    def round_before(self, previous_task_id, task):
        self.rounds = [
            SimpleNamespace(round_id="r1", round_number=1),
            SimpleNamespace(round_id="r2", round_number=2),
        ]
        self.round_tasks = {"r1": [str(previous_task_id)], "r2": [str(task.task_id)]}


@pytest.fixture
def db(monkeypatch):
    prepass._IN_PROGRESS.clear()
    prepass._CONSECUTIVE_FAILURES.clear()
    prepass._EVAL_SET_FINGERPRINTS.clear()
    monkeypatch.setattr(prepass, "compute_required_gpus", lambda task: 1)
    monkeypatch.setattr(prepass.gpu_cost_sql, "start_cost_run", AsyncMock(return_value=True))
    monkeypatch.setattr(prepass.gpu_cost_sql, "finish_cost_run", AsyncMock(return_value=None))
    return FakeDb(monkeypatch)


CONFIG = SimpleNamespace(psql_db=object(), discord_url="https://discord.example")


class TestCandidateSelection:
    async def test_own_model_task_hotkey_is_missing(self, db):
        task = _task()
        db.put(task)

        missing = await prepass.references_to_measure(CONFIG)

        assert [(m.task.task_id, m.hotkey, m.repo) for m in missing] == [(task.task_id, HOTKEY, REPO)]

    async def test_both_statuses_are_scanned(self, db):
        looking, training = _task(), _task()
        db.put(looking, status=TaskStatus.LOOKING_FOR_NODES)
        db.put(training, status=TaskStatus.TRAINING)

        assert len(await prepass.references_to_measure(CONFIG)) == 2

    async def test_task_common_task_is_measured_once_under_the_empty_hotkey(self, db):
        task = _task(training_start_point=TrainingStartPoint.DEFAULT)
        db.put(task, hotkeys=(HOTKEY, OTHER))
        db.starting.clear()  # no per-miner starting model: the common input is the task's own model

        [missing] = await prepass.references_to_measure(CONFIG)

        assert (missing.task.task_id, missing.hotkey, missing.repo, missing.calibration) == (task.task_id, "", "org/model", False)
        assert missing.key == (str(task.task_id), "")

    async def test_task_common_task_without_nodes_yet_is_still_measured(self, db):
        db.put(_task(training_start_point=TrainingStartPoint.DEFAULT), hotkeys=())

        assert len(await prepass.references_to_measure(CONFIG)) == 1

    async def test_task_common_reference_is_the_augmented_model_when_there_is_one(self, db):
        db.put(_task(training_start_point=TrainingStartPoint.DEFAULT, augmented_model_id="org/model-aug"))

        [missing] = await prepass.references_to_measure(CONFIG)

        assert missing.repo == "org/model-aug"

    async def test_task_common_stored_row_is_not_remeasured(self, db):
        task = _task(training_start_point=TrainingStartPoint.DEFAULT)
        db.put(task)
        db.references[task.task_id] = {("", SKU): _reference(task.task_id, hotkey="", model="org/model")}

        assert await prepass.references_to_measure(CONFIG) == []

    async def test_task_common_calibration_follows_the_common_test_row_and_gives_up(self, db):
        task = _cal_task(training_start_point=TrainingStartPoint.DEFAULT)
        db.put(task)
        assert not (await prepass.references_to_measure(CONFIG))[0].calibration
        db.references[task.task_id] = {("", SKU): _reference(task.task_id, hotkey="", model="org/model")}

        [missing] = await prepass.references_to_measure(CONFIG)
        assert missing.calibration and missing.key == (str(task.task_id), "", "calibration")

        prepass._CONSECUTIVE_FAILURES[missing.key] = t_cst.QUANT_CALIBRATION_REFERENCE_MAX_FAILURES
        assert prepass.calibration_reference_gave_up(task.task_id, "")
        assert await prepass.references_to_measure(CONFIG) == []

    async def test_task_common_task_copies_the_sibling_slot_row_on_the_same_model_and_eval_set(self, db, evaluator):
        sibling = _task(QuantFixedRawTask, training_start_point=TrainingStartPoint.DEFAULT)
        task = _task(training_start_point=TrainingStartPoint.DEFAULT)
        db.put(task)
        db.rounds = [SimpleNamespace(round_id="r1", round_number=1)]
        db.round_tasks = {"r1": [str(sibling.task_id), str(task.task_id)]}
        db.references[sibling.task_id] = {("", SKU): _reference(sibling.task_id, hotkey="", model="org/model")}
        prepass._EVAL_SET_FINGERPRINTS[str(task.task_id)] = FINGERPRINT

        await prepass.measure_missing_quant_references(CONFIG)

        evaluator.assert_not_called()
        assert [(ref.task_id, ref.hotkey) for ref in db.added] == [(task.task_id, "")]

    async def test_own_model_task_does_not_copy_from_a_sibling(self, db, evaluator):
        sibling = _task(QuantFixedRawTask)
        task = _task()
        db.put(task)
        db.rounds = [SimpleNamespace(round_id="r1", round_number=1)]
        db.round_tasks = {"r1": [str(sibling.task_id), str(task.task_id)]}
        db.references[sibling.task_id] = {(HOTKEY, SKU): _reference(sibling.task_id)}
        prepass._EVAL_SET_FINGERPRINTS[str(task.task_id)] = FINGERPRINT
        evaluator.return_value = SimpleNamespace(results={REPO: _result()})

        await prepass.measure_missing_quant_references(CONFIG)

        evaluator.assert_called_once()
        # A bare bf16 measurement may run on an A100 (QUANT_REFERENCE_ALLOWED_SKUS); entries never do.
        assert evaluator.await_args.kwargs["reference_only"] is True

    async def test_unprepped_task_is_skipped(self, db):
        db.put(_task(test_data=None))

        assert await prepass.references_to_measure(CONFIG) == []

    async def test_hotkey_without_a_starting_model_is_skipped(self, db):
        task = _task()
        db.put(task, hotkeys=(HOTKEY, OTHER))
        del db.starting[(str(task.task_id), OTHER)]

        assert [m.hotkey for m in await prepass.references_to_measure(CONFIG)] == [HOTKEY]

    async def test_hotkey_with_a_quant_failure_is_skipped(self, db):
        task = _task()
        db.put(task, hotkeys=(HOTKEY, OTHER))
        db.failed[str(task.task_id)] = {OTHER}

        assert [m.hotkey for m in await prepass.references_to_measure(CONFIG)] == [HOTKEY]

    async def test_already_measured_hotkey_is_skipped(self, db):
        task = _task()
        db.put(task, hotkeys=(HOTKEY, OTHER))
        db.references[task.task_id] = {(HOTKEY, SKU): _reference(task.task_id)}

        assert [m.hotkey for m in await prepass.references_to_measure(CONFIG)] == [OTHER]


@pytest.fixture
def evaluator(monkeypatch):
    run = AsyncMock(return_value=SimpleNamespace(results={REPO: _result()}))
    monkeypatch.setattr(prepass, "run_evaluation_basilica_quant", run)
    return run


@pytest.fixture
def local_fingerprint(monkeypatch):
    fingerprint = AsyncMock(return_value=FINGERPRINT)
    monkeypatch.setattr(prepass, "_eval_set_fingerprint", fingerprint)
    return fingerprint


class TestPreviousRoundCopy:
    async def test_same_eval_set_copies_rows_without_measuring(self, db, evaluator, local_fingerprint):
        previous, task = _task(), _task(QuantFixedRawTask)
        db.put(task)
        db.round_before(previous.task_id, task)
        db.references[previous.task_id] = {(HOTKEY, SKU): _reference(previous.task_id)}

        await prepass.measure_missing_quant_references(CONFIG)

        evaluator.assert_not_called()
        [copied] = db.added
        assert copied.task_id == task.task_id and copied.hotkey == HOTKEY and copied.sku == SKU
        assert copied.loss == 2.0 and copied.measured_at is None

    async def test_different_eval_set_is_measured(self, db, evaluator, local_fingerprint):
        previous, task = _task(), _task(QuantFixedRawTask)
        db.put(task)
        db.round_before(previous.task_id, task)
        db.references[previous.task_id] = {(HOTKEY, SKU): _reference(previous.task_id, fingerprint="fp-other")}

        await prepass.measure_missing_quant_references(CONFIG)

        evaluator.assert_called_once()
        assert [r.eval_set_fingerprint for r in db.added] == ["fp-measured"]

    async def test_different_model_is_measured(self, db, evaluator, local_fingerprint):
        previous, task = _task(), _task(QuantFixedRawTask)
        db.put(task)
        db.round_before(previous.task_id, task)
        db.references[previous.task_id] = {(HOTKEY, SKU): _reference(previous.task_id, model="miner/another")}

        await prepass.measure_missing_quant_references(CONFIG)

        evaluator.assert_called_once()

    async def test_previous_rows_of_another_hotkey_do_not_count(self, db, evaluator, local_fingerprint):
        previous, task = _task(), _task(QuantFixedRawTask)
        db.put(task)
        db.round_before(previous.task_id, task)
        db.references[previous.task_id] = {(OTHER, SKU): _reference(previous.task_id, hotkey=OTHER)}

        await prepass.measure_missing_quant_references(CONFIG)

        evaluator.assert_called_once()
        local_fingerprint.assert_not_called()

    async def test_first_round_task_is_measured(self, db, evaluator, local_fingerprint):
        task = _task()
        db.put(task)
        db.rounds = [SimpleNamespace(round_id="r1", round_number=1)]
        db.round_tasks = {"r1": [str(task.task_id)]}

        await prepass.measure_missing_quant_references(CONFIG)

        evaluator.assert_called_once()

    async def test_fingerprint_is_computed_from_the_downloaded_eval_set(self, monkeypatch, tmp_path):
        prepass._EVAL_SET_FINGERPRINTS.clear()
        dataset = tmp_path / "test.jsonl"
        dataset.write_text('{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}\n')
        download = AsyncMock(return_value=str(dataset))
        monkeypatch.setattr(prepass, "download_s3_file", download)
        task = _task()

        first = await prepass._eval_set_fingerprint(task)
        second = await prepass._eval_set_fingerprint(task)

        assert first == second and len(first) == 32
        download.assert_awaited_once()


class TestMeasurement:
    async def test_measurement_stores_a_row_keyed_by_hotkey_with_the_eval_sku(self, db, evaluator):
        task = _task()
        db.put(task)

        await prepass.measure_missing_quant_references(CONFIG)

        [stored] = db.added
        assert (stored.task_id, stored.hotkey, stored.sku) == (task.task_id, HOTKEY, SKU)
        assert (stored.model, stored.loss, stored.eval_gpu) == (REPO, 2.0, GPU)

    async def test_task_common_measurement_stores_one_row_under_the_empty_hotkey(self, db, evaluator):
        task = _task(QuantFixedRawTask, training_start_point=TrainingStartPoint.DEFAULT)
        db.put(task, hotkeys=(HOTKEY, OTHER))
        evaluator.return_value = SimpleNamespace(results={"org/model": _result()})

        await prepass.measure_missing_quant_references(CONFIG)

        [stored] = db.added
        assert (stored.task_id, stored.hotkey, stored.sku) == (task.task_id, "", SKU)
        evaluator.assert_called_once()
        assert evaluator.call_args.kwargs["models"] == ["org/model"]
        assert evaluator.call_args.kwargs["dataset"] == task.test_data

    async def test_task_common_calibration_measurement_runs_on_the_calibration_rows(self, db, evaluator):
        task = _cal_task(training_start_point=TrainingStartPoint.DEFAULT)
        db.put(task)
        db.references[task.task_id] = {("", SKU): _reference(task.task_id, hotkey="", model="org/model")}
        evaluator.return_value = SimpleNamespace(results={"org/model": _cal_result()})

        await prepass.measure_missing_quant_references(CONFIG)

        assert evaluator.call_args.kwargs["dataset"] == CAL_URL
        [stored] = db.calibration_added
        assert (stored.task_id, stored.hotkey) == (task.task_id, "")

    async def test_evaluator_gets_no_database(self, db, evaluator):
        task = _task()
        db.put(task)

        await prepass.measure_missing_quant_references(CONFIG)

        kwargs = evaluator.call_args.kwargs
        assert kwargs["psql_db"] is None
        assert kwargs["models"] == [REPO] and kwargs["original_model"] == REPO
        assert kwargs["reference_models"] == {REPO: REPO} and kwargs["cached_references"] == {REPO: {}}
        assert kwargs["dataset"] == task.test_data and kwargs["file_format"] == FileFormat.JSON
        assert kwargs["quant_spec"] is None and kwargs["decoded_model_urls"] is None

    async def test_a_second_cycle_does_not_remeasure(self, db, evaluator):
        db.put(_task())

        await prepass.measure_missing_quant_references(CONFIG)
        await prepass.measure_missing_quant_references(CONFIG)

        evaluator.assert_called_once()

    async def test_in_progress_pair_is_not_launched_twice(self, db, evaluator):
        task = _task()
        db.put(task)
        prepass._IN_PROGRESS.add((str(task.task_id), HOTKEY))

        await prepass.measure_missing_quant_references(CONFIG)

        evaluator.assert_not_called()

    async def test_in_progress_is_released_after_the_run(self, db, evaluator):
        db.put(_task())

        await prepass.measure_missing_quant_references(CONFIG)

        assert prepass._IN_PROGRESS == set()


class TestFailures:
    @pytest.fixture
    def alert(self, monkeypatch):
        notify = AsyncMock()
        monkeypatch.setattr(prepass, "notify_evaluation_exception", notify)
        return notify

    @pytest.mark.parametrize(
        "outcome",
        [
            Exception("pod died"),
            _result(with_reference=False),
            _result(quant_failure="did not load"),
        ],
    )
    async def test_failure_stores_nothing_and_retries_next_cycle(self, db, evaluator, alert, outcome):
        task = _task()
        db.put(task)
        evaluator.return_value = SimpleNamespace(results={REPO: outcome})

        await prepass.measure_missing_quant_references(CONFIG)
        await prepass.measure_missing_quant_references(CONFIG)

        assert db.added == []
        assert evaluator.call_count == 2
        assert prepass._CONSECUTIVE_FAILURES[(str(task.task_id), HOTKEY)] == 2

    async def test_evaluator_exception_counts_as_a_failure(self, db, evaluator, alert):
        db.put(_task())
        evaluator.side_effect = RuntimeError("basilica down")

        await prepass.measure_missing_quant_references(CONFIG)

        assert db.added == [] and prepass._IN_PROGRESS == set()

    async def test_waiting_for_capacity_is_not_a_failure(self, db, evaluator, alert):
        task = _task()
        db.put(task)
        evaluator.side_effect = EvaluationCapacityUnavailable("no H100 free")

        await prepass.measure_missing_quant_references(CONFIG)

        assert (str(task.task_id), HOTKEY) not in prepass._CONSECUTIVE_FAILURES
        assert db.added == [] and prepass._IN_PROGRESS == set()

    async def test_an_empty_message_error_is_recorded_with_its_type(self, db, evaluator, alert):
        db.put(_task())
        evaluator.return_value = SimpleNamespace(results={REPO: httpx.ReadTimeout("")})

        for _ in range(t_cst.QUANT_REFERENCE_PREPASS_FAILURES_BEFORE_ALERT):
            await prepass.measure_missing_quant_references(CONFIG)

        assert "ReadTimeout (no message)" in alert.await_args.kwargs["error"]

    async def test_alert_fires_once_at_the_threshold(self, db, evaluator, alert):
        db.put(_task())
        evaluator.return_value = SimpleNamespace(results={REPO: Exception("pod died")})

        for _ in range(t_cst.QUANT_REFERENCE_PREPASS_FAILURES_BEFORE_ALERT + 2):
            await prepass.measure_missing_quant_references(CONFIG)

        alert.assert_awaited_once()
        assert alert.await_args.kwargs["hotkeys"] == [HOTKEY]

    async def test_success_resets_the_failure_count(self, db, evaluator, alert):
        task = _task()
        db.put(task)
        evaluator.return_value = SimpleNamespace(results={REPO: Exception("pod died")})
        await prepass.measure_missing_quant_references(CONFIG)
        evaluator.return_value = SimpleNamespace(results={REPO: _result()})

        await prepass.measure_missing_quant_references(CONFIG)

        assert (str(task.task_id), HOTKEY) not in prepass._CONSECUTIVE_FAILURES
        assert len(db.added) == 1


CAL_URL = "https://s3.example/calibration.jsonl"
CAL_FINGERPRINT = "fp-calibration"


def _cal_task(cls=QuantFreeformRawTask, **overrides):
    return _task(cls, calibration_data=CAL_URL, calibration_fingerprint=CAL_FINGERPRINT, calibration_n_examples=1, **overrides)


def _cal_record(task_id, hotkey=HOTKEY, fingerprint=CAL_FINGERPRINT, model=REPO) -> QuantCalibrationReferenceRecord:
    return QuantCalibrationReferenceRecord(
        task_id=task_id,
        hotkey=hotkey,
        model=model,
        loss=1.5,
        per_example_losses=[1.5],
        eval_set_fingerprint=fingerprint,
        eval_gpu=GPU,
        measured_at=datetime(2026, 10, 20),
    )


def _cal_result(fingerprint=CAL_FINGERPRINT) -> EvaluationResultText:
    reference = QuantReferenceScore(
        model=REPO, eval_loss=1.5, per_example_losses=[1.5], eval_set_fingerprint=fingerprint, eval_gpu=GPU
    )
    return EvaluationResultText(is_finetune=True, eval_loss=1.5, quant_reference=reference)


class TestCalibrationReference:
    async def test_freeform_with_calibration_data_needs_a_second_measurement_after_the_test_one(self, db):
        task = _cal_task()
        db.put(task)
        db.references[task.task_id] = {(HOTKEY, SKU): _reference(task.task_id)}

        [missing] = await prepass.references_to_measure(CONFIG)

        assert missing.calibration and missing.dataset == CAL_URL and missing.key == (str(task.task_id), HOTKEY, "calibration")

    async def test_the_test_reference_comes_first(self, db):
        task = _cal_task()
        db.put(task)

        [missing] = await prepass.references_to_measure(CONFIG)

        assert not missing.calibration and missing.dataset == task.test_data

    @pytest.mark.parametrize(
        "make",
        [
            lambda: _task(),  # freeform without calibration_data
        ],
    )
    async def test_no_calibration_measurement_for_a_legacy_task(self, db, make):
        task = make()
        db.put(task)
        db.references[task.task_id] = {(HOTKEY, SKU): _reference(task.task_id)}

        assert await prepass.references_to_measure(CONFIG) == []

    async def test_a_fixed_task_with_calibration_data_is_measured_like_a_freeform_one(self, db):
        task = _cal_task(QuantFixedRawTask)
        db.put(task)
        db.references[task.task_id] = {(HOTKEY, SKU): _reference(task.task_id)}

        [missing] = await prepass.references_to_measure(CONFIG)

        assert missing.calibration and missing.dataset == CAL_URL

    async def test_a_stored_calibration_row_is_not_remeasured(self, db):
        task = _cal_task()
        db.put(task)
        db.references[task.task_id] = {(HOTKEY, SKU): _reference(task.task_id)}
        db.calibration[task.task_id] = {HOTKEY: _cal_record(task.task_id)}

        assert await prepass.references_to_measure(CONFIG) == []

    async def test_measures_the_calibration_dataset_and_stores_a_record(self, db, evaluator):
        task = _cal_task()
        db.put(task)
        db.references[task.task_id] = {(HOTKEY, SKU): _reference(task.task_id)}
        evaluator.return_value = SimpleNamespace(results={REPO: _cal_result()})

        await prepass.measure_missing_quant_references(CONFIG)

        assert evaluator.call_args.kwargs["dataset"] == CAL_URL
        assert evaluator.call_args.kwargs["psql_db"] is None
        [record] = db.calibration_added
        assert (record.task_id, record.hotkey, record.model, record.loss, record.sku) == (task.task_id, HOTKEY, REPO, 1.5, SKU)
        assert db.added == []  # no SKU reference row from the calibration pass

    async def test_a_mismatched_fingerprint_is_a_failure_not_a_row(self, db, evaluator):
        task = _cal_task()
        db.put(task)
        db.references[task.task_id] = {(HOTKEY, SKU): _reference(task.task_id)}
        evaluator.return_value = SimpleNamespace(results={REPO: _cal_result(fingerprint="fp-other")})

        await prepass.measure_missing_quant_references(CONFIG)

        assert db.calibration_added == []
        assert prepass._CONSECUTIVE_FAILURES[(str(task.task_id), HOTKEY, "calibration")] == 1

    async def test_gives_up_after_the_max_failures_and_stops_measuring(self, db, evaluator, monkeypatch):
        monkeypatch.setattr(prepass, "notify_evaluation_exception", AsyncMock())
        task = _cal_task()
        db.put(task)
        db.references[task.task_id] = {(HOTKEY, SKU): _reference(task.task_id)}
        evaluator.return_value = SimpleNamespace(results={REPO: Exception("pod died")})

        assert not prepass.calibration_reference_gave_up(task.task_id, HOTKEY)
        for _ in range(t_cst.QUANT_CALIBRATION_REFERENCE_MAX_FAILURES + 2):
            await prepass.measure_missing_quant_references(CONFIG)

        assert prepass.calibration_reference_gave_up(task.task_id, HOTKEY)
        assert evaluator.call_count == t_cst.QUANT_CALIBRATION_REFERENCE_MAX_FAILURES
        assert await prepass.references_to_measure(CONFIG) == []

    async def test_calibration_failures_do_not_count_against_the_test_reference(self, db, evaluator, monkeypatch):
        monkeypatch.setattr(prepass, "notify_evaluation_exception", AsyncMock())
        task = _cal_task()
        db.put(task)
        db.references[task.task_id] = {(HOTKEY, SKU): _reference(task.task_id)}
        evaluator.return_value = SimpleNamespace(results={REPO: Exception("pod died")})

        await prepass.measure_missing_quant_references(CONFIG)

        assert (str(task.task_id), HOTKEY) not in prepass._CONSECUTIVE_FAILURES

    async def test_previous_round_row_with_the_same_model_and_fingerprint_is_copied(self, db, evaluator):
        previous, task = _cal_task(), _cal_task()
        db.put(task)
        db.round_before(previous.task_id, task)
        db.references[task.task_id] = {(HOTKEY, SKU): _reference(task.task_id)}
        db.calibration[previous.task_id] = {HOTKEY: _cal_record(previous.task_id)}

        await prepass.measure_missing_quant_references(CONFIG)

        evaluator.assert_not_called()
        [copied] = db.calibration_added
        assert copied.task_id == task.task_id and copied.measured_at is None and copied.loss == 1.5

    @pytest.mark.parametrize("override", [{"fingerprint": "fp-other"}, {"model": "miner/another"}])
    async def test_previous_round_row_on_another_set_or_model_is_not_copied(self, db, evaluator, override):
        previous, task = _cal_task(), _cal_task()
        db.put(task)
        db.round_before(previous.task_id, task)
        db.references[task.task_id] = {(HOTKEY, SKU): _reference(task.task_id)}
        db.calibration[previous.task_id] = {HOTKEY: _cal_record(previous.task_id, **override)}
        evaluator.return_value = SimpleNamespace(results={REPO: _cal_result()})

        await prepass.measure_missing_quant_references(CONFIG)

        evaluator.assert_called_once()
