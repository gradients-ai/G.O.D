"""Quant trainer request: TrainRequestQuant carries the task's input model, spec and dataset.

Freeform entrants also carry the decode request for their registration decoder snapshot. The chained
tasks' shared input is covered in test_chained_training_requests.py.
"""

import logging
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.constants.environments import TrainingStartPoint
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainRequestQuant
from core.models.payload_models import TrainRequestText
from core.models.quant_artifact_models import QuantDecodeRequest
from core.models.quant_models import QuantCalibrationReference
from core.models.quant_models import QuantCalibrationRequest
from core.models.quant_models import QuantEvalGpu
from core.models.quant_models import QuantFormat
from core.models.quant_models import QuantTarget
from core.models.task_models import TaskType
from trainer.runtime import get_task_type
from validator.evaluation.quant_artifacts import decoded_model_prefix
from validator.scoring.models import QuantCalibrationReferenceRecord
from validator.scoring.models import QuantSkuReference
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tournament import constants as t_cst
from validator.tournament import orchestrator
from validator.tournament import quant_reference_prepass as prepass
from validator.tournament.models import TournamentParticipant


SPEC = QuantFormat.W4A8_FP8_G128.spec()
EXPECTED_REPO = "miner-org/expected-repo"
CONTINUATION_REPO = "miner-org/round-1-winner"
AUGMENTED_MODEL = "validator-org/augmented-model"
HOTKEY = "5Hotkey"
TOURNAMENT_ID = "tourn_1"
PRESIGNED_URL = "https://minio.example/quant-decoders/abc/sha.tar.xz?X-Amz-Signature=secret"
BUCKET = "test-bucket"
SNAPSHOT_FIELDS = {"decoder_object_name": "quant-decoders/abc/sha.tar.xz", "decoder_xz_bytes": 4096, "decoder_sha256": "sha"}


def _base_fields() -> dict:
    return {
        "is_organic": False,
        "task_id": uuid.uuid4(),
        "status": "looking_for_nodes",
        "model_id": "org/model",
        "ds": "org/dataset",
        "account_id": uuid.uuid4(),
        "hours_to_complete": 2.0,
        "training_data": "s3://bucket/train.json",
        "created_at": datetime(2026, 10, 19),
    }


def _dataset_type() -> InstructTextDatasetType:
    return InstructTextDatasetType(field_instruction="instruction", field_output="output")


def _fixed_task(**overrides) -> QuantFixedRawTask:
    return QuantFixedRawTask(**{**_base_fields(), "dataset_type": _dataset_type(), "quant_spec": SPEC, **overrides})


def _freeform_task(**overrides) -> QuantFreeformRawTask:
    return QuantFreeformRawTask(**{**_base_fields(), "dataset_type": _dataset_type(), **overrides})


def _instruct_task() -> InstructTextRawTask:
    return InstructTextRawTask(**_base_fields(), field_instruction="instruction", field_output="output")


class TestQuantTrainingData:
    def test_fixed_task_carries_spec_and_dataset(self):
        task = _fixed_task()
        request = orchestrator._quant_training_data(task, EXPECTED_REPO, None, model=task.model_id, quant_target=None)

        assert isinstance(request, TrainRequestQuant)
        assert request.task_type == TaskType.QUANTFIXEDTASK
        assert request.quant_spec == task.quant_spec
        assert request.dataset_type is task.dataset_type
        assert request.file_format == FileFormat.S3
        assert request.dataset == task.training_data
        assert request.model == task.model_id
        assert request.expected_repo_name == EXPECTED_REPO

    def test_freeform_task_has_no_spec(self):
        task = _freeform_task()
        request = orchestrator._quant_training_data(task, EXPECTED_REPO, None, model=task.model_id, quant_target=None)

        assert request.task_type == TaskType.QUANTFREEFORMTASK
        assert request.quant_spec is None

    def test_model_is_the_given_starting_model(self):
        request = orchestrator._quant_training_data(
            _fixed_task(), EXPECTED_REPO, None, model=AUGMENTED_MODEL, quant_target=None
        )

        assert request.model == AUGMENTED_MODEL


def _participant(**overrides) -> TournamentParticipant:
    return TournamentParticipant(tournament_id=TOURNAMENT_ID, hotkey=HOTKEY, **overrides)


@pytest.fixture
def db_lookups(monkeypatch):
    lookups = SimpleNamespace(
        starting_model=AsyncMock(return_value=CONTINUATION_REPO),
        participant=AsyncMock(return_value=_participant(**SNAPSHOT_FIELDS)),
        presigned_url=AsyncMock(return_value=PRESIGNED_URL),
    )
    monkeypatch.setattr(orchestrator.task_sql, "get_expected_repo_name", AsyncMock(return_value=EXPECTED_REPO))
    monkeypatch.setattr(orchestrator.task_sql, "get_starting_model_repo", lookups.starting_model)
    monkeypatch.setattr(orchestrator.task_sql, "get_miner_baseline_stats", AsyncMock(return_value=None))
    monkeypatch.setattr(orchestrator, "get_quant_sku_references", AsyncMock(return_value={}))
    monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=TOURNAMENT_ID))
    monkeypatch.setattr(orchestrator.tournament_sql, "get_tournament_participant", lookups.participant)
    monkeypatch.setattr(orchestrator.async_minio_client, "get_presigned_url", lookups.presigned_url)
    monkeypatch.setattr(orchestrator.service_cst, "BUCKET_NAME", BUCKET)
    return lookups


async def _request(task, hotkey: str = HOTKEY):
    return await orchestrator._create_training_request(
        task,
        hotkey=hotkey,
        available_gpu_ids=[0, 1],
        training_repo="https://github.com/miner/repo",
        training_commit_hash="abc123",
        github_token=None,
        requested_datasets=None,
        config=SimpleNamespace(psql_db=None, discord_url=None),
    )


class TestCreateTrainingRequest:
    async def test_text_task_still_uses_starting_model(self, db_lookups):
        request = await _request(_instruct_task())

        assert isinstance(request.training_data, TrainRequestText)
        assert request.training_data.model == CONTINUATION_REPO

    async def test_quant_task_uses_the_starting_model(self, db_lookups):
        request = await _request(_fixed_task(augmented_model_id=AUGMENTED_MODEL))

        assert request.training_data.model == CONTINUATION_REPO

    async def test_quant_task_without_a_starting_model_uses_augmented_then_base(self, db_lookups):
        db_lookups.starting_model.return_value = None

        assert (await _request(_fixed_task(augmented_model_id=AUGMENTED_MODEL))).training_data.model == AUGMENTED_MODEL
        assert (await _request(_freeform_task())).training_data.model == "org/model"

    @pytest.mark.parametrize("make_task", [_fixed_task, _freeform_task])
    async def test_quant_target_is_none_without_a_stored_reference(self, db_lookups, make_task):
        assert (await _request(make_task())).training_data.quant_target is None

    async def test_quant_target_is_published_from_the_stored_reference(self, db_lookups, monkeypatch):
        task = _freeform_task(training_start_point=TrainingStartPoint.OWN_R1_MODEL, bar_cap=0.05)
        reference = QuantSkuReference(
            task_id=task.task_id,
            hotkey=HOTKEY,
            sku="h100-sxm",
            model="org/own-bf16",
            loss=2.0,
            per_example_losses=[2.0],
            eval_set_fingerprint="fp",
            eval_gpu=QuantEvalGpu(name="NVIDIA H100", sm_count=132, sku="h100-sxm"),
        )
        monkeypatch.setattr(
            orchestrator, "get_quant_sku_references", AsyncMock(return_value={reference.key: reference})
        )

        target = (await _request(task)).training_data.quant_target

        assert target == orchestrator.quant_target_from_reference(task, reference)

    @pytest.mark.parametrize("make_task", [_fixed_task, _freeform_task])
    async def test_proxy_request_resolves_quant_task_type(self, db_lookups, make_task):
        task = make_task()
        request = await _request(task)

        assert isinstance(request.training_data, TrainRequestQuant)
        assert get_task_type(request) == task.task_type

    async def test_proxy_request_round_trips_as_quant(self, db_lookups):
        request = await _request(_fixed_task())
        reparsed = type(request).model_validate(request.model_dump())

        assert isinstance(reparsed.training_data, TrainRequestQuant)
        assert get_task_type(reparsed) == TaskType.QUANTFIXEDTASK


class TestQuantTargetFromReference:
    def _reference(self, loss: float = 2.0) -> QuantSkuReference:
        return QuantSkuReference(
            task_id=uuid.uuid4(),
            sku="h100-sxm",
            model="org/own-bf16",
            loss=loss,
            per_example_losses=[loss],
            eval_set_fingerprint="fp",
            eval_gpu=QuantEvalGpu(name="NVIDIA H100", sm_count=132, sku="h100-sxm"),
        )

    def test_fixed_target_carries_only_the_reference(self):
        target = orchestrator.quant_target_from_reference(_fixed_task(), self._reference())

        assert target == QuantTarget(reference_model="org/own-bf16", reference_loss=2.0, reference_sku="h100-sxm")
        assert target.bar_cap is None and target.bar is None
        assert target.knee is None and target.knee_loss is None
        assert target.penalty_curvature is None and target.cap is None and target.cap_loss is None

    def test_freeform_target_publishes_the_penalty(self):
        task = _freeform_task(penalty_knee=0.04, penalty_curvature=2500.0, bar_cap=0.08)
        target = orchestrator.quant_target_from_reference(task, self._reference(2.0))

        assert (target.bar_cap, target.knee, target.penalty_curvature, target.cap) == (0.04, 1.04, 2500.0, 1.08)
        assert target.bar == pytest.approx(2.08) and target.knee_loss == pytest.approx(2.08)
        assert target.cap_loss == pytest.approx(2.16)

    def test_legacy_freeform_target_is_the_cliff(self):
        task = _freeform_task(penalty_knee=None, penalty_curvature=None, bar_cap=0.05)
        target = orchestrator.quant_target_from_reference(task, self._reference(2.0))

        assert (target.bar_cap, target.knee, target.penalty_curvature, target.cap) == (0.05, 1.05, 0.0, 1.05)
        assert target.bar == pytest.approx(2.1) and target.knee_loss == pytest.approx(2.1)
        assert target.cap_loss == pytest.approx(2.1)

    @pytest.mark.parametrize("make_task", [_fixed_task, _freeform_task])
    def test_target_fits_the_train_request(self, make_task):
        task = make_task()
        target = orchestrator.quant_target_from_reference(task, self._reference())

        request = orchestrator._quant_training_data(task, EXPECTED_REPO, None, model="org/own-bf16", quant_target=target)

        assert request.quant_target == target


class TestFreeformDecodeRequest:
    async def test_freeform_participant_with_snapshot_carries_decode(self, db_lookups):
        task = _freeform_task()
        request = await _request(task)

        assert request.training_data.decode == QuantDecodeRequest(
            decoder_snapshot_url=PRESIGNED_URL,
            decoder_sha256=SNAPSHOT_FIELDS["decoder_sha256"],
            bucket=BUCKET,
            object_prefix=decoded_model_prefix(task.task_id, HOTKEY),
        )
        db_lookups.presigned_url.assert_awaited_once_with(BUCKET, SNAPSHOT_FIELDS["decoder_object_name"])

    async def test_fixed_task_has_no_decode(self, db_lookups):
        request = await _request(_fixed_task())

        assert request.training_data.decode is None
        db_lookups.participant.assert_not_awaited()

    async def test_participant_without_snapshot_has_no_decode(self, db_lookups):
        db_lookups.participant.return_value = _participant()
        request = await _request(_freeform_task())

        assert request.training_data.decode is None
        db_lookups.presigned_url.assert_not_awaited()

    async def test_organic_task_has_no_decode(self, db_lookups, monkeypatch):
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=None))
        request = await _request(_freeform_task())

        assert request.training_data.decode is None
        db_lookups.participant.assert_not_awaited()


CAL_URL = "https://minio.example/bucket/abc_quant_calibration_set.jsonl?X-Amz-Signature=old"
CAL_FRESH_URL = "https://minio.example/bucket/abc_quant_calibration_set.jsonl?X-Amz-Signature=fresh"
CAL_FINGERPRINT = "fp-cal"


def _calibration_task(cls=_freeform_task, **overrides):
    fields = dict(
        calibration_data=CAL_URL,
        calibration_fingerprint=CAL_FINGERPRINT,
        calibration_n_examples=3,
        training_start_point=TrainingStartPoint.OWN_R1_MODEL,
    )
    return cls(**{**fields, **overrides})


def _calibration_record(task, hotkey=HOTKEY, fingerprint=CAL_FINGERPRINT, losses=(1.0, float("nan"), 3.0)):
    return QuantCalibrationReferenceRecord(
        task_id=task.task_id,
        hotkey=hotkey,
        model="org/own-bf16",
        loss=2.0,
        per_example_losses=list(losses),
        eval_set_fingerprint=fingerprint,
        eval_gpu=QuantEvalGpu(name="NVIDIA H100", sm_count=132, sku="h100-sxm"),
    )


@pytest.fixture
def calibration_store(monkeypatch, db_lookups):
    store: dict[uuid.UUID, dict[str, QuantCalibrationReferenceRecord]] = {}

    async def get_records(task_id, psql_db):
        return dict(store.get(task_id, {}))

    new_url = AsyncMock(return_value=CAL_FRESH_URL)
    monkeypatch.setattr(orchestrator, "get_quant_calibration_references", get_records)
    monkeypatch.setattr(orchestrator.async_minio_client, "get_new_presigned_url", new_url)
    store_namespace = SimpleNamespace(records=store, new_url=new_url)
    return store_namespace


class TestCalibrationRequest:
    async def test_freeform_request_carries_the_rows_and_the_hotkeys_reference(self, calibration_store):
        task = _calibration_task()
        calibration_store.records[task.task_id] = {HOTKEY: _calibration_record(task)}

        calibration = (await _request(task)).training_data.calibration

        assert calibration == QuantCalibrationRequest(
            examples_url=CAL_FRESH_URL,
            eval_set_fingerprint=CAL_FINGERPRINT,
            n_examples=3,
            reference=QuantCalibrationReference(
                model="org/own-bf16", loss=2.0, per_example_losses=[1.0, None, 3.0], sku="h100-sxm"
            ),
        )
        calibration_store.new_url.assert_awaited_once_with(CAL_URL)

    async def test_without_a_measured_reference_the_rows_are_published_with_reference_none(self, calibration_store):
        task = _calibration_task()

        calibration = (await _request(task)).training_data.calibration

        assert calibration.reference is None and calibration.n_examples == 3

    async def test_another_hotkeys_or_another_sets_reference_is_not_published(self, calibration_store):
        task = _calibration_task()
        calibration_store.records[task.task_id] = {"5Other": _calibration_record(task, hotkey="5Other")}
        assert (await _request(task)).training_data.calibration.reference is None

        calibration_store.records[task.task_id] = {HOTKEY: _calibration_record(task, fingerprint="fp-other")}
        assert (await _request(task)).training_data.calibration.reference is None

    async def test_a_presign_failure_publishes_no_calibration_and_the_request_still_builds(self, calibration_store):
        calibration_store.new_url.return_value = None

        request = await _request(_calibration_task())

        assert request.training_data.calibration is None

    async def test_legacy_freeform_request_carries_none(self, calibration_store):
        assert (await _request(_freeform_task())).training_data.calibration is None
        calibration_store.new_url.assert_not_called()

    async def test_legacy_fixed_request_carries_none(self, calibration_store):
        assert (await _request(_fixed_task())).training_data.calibration is None
        calibration_store.new_url.assert_not_called()

    async def test_fixed_request_carries_the_rows_and_the_hotkeys_reference(self, calibration_store):
        task = _calibration_task(_fixed_task)
        calibration_store.records[task.task_id] = {HOTKEY: _calibration_record(task)}

        request = await _request(task)

        assert request.training_data.task_type == TaskType.QUANTFIXEDTASK
        assert request.training_data.calibration == QuantCalibrationRequest(
            examples_url=CAL_FRESH_URL,
            eval_set_fingerprint=CAL_FINGERPRINT,
            n_examples=3,
            reference=QuantCalibrationReference(
                model="org/own-bf16", loss=2.0, per_example_losses=[1.0, None, 3.0], sku="h100-sxm"
            ),
        )

    @pytest.mark.parametrize("make_task", [_fixed_task, _freeform_task])
    async def test_task_common_request_reads_the_empty_hotkey_row(self, calibration_store, make_task):
        task = _calibration_task(make_task, training_start_point=TrainingStartPoint.DEFAULT)
        calibration_store.records[task.task_id] = {"": _calibration_record(task, hotkey="")}

        calibration = (await _request(task)).training_data.calibration

        assert calibration.reference is not None and calibration.reference.model == "org/own-bf16"

    @pytest.mark.parametrize("make_task", [_fixed_task, _freeform_task])
    async def test_a_boss_compression_task_hands_each_player_its_own_calibration_reference(self, calibration_store, make_task):
        """Its freeform / fixed compression tasks: own-upstream rows, the calibration set inherited from the upstream instruct
        task, each player's own bf16 reference."""
        task = _calibration_task(make_task, training_start_point=TrainingStartPoint.OWN_UPSTREAM_MODEL)
        challenger = "5Challenger"
        calibration_store.records[task.task_id] = {
            HOTKEY: _calibration_record(task),
            challenger: _calibration_record(task, hotkey=challenger).model_copy(update={"model": "org/challenger-bf16"}),
        }

        boss_side = (await _request(task, HOTKEY)).training_data.calibration
        challenger_side = (await _request(task, challenger)).training_data.calibration

        assert boss_side.examples_url == challenger_side.examples_url == CAL_FRESH_URL
        assert boss_side.eval_set_fingerprint == challenger_side.eval_set_fingerprint == CAL_FINGERPRINT
        assert (boss_side.reference.model, challenger_side.reference.model) == ("org/own-bf16", "org/challenger-bf16")

    def test_a_fixed_request_may_carry_a_calibration(self):
        calibration = QuantCalibrationRequest(examples_url=CAL_URL, eval_set_fingerprint="fp", n_examples=1, reference=None)

        request = TrainRequestQuant(
            model="m",
            task_id="t",
            hours_to_complete=1.0,
            dataset="d",
            dataset_type=_dataset_type(),
            file_format=FileFormat.S3,
            task_type=TaskType.QUANTFIXEDTASK,
            quant_spec=SPEC,
            calibration=calibration,
        )

        assert request.calibration == calibration

    async def test_a_legacy_request_json_only_adds_a_null_calibration_key(self, db_lookups):
        request = await _request(_freeform_task())
        dumped = request.training_data.model_dump(mode="json")

        assert dumped.pop("calibration") is None
        assert set(dumped) == {
            "model", "task_id", "hours_to_complete", "expected_repo_name", "baseline_stats", "dataset", "dataset_type",
            "file_format", "task_type", "quant_spec", "decode", "quant_target",
        }  # fmt: skip


class TestCalibrationReadiness:
    @pytest.fixture
    def sku_rows(self, monkeypatch):
        async def get_references(task_id, psql_db):
            return {(HOTKEY, "h100-sxm"): object()}

        monkeypatch.setattr(orchestrator, "get_quant_sku_references", get_references)

    async def test_waits_for_the_calibration_row(self, calibration_store, sku_rows):
        task = _calibration_task()

        assert not await orchestrator._reference_ready(task, HOTKEY, SimpleNamespace(psql_db=None))

        calibration_store.records[task.task_id] = {HOTKEY: _calibration_record(task)}
        assert await orchestrator._reference_ready(task, HOTKEY, SimpleNamespace(psql_db=None))

    async def test_stops_waiting_once_the_prepass_gave_up(self, calibration_store, sku_rows, monkeypatch):
        task = _calibration_task()
        monkeypatch.setattr(orchestrator, "calibration_reference_gave_up", lambda task_id, hotkey: True)

        assert await orchestrator._reference_ready(task, HOTKEY, SimpleNamespace(psql_db=None))

    async def test_still_needs_the_test_reference_first(self, calibration_store, monkeypatch):
        monkeypatch.setattr(orchestrator, "get_quant_sku_references", AsyncMock(return_value={}))
        monkeypatch.setattr(orchestrator, "calibration_reference_gave_up", lambda task_id, hotkey: True)

        assert not await orchestrator._reference_ready(_calibration_task(), HOTKEY, SimpleNamespace(psql_db=None))

    async def test_legacy_freeform_never_reads_calibration_rows(self, sku_rows, monkeypatch):
        read = AsyncMock()
        monkeypatch.setattr(orchestrator, "get_quant_calibration_references", read)
        task = _freeform_task(training_start_point=TrainingStartPoint.OWN_R1_MODEL)

        assert await orchestrator._reference_ready(task, HOTKEY, SimpleNamespace(psql_db=None))
        read.assert_not_called()

    async def test_task_common_task_waits_for_the_empty_hotkey_rows(self, calibration_store, monkeypatch):
        task = _calibration_task(training_start_point=TrainingStartPoint.DEFAULT)
        config = SimpleNamespace(psql_db=None)
        rows = {}

        async def get_references(task_id, psql_db):
            return rows

        monkeypatch.setattr(orchestrator, "get_quant_sku_references", get_references)
        assert not await orchestrator._reference_ready(task, HOTKEY, config)  # no common test row yet

        rows[("", "h100-sxm")] = object()
        assert not await orchestrator._reference_ready(task, HOTKEY, config)  # test row, no common calibration row

        calibration_store.records[task.task_id] = {HOTKEY: _calibration_record(task)}  # a miner's row is not the common one
        assert not await orchestrator._reference_ready(task, HOTKEY, config)

        calibration_store.records[task.task_id] = {"": _calibration_record(task, hotkey="")}
        assert await orchestrator._reference_ready(task, HOTKEY, config)

    async def test_task_common_calibration_give_up_is_keyed_by_the_empty_hotkey(self, calibration_store, monkeypatch):
        task = _calibration_task(training_start_point=TrainingStartPoint.DEFAULT)
        monkeypatch.setattr(orchestrator, "get_quant_sku_references", AsyncMock(return_value={("", "h100-sxm"): object()}))
        monkeypatch.setattr(orchestrator, "calibration_reference_gave_up", lambda task_id, hotkey: hotkey == "")

        assert await orchestrator._reference_ready(task, HOTKEY, SimpleNamespace(psql_db=None))


class TestTaskCommonReferenceGiveUp:
    CONFIG = SimpleNamespace(psql_db=None)

    @pytest.fixture(autouse=True)
    def no_rows(self, monkeypatch):
        prepass._CONSECUTIVE_FAILURES.clear()
        monkeypatch.setattr(orchestrator, "get_quant_sku_references", AsyncMock(return_value={}))
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=TOURNAMENT_ID))

    async def test_organic_task_never_waits(self, monkeypatch):
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=None))
        task = _fixed_task(training_start_point=TrainingStartPoint.DEFAULT)

        assert await orchestrator._reference_ready(task, HOTKEY, self.CONFIG)

    async def test_task_common_waits_below_the_failure_limit(self):
        task = _fixed_task(training_start_point=TrainingStartPoint.DEFAULT)
        prepass._CONSECUTIVE_FAILURES[(str(task.task_id), "")] = t_cst.QUANT_TASK_COMMON_REFERENCE_MAX_FAILURES - 1

        assert not await orchestrator._reference_ready(task, HOTKEY, self.CONFIG)

    async def test_task_common_gives_up_at_the_limit_and_trains_with_no_target(self):
        task = _fixed_task(training_start_point=TrainingStartPoint.DEFAULT)
        prepass._CONSECUTIVE_FAILURES[(str(task.task_id), "")] = t_cst.QUANT_TASK_COMMON_REFERENCE_MAX_FAILURES

        assert await orchestrator._reference_ready(task, HOTKEY, self.CONFIG)
        assert await orchestrator._quant_target_for(task, HOTKEY, self.CONFIG) is None

    async def test_own_model_never_gives_up(self):
        task = _fixed_task(training_start_point=TrainingStartPoint.OWN_R1_MODEL)
        prepass._CONSECUTIVE_FAILURES[(str(task.task_id), HOTKEY)] = 1000
        prepass._CONSECUTIVE_FAILURES[(str(task.task_id), "")] = 1000

        assert not await orchestrator._reference_ready(task, HOTKEY, self.CONFIG)


@pytest.fixture
def missing_calibration_alert(monkeypatch, calibration_store):
    notify = AsyncMock()
    monkeypatch.setattr(orchestrator, "notify_evaluation_exception", notify)
    monkeypatch.setattr(orchestrator, "_QUANT_CALIBRATION_MISSING_ALERTED", set())
    return notify


class TestMissingCalibrationAlert:
    """A quant row dispatched without calibration rows is never silent: an ERROR per row, one alert per task."""

    @pytest.mark.parametrize("make_task", [_fixed_task, _freeform_task])
    async def test_a_task_without_a_calibration_set_alerts_once_and_still_builds(
        self, missing_calibration_alert, make_task, caplog
    ):
        task = make_task()

        with caplog.at_level(logging.ERROR):
            first = await _request(task, HOTKEY)
            second = await _request(task, "5Challenger")

        assert first.training_data.calibration is None and second.training_data.calibration is None
        assert caplog.text.count("WITHOUT calibration rows") == 2
        missing_calibration_alert.assert_awaited_once()
        kwargs = missing_calibration_alert.await_args.kwargs
        assert kwargs["task_id"] == str(task.task_id) and kwargs["hotkeys"] == [HOTKEY]
        assert "no calibration set" in kwargs["error"]

    async def test_an_unpublishable_calibration_set_alerts(self, missing_calibration_alert, calibration_store):
        calibration_store.new_url.return_value = None

        await _request(_calibration_task())

        missing_calibration_alert.assert_awaited_once()
        assert "could not be published" in missing_calibration_alert.await_args.kwargs["error"]

    async def test_a_published_calibration_set_does_not_alert(self, missing_calibration_alert):
        assert (await _request(_calibration_task())).training_data.calibration is not None

        missing_calibration_alert.assert_not_awaited()

    async def test_a_text_task_never_alerts(self, missing_calibration_alert):
        await _request(_instruct_task())

        missing_calibration_alert.assert_not_awaited()
