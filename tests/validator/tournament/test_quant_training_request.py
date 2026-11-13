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


CAL_URL = "https://minio.example/bucket/abc_quant_calibration_set.jsonl?X-Amz-Signature=old"
CAL_FRESH_URL = "https://minio.example/bucket/abc_quant_calibration_set.jsonl?X-Amz-Signature=fresh"
CAL_FINGERPRINT = "fp-cal"
CALIBRATION = QuantCalibrationRequest(
    examples_url=CAL_FRESH_URL,
    eval_set_fingerprint=CAL_FINGERPRINT,
    n_examples=3,
    reference=QuantCalibrationReference(model="org/own-bf16", loss=2.0, per_example_losses=[1.0, None, 3.0], sku="h100-sxm"),
)
FIXED_TARGET = QuantTarget(reference_model="org/own-bf16", reference_loss=2.0, reference_sku="h100-sxm")
UNCALIBRATED = dict(calibration_data=None, calibration_fingerprint=None, calibration_n_examples=None)


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
        "calibration_data": CAL_URL,
        "calibration_fingerprint": CAL_FINGERPRINT,
        "calibration_n_examples": 3,
    }


def _dataset_type() -> InstructTextDatasetType:
    return InstructTextDatasetType(field_instruction="instruction", field_output="output")


def _fixed_task(**overrides) -> QuantFixedRawTask:
    return QuantFixedRawTask(**{**_base_fields(), "dataset_type": _dataset_type(), "quant_spec": SPEC, **overrides})


def _freeform_task(**overrides) -> QuantFreeformRawTask:
    return QuantFreeformRawTask(**{**_base_fields(), "dataset_type": _dataset_type(), **overrides})


def _target_of(task) -> QuantTarget:
    reference = _sku_reference(task, "")
    return orchestrator.quant_target_from_reference(task, reference)


def _sku_reference(task, hotkey: str, sku="h100-sxm", loss=2.0) -> QuantSkuReference:
    return QuantSkuReference(
        task_id=task.task_id,
        hotkey=hotkey,
        sku=sku,
        model="org/own-bf16",
        loss=loss,
        per_example_losses=[loss],
        eval_set_fingerprint="fp",
        eval_gpu=QuantEvalGpu(name="NVIDIA H100", sm_count=132, sku=sku),
    )


def _instruct_task() -> InstructTextRawTask:
    return InstructTextRawTask(**_base_fields(), field_instruction="instruction", field_output="output")


class TestQuantTrainingData:
    def test_fixed_task_carries_spec_and_dataset(self):
        task = _fixed_task()
        request = orchestrator._quant_training_data(
            task, EXPECTED_REPO, None, model=task.model_id, quant_target=_target_of(task), calibration=CALIBRATION
        )

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
        request = orchestrator._quant_training_data(
            task, EXPECTED_REPO, None, model=task.model_id, quant_target=_target_of(task), calibration=CALIBRATION
        )

        assert request.task_type == TaskType.QUANTFREEFORMTASK
        assert request.quant_spec is None

    def test_model_is_the_given_starting_model(self):
        request = orchestrator._quant_training_data(
            _fixed_task(), EXPECTED_REPO, None, model=AUGMENTED_MODEL, quant_target=FIXED_TARGET, calibration=CALIBRATION
        )

        assert request.model == AUGMENTED_MODEL


def _participant(**overrides) -> TournamentParticipant:
    return TournamentParticipant(tournament_id=TOURNAMENT_ID, hotkey=HOTKEY, **overrides)


MEASURED_HOTKEYS = (HOTKEY, "", orchestrator.EMISSION_BURN_HOTKEY, "5Challenger")


@pytest.fixture(autouse=True)
def sku_store(monkeypatch):
    """The stored bf16 references by task; a task nobody stored anything for is measured for MEASURED_HOTKEYS."""
    records: dict[uuid.UUID, list[QuantSkuReference]] = {}

    async def get_references(task_id, psql_db):
        task = SimpleNamespace(task_id=task_id)
        rows = records[task_id] if task_id in records else [_sku_reference(task, hotkey) for hotkey in MEASURED_HOTKEYS]
        return {row.key: row for row in rows}

    monkeypatch.setattr(orchestrator, "get_quant_sku_references", get_references)
    return records


@pytest.fixture(autouse=True)
def db_lookups(monkeypatch, calibration_store):
    lookups = SimpleNamespace(
        starting_model=AsyncMock(return_value=CONTINUATION_REPO),
        participant=AsyncMock(return_value=_participant(**SNAPSHOT_FIELDS)),
        presigned_url=AsyncMock(return_value=PRESIGNED_URL),
    )
    monkeypatch.setattr(orchestrator.task_sql, "get_expected_repo_name", AsyncMock(return_value=EXPECTED_REPO))
    monkeypatch.setattr(orchestrator.task_sql, "get_starting_model_repo", lookups.starting_model)
    monkeypatch.setattr(orchestrator.task_sql, "get_miner_baseline_stats", AsyncMock(return_value=None))
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
    async def test_without_a_stored_reference_there_is_no_request(self, db_lookups, sku_store, make_task, quant_inputs_alert):
        task = make_task()
        sku_store[task.task_id] = []

        with pytest.raises(orchestrator.QuantInputsUnavailable, match="no target to publish"):
            await _request(task)

        quant_inputs_alert.assert_awaited_once()

    async def test_quant_target_is_published_from_the_stored_reference(self, db_lookups, sku_store):
        task = _freeform_task(training_start_point=TrainingStartPoint.OWN_R1_MODEL, bar_cap=0.05)
        reference = _sku_reference(task, HOTKEY)
        sku_store[task.task_id] = [reference]

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

    def test_freeform_target_without_penalty_is_the_cliff(self):
        task = _freeform_task(penalty_knee=None, penalty_curvature=None, bar_cap=0.05)
        target = orchestrator.quant_target_from_reference(task, self._reference(2.0))

        assert (target.bar_cap, target.knee, target.penalty_curvature, target.cap) == (0.05, 1.05, 0.0, 1.05)
        assert target.bar == pytest.approx(2.1) and target.knee_loss == pytest.approx(2.1)
        assert target.cap_loss == pytest.approx(2.1)

    @pytest.mark.parametrize("make_task", [_fixed_task, _freeform_task])
    def test_target_fits_the_train_request(self, make_task):
        task = make_task()
        target = orchestrator.quant_target_from_reference(task, self._reference())

        request = orchestrator._quant_training_data(
            task, EXPECTED_REPO, None, model="org/own-bf16", quant_target=target, calibration=CALIBRATION
        )

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
def calibration_store(monkeypatch):
    """The stored calibration references by task; a task nobody stored anything for is measured for MEASURED_HOTKEYS."""
    store: dict[uuid.UUID, dict[str, QuantCalibrationReferenceRecord]] = {}

    async def get_records(task_id, psql_db):
        if task_id in store:
            return dict(store[task_id])
        return {h: _calibration_record(SimpleNamespace(task_id=task_id), hotkey=h) for h in MEASURED_HOTKEYS}

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

    async def test_without_a_measured_reference_there_is_no_calibration_request(self, calibration_store):
        task = _calibration_task()
        calibration_store.records[task.task_id] = {}

        with pytest.raises(orchestrator.QuantInputsUnavailable, match="not measured on the calibration rows"):
            await orchestrator._quant_calibration_for(task, HOTKEY, SimpleNamespace(psql_db=None))
        calibration_store.new_url.assert_not_called()

    async def test_another_hotkeys_reference_is_not_published(self, calibration_store):
        task = _calibration_task()
        calibration_store.records[task.task_id] = {"5Other": _calibration_record(task, hotkey="5Other")}

        with pytest.raises(orchestrator.QuantInputsUnavailable, match="not measured on the calibration rows"):
            await orchestrator._quant_calibration_for(task, HOTKEY, SimpleNamespace(psql_db=None))

    async def test_a_reference_on_another_example_set_is_not_published(self, calibration_store):
        task = _calibration_task()
        calibration_store.records[task.task_id] = {HOTKEY: _calibration_record(task, fingerprint="fp-other")}

        with pytest.raises(orchestrator.QuantInputsUnavailable, match="another example set"):
            await orchestrator._quant_calibration_for(task, HOTKEY, SimpleNamespace(psql_db=None))
        calibration_store.new_url.assert_not_called()

    async def test_a_presign_failure_raises_and_builds_no_request(self, calibration_store):
        calibration_store.new_url.return_value = None
        task = _calibration_task()

        with pytest.raises(orchestrator.QuantInputsUnavailable, match="presigned URL"):
            await orchestrator._quant_calibration_for(task, HOTKEY, SimpleNamespace(psql_db=None))

    @pytest.mark.parametrize("make_task", [_fixed_task, _freeform_task])
    async def test_a_task_without_a_published_set_raises_without_presigning(self, calibration_store, make_task):
        with pytest.raises(orchestrator.QuantInputsUnavailable, match="no published calibration set"):
            await orchestrator._quant_calibration_for(make_task(**UNCALIBRATED), HOTKEY, SimpleNamespace(psql_db=None))
        calibration_store.new_url.assert_not_called()

    async def test_a_set_without_a_fingerprint_or_count_is_unpublished(self, calibration_store):
        for missing in ("calibration_fingerprint", "calibration_n_examples"):
            with pytest.raises(orchestrator.QuantInputsUnavailable):
                await orchestrator._quant_calibration_for(
                    _calibration_task(**{missing: None}), HOTKEY, SimpleNamespace(psql_db=None)
                )

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
        calibration = CALIBRATION

        request = TrainRequestQuant(
            model="m",
            task_id="t",
            hours_to_complete=1.0,
            dataset="d",
            dataset_type=_dataset_type(),
            file_format=FileFormat.S3,
            task_type=TaskType.QUANTFIXEDTASK,
            quant_spec=SPEC,
            quant_target=FIXED_TARGET,
            calibration=calibration,
        )

        assert request.calibration == calibration

    async def test_the_request_json_carries_the_calibration(self, db_lookups):
        request = await _request(_freeform_task())
        dumped = request.training_data.model_dump(mode="json")

        assert dumped.pop("calibration")["examples_url"] == CAL_FRESH_URL
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
        calibration_store.records[task.task_id] = {}

        assert not await orchestrator._reference_ready(task, HOTKEY, SimpleNamespace(psql_db=None))

        calibration_store.records[task.task_id] = {HOTKEY: _calibration_record(task)}
        assert await orchestrator._reference_ready(task, HOTKEY, SimpleNamespace(psql_db=None))

    async def test_never_ready_without_the_calibration_row_however_many_measurements_failed(self, calibration_store, sku_rows):
        task = _calibration_task()
        calibration_store.records[task.task_id] = {}
        prepass._CONSECUTIVE_FAILURES.clear()
        for key in ((str(task.task_id), HOTKEY, "calibration"), (str(task.task_id), HOTKEY), (str(task.task_id), "")):
            prepass._CONSECUTIVE_FAILURES[key] = 1000

        assert not await orchestrator._reference_ready(task, HOTKEY, SimpleNamespace(psql_db=None))
        prepass._CONSECUTIVE_FAILURES.clear()

    async def test_a_calibration_row_on_another_example_set_is_not_ready(self, calibration_store, sku_rows):
        task = _calibration_task()
        calibration_store.records[task.task_id] = {HOTKEY: _calibration_record(task, fingerprint="fp-stale")}

        assert not await orchestrator._reference_ready(task, HOTKEY, SimpleNamespace(psql_db=None))

        calibration_store.records[task.task_id] = {HOTKEY: _calibration_record(task)}
        assert await orchestrator._reference_ready(task, HOTKEY, SimpleNamespace(psql_db=None))

    async def test_still_needs_the_test_reference_first(self, calibration_store, sku_store):
        task = _calibration_task()
        sku_store[task.task_id] = []

        assert not await orchestrator._reference_ready(task, HOTKEY, SimpleNamespace(psql_db=None))

    @pytest.mark.parametrize("make_task", [_fixed_task, _freeform_task])
    async def test_a_task_without_a_published_set_is_never_ready_and_reads_no_calibration_rows(
        self, sku_rows, monkeypatch, make_task
    ):
        read = AsyncMock()
        monkeypatch.setattr(orchestrator, "get_quant_calibration_references", read)
        task = make_task(training_start_point=TrainingStartPoint.OWN_R1_MODEL, **UNCALIBRATED)

        assert not await orchestrator._reference_ready(task, HOTKEY, SimpleNamespace(psql_db=None))
        read.assert_not_called()

    async def test_task_common_task_waits_for_the_empty_hotkey_rows(self, calibration_store, monkeypatch):
        task = _calibration_task(training_start_point=TrainingStartPoint.DEFAULT)
        config = SimpleNamespace(psql_db=None)
        rows = {}
        calibration_store.records[task.task_id] = {}

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


class TestAReferenceIsNeverGivenUp:
    CONFIG = SimpleNamespace(psql_db=None)
    LIMIT = 1000

    @pytest.fixture(autouse=True)
    def no_failures(self):
        prepass._CONSECUTIVE_FAILURES.clear()
        yield
        prepass._CONSECUTIVE_FAILURES.clear()

    @pytest.mark.parametrize("training_start_point", [TrainingStartPoint.DEFAULT, TrainingStartPoint.OWN_R1_MODEL])
    @pytest.mark.parametrize("failures", [0, 5, 6, LIMIT])
    async def test_a_row_without_its_reference_waits_however_many_measurements_failed(
        self, sku_store, training_start_point, failures
    ):
        task = _fixed_task(training_start_point=training_start_point)
        sku_store[task.task_id] = []
        prepass._CONSECUTIVE_FAILURES[(str(task.task_id), "")] = failures
        prepass._CONSECUTIVE_FAILURES[(str(task.task_id), HOTKEY)] = failures

        assert not await orchestrator._reference_ready(task, HOTKEY, self.CONFIG)

    async def test_a_non_tournament_task_waits_too(self, sku_store, monkeypatch):
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=None))
        task = _fixed_task(training_start_point=TrainingStartPoint.DEFAULT)
        sku_store[task.task_id] = []

        assert not await orchestrator._reference_ready(task, HOTKEY, self.CONFIG)

    async def test_the_row_is_ready_once_its_references_are_measured(self):
        task = _fixed_task(training_start_point=TrainingStartPoint.DEFAULT)

        assert await orchestrator._reference_ready(task, HOTKEY, self.CONFIG)


@pytest.fixture
def quant_inputs_alert(monkeypatch):
    notify = AsyncMock()
    monkeypatch.setattr(orchestrator, "notify_evaluation_exception", notify)
    monkeypatch.setattr(orchestrator, "_QUANT_INPUTS_MISSING_HELD_ROWS", {})
    return notify


class TestMissingQuantInputsBackstop:
    """A quant row whose task cannot hand over its calibration rows, target or reference losses is never dispatched: it raises,
    loudly."""

    @pytest.mark.parametrize("hotkey", [HOTKEY, orchestrator.EMISSION_BURN_HOTKEY])
    @pytest.mark.parametrize("make_task", [_fixed_task, _freeform_task])
    async def test_a_task_without_a_calibration_set_raises_and_alerts(
        self, quant_inputs_alert, make_task, hotkey, caplog
    ):
        task = make_task(**UNCALIBRATED)

        with caplog.at_level(logging.ERROR), pytest.raises(orchestrator.QuantInputsUnavailable):
            await _request(task, hotkey)

        assert "HELD and NOT dispatched" in caplog.text
        quant_inputs_alert.assert_awaited_once()
        kwargs = quant_inputs_alert.await_args.kwargs
        assert kwargs["task_id"] == str(task.task_id) and kwargs["hotkeys"] == [hotkey]
        error = kwargs["error"]
        assert "HELD" in error and "NOT dispatched" in error and "no published calibration set" in error

    @pytest.mark.parametrize("hotkey", [HOTKEY, orchestrator.EMISSION_BURN_HOTKEY])
    @pytest.mark.parametrize("make_task", [_fixed_task, _freeform_task])
    async def test_a_row_without_a_target_raises_and_alerts_with_that_reason(
        self, quant_inputs_alert, sku_store, make_task, hotkey
    ):
        task = _calibration_task(make_task)
        sku_store[task.task_id] = []

        with pytest.raises(orchestrator.QuantInputsUnavailable):
            await _request(task, hotkey)

        quant_inputs_alert.assert_awaited_once()
        assert "no target to publish" in quant_inputs_alert.await_args.kwargs["error"]
        assert quant_inputs_alert.await_args.kwargs["hotkeys"] == [hotkey]

    @pytest.mark.parametrize("stale", [False, True])
    async def test_a_row_without_calibration_reference_losses_raises_and_alerts(
        self, quant_inputs_alert, calibration_store, stale
    ):
        task = _calibration_task()
        records = {HOTKEY: _calibration_record(task, fingerprint="fp-stale")} if stale else {}
        calibration_store.records[task.task_id] = records

        with pytest.raises(orchestrator.QuantInputsUnavailable):
            await _request(task)

        quant_inputs_alert.assert_awaited_once()
        reason = "another example set" if stale else "not measured on the calibration rows"
        assert reason in quant_inputs_alert.await_args.kwargs["error"]

    async def test_an_unpublishable_calibration_set_raises_and_alerts(self, quant_inputs_alert, calibration_store):
        calibration_store.new_url.return_value = None

        with pytest.raises(orchestrator.QuantInputsUnavailable):
            await _request(_calibration_task())

        quant_inputs_alert.assert_awaited_once()
        assert "could not be published" in quant_inputs_alert.await_args.kwargs["error"]

    async def test_a_presign_that_raises_is_unavailable_and_leaks_neither_url_nor_error_text(
        self, quant_inputs_alert, calibration_store
    ):
        calibration_store.new_url.side_effect = ConnectionError(f"cannot reach {CAL_URL}")

        with pytest.raises(orchestrator.QuantInputsUnavailable) as raised:
            await _request(_calibration_task())

        message = str(raised.value)
        assert "ConnectionError" in message and "minio.example" not in message and "cannot reach" not in message
        quant_inputs_alert.assert_awaited_once()
        error = quant_inputs_alert.await_args.kwargs["error"]
        assert "ConnectionError" in error and "minio.example" not in error and "cannot reach" not in error

    async def test_a_built_calibration_clears_the_tasks_held_row_count(self, quant_inputs_alert):
        task = _calibration_task()
        orchestrator._QUANT_INPUTS_MISSING_HELD_ROWS[str(task.task_id)] = 5
        orchestrator._QUANT_INPUTS_MISSING_HELD_ROWS["other"] = 2

        await _request(task)

        assert orchestrator._QUANT_INPUTS_MISSING_HELD_ROWS == {"other": 2}

    async def test_a_published_calibration_set_does_not_alert(self, quant_inputs_alert):
        assert (await _request(_calibration_task())).training_data.calibration is not None

        quant_inputs_alert.assert_not_awaited()

    async def test_a_text_task_never_alerts(self, quant_inputs_alert):
        await _request(_instruct_task())

        quant_inputs_alert.assert_not_awaited()

    async def test_the_webhook_hears_the_first_held_row_and_every_nth_after_it(self, quant_inputs_alert, caplog):
        task = _freeform_task(**UNCALIBRATED)
        every = orchestrator.QUANT_INPUTS_MISSING_REALERT_HELD_ROWS

        with caplog.at_level(logging.ERROR):
            for _ in range(2 * every):
                await orchestrator._alert_quant_inputs_missing(task, HOTKEY, SimpleNamespace(psql_db=None))

        assert caplog.text.count("HELD and NOT dispatched") == 2 * every  # the ERROR is logged every time
        assert quant_inputs_alert.await_count == 3  # held row 1, row `every`, row `2 * every`
