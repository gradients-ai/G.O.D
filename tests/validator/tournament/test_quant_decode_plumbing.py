"""Freeform decode plumbing in the orchestrator: decode-result persistence and the PREEVALUATION hold for
unsettled decodes.

Only the miner's fault is scored (no decoder at registration, or the trainer's quant_failure); infra
trouble and a snapshot mismatch hold the task and alert once per (task, hotkey), never zeroing anyone.
"""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import httpx
import pytest

from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestQuant
from core.models.quant_artifact_models import QuantDecodedFile
from core.models.quant_artifact_models import QuantDecodeResult
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tournament import orchestrator
from validator.tournament.models import QuantDecodeRow
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentTaskTraining
from validator.tournament.models import TrainingStatus


HOTKEY = "5Miner"
OTHER_HOTKEY = "5Other"
TOURNAMENT_ID = "tourn_1"
TRAINER_IP = "10.0.0.1"
DECODER_SHA = "a" * 64
DECODER_BYTES = 4096
SNAPSHOT_FIELDS = {
    "decoder_object_name": "quant-decoders/c/a.tar.xz",
    "decoder_xz_bytes": DECODER_BYTES,
    "decoder_sha256": DECODER_SHA,
}
CONFIG = SimpleNamespace(psql_db=None, discord_url="https://discord.example/webhook")


def _base_fields() -> dict:
    return {
        "is_organic": False,
        "task_id": uuid.uuid4(),
        "status": "training",
        "model_id": "org/model",
        "ds": "org/dataset",
        "account_id": uuid.uuid4(),
        "hours_to_complete": 2.0,
        "training_data": "s3://bucket/train.json",
        "created_at": datetime(2026, 10, 19),
    }


def _dataset_type() -> InstructTextDatasetType:
    return InstructTextDatasetType(field_instruction="instruction", field_output="output")


def _freeform_task() -> QuantFreeformRawTask:
    return QuantFreeformRawTask(**_base_fields(), dataset_type=_dataset_type())


def _fixed_task() -> QuantFixedRawTask:
    return QuantFixedRawTask(**_base_fields(), dataset_type=_dataset_type(), quant_spec=QuantFormat.W4A8_FP8_G128.spec())


def _instruct_task() -> InstructTextRawTask:
    return InstructTextRawTask(**_base_fields(), field_instruction="instruction", field_output="output")


def _training(task, hotkey: str = HOTKEY, trainer_ip: str | None = TRAINER_IP) -> TournamentTaskTraining:
    return TournamentTaskTraining(
        task=task,
        hotkey=hotkey,
        training_status=TrainingStatus.SUCCESS,
        n_training_attempts=1,
        created_at=datetime(2026, 10, 19),
        updated_at=datetime(2026, 10, 19),
        trainer_ip=trainer_ip,
    )


def _task_log(task, hotkey: str = HOTKEY, decode: QuantDecodeResult | None = None) -> TrainerTaskLog:
    return TrainerTaskLog(
        gpu_ids=[0],
        status=TaskStatus.SUCCESS,
        training_data=TrainRequestQuant(
            model=task.model_id,
            task_id=str(task.task_id),
            hours_to_complete=task.hours_to_complete,
            dataset=task.training_data,
            dataset_type=task.dataset_type,
            file_format="s3",
            task_type=task.task_type,
            quant_spec=getattr(task, "quant_spec", None),
        ),
        github_repo="https://github.com/miner/repo",
        hotkey=hotkey,
        quant_decode=decode,
    )


def _ok(sha: str = DECODER_SHA, decoder_bytes: int = DECODER_BYTES) -> QuantDecodeResult:
    return QuantDecodeResult(ok=True, artifact_bytes=10_000, decoder_bytes=decoder_bytes, decoder_sha256=sha)


def _participant(with_snapshot: bool = True) -> TournamentParticipant:
    return TournamentParticipant(tournament_id=TOURNAMENT_ID, hotkey=HOTKEY, **(SNAPSHOT_FIELDS if with_snapshot else {}))


def _row(hotkey: str, *, decoder_sha256: str | None = None, quant_failure: str | None = None) -> QuantDecodeRow:
    return QuantDecodeRow(
        hotkey=hotkey, artifact_bytes=None, decoder_bytes=None, decoder_sha256=decoder_sha256, quant_failure=quant_failure
    )


@pytest.fixture
def decode_sql(monkeypatch):
    mocks = SimpleNamespace(
        participant=AsyncMock(return_value=_participant()),
        persist=AsyncMock(),
        notify=AsyncMock(),
    )
    monkeypatch.setattr(orchestrator, "_QUANT_DECODE_ALERTED", set())
    monkeypatch.setattr(orchestrator, "_QUANT_DECODE_REPOLL_FAILURES", {})
    monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=TOURNAMENT_ID))
    monkeypatch.setattr(orchestrator.tournament_sql, "get_tournament_participant", mocks.participant)
    monkeypatch.setattr(orchestrator.submissions_and_scoring_sql, "set_quant_decode_result", mocks.persist)
    monkeypatch.setattr(orchestrator, "notify_quant_decode_infra_error", mocks.notify)
    return mocks


class TestRecordQuantDecode:
    async def test_ok_persists_bytes_and_sha(self, decode_sql):
        task = _freeform_task()
        settled = await orchestrator._record_quant_decode(_training(task), _task_log(task, decode=_ok()), CONFIG)

        assert settled is True
        decode_sql.persist.assert_awaited_once_with(
            task.task_id,
            HOTKEY,
            artifact_bytes=10_000,
            decoder_bytes=DECODER_BYTES,
            decoder_sha256=DECODER_SHA,
            quant_failure=None,
            psql_db=None,
            build_warning=None,
            artifact_files=[],
            decoded_files=[],
        )
        decode_sql.notify.assert_not_awaited()

    async def test_ok_persists_artifact_and_decoded_file_lists(self, decode_sql):
        task = _freeform_task()
        artifact = [QuantDecodedFile(name=".gitattributes", bytes=1519, sha256="a" * 64)]
        decoded = [QuantDecodedFile(name="model-00001.safetensors", bytes=8, sha256="b" * 64)]
        decode = _ok().model_copy(update={"artifact_files": artifact, "decoded_files": decoded})

        settled = await orchestrator._record_quant_decode(_training(task), _task_log(task, decode=decode), CONFIG)

        assert settled is True
        kwargs = decode_sql.persist.await_args.kwargs
        assert kwargs["artifact_files"] == artifact
        assert kwargs["decoded_files"] == decoded

    async def test_scored_fail_keeps_the_artifact_listing_it_downloaded(self, decode_sql):
        task = _freeform_task()
        artifact = [QuantDecodedFile(name="m.bin", bytes=5, sha256="a" * 64)]
        decode = QuantDecodeResult(quant_failure="decoder crashed", artifact_files=artifact)

        await orchestrator._record_quant_decode(_training(task), _task_log(task, decode=decode), CONFIG)

        kwargs = decode_sql.persist.await_args.kwargs
        assert kwargs["artifact_files"] == artifact
        assert kwargs["decoded_files"] is None

    async def test_build_warning_reaches_the_task_node_row_without_failing_it(self, decode_sql):
        task = _freeform_task()
        warning = "build.sh exited 0 but left /build empty"
        decode = _ok().model_copy(update={"build_warning": warning})

        settled = await orchestrator._record_quant_decode(_training(task), _task_log(task, decode=decode), CONFIG)

        assert settled is True
        kwargs = decode_sql.persist.await_args.kwargs
        assert (kwargs["build_warning"], kwargs["quant_failure"]) == (warning, None)

    @pytest.mark.parametrize("decode", [_ok(sha="b" * 64), _ok(decoder_bytes=DECODER_BYTES + 1)])
    async def test_snapshot_mismatch_alerts_and_persists_nothing(self, decode_sql, decode):
        task = _freeform_task()
        settled = await orchestrator._record_quant_decode(_training(task), _task_log(task, decode=decode), CONFIG)

        assert settled is False
        decode_sql.persist.assert_not_awaited()
        decode_sql.notify.assert_awaited_once()
        assert "decoder snapshot mismatch" in decode_sql.notify.await_args.args[2]

    async def test_quant_failure_persists(self, decode_sql):
        task = _freeform_task()
        decode = QuantDecodeResult(quant_failure="decoder crashed", decoder_bytes=DECODER_BYTES, decoder_sha256=DECODER_SHA)
        settled = await orchestrator._record_quant_decode(_training(task), _task_log(task, decode=decode), CONFIG)

        assert settled is True
        kwargs = decode_sql.persist.await_args.kwargs
        assert kwargs["quant_failure"] == "decoder crashed"
        assert kwargs["decoder_bytes"] == DECODER_BYTES
        assert kwargs["decoder_sha256"] == DECODER_SHA

    async def test_infra_error_alerts_once_per_task_and_hotkey(self, decode_sql):
        task = _freeform_task()
        log = _task_log(task, decode=QuantDecodeResult(infra_error="minio unreachable"))

        first = await orchestrator._record_quant_decode(_training(task), log, CONFIG)
        second = await orchestrator._record_quant_decode(_training(task), log, CONFIG)

        assert first is False and second is False
        decode_sql.persist.assert_not_awaited()
        decode_sql.notify.assert_awaited_once_with(str(task.task_id), HOTKEY, "minio unreachable", CONFIG.discord_url)

    async def test_no_result_without_snapshot_is_scored_fail(self, decode_sql):
        decode_sql.participant.return_value = _participant(with_snapshot=False)
        task = _freeform_task()
        settled = await orchestrator._record_quant_decode(_training(task), _task_log(task), CONFIG)

        assert settled is True
        assert decode_sql.persist.await_args.kwargs["quant_failure"] == orchestrator.NO_DECODER_SNAPSHOT_FAILURE

    async def test_no_result_with_snapshot_holds(self, decode_sql):
        task = _freeform_task()
        settled = await orchestrator._record_quant_decode(_training(task), _task_log(task), CONFIG)

        assert settled is False
        decode_sql.persist.assert_not_awaited()
        decode_sql.notify.assert_awaited_once()

    async def test_fixed_task_is_noop(self, decode_sql):
        task = _fixed_task()
        settled = await orchestrator._record_quant_decode(_training(task), _task_log(task), CONFIG)

        assert settled is True
        decode_sql.participant.assert_not_awaited()
        decode_sql.persist.assert_not_awaited()

    async def test_organic_task_is_settled_without_a_scored_fail(self, decode_sql, monkeypatch):
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=None))
        task = _freeform_task()
        settled = await orchestrator._record_quant_decode(_training(task), _task_log(task), CONFIG)

        assert settled is True
        decode_sql.participant.assert_not_awaited()
        decode_sql.persist.assert_not_awaited()
        decode_sql.notify.assert_not_awaited()


@pytest.fixture
def preevaluation(monkeypatch, decode_sql):
    mocks = SimpleNamespace(
        decode_rows=AsyncMock(return_value={}),
        task_details=AsyncMock(),
        update_task=AsyncMock(),
        statuses=AsyncMock(
            return_value={HOTKEY: "success", OTHER_HOTKEY: "success", "5Failed": "failure"}
        ),
        successful=AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(orchestrator.task_sql, "update_task", mocks.update_task)
    monkeypatch.setattr(orchestrator.tournament_sql, "get_training_status_for_task", mocks.statuses)
    monkeypatch.setattr(orchestrator.tournament_sql, "get_tournament_training_tasks", mocks.successful)
    monkeypatch.setattr(orchestrator.submissions_and_scoring_sql, "get_quant_decode_rows", mocks.decode_rows)
    monkeypatch.setattr(orchestrator, "get_training_task_details", mocks.task_details)
    monkeypatch.setattr(orchestrator.seeds_sql, "boss_round_seed_ready_to_finalise", AsyncMock(return_value=True))
    return mocks


def _serve_tasks(monkeypatch, task) -> None:
    monkeypatch.setattr(
        orchestrator.tournament_sql, "get_tasks_with_all_training_completed", AsyncMock(return_value=[task.task_id])
    )
    monkeypatch.setattr(orchestrator.task_sql, "get_task", AsyncMock(return_value=task))


class TestPreevaluationHold:
    async def test_holds_then_moves_once_the_poll_settles(self, monkeypatch, preevaluation):
        task = _freeform_task()
        _serve_tasks(monkeypatch, task)
        preevaluation.decode_rows.return_value = {
            HOTKEY: _row(HOTKEY),
            OTHER_HOTKEY: _row(OTHER_HOTKEY, decoder_sha256=DECODER_SHA),
        }
        preevaluation.successful.return_value = [_training(task), _training(_fixed_task(), hotkey=HOTKEY)]
        preevaluation.task_details.return_value = _task_log(task, decode=QuantDecodeResult(infra_error="image missing"))

        await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)

        preevaluation.task_details.assert_awaited_once_with(TRAINER_IP, str(task.task_id), HOTKEY)
        preevaluation.update_task.assert_not_awaited()

        preevaluation.task_details.return_value = _task_log(task, decode=_ok())
        await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)

        preevaluation.update_task.assert_awaited_once()
        assert task.status == TaskStatus.PREEVALUATION

    async def test_repoll_http_failures_alert_after_the_threshold(self, monkeypatch, preevaluation, decode_sql):
        task = _freeform_task()
        _serve_tasks(monkeypatch, task)
        preevaluation.statuses.return_value = {HOTKEY: "success"}
        preevaluation.successful.return_value = [_training(task)]
        preevaluation.task_details.side_effect = httpx.ConnectError("trainer gone")

        for _ in range(orchestrator.QUANT_DECODE_REPOLL_FAILURES_BEFORE_ALERT - 1):
            await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)
        decode_sql.notify.assert_not_awaited()

        await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)

        decode_sql.notify.assert_awaited_once()
        assert "trainer gone" in decode_sql.notify.await_args.args[2]
        preevaluation.update_task.assert_not_awaited()

    async def test_successful_repoll_resets_the_failure_count(self, monkeypatch, preevaluation, decode_sql):
        task = _freeform_task()
        _serve_tasks(monkeypatch, task)
        preevaluation.statuses.return_value = {HOTKEY: "success"}
        preevaluation.successful.return_value = [_training(task)]
        held_log = _task_log(task, decode=QuantDecodeResult(infra_error="image missing"))
        error = httpx.ConnectError("blip")
        preevaluation.task_details.side_effect = [error, error, held_log, error, error]

        for _ in range(5):
            await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)

        assert [call.args[2] for call in decode_sql.notify.await_args_list] == ["image missing"]

    async def test_hotkey_without_trainer_ip_holds_and_alerts(self, monkeypatch, preevaluation, decode_sql):
        task = _freeform_task()
        _serve_tasks(monkeypatch, task)
        preevaluation.decode_rows.return_value = {OTHER_HOTKEY: _row(OTHER_HOTKEY, quant_failure="bad output")}
        preevaluation.successful.return_value = [_training(task, trainer_ip=None)]

        await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)

        preevaluation.update_task.assert_not_awaited()
        preevaluation.task_details.assert_not_awaited()
        decode_sql.notify.assert_awaited_once()

    async def test_summary_counts_only_tasks_actually_moved(self, monkeypatch, preevaluation):
        held, moved = _freeform_task(), _fixed_task()
        tasks = {held.task_id: held, moved.task_id: moved}
        monkeypatch.setattr(
            orchestrator.tournament_sql,
            "get_tasks_with_all_training_completed",
            AsyncMock(return_value=list(tasks)),
        )
        monkeypatch.setattr(orchestrator.task_sql, "get_task", AsyncMock(side_effect=lambda task_id, _db: tasks[task_id]))
        preevaluation.statuses.return_value = {HOTKEY: "success"}
        log = MagicMock()
        monkeypatch.setattr(orchestrator, "logger", log)

        await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)

        assert held.status != TaskStatus.PREEVALUATION
        assert moved.status == TaskStatus.PREEVALUATION
        assert "Successfully moved 1 tasks to preevaluation status" in log.info.call_args.args[0]

    @pytest.mark.parametrize("make_task", [_fixed_task, _instruct_task])
    async def test_never_holds_other_task_types(self, monkeypatch, preevaluation, make_task):
        task = make_task()
        _serve_tasks(monkeypatch, task)

        await orchestrator._move_completed_tasks_to_preevaluation(CONFIG)

        preevaluation.update_task.assert_awaited_once()
        preevaluation.decode_rows.assert_not_awaited()
