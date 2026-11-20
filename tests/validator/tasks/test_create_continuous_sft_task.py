"""create_continuous_sft_task builds the weekly ChatRawTask correctly.

Mocks the three I/O seams (state read, content-service call, add_task) and asserts on the
ChatRawTask handed to add_task. The critical invariants: base = carried winner else seed,
train_index forwarded to the content service, and chat_template == "tokenizer_default" (a silent
revert to chatml would corrupt every continuous-SFT eval loss).
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from tenacity import stop_after_attempt

from core.constants.environments import TrainingStartPoint
from core.models.dataset_models import FileFormat
from core.models.task_models import TaskType
from validator.tasks.synthetics import scheduler
from validator.tournament import constants as t_cst
from validator.tournament.models import ChunkScheme
from validator.tournament.models import ContinuousSftState
from validator.tournament.models import ContinuousSftTaskChunk


SEED = "Qwen/Qwen3-8B-Base"


async def _assign_task_id(task, psql_db):
    task.task_id = uuid4()  # add_task assigns the id of the stored task
    return task


def _patch(monkeypatch, *, state, response):
    """Wire the three seams; return the AsyncMocks for call_content_service and add_task."""
    monkeypatch.setattr(scheduler, "get_continuous_sft_state", AsyncMock(return_value=state))
    ccs = AsyncMock(return_value=response)
    monkeypatch.setattr(scheduler, "call_content_service", ccs)
    add_task = AsyncMock(side_effect=_assign_task_id)
    monkeypatch.setattr(scheduler, "add_task", add_task)
    monkeypatch.setattr(scheduler, "record_continuous_sft_task_chunk", AsyncMock())
    return ccs, add_task


def _good_response():
    return {"train_s3_url": "https://s3/train", "test_s3_url": "https://s3/test", "ds": "chunk-00003"}


async def test_builds_expected_chatrawtask(monkeypatch):
    state = ContinuousSftState(lineage="qwen", train_index=3, last_winner_repo="org/winner")
    _, add_task = _patch(monkeypatch, state=state, response=_good_response())

    await scheduler.create_continuous_sft_task(MagicMock(), "qwen", SEED)

    task = add_task.call_args[0][0]
    assert task.task_type == TaskType.CHATTASK
    assert task.training_start_point == TrainingStartPoint.CONTINUOUS_SFT
    assert task.chat_template == "tokenizer_default"
    assert task.file_format == FileFormat.S3
    assert task.training_data == "https://s3/train"
    assert task.test_data == "https://s3/test"
    assert task.ds == "continuous-sft:qwen:chunk-00003"
    assert task.hours_to_complete == t_cst.CONTINUOUS_SFT_TRAINING_HOURS  # fixed budget, unchanged after prep
    assert task.is_organic is False
    assert task.augmentation_config is None


async def test_base_model_is_winner_when_present(monkeypatch):
    state = ContinuousSftState(lineage="qwen", train_index=3, last_winner_repo="org/winner")
    _, add_task = _patch(monkeypatch, state=state, response=_good_response())
    await scheduler.create_continuous_sft_task(MagicMock(), "qwen", SEED)
    assert add_task.call_args[0][0].model_id == "org/winner"


async def test_base_model_is_seed_on_first_run(monkeypatch):
    state = ContinuousSftState(lineage="qwen", train_index=0, last_winner_repo=None)
    _, add_task = _patch(monkeypatch, state=state, response=_good_response())
    await scheduler.create_continuous_sft_task(MagicMock(), "qwen", SEED)
    assert add_task.call_args[0][0].model_id == SEED


async def test_forwards_train_index_to_content_service(monkeypatch):
    state = ContinuousSftState(lineage="qwen", train_index=7, last_winner_repo=None)
    ccs, _ = _patch(monkeypatch, state=state, response=_good_response())
    await scheduler.create_continuous_sft_task(MagicMock(), "qwen", SEED)
    # params dict is the third positional arg to call_content_service(endpoint, keypair, params)
    assert ccs.call_args[0][2] == {"train_index": 7}


async def test_ds_label_falls_back_when_no_ds_field(monkeypatch):
    state = ContinuousSftState(lineage="qwen", train_index=5, last_winner_repo=None)
    resp = {"train_s3_url": "https://s3/train", "test_s3_url": "https://s3/test"}  # no "ds"
    _, add_task = _patch(monkeypatch, state=state, response=resp)
    await scheduler.create_continuous_sft_task(MagicMock(), "qwen", SEED)
    assert add_task.call_args[0][0].ds == "continuous-sft:qwen:train-index-5"


@pytest.mark.parametrize(
    "bad_response",
    [
        ["not", "a", "dict"],
        {"train_s3_url": "https://s3/train"},  # missing test url
        {"test_s3_url": "https://s3/test"},  # missing train url
        {"train_s3_url": "", "test_s3_url": ""},  # empty urls
    ],
)
async def test_raises_on_malformed_content_service_response(monkeypatch, bad_response):
    state = ContinuousSftState(lineage="qwen", train_index=1, last_winner_repo=None)
    _, add_task = _patch(monkeypatch, state=state, response=bad_response)
    with pytest.raises(ValueError):
        await scheduler.create_continuous_sft_task(MagicMock(), "qwen", SEED)
    add_task.assert_not_called()  # never persist a task pointing at nothing


async def test_records_the_task_chunk_after_add_task(monkeypatch):
    state = ContinuousSftState(lineage="qwen", train_index=7, last_winner_repo=None)
    _, add_task = _patch(monkeypatch, state=state, response=_good_response())
    config = MagicMock()
    task = await scheduler.create_continuous_sft_task(config, "qwen", SEED)
    record = scheduler.record_continuous_sft_task_chunk
    record.assert_awaited_once_with(
        ContinuousSftTaskChunk(task_id=task.task_id, lineage="qwen", train_index=7, chunk_scheme=ChunkScheme.V2),
        config.psql_db,
    )


async def test_failed_chunk_record_deletes_the_task_and_reraises(monkeypatch):
    state = ContinuousSftState(lineage="qwen", train_index=7, last_winner_repo=None)
    _, add_task = _patch(monkeypatch, state=state, response=_good_response())
    monkeypatch.setattr(scheduler, "record_continuous_sft_task_chunk", AsyncMock(side_effect=RuntimeError("write failed")))
    delete_task = AsyncMock()
    monkeypatch.setattr(scheduler, "delete_task", delete_task)
    config = MagicMock()

    # One attempt only: the creator's retry wrapper (which re-raises the original error) would otherwise sleep between tries.
    single_attempt = scheduler.create_continuous_sft_task.retry_with(stop=stop_after_attempt(1))
    with pytest.raises(RuntimeError, match="write failed"):
        await single_attempt(config, "qwen", SEED)

    # Every task stored along the way is deleted again.
    stored = [call.args[0].task_id for call in add_task.call_args_list]
    assert len(stored) == 1
    assert [call.args for call in delete_task.await_args_list] == [(stored[0], config.psql_db)]
