"""create_continuous_sft_task builds the weekly ChatRawTask correctly.

Mocks the three I/O seams (state read, content-service call, add_task) and asserts on the
ChatRawTask handed to add_task. The critical invariants: base = carried winner else seed,
train_index forwarded to the content service, and chat_template == "tokenizer_default" (a silent
revert to chatml would corrupt every quasar eval loss).
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from core.constants.environments import TrainingStartPoint
from core.models.dataset_models import FileFormat
from core.models.task_models import TaskType
from validator.tasks.synthetics import scheduler
from validator.tournament.models import ContinuousSftState


SEED = "Qwen/Qwen3-8B-Base"


def _patch(monkeypatch, *, state, response):
    """Wire the three seams; return the AsyncMocks for call_content_service and add_task."""
    monkeypatch.setattr(scheduler, "get_continuous_sft_state", AsyncMock(return_value=state))
    ccs = AsyncMock(return_value=response)
    monkeypatch.setattr(scheduler, "call_content_service", ccs)
    add_task = AsyncMock(side_effect=lambda task, psql_db: task)  # echo the task back
    monkeypatch.setattr(scheduler, "add_task", add_task)
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
    assert task.hours_to_complete == 4.0  # fallback budget; prep resizes via the throughput pipeline
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
    state = ContinuousSftState(lineage="quasar", train_index=5, last_winner_repo=None)
    resp = {"train_s3_url": "https://s3/train", "test_s3_url": "https://s3/test"}  # no "ds"
    _, add_task = _patch(monkeypatch, state=state, response=resp)
    await scheduler.create_continuous_sft_task(MagicMock(), "quasar", SEED)
    assert add_task.call_args[0][0].ds == "continuous-sft:quasar:train-index-5"


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
