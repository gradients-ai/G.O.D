"""Where calibration is carved, inherited or left out: the task flag, quant task construction, recreation, hiding, copying."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from core.models.dataset_models import InstructTextDatasetType
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.db.sql import tasks as task_sql
from validator.tasks.details import hide_sensitive_data_till_finished
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import QuantPreppedSplit
from validator.tasks.synthetics import scheduler


CAL_URL = "https://minio/r1_calibration_data.json"
DATASET_TYPE = InstructTextDatasetType(field_instruction="q", field_output="a")


@pytest.mark.parametrize("carve", [True, False])
async def test_create_synthetic_instruct_task_records_the_carve_flag(monkeypatch, carve):
    monkeypatch.setattr(scheduler, "get_dataset", AsyncMock(return_value=SimpleNamespace(dataset_id="ds", num_rows=50_000)))
    columns = SimpleNamespace(field_instruction="q", field_input="c", field_output="a")
    monkeypatch.setattr(scheduler, "_get_columns_for_instruct_dataset", AsyncMock(return_value=columns))
    monkeypatch.setattr(scheduler, "get_model_num_params", lambda model_id: 1_000_000_000)
    monkeypatch.setattr(scheduler, "add_task", AsyncMock(side_effect=lambda task, psql_db: task))

    kwargs = {"carve_calibration": True} if carve else {}
    task = await scheduler.create_synthetic_instruct_text_task(
        MagicMock(), None, MagicMock(), model_id_override="m", allow_augmentation=False, allow_yarn=False, **kwargs
    )

    assert task.carve_calibration is carve and task.calibration_data is None


def test_new_quant_task_always_carves_and_carries_the_given_calibration_rows():
    kwargs = dict(model_id="m", ds="ds", dataset_type=DATASET_TYPE, hours_to_complete=2.0, calibration_data=CAL_URL)

    freeform = scheduler._new_quant_task(TaskType.QUANTFREEFORMTASK, **kwargs)
    fixed = scheduler._new_quant_task(TaskType.QUANTFIXEDTASK, quant_spec=QuantFormat.W4A16_SYM_G128.spec(), **kwargs)

    for task in (freeform, fixed):
        assert task.calibration_data == CAL_URL and task.carve_calibration is True


def test_new_quant_task_has_no_carve_parameter_and_carves_a_given_split():
    kwargs = dict(model_id="m", ds="ds", dataset_type=DATASET_TYPE, hours_to_complete=2.0, test_data="https://minio/t.json")
    spec = QuantFormat.W4A16_SYM_G128.spec()

    for task_type in (TaskType.QUANTFREEFORMTASK, TaskType.QUANTFIXEDTASK):
        assert scheduler._new_quant_task(task_type, quant_spec=spec, **kwargs).carve_calibration is True
        with pytest.raises(TypeError):
            scheduler._new_quant_task(task_type, quant_spec=spec, carve_calibration=False, **kwargs)


def test_a_default_quant_task_has_no_calibration_yet_and_carves():
    task = scheduler._new_quant_task(
        TaskType.QUANTFREEFORMTASK, model_id="m", ds="ds", dataset_type=DATASET_TYPE, hours_to_complete=2.0
    )

    assert (task.calibration_data, task.calibration_fingerprint, task.calibration_n_examples, task.carve_calibration) == (
        None, None, None, True,
    )  # fmt: skip


async def test_recreate_quant_task_re_resolves_the_calibration_rows_with_the_split(monkeypatch):
    monkeypatch.setattr(scheduler, "add_task", AsyncMock(side_effect=lambda task, psql_db: task))
    source = scheduler._new_quant_task(
        TaskType.QUANTFREEFORMTASK, model_id="m", ds="ds", dataset_type=DATASET_TYPE, hours_to_complete=2.0
    )
    split = QuantPreppedSplit(training_data="https://t", test_data="https://e", calibration_data=CAL_URL)

    fresh = await scheduler.recreate_quant_task(MagicMock(), source, prepped_split=split)
    plain = await scheduler.recreate_quant_task(MagicMock(), source)

    assert fresh.calibration_data == CAL_URL and fresh.carve_calibration is True
    assert plain.calibration_data is None and plain.carve_calibration is True


def _instruct_task(status: TaskStatus = TaskStatus.TRAINING) -> InstructTextRawTask:
    return InstructTextRawTask(
        task_id=uuid4(),
        is_organic=False,
        status=status,
        model_id="m",
        ds="ds",
        account_id=uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 11, 6),
        field_instruction="q",
        field_output="a",
        test_data="https://t",
        training_data="https://tr",
        calibration_data=CAL_URL,
        calibration_fingerprint="fp",
        calibration_n_examples=3,
    )


def test_calibration_is_hidden_while_the_task_is_in_flight_and_shown_after():
    in_flight = hide_sensitive_data_till_finished(_instruct_task())
    finished = hide_sensitive_data_till_finished(_instruct_task(TaskStatus.SUCCESS))

    assert (in_flight.calibration_data, in_flight.calibration_fingerprint, in_flight.calibration_n_examples) == (None, None, None)
    assert finished.calibration_data == CAL_URL and finished.calibration_fingerprint == "fp"


async def test_the_copy_path_regenerates_the_calibration_presigned_url(monkeypatch):
    regenerate = AsyncMock(side_effect=lambda url: url + "?fresh")
    monkeypatch.setattr(task_sql.async_minio_client, "get_new_presigned_url", regenerate)
    monkeypatch.setattr(task_sql, "add_task", AsyncMock(side_effect=lambda task, psql_db: task))

    copied = await task_sql.copy_task_for_benchmark(_instruct_task(), MagicMock())

    assert copied.calibration_data == CAL_URL + "?fresh"
    assert {call.args[0] for call in regenerate.await_args_list} == {"https://t", "https://tr", CAL_URL}
