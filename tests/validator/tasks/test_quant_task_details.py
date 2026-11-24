"""convert_task_to_task_details for quant tasks."""

from datetime import datetime
from uuid import uuid4

from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import InstructTextTaskDetails
from core.models.payload_models import QuantFixedTaskDetails
from core.models.payload_models import QuantFreeformTaskDetails
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.tasks.details import convert_task_to_task_details
from validator.tasks.models import InstructTextTask
from validator.tasks.models import QuantFixedTask
from validator.tasks.models import QuantFreeformTask


SPEC = QuantFormat.W4A8_FP8_G128.spec()


def _common() -> dict:
    return dict(
        task_id=uuid4(),
        is_organic=False,
        status=TaskStatus.SUCCESS,
        model_id="org/base",
        ds="secret/ds",
        account_id=uuid4(),
        hours_to_complete=3.0,
        created_at=datetime(2026, 1, 1),
        started_at=datetime(2026, 1, 2),
        termination_at=datetime(2026, 1, 3),
        trained_model_repository="org/trained",
        calibration_data="https://signed.example/cal?sig=abc",
    )


def _quant_kwargs() -> dict:
    return dict(_common(), dataset_type=InstructTextDatasetType(field_instruction="instruction", field_output="output"))


def _freeform() -> QuantFreeformTask:
    return QuantFreeformTask(**_quant_kwargs())


def _fixed() -> QuantFixedTask:
    return QuantFixedTask(**_quant_kwargs(), quant_spec=SPEC)


def _text() -> InstructTextTask:
    return InstructTextTask(
        **_common(),
        field_instruction="instruction",
        field_output="output",
    )


def test_freeform_task_converts_without_internal_fields():
    task = _freeform()

    details = convert_task_to_task_details(task)

    assert isinstance(details, QuantFreeformTaskDetails)
    assert (details.id, details.account_id, details.status) == (task.task_id, task.account_id, TaskStatus.SUCCESS)
    assert details.task_type == TaskType.QUANTFREEFORMTASK and details.base_model_repository == "org/base"
    assert (details.created_at, details.started_at) == (task.created_at, task.started_at)
    assert details.finished_at == task.termination_at
    assert details.hours_to_complete == 3.0 and details.trained_model_repository == "org/trained"
    dumped = details.model_dump_json()
    assert "signed.example" not in dumped and "secret/ds" not in dumped and "calibration" not in dumped


def test_fixed_task_converts_with_its_format():
    details = convert_task_to_task_details(_fixed())

    assert isinstance(details, QuantFixedTaskDetails)
    assert details.task_type == TaskType.QUANTFIXEDTASK and details.quant_spec == SPEC


def test_text_task_is_unchanged():
    details = convert_task_to_task_details(_text())

    assert isinstance(details, InstructTextTaskDetails) and details.ds_repo == "secret/ds"


