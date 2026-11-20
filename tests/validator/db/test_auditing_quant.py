"""get_task_with_hotkey_details for quant tasks: both types return a task (not None); freeform rows carry the scorecard."""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import validator.db.constants as cst
from core.models.dataset_models import InstructTextDatasetType
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskType
from validator.db.sql import auditing
from validator.tasks.models import QuantFixedTask
from validator.tasks.models import QuantFixedTaskWithHotkeyDetails
from validator.tasks.models import QuantFreeformScorecard
from validator.tasks.models import QuantFreeformTask
from validator.tasks.models import QuantFreeformTaskWithHotkeyDetails
from validator.tasks.models import ScorecardArtifact
from validator.tasks.models import ScorecardBytes
from validator.tasks.models import ScorecardDecoded
from validator.tasks.models import ScorecardDecoder
from validator.tasks.models import ScorecardQuality


TASK_ID = uuid.uuid4()


def _fields() -> dict:
    return {
        "task_id": TASK_ID,
        "is_organic": False,
        "status": "success",
        "model_id": "org/base",
        "ds": "org/dataset",
        "account_id": uuid.uuid4(),
        "hours_to_complete": 2.0,
        "created_at": datetime(2026, 10, 19),
        "test_data": "https://bucket/test.jsonl",
        "dataset_type": InstructTextDatasetType(field_instruction="instruction", field_output="output"),
    }


def _row(hotkey: str) -> dict:
    return {
        cst.HOTKEY: hotkey,
        cst.SUBMISSION_ID: None,
        cst.QUALITY_SCORE: 1.0,
        cst.TEST_LOSS: 2.0,
        cst.SYNTH_LOSS: 2.0,
        cst.SCORE_REASON: None,
        "rank": 1,
        cst.REPO: "org/repo",
        cst.OFFER_RESPONSE: None,
    }


def _scorecard() -> QuantFreeformScorecard:
    return QuantFreeformScorecard(
        bytes=ScorecardBytes(),
        quality=ScorecardQuality(test_loss=2.0),
        artifact=ScorecardArtifact(),
        decoder=ScorecardDecoder(),
        decoded=ScorecardDecoded(),
    )


@pytest.fixture
def audit(monkeypatch):
    state = SimpleNamespace(task=None, rows=[_row("A"), _row("B")], scorecards={}, scorecard_calls=[])
    conn = MagicMock()
    conn.fetch = AsyncMock(side_effect=lambda *_: state.rows)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    config = SimpleNamespace(psql_db=SimpleNamespace(connection=AsyncMock(return_value=cm)))

    async def scorecards(task, cfg):
        state.scorecard_calls.append(task.task_type)
        return state.scorecards

    monkeypatch.setattr(auditing.tasks_sql, "get_task_by_id", AsyncMock(side_effect=lambda *_: state.task))
    monkeypatch.setattr(auditing, "quant_scorecards_for_task", scorecards)
    monkeypatch.setattr(auditing.seeds_sql, "get_hidden_seed_task_ids", AsyncMock(return_value=set()))
    state.config = config
    return state


async def test_fixed_task_returns_details_without_scorecards(audit) -> None:
    audit.task = QuantFixedTask(**_fields(), quant_spec=QuantFormat.W4A8_FP8_G128.spec())

    result = await auditing.get_task_with_hotkey_details(str(TASK_ID), audit.config)

    assert isinstance(result, QuantFixedTaskWithHotkeyDetails)
    assert result.task_type == TaskType.QUANTFIXEDTASK
    assert [d.hotkey for d in result.hotkey_details] == ["A", "B"]
    assert all(d.quant_scorecard is None for d in result.hotkey_details)
    assert audit.scorecard_calls == []


async def test_a_seed_task_is_hidden_in_the_audit_while_its_rounds_are_unfinished(audit, monkeypatch) -> None:
    audit.task = QuantFreeformTask(**_fields())
    hidden = AsyncMock(return_value={str(TASK_ID)})
    monkeypatch.setattr(auditing.seeds_sql, "get_hidden_seed_task_ids", hidden)

    result = await auditing.get_task_with_hotkey_details(str(TASK_ID), audit.config)

    assert result.status == "success"  # finished, and still blanked
    assert result.test_data is None and result.ds == "Hidden"
    assert hidden.await_args.args[0] == [str(TASK_ID)]


async def test_freeform_task_attaches_scorecard_per_hotkey(audit) -> None:
    audit.task = QuantFreeformTask(**_fields())
    audit.scorecards = {"A": _scorecard()}

    result = await auditing.get_task_with_hotkey_details(str(TASK_ID), audit.config)

    assert isinstance(result, QuantFreeformTaskWithHotkeyDetails)
    by_hotkey = {d.hotkey: d for d in result.hotkey_details}
    assert by_hotkey["A"].quant_scorecard == _scorecard()
    assert by_hotkey["B"].quant_scorecard is None


async def test_unknown_task_is_404(audit) -> None:
    with pytest.raises(HTTPException) as e:
        await auditing.get_task_with_hotkey_details(str(TASK_ID), audit.config)
    assert e.value.status_code == 404
