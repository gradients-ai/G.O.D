"""A non-finite number in a freeform scorecard must reach the public as JSON null, never as a 500 on the whole task.

A scored fail is persisted with test_loss = NaN. FastAPI serialises responses with allow_nan=False, so the card models
turn NaN and +/-inf into None themselves; the endpoint test goes through the real router, the real
get_task_with_hotkey_details and the real quant_scorecards_for_task and fakes only the SQL layer beneath them.
"""

import math
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.models.dataset_models import InstructTextDatasetType
from core.models.quant_models import QuantEvalGpu
from validator.db import constants as db_cst
from validator.db.sql import auditing as auditing_sql
from validator.endpoints import auditing as auditing_endpoint
from validator.scoring import scorecard as scorecard_module
from validator.scoring.models import QuantScorecardRow
from validator.scoring.models import QuantSkuReference
from validator.scoring.scorecard import build_quant_scorecard
from validator.tasks.models import QuantFreeformTask
from validator.tasks.models import ScorecardBytes
from validator.tasks.models import ScorecardQuality


TASK_ID = uuid.uuid4()
HEALTHY = "5Healthy"
FAILED = "5Failed"
PCIE = QuantEvalGpu(name="NVIDIA H100 PCIe", sm_count=114, sku="H100 PCIe")
REFERENCE_LOSS = 2.0
NON_FINITE = [math.nan, math.inf, -math.inf]


def _row(hotkey: str, **overrides) -> QuantScorecardRow:
    fields = {
        "hotkey": hotkey,
        "test_loss": 2.04,
        "artifact_bytes": 4_001_519,
        "decoder_bytes": 2048,
        "passed_bar": True,
        "freeform_penalty": math.log(1.5),
        "eval_gpu": PCIE,
    }
    return QuantScorecardRow(**{**fields, **overrides})


def _task() -> QuantFreeformTask:
    return QuantFreeformTask(
        task_id=TASK_ID,
        is_organic=False,
        status="success",
        model_id="org/base",
        ds="org/dataset",
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 11, 1),
        test_data="https://bucket/test.jsonl",
        training_data="s3://bucket/train.json",
        dataset_type=InstructTextDatasetType(field_instruction="instruction", field_output="output"),
    )


def _audit_row(hotkey: str, rank: int, test_loss: float) -> dict:
    return {
        db_cst.HOTKEY: hotkey,
        db_cst.SUBMISSION_ID: None,
        db_cst.QUALITY_SCORE: 1.0,
        db_cst.TEST_LOSS: test_loss,
        db_cst.SYNTH_LOSS: test_loss,
        db_cst.SCORE_REASON: None,
        "rank": rank,
        db_cst.REPO: None,
        db_cst.OFFER_RESPONSE: None,
    }


@pytest.fixture
def client(monkeypatch) -> TestClient:
    rows = {
        HEALTHY: _row(HEALTHY),
        FAILED: _row(FAILED, test_loss=math.nan, passed_bar=None, quant_failure="decoder exited with code 1"),
    }
    reference = QuantSkuReference(
        task_id=TASK_ID,
        sku=PCIE.reference_key,
        model="org/base",
        loss=REFERENCE_LOSS,
        per_example_losses=[REFERENCE_LOSS],
        eval_set_fingerprint="fp",
        eval_gpu=PCIE,
    )
    monkeypatch.setattr(auditing_sql.tasks_sql, "get_task_by_id", AsyncMock(return_value=_task()))
    monkeypatch.setattr(auditing_sql.seeds_sql, "get_hidden_seed_task_ids", AsyncMock(return_value=set()))
    monkeypatch.setattr(scorecard_module.scoring_sql, "get_quant_scorecard_rows", AsyncMock(return_value=rows))
    monkeypatch.setattr(
        scorecard_module, "get_quant_sku_references", AsyncMock(return_value={("", PCIE.reference_key): reference})
    )
    monkeypatch.setattr(scorecard_module.tournament_sql, "get_tournament_id_by_task_id", AsyncMock(return_value=None))

    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[_audit_row(HEALTHY, 1, 2.04), _audit_row(FAILED, 2, math.nan)])
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    psql_db = SimpleNamespace(connect=AsyncMock(), connection=AsyncMock(return_value=cm))

    app = FastAPI()
    app.state.config = SimpleNamespace(psql_db=psql_db)
    app.include_router(auditing_endpoint.router)
    return TestClient(app)


def test_scored_fail_with_nan_loss_does_not_500_the_task(client: TestClient) -> None:
    response = client.get(f"/auditing/tasks/{TASK_ID}")

    assert response.status_code == 200
    details = {d["hotkey"]: d["quant_scorecard"] for d in response.json()["hotkey_details"]}
    failed = details[FAILED]["quality"]
    assert failed["test_loss"] is None
    assert failed["ratio"] is None
    assert failed["quant_failure"] == "decoder exited with code 1"
    assert details[FAILED]["bytes"]["total_bytes"] == 4_003_567

    healthy = details[HEALTHY]
    assert healthy["quality"]["test_loss"] == pytest.approx(2.04)
    assert healthy["quality"]["ratio"] == pytest.approx(1.02)
    assert healthy["quality"]["reference_loss"] == REFERENCE_LOSS
    assert healthy["bytes"]["effective_bytes"] == pytest.approx(4_003_567 * 1.5)


@pytest.mark.parametrize("bad", NON_FINITE)
def test_quality_floats_become_none(bad: float) -> None:
    quality = ScorecardQuality(test_loss=bad, reference_loss=bad, ratio=bad, passed_bar=True)

    assert (quality.test_loss, quality.reference_loss, quality.ratio) == (None, None, None)
    assert quality.passed_bar is True
    assert "NaN" not in quality.model_dump_json() and "Infinity" not in quality.model_dump_json()
    assert ScorecardQuality(test_loss=2.5).test_loss == 2.5


@pytest.mark.parametrize("bad", NON_FINITE)
def test_bytes_floats_become_none(bad: float) -> None:
    card = ScorecardBytes(ln_penalty=bad, effective_bytes=bad)

    assert (card.ln_penalty, card.effective_bytes) == (None, None)
    assert ScorecardBytes(ln_penalty=0.5, effective_bytes=3.0).effective_bytes == 3.0


@pytest.mark.parametrize("bad", NON_FINITE)
def test_builder_with_non_finite_penalty_gives_no_effective_bytes(bad: float) -> None:
    card = build_quant_scorecard(_row(HEALTHY, freeform_penalty=bad), None, REFERENCE_LOSS)

    assert card.bytes.ln_penalty is None
    assert card.bytes.effective_bytes is None
    assert card.bytes.total_bytes == 4_003_567


@pytest.mark.parametrize("bad", NON_FINITE)
def test_builder_with_non_finite_loss_and_reference(bad: float) -> None:
    card = build_quant_scorecard(_row(HEALTHY, test_loss=bad, quant_failure="x"), None, bad)

    assert (card.quality.test_loss, card.quality.reference_loss, card.quality.ratio) == (None, None, None)
    assert card.quality.quant_failure == "x"


def test_builder_with_overflowing_penalty_gives_no_effective_bytes() -> None:
    card = build_quant_scorecard(_row(HEALTHY, freeform_penalty=1e6), None, REFERENCE_LOSS)

    assert card.bytes.ln_penalty == 1e6
    assert card.bytes.effective_bytes is None


def test_builder_with_huge_but_finite_product_gives_no_effective_bytes() -> None:
    # exp(709) is finite, the product with the byte total overflows to inf.
    card = build_quant_scorecard(_row(HEALTHY, freeform_penalty=709.0), None, REFERENCE_LOSS)

    assert card.bytes.effective_bytes is None


def test_none_penalty_keeps_factor_one() -> None:
    card = build_quant_scorecard(_row(HEALTHY, freeform_penalty=None), None, REFERENCE_LOSS)

    assert card.bytes.effective_bytes == pytest.approx(4_003_567)
