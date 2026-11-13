"""The chained freeform bar is the task's common bar on the shared input, and the finalize-time quant_failure write."""

import math
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

import validator.db.constants as db_cst
from core.constants.environments import TrainingStartPoint
from core.models.quant_models import QuantEvalGpu
from core.models.task_models import TaskType
from validator.db.sql import submissions_and_scoring as scoring_sql
from validator.scoring import constants as scoring_cst
from validator.scoring import tasks as scoring_tasks
from validator.scoring.models import MinerResultsText
from validator.scoring.models import QuantSkuReference
from validator.scoring.quant_ranking import scored_fail_reason
from validator.scoring.tasks import finalize_task_scores_from_raw_losses
from validator.tournament import constants as t_cst


CONFIG = SimpleNamespace(psql_db=None)
SXM = QuantEvalGpu(name="NVIDIA H100 80GB HBM3", sm_count=132, sku="H100 SXM")
# The shared input's bf16 loss on SXM.
SXM_REFERENCES = {
    ("", SXM.reference_key): QuantSkuReference(
        task_id=uuid.uuid4(),
        sku=SXM.reference_key,
        model="org/shared-input",
        loss=2.0,
        per_example_losses=[2.0],
        eval_set_fingerprint="fp",
        eval_gpu=SXM,
    )
}


def _freeform_task():
    return SimpleNamespace(
        training_start_point=TrainingStartPoint.DEFAULT,
        task_id=uuid.uuid4(),
        task_type=TaskType.QUANTFREEFORMTASK,
        bar_cap=0.05,
        penalty_knee=None,
        penalty_curvature=None,
    )


def _row(hotkey, test_loss, artifact_bytes=1000, quant_failure=None):
    return {
        db_cst.HOTKEY: hotkey,
        db_cst.TEST_LOSS: test_loss,
        db_cst.SYNTH_LOSS: test_loss,
        db_cst.SCORE_REASON: None,
        db_cst.ARTIFACT_BYTES: artifact_bytes,
        db_cst.DECODER_BYTES: 0,
        db_cst.PASSED_BAR: None,
        db_cst.QUANT_FAILURE: quant_failure,
        db_cst.EVAL_GPU: SXM.model_dump(),
    }


def _result(hotkey, test_loss=2.0, quant_failure=None):
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=test_loss,
        synth_loss=test_loss,
        is_finetune=True,
        task_type=TaskType.QUANTFREEFORMTASK,
        artifact_bytes=1000,
        decoder_bytes=0,
        quant_failure=quant_failure,
        reference_loss=2.0,
    )


@pytest.fixture
def persisted(monkeypatch):
    """Patch the raw-loss and reference reads and the score write; returns the rows to fill."""
    rows: list[dict] = []
    monkeypatch.setattr(scoring_tasks, "get_task_node_losses", AsyncMock(side_effect=lambda *_: rows))
    monkeypatch.setattr(scoring_tasks, "get_quant_sku_references", AsyncMock(return_value=SXM_REFERENCES))
    monkeypatch.setattr(scoring_tasks, "_update_scores", AsyncMock())
    return rows


class TestChainedFinalize:
    async def test_every_player_is_held_to_the_shared_input_bar(self, persisted) -> None:
        # One bf16 reference of 2.0 on the shared input: a 2.1 bar for both players.
        persisted.extend([_row("boss", 2.08, artifact_bytes=9000), _row("challenger", 2.12, artifact_bytes=10)])

        results = await finalize_task_scores_from_raw_losses(_freeform_task(), ["boss", "challenger"], CONFIG)

        by_hotkey = {result.hotkey: result for result in results}
        assert (by_hotkey["boss"].passed_bar, by_hotkey["challenger"].passed_bar) == (True, False)
        assert by_hotkey["boss"].score == scoring_cst.FIRST_PLACE_SCORE

    async def test_smaller_pass_wins_under_the_common_bar(self, persisted) -> None:
        persisted.extend([_row("boss", 2.0, artifact_bytes=9000), _row("challenger", 2.09, artifact_bytes=10)])

        results = await finalize_task_scores_from_raw_losses(_freeform_task(), ["boss", "challenger"], CONFIG)

        assert {result.hotkey: result.rank for result in results} == {"challenger": 1, "boss": 2}

    async def test_upstream_failed_player_is_a_scored_fail(self, persisted) -> None:
        persisted.extend([_row("ok", 2.05), _row("failed", None, quant_failure=t_cst.UPSTREAM_FAILURE_REASON)])

        results = await finalize_task_scores_from_raw_losses(_freeform_task(), ["ok", "failed"], CONFIG)

        by_hotkey = {result.hotkey: result for result in results}
        assert by_hotkey["ok"].rank == 1
        assert by_hotkey["failed"].score == scoring_cst.SCORE_PENALTY

    def test_scored_fail_reason_is_the_quant_failure(self) -> None:
        result = _result("failed", test_loss=math.nan, quant_failure="does not load")
        assert scored_fail_reason(result, 0.05) == "does not load"


class TestQuantFailureWrite:
    async def test_sql_never_overwrites_an_existing_failure_with_null(self) -> None:
        conn = MagicMock()
        conn.execute = AsyncMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        psql = MagicMock()
        psql.connection = AsyncMock(return_value=cm)

        await scoring_sql.set_task_node_quality_score(uuid.uuid4(), "hk", 0.0, 2.0, 2.0, psql)

        query, *args = conn.execute.await_args.args
        assert f"{db_cst.QUANT_FAILURE} = COALESCE($9, {db_cst.TASK_NODES_TABLE}.{db_cst.QUANT_FAILURE})" in query
        assert args[8] is None
