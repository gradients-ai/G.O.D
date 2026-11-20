"""quant_sku_references persistence: first writer wins per (task, hotkey, SKU), rows keyed by (hotkey, SKU) (mocked DB)."""

import math
import uuid
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import validator.db.constants as cst
from core.models.quant_models import QuantEvalGpu
from validator.db.sql import quant_references as references_sql
from validator.scoring.models import QuantSkuReference


TASK_ID = uuid.uuid4()
PCIE = QuantEvalGpu(name="NVIDIA H100 PCIe", sm_count=114, sku="H100 PCIe")
MEASURED_AT = datetime(2026, 10, 27, 12)


def _mock_psql(*, fetchrow=None, fetch=None):
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow)
    conn.fetch = AsyncMock(return_value=fetch or [])
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    psql = MagicMock()
    psql.connection = AsyncMock(return_value=cm)
    return psql, conn


def _reference(loss: float = 1.41, hotkey: str = "") -> QuantSkuReference:
    return QuantSkuReference(
        task_id=TASK_ID,
        hotkey=hotkey,
        sku=PCIE.reference_key,
        model="org/base",
        loss=loss,
        per_example_losses=[loss, loss],
        eval_set_fingerprint="fp",
        eval_gpu=PCIE,
    )


def _row(reference: QuantSkuReference) -> dict:
    return {
        cst.TASK_ID: reference.task_id,
        cst.HOTKEY: reference.hotkey,
        cst.SKU: reference.sku,
        cst.REFERENCE_MODEL: reference.model,
        cst.LOSS: reference.loss,
        cst.PER_EXAMPLE_LOSSES: reference.per_example_losses,
        cst.EVAL_SET_FINGERPRINT: reference.eval_set_fingerprint,
        cst.EVAL_GPU: reference.eval_gpu.model_dump(),
        cst.MEASURED_AT: MEASURED_AT,
    }


async def test_add_never_overwrites_and_returns_the_stored_row():
    stored = _reference(1.41)
    psql, conn = _mock_psql(fetchrow=_row(stored))

    returned = await references_sql.add_quant_sku_reference(_reference(1.42), psql)

    query, *args = conn.execute.await_args.args
    assert f"ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}, {cst.SKU}) DO NOTHING" in query
    # The vector and GPU bind directly through the jsonb codec.
    assert args == [TASK_ID, "", "H100 PCIe", "org/base", 1.42, [1.42, 1.42], "fp", PCIE.model_dump()]
    assert returned.loss == 1.41
    assert not returned.matches(_reference(1.42))
    assert returned.matches(_reference(1.41))


async def test_add_is_first_writer_wins_per_hotkey_and_rereads_by_the_same_key():
    stored = _reference(1.41, hotkey="miner")
    psql, conn = _mock_psql(fetchrow=_row(stored))

    returned = await references_sql.add_quant_sku_reference(_reference(1.42, hotkey="miner"), psql)

    insert_args = conn.execute.await_args.args[1:]
    assert insert_args[:3] == (TASK_ID, "miner", "H100 PCIe")
    select_query, *select_args = conn.fetchrow.await_args.args
    assert f"{cst.HOTKEY} = $2" in select_query
    assert select_args == [TASK_ID, "miner", "H100 PCIe"]
    assert returned.hotkey == "miner" and returned.loss == 1.41


async def test_get_returns_the_cache_by_hotkey_and_reference_key():
    common, own = _reference(), _reference(1.5, hotkey="miner")
    psql, conn = _mock_psql(fetch=[_row(common), _row(own)])

    references = await references_sql.get_quant_sku_references(TASK_ID, psql)

    assert list(references) == [("", "H100 PCIe"), ("miner", "H100 PCIe")]
    assert references[("", "H100 PCIe")].eval_gpu == PCIE
    assert references[("", "H100 PCIe")].measured_at == MEASURED_AT
    assert references[("miner", "H100 PCIe")].loss == 1.5
    assert conn.fetch.await_args.args[1] == TASK_ID


def _reference_with_unscored_example(loss: float = 1.41) -> QuantSkuReference:
    """The scorer keeps NaN in place for an example with no supervised token."""
    return _reference(loss).model_copy(update={"per_example_losses": [loss, math.nan, loss]})


async def test_unscored_example_binds_as_null_and_reads_back_as_nan():
    # json.dumps would emit a bare NaN token, which PostgreSQL rejects and would fail the INSERT, and
    # with it the whole evaluation batch - as set_task_node_losses already guards for a submission.
    stored_row = _row(_reference_with_unscored_example())
    stored_row[cst.PER_EXAMPLE_LOSSES] = [1.41, None, 1.41]
    psql, conn = _mock_psql(fetchrow=stored_row, fetch=[stored_row])

    returned = await references_sql.add_quant_sku_reference(_reference_with_unscored_example(), psql)
    bound_losses = conn.execute.await_args.args[6]
    assert bound_losses == [1.41, None, 1.41]
    assert returned.per_example_losses[0] == 1.41 and math.isnan(returned.per_example_losses[1])

    references = await references_sql.get_quant_sku_references(TASK_ID, psql)
    assert math.isnan(references[("", "H100 PCIe")].per_example_losses[1])


def test_matches_treats_two_unscored_examples_as_the_same_entry():
    measured = _reference_with_unscored_example()
    assert measured.matches(_reference_with_unscored_example())
    assert not measured.matches(_reference(1.41))  # a vector of another length
    assert not measured.matches(measured.model_copy(update={"per_example_losses": [1.41, math.nan, 1.42]}))
    assert not measured.matches(measured.model_copy(update={"per_example_losses": [1.41, 1.41, 1.41]}))
