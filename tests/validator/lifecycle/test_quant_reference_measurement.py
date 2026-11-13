"""ensure_quant_references: every scored quant row needs the bf16 reference cached for its own SKU."""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.db.constants as db_cst
from core.constants.environments import TrainingStartPoint
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.quant_models import QuantEvalGpu
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from validator.lifecycle import quant_reference
from validator.lifecycle import tasks as lifecycle_tasks
from validator.scoring.models import QuantSkuReference
from validator.scoring.quant_ranking import QuantReferenceUnavailable
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tournament.models import BOSS_ROUND_SEED_ROLES
from validator.tournament.models import SeedRole


CONFIG = SimpleNamespace(psql_db="db")
SXM = QuantEvalGpu(name="NVIDIA H100 80GB HBM3", sm_count=132, sku="H100 SXM")
PCIE = QuantEvalGpu(name="NVIDIA H100 PCIe", sm_count=114, sku="H100 PCIe")
OVERRIDE = QuantEvalGpu(name="NVIDIA A100-SXM4-80GB", sm_count=108, override=True)


def _fields(**overrides) -> dict:
    defaults = dict(
        task_id=uuid.uuid4(),
        is_organic=False,
        status=TaskStatus.EVALUATING,
        model_id="org/base",
        ds="org/data",
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 10, 19),
        test_data="https://minio/quant_eval_set.jsonl",
    )
    return {**defaults, **overrides}


def _freeform(**overrides) -> QuantFreeformRawTask:
    return QuantFreeformRawTask(**_fields(**overrides), dataset_type=ChatTemplateDatasetType())


def _fixed(**overrides) -> QuantFixedRawTask:
    return QuantFixedRawTask(
        **_fields(**overrides), dataset_type=ChatTemplateDatasetType(), quant_spec=QuantFormat.W4A16_SYM_G128.spec()
    )


def _reference(gpu: QuantEvalGpu, hotkey: str = "") -> QuantSkuReference:
    return QuantSkuReference(
        task_id=uuid.uuid4(),
        hotkey=hotkey,
        sku=gpu.reference_key,
        model="org/base",
        loss=1.5,
        per_example_losses=[1.5],
        eval_set_fingerprint="fp",
        eval_gpu=gpu,
    )


def _row(hotkey: str, gpu: QuantEvalGpu | None, test_loss: float | None = 1.6) -> dict:
    return {db_cst.HOTKEY: hotkey, db_cst.TEST_LOSS: test_loss, db_cst.EVAL_GPU: gpu.model_dump() if gpu else None}


@pytest.fixture
def seams(monkeypatch):
    """Cache, raw-row, reference-model, SQL and alert seams."""
    state = SimpleNamespace(rows=[], cached={}, model="org/base")
    state.set_status = AsyncMock()
    state.clear = AsyncMock()
    state.update_task = AsyncMock()
    state.notify = AsyncMock()
    monkeypatch.setattr(quant_reference, "get_task_node_losses", AsyncMock(side_effect=lambda *_: state.rows))
    monkeypatch.setattr(quant_reference, "get_quant_sku_references", AsyncMock(side_effect=lambda *_: state.cached))
    monkeypatch.setattr(quant_reference, "quant_reference_model", AsyncMock(side_effect=lambda *_: state.model))
    monkeypatch.setattr(quant_reference.tasks_sql, "update_task_evaluations_status", state.set_status)
    monkeypatch.setattr(quant_reference.tasks_sql, "clear_task_evaluation_deployments", state.clear)
    monkeypatch.setattr(quant_reference.tasks_sql, "update_task", state.update_task)
    monkeypatch.setattr(quant_reference, "notify_evaluation_exception", state.notify)
    monkeypatch.setattr(quant_reference, "_QUANT_REFERENCE_ALERTED", set())
    return state


@pytest.mark.parametrize("make_task", [_fixed, _freeform])
async def test_every_row_with_its_skus_reference_is_ready(seams, make_task):
    seams.rows = [_row("on_sxm", SXM), _row("on_pcie", PCIE), _row("scored_fail", PCIE, test_loss=None)]
    seams.cached = {("", "H100 SXM"): _reference(SXM), ("", "H100 PCIe"): _reference(PCIE)}

    assert await quant_reference.ensure_quant_references(make_task(), ["on_sxm", "on_pcie", "scored_fail"], CONFIG)

    seams.set_status.assert_not_awaited()


@pytest.mark.parametrize("gpu", [PCIE, None, OVERRIDE], ids=["uncached-sku", "no-eval-gpu", "override-gpu"])
async def test_a_row_without_its_skus_reference_is_re_evaluated(seams, gpu):
    seams.rows = [_row("cached", SXM), _row("missing", gpu)]
    seams.cached = {("", "H100 SXM"): _reference(SXM)}
    task = _fixed()

    assert await quant_reference.ensure_quant_references(task, ["cached", "missing"], CONFIG) is False

    seams.set_status.assert_awaited_once_with(task.task_id, ["missing"], "pending", "db")
    seams.clear.assert_awaited_once_with(task.task_id, ["missing"], "db")
    assert task.n_eval_attempts == 1
    seams.update_task.assert_awaited_once_with(task, "db")
    seams.notify.assert_awaited_once()


async def test_an_override_row_is_ready_with_a_reference_from_the_same_unlisted_gpu(seams):
    seams.rows = [_row("test_box", OVERRIDE)]
    seams.cached = {("", OVERRIDE.reference_key): _reference(OVERRIDE)}

    assert await quant_reference.ensure_quant_references(_fixed(), ["test_box"], CONFIG) is True


def _own_model(**overrides) -> QuantFreeformRawTask:
    return _freeform(training_start_point=TrainingStartPoint.OWN_R1_MODEL, **overrides)


async def test_own_model_rows_are_looked_up_by_their_own_hotkey(seams):
    seams.rows = [_row("a", SXM), _row("b", SXM)]
    seams.cached = {("a", "H100 SXM"): _reference(SXM, "a")}
    task = _own_model()

    assert await quant_reference.ensure_quant_references(task, ["a", "b"], CONFIG) is False

    seams.set_status.assert_awaited_once_with(task.task_id, ["b"], "pending", "db")


async def test_own_model_rows_do_not_use_a_task_common_reference_of_another_hotkey(seams):
    seams.rows = [_row("a", SXM), _row("b", SXM)]
    seams.cached = {("a", "H100 SXM"): _reference(SXM, "a"), ("b", "H100 SXM"): _reference(SXM, "b")}

    assert await quant_reference.ensure_quant_references(_own_model(), ["a", "b"], CONFIG) is True


async def test_own_model_miner_without_a_reference_model_is_a_persisted_scored_fail(seams, monkeypatch):
    seams.rows = [_row("a", SXM), _row("b", SXM)]
    seams.cached = {}
    models = {"a": "org/a-r1"}
    monkeypatch.setattr(
        quant_reference, "quant_reference_model", AsyncMock(side_effect=lambda _task, hotkey, _db: models.get(hotkey))
    )
    persist = AsyncMock()
    monkeypatch.setattr(quant_reference, "set_task_node_quant_failure", persist)
    task = _own_model()

    assert await quant_reference.ensure_quant_references(task, ["a", "b"], CONFIG) is False

    persist.assert_awaited_once_with(task.task_id, "b", quant_reference.NO_REFERENCE_MODEL_FAILURE, "db")
    # a's missing reference is still checked and re-evaluated; b is neither checked nor re-evaluated.
    seams.set_status.assert_awaited_once_with(task.task_id, ["a"], "pending", "db")


async def test_own_model_task_proceeds_when_the_only_other_miner_has_its_reference(seams, monkeypatch):
    seams.rows = [_row("a", SXM), _row("b", SXM)]
    seams.cached = {("a", "H100 SXM"): _reference(SXM, "a")}
    models = {"a": "org/a-r1"}
    monkeypatch.setattr(
        quant_reference, "quant_reference_model", AsyncMock(side_effect=lambda _task, hotkey, _db: models.get(hotkey))
    )
    monkeypatch.setattr(quant_reference, "set_task_node_quant_failure", AsyncMock())

    assert await quant_reference.ensure_quant_references(_own_model(), ["a", "b"], CONFIG) is True


async def test_failed_evaluations_need_no_reference(seams):
    seams.rows = [_row("failed", PCIE)]

    assert await quant_reference.ensure_quant_references(_freeform(), [], CONFIG) is True


async def test_spent_retries_make_the_reference_unavailable(seams):
    seams.rows = [_row("missing", PCIE)]

    with pytest.raises(QuantReferenceUnavailable, match="missing"):
        await quant_reference.ensure_quant_references(_fixed(n_eval_attempts=1), ["missing"], CONFIG)

    seams.set_status.assert_not_awaited()


async def test_no_reference_model_is_unavailable(seams):
    seams.model = None

    with pytest.raises(QuantReferenceUnavailable, match="no bf16 reference model"):
        await quant_reference.ensure_quant_references(_freeform(), [], CONFIG)


async def test_the_alert_fires_once_per_task(seams):
    seams.rows = [_row("missing", PCIE)]
    task = _fixed()

    await quant_reference.ensure_quant_references(task, ["missing"], CONFIG)
    task.n_eval_attempts = 0
    await quant_reference.ensure_quant_references(task, ["missing"], CONFIG)

    seams.notify.assert_awaited_once()


@pytest.fixture
def finalize(monkeypatch):
    """A tournament quant task whose single entrant trained and evaluated successfully."""
    seams = SimpleNamespace(finalize=AsyncMock(), ensure=AsyncMock(return_value=False), update_task=AsyncMock())
    monkeypatch.setattr(
        lifecycle_tasks.tournament_sql, "get_training_status_for_task", AsyncMock(return_value={"miner": "success"})
    )
    monkeypatch.setattr(
        lifecycle_tasks.tasks_sql,
        "get_task_evaluation_rows",
        AsyncMock(return_value=[{"hotkey": "miner", "evaluation_status": "success"}]),
    )
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "update_task", seams.update_task)
    monkeypatch.setattr(lifecycle_tasks, "finalize_task_scores_from_raw_losses", seams.finalize)
    monkeypatch.setattr(lifecycle_tasks, "ensure_quant_references", seams.ensure)
    monkeypatch.setattr(lifecycle_tasks.seeds_sql, "get_seed_task_by_task_id", AsyncMock(return_value=None))
    seams.notify = AsyncMock()
    monkeypatch.setattr(lifecycle_tasks.boss_seed_eval, "notify_evaluation_exception", seams.notify)
    return seams


async def test_finalize_defers_while_references_are_missing(finalize):
    task = _freeform()

    assert await lifecycle_tasks._finalize_task_status_from_evaluations(task, CONFIG) is False

    finalize.ensure.assert_awaited_once_with(task, ["miner"], CONFIG)
    finalize.finalize.assert_not_awaited()


async def test_finalize_proceeds_once_references_are_ready(finalize):
    finalize.ensure.return_value = True
    task = _freeform()

    assert await lifecycle_tasks._finalize_task_status_from_evaluations(task, CONFIG) is True

    finalize.finalize.assert_awaited_once_with(task, ["miner"], CONFIG)
    assert task.status == TaskStatus.SUCCESS


async def test_an_unavailable_reference_swaps_the_task(finalize):
    finalize.ensure.side_effect = QuantReferenceUnavailable("no bf16 reference model")
    task = _freeform()

    assert await lifecycle_tasks._finalize_task_status_from_evaluations(task, CONFIG) is True

    finalize.finalize.assert_not_awaited()
    assert task.status == TaskStatus.PREP_TASK_FAILURE
    finalize.update_task.assert_awaited_once_with(task, "db")


@pytest.mark.parametrize("role", sorted(BOSS_ROUND_SEED_ROLES, key=lambda role: role.value))
async def test_an_unavailable_reference_holds_a_boss_round_seed_instead_of_failing_it(finalize, monkeypatch, role):
    seed = SimpleNamespace(role=role)
    monkeypatch.setattr(lifecycle_tasks.seeds_sql, "get_seed_task_by_task_id", AsyncMock(return_value=seed))
    finalize.ensure.side_effect = QuantReferenceUnavailable("no same-SKU bf16 reference")
    task = _freeform()
    status_before = task.status

    for _ in range(2):
        assert await lifecycle_tasks._finalize_task_status_from_evaluations(task, CONFIG) is False

    assert task.status == status_before
    finalize.update_task.assert_not_awaited()
    finalize.notify.assert_awaited_once()


async def test_an_unavailable_reference_still_swaps_an_input_seed(finalize, monkeypatch):
    seed = SimpleNamespace(role=SeedRole.INPUT_SMALL)
    monkeypatch.setattr(lifecycle_tasks.seeds_sql, "get_seed_task_by_task_id", AsyncMock(return_value=seed))
    finalize.ensure.side_effect = QuantReferenceUnavailable("no bf16 reference model")
    task = _freeform()

    assert await lifecycle_tasks._finalize_task_status_from_evaluations(task, CONFIG) is True
    assert task.status == TaskStatus.PREP_TASK_FAILURE
