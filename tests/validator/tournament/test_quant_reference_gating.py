"""Own-model quant training gating: scored fails get no slot, a miner waits for its bf16 reference, and gets its target."""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.db.constants as db_cst
from core.constants.environments import TrainingStartPoint
from core.models.dataset_models import InstructTextDatasetType
from core.models.quant_models import QuantEvalGpu
from core.models.quant_models import QuantFormat
from core.models.quant_models import QuantTarget
from core.models.task_models import TaskStatus
from validator.scoring.models import QuantSkuReference
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tournament import orchestrator
from validator.tournament.models import TaskTrainingAssignment
from validator.tournament.models import TournamentTaskTraining
from validator.tournament.models import TrainingRepoInfo
from validator.tournament.models import TrainingStatus
from validator.tournament.seed_tasks import NON_SEED_TRAINING_RANK


HOTKEY = "5Miner"
OTHER = "5Other"
CONFIG = SimpleNamespace(psql_db=None)


def _task(cls=QuantFixedRawTask, **overrides):
    fields = {
        "is_organic": False,
        "task_id": uuid.uuid4(),
        "status": "looking_for_nodes",
        "model_id": "org/model",
        "ds": "org/dataset",
        "account_id": uuid.uuid4(),
        "hours_to_complete": 2.0,
        "training_data": "s3://bucket/train.json",
        "created_at": datetime(2026, 10, 19),
        "dataset_type": InstructTextDatasetType(field_instruction="instruction", field_output="output"),
        "training_start_point": TrainingStartPoint.OWN_R1_MODEL,
        **overrides,
    }
    if cls is QuantFixedRawTask:
        fields["quant_spec"] = QuantFormat.W4A8_FP8_G128.spec()
    return cls(**fields)


def _reference(task, hotkey=HOTKEY, sku="h100-sxm", loss=2.0, measured_at=datetime(2026, 10, 20)) -> QuantSkuReference:
    return QuantSkuReference(
        task_id=task.task_id,
        hotkey=hotkey,
        sku=sku,
        model="miner/own-bf16",
        loss=loss,
        per_example_losses=[loss],
        eval_set_fingerprint="fp",
        eval_gpu=QuantEvalGpu(name="NVIDIA H100", sm_count=132, sku=sku),
        measured_at=measured_at,
    )


@pytest.fixture
def references(monkeypatch):
    """The stored quant_sku_references of whatever task is asked about."""
    store: dict[uuid.UUID, list[QuantSkuReference]] = {}

    async def get_references(task_id, psql_db):
        return {row.key: row for row in store.get(task_id, [])}

    monkeypatch.setattr(orchestrator, "get_quant_sku_references", get_references)
    return store


class TestQuantTargetFor:
    async def test_freeform_target_carries_the_bar(self, references):
        task = _task(QuantFreeformRawTask)
        references[task.task_id] = [_reference(task, loss=2.0)]

        target = await orchestrator._quant_target_for(task, HOTKEY, CONFIG)

        assert target is not None
        assert (target.reference_model, target.reference_loss, target.reference_sku) == ("miner/own-bf16", 2.0, "h100-sxm")
        assert target.bar_cap == 0.04 and target.bar == pytest.approx(2.0 * 1.04)

    async def test_fixed_target_has_no_bar(self, references):
        task = _task()
        references[task.task_id] = [_reference(task)]

        target = await orchestrator._quant_target_for(task, HOTKEY, CONFIG)

        assert target == QuantTarget(reference_model="miner/own-bf16", reference_loss=2.0, reference_sku="h100-sxm")

    async def test_none_without_a_row(self, references):
        assert await orchestrator._quant_target_for(_task(), HOTKEY, CONFIG) is None

    async def test_other_miners_rows_are_not_used(self, references):
        task = _task()
        references[task.task_id] = [_reference(task, hotkey=OTHER)]

        assert await orchestrator._quant_target_for(task, HOTKEY, CONFIG) is None

    async def test_earliest_measured_sku_wins(self, references):
        task = _task()
        references[task.task_id] = [
            _reference(task, sku="h100-pcie", loss=3.0, measured_at=datetime(2026, 10, 22)),
            _reference(task, sku="h100-sxm", loss=2.0, measured_at=datetime(2026, 10, 20)),
        ]

        target = await orchestrator._quant_target_for(task, HOTKEY, CONFIG)

        assert target.reference_sku == "h100-sxm"

    async def test_task_common_task_without_a_row_has_no_target(self, references):
        task = _task(training_start_point=TrainingStartPoint.DEFAULT)
        references[task.task_id] = [_reference(task, hotkey=OTHER)]

        assert await orchestrator._quant_target_for(task, HOTKEY, CONFIG) is None

    async def test_task_common_row_is_the_target_for_every_miner(self, references):
        task = _task(training_start_point=TrainingStartPoint.DEFAULT)
        references[task.task_id] = [_reference(task, hotkey="")]

        assert (await orchestrator._quant_target_for(task, HOTKEY, CONFIG)).reference_loss == 2.0


class TestNoTrainingRowForScoredFails:
    @pytest.fixture
    def training_sql(self, monkeypatch):
        add_pairs = AsyncMock()
        failed: set[str] = set()

        async def node_losses(task_id, psql_db):
            return [
                {db_cst.HOTKEY: h, db_cst.QUANT_FAILURE: "no usable R1 model" if h in failed else None} for h in (HOTKEY, OTHER)
            ]

        monkeypatch.setattr(
            orchestrator.task_sql,
            "get_nodes_assigned_to_task",
            AsyncMock(return_value=[SimpleNamespace(hotkey=HOTKEY), SimpleNamespace(hotkey=OTHER)]),
        )
        monkeypatch.setattr(orchestrator.submissions_and_scoring_sql, "get_task_node_losses", node_losses)
        monkeypatch.setattr(orchestrator.task_sql, "get_expected_repo_name", AsyncMock(return_value=None))
        monkeypatch.setattr(orchestrator.task_sql, "update_task", AsyncMock())
        monkeypatch.setattr(orchestrator.tournament_sql, "get_tournament_id_by_task_id", AsyncMock(return_value="tourn"))
        monkeypatch.setattr(
            orchestrator.tournament_sql,
            "get_tournament_training_repo_and_commit",
            AsyncMock(return_value=TrainingRepoInfo("https://github.com/x/repo", "c" * 40, None, None)),
        )
        monkeypatch.setattr(orchestrator.tournament_sql, "add_tournament_task_hotkey_pairs_for_training", add_pairs)
        return SimpleNamespace(add_pairs=add_pairs, failed=failed)

    def _hotkeys(self, training_sql) -> list[str]:
        assignments: list[TaskTrainingAssignment] = training_sql.add_pairs.await_args.args[0]
        return [assignment.hotkey for assignment in assignments]

    async def test_failed_hotkey_gets_no_row(self, training_sql):
        training_sql.failed.add(OTHER)

        await orchestrator._process_tasks_for_training([_task()], CONFIG, priority=2)

        assert self._hotkeys(training_sql) == [HOTKEY]

    async def test_task_common_task_keeps_every_row(self, training_sql):
        training_sql.failed.add(OTHER)

        await orchestrator._process_tasks_for_training(
            [_task(training_start_point=TrainingStartPoint.DEFAULT)], CONFIG, priority=2
        )

        assert self._hotkeys(training_sql) == [HOTKEY, OTHER]


class TestEveryOwnModelUnusable:
    @pytest.fixture
    def sql(self, monkeypatch):
        failed: set[str] = set()

        async def node_losses(task_id, psql_db):
            return [
                {db_cst.HOTKEY: h, db_cst.QUANT_FAILURE: "no usable R1 model" if h in failed else None} for h in (HOTKEY, OTHER)
            ]

        update_task = AsyncMock()
        add_pairs = AsyncMock()
        monkeypatch.setattr(
            orchestrator.task_sql,
            "get_nodes_assigned_to_task",
            AsyncMock(return_value=[SimpleNamespace(hotkey=HOTKEY), SimpleNamespace(hotkey=OTHER)]),
        )
        monkeypatch.setattr(orchestrator.submissions_and_scoring_sql, "get_task_node_losses", node_losses)
        monkeypatch.setattr(orchestrator.task_sql, "update_task", update_task)
        monkeypatch.setattr(orchestrator.tournament_sql, "get_tournament_id_by_task_id", AsyncMock(return_value="tourn"))
        monkeypatch.setattr(
            orchestrator.tournament_sql,
            "get_tournament_training_repo_and_commit",
            AsyncMock(return_value=TrainingRepoInfo("https://github.com/x/repo", "c" * 40, None, None)),
        )
        monkeypatch.setattr(orchestrator.tournament_sql, "add_tournament_task_hotkey_pairs_for_training", add_pairs)
        return SimpleNamespace(update_task=update_task, add_pairs=add_pairs, failed=failed)

    async def test_all_failed_fails_the_task_without_training_rows(self, sql):
        sql.failed.update({HOTKEY, OTHER})
        task = _task()

        await orchestrator._process_tasks_for_training([task], CONFIG, priority=2)

        assert task.status == TaskStatus.FAILURE
        sql.update_task.assert_awaited_once()
        sql.add_pairs.assert_not_awaited()

    async def test_some_failed_still_trains_the_rest(self, sql):
        sql.failed.add(OTHER)
        task = _task()

        await orchestrator._process_tasks_for_training([task], CONFIG, priority=2)

        assert task.status == TaskStatus.TRAINING
        assignments: list[TaskTrainingAssignment] = sql.add_pairs.await_args.args[0]
        assert [assignment.hotkey for assignment in assignments] == [HOTKEY]


class TestSchedulingWaitsForTheReference:
    @pytest.fixture
    def scheduling(self, monkeypatch):
        mocks = SimpleNamespace(
            create_request=AsyncMock(return_value=SimpleNamespace()),
            start=AsyncMock(return_value=True),
            check_gpus=AsyncMock(return_value=("1.2.3.4", [0], "NVLINK")),
            set_status=AsyncMock(),
        )
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value="tourn"))
        monkeypatch.setattr(
            orchestrator,
            "training_ranks",
            AsyncMock(side_effect=lambda task_ids, _db: {task_id: NON_SEED_TRAINING_RANK for task_id in task_ids}),
        )
        monkeypatch.setattr(orchestrator, "_update_all_trainers_gpu_availability", AsyncMock())
        monkeypatch.setattr(orchestrator, "_check_suitable_gpus", mocks.check_gpus)
        monkeypatch.setattr(orchestrator, "_create_training_request", mocks.create_request)
        monkeypatch.setattr(orchestrator, "start_training_task", mocks.start)
        monkeypatch.setattr(orchestrator.tournament_sql, "update_tournament_task_training_status", mocks.set_status)
        monkeypatch.setattr(orchestrator.tournament_sql, "update_gpu_availability", AsyncMock())
        monkeypatch.setattr(orchestrator.gpu_cost_sql, "start_cost_run", AsyncMock())
        monkeypatch.setattr(orchestrator.asyncio, "sleep", AsyncMock())
        return mocks

    def _training(self, task, hotkey=HOTKEY) -> TournamentTaskTraining:
        return TournamentTaskTraining(
            task=task,
            hotkey=hotkey,
            training_status=TrainingStatus.PENDING,
            n_training_attempts=0,
            created_at=datetime(2026, 10, 19),
            updated_at=datetime(2026, 10, 19),
            training_repo="https://github.com/x/repo",
            training_commit_hash="c" * 40,
        )

    async def test_pair_without_a_reference_is_deferred_before_any_gpu_lookup(self, scheduling, references):
        pending = [self._training(_task())]

        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        scheduling.check_gpus.assert_not_called()
        scheduling.start.assert_not_called()
        scheduling.set_status.assert_not_called()
        assert pending == []

    async def test_pair_is_scheduled_once_its_reference_exists(self, scheduling, references):
        task = _task()
        references[task.task_id] = [_reference(task)]

        await orchestrator.schedule_tasks_for_training([self._training(task)], CONFIG)

        scheduling.start.assert_awaited_once()

    async def test_only_the_pair_without_a_row_waits(self, scheduling, references):
        task = _task()
        references[task.task_id] = [_reference(task, hotkey=HOTKEY)]
        pending = [self._training(task, OTHER), self._training(task, HOTKEY)]  # the last one is taken first

        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        assert scheduling.start.await_count == 1
        assert scheduling.create_request.await_args.args[1] == HOTKEY

    async def test_task_common_task_waits_for_the_common_row_not_a_miners(self, scheduling, references):
        task = _task(training_start_point=TrainingStartPoint.DEFAULT)
        pending = [self._training(task)]
        references[task.task_id] = [_reference(task, hotkey=HOTKEY)]  # a per-miner row is not the task-common one

        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        scheduling.start.assert_not_called()
        assert pending == []

    async def test_task_common_task_is_scheduled_once_the_common_row_exists(self, scheduling, references):
        task = _task(training_start_point=TrainingStartPoint.DEFAULT)
        references[task.task_id] = [_reference(task, hotkey="")]

        await orchestrator.schedule_tasks_for_training([self._training(task, OTHER), self._training(task, HOTKEY)], CONFIG)

        assert scheduling.start.await_count == 2
