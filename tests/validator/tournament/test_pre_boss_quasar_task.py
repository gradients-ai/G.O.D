"""The pre-boss knockout (2 competitors left) always plays on quasar.

Guards the routing in _create_probability_based_text_tasks: the pre-boss round is detected by
COMPETITOR count (a small-tournament round 1 also creates a single task, so task count is not a
valid key), and its task is a standard instruct task with only the model forced to the quasar
seed — no KL, no augmentation, normal dataset pull.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from core.constants.environments import TrainingStartPoint
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.tournament import constants as t_cst
from validator.tournament import task_creator
from validator.tournament import tournament_manager
from validator.tournament.models import KnockoutRound


def _knockout(pairs: list[tuple[str, str]]) -> KnockoutRound:
    return KnockoutRound(round_id="tourn_round_003", round_number=3, pairs=pairs)


def _patch_seams(monkeypatch):
    """Stub the pool generators, DB lookups and registration; return the two creation mocks."""
    for name in ("_get_text_models", "_get_instruct_text_datasets", "_get_dpo_datasets"):
        monkeypatch.setattr(task_creator, name, lambda *a, **k: MagicMock())
    monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", AsyncMock())

    quasar_task = SimpleNamespace(task_id="quasar-task", task_type=TaskType.INSTRUCTTEXTTASK)
    instruct_mock = AsyncMock(return_value=quasar_task)
    monkeypatch.setattr(task_creator, "create_synthetic_instruct_text_task", instruct_mock)

    probability_task = SimpleNamespace(task_id="prob-task", task_type=TaskType.DPOTASK)
    probability_mock = AsyncMock(return_value=probability_task)
    monkeypatch.setattr(task_creator, "_create_single_probability_task", probability_mock)
    return instruct_mock, probability_mock


async def test_two_competitors_forces_the_quasar_task(monkeypatch):
    instruct_mock, probability_mock = _patch_seams(monkeypatch)

    tasks = await task_creator._create_probability_based_text_tasks(
        _knockout([("miner-a", "miner-b")]), "tourn", MagicMock()
    )

    probability_mock.assert_not_awaited()
    assert [t.task_id for t in tasks] == ["quasar-task"]
    args, kwargs = instruct_mock.call_args
    assert args[1] is None  # no model pool: the model is forced
    assert kwargs["model_id_override"] == t_cst.PRE_BOSS_QUASAR_MODEL
    assert kwargs["enable_kl"] is False
    assert kwargs["allow_augmentation"] is False
    assert kwargs["allow_yarn"] is False


async def test_more_than_two_competitors_keeps_probability_routing(monkeypatch):
    instruct_mock, probability_mock = _patch_seams(monkeypatch)

    tasks = await task_creator._create_probability_based_text_tasks(
        _knockout([("miner-a", "miner-b"), ("miner-c", "miner-d")]), "tourn", MagicMock()
    )

    instruct_mock.assert_not_awaited()
    assert probability_mock.await_count == 2
    assert len(tasks) == 2


class TestReplacementRouting:
    """Prep-failure replacement must preserve forced-model tasks: continuous-SFT recreates the
    same lineage (same carried base, chunk re-materialized), and the pre-boss quasar task
    re-forces the seed with everything else fresh. Neither may fall through to
    create_new_task_of_same_type, which draws a random model (and has no CHATTASK route at all)."""

    def _patch_replace_seams(self, monkeypatch, original):
        monkeypatch.setattr(task_creator.task_sql, "get_task", AsyncMock(return_value=original))
        monkeypatch.setattr(task_creator.task_sql, "get_nodes_assigned_to_task", AsyncMock(return_value=[]))
        monkeypatch.setattr(task_creator.task_sql, "delete_task", AsyncMock())
        monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", AsyncMock())
        monkeypatch.setattr(task_creator, "_get_instruct_text_datasets", lambda *a, **k: MagicMock())
        same_type_mock = AsyncMock()
        monkeypatch.setattr(task_creator, "create_new_task_of_same_type", same_type_mock)
        return same_type_mock

    async def test_continuous_sft_replacement_recreates_the_same_lineage(self, monkeypatch):
        original = SimpleNamespace(
            task_id="orig-task",
            task_type=TaskType.CHATTASK,
            training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
            ds="continuous-sft:qwen:chunk-00003",
            status=TaskStatus.PREP_TASK_FAILURE.value,
            model_id="miner-org/carried-winner",
            model_params_count=0,
        )
        same_type_mock = self._patch_replace_seams(monkeypatch, original)
        recreate_mock = AsyncMock(return_value=SimpleNamespace(task_id="new-task", task_type=TaskType.CHATTASK))
        monkeypatch.setattr(task_creator, "create_continuous_sft_task", recreate_mock)

        new_task_id = await task_creator.replace_tournament_task(
            "orig-task", "tourn", "round-4", None, "pair-1", MagicMock()
        )

        same_type_mock.assert_not_awaited()
        assert new_task_id == "new-task"
        _, lineage, seed_model = recreate_mock.call_args.args
        assert lineage == "qwen"
        assert seed_model == t_cst.CONTINUOUS_SFT_LINEAGES["qwen"]

    async def test_pre_boss_quasar_replacement_reforces_the_seed_model(self, monkeypatch):
        original = SimpleNamespace(
            task_id="orig-task",
            task_type=TaskType.INSTRUCTTEXTTASK,
            training_start_point=TrainingStartPoint.DEFAULT,
            ds="tatsu-lab/alpaca",
            status=TaskStatus.PREP_TASK_FAILURE.value,
            model_id=t_cst.PRE_BOSS_QUASAR_MODEL,
            model_params_count=0,
        )
        same_type_mock = self._patch_replace_seams(monkeypatch, original)
        instruct_mock = AsyncMock(return_value=SimpleNamespace(task_id="new-task", task_type=TaskType.INSTRUCTTEXTTASK))
        monkeypatch.setattr(task_creator, "create_synthetic_instruct_text_task", instruct_mock)

        new_task_id = await task_creator.replace_tournament_task(
            "orig-task", "tourn", "round-3", None, "pair-1", MagicMock()
        )

        same_type_mock.assert_not_awaited()
        assert new_task_id == "new-task"
        assert instruct_mock.call_args.kwargs["model_id_override"] == t_cst.PRE_BOSS_QUASAR_MODEL
        assert instruct_mock.call_args.kwargs["allow_augmentation"] is False


class TestPreBossBothMinersFailed:
    """When both pre-boss miners fail the quasar task, the round must COMPLETE (not stall for
    investigation): with no positive quality scores get_task_winner yields None, winners come back
    empty, and advance_tournament's zero-winner path retains the boss with no boss round and no
    emission shift. These tests pin the completion gate in is_tourn_task_completed."""

    def _tournament_task(self):
        return SimpleNamespace(task_id="task-1", round_id="round-3", tournament_id="tourn", group_id=None, pair_id="p1")

    def _task_obj(self, status, model_id=t_cst.PRE_BOSS_QUASAR_MODEL):
        return SimpleNamespace(
            task_id="task-1", status=status, task_type=TaskType.INSTRUCTTEXTTASK, model_id=model_id
        )

    def _config(self):
        return SimpleNamespace(psql_db=MagicMock(), discord_url=None)

    def _patch_trainings(self, monkeypatch, statuses: dict[str, str]):
        monkeypatch.setattr(tournament_manager, "get_training_status_for_task", AsyncMock(return_value=statuses))

    async def test_both_failed_completes_round_on_task_success(self, monkeypatch):
        self._patch_trainings(monkeypatch, {"miner-a": "failure", "miner-b": "failure"})
        completed, reason = await tournament_manager.is_tourn_task_completed(
            self._tournament_task(), self._task_obj(TaskStatus.SUCCESS.value), self._config()
        )
        assert completed is True
        assert "boss is retained" in reason

    async def test_both_failed_completes_round_on_task_failure(self, monkeypatch):
        self._patch_trainings(monkeypatch, {"miner-a": "failure", "miner-b": "failure"})
        completed, reason = await tournament_manager.is_tourn_task_completed(
            self._tournament_task(), self._task_obj(TaskStatus.FAILURE.value), self._config()
        )
        assert completed is True
        assert "boss is retained" in reason

    async def test_non_quasar_task_still_stalls_for_investigation(self, monkeypatch):
        self._patch_trainings(monkeypatch, {"miner-a": "failure", "miner-b": "failure"})
        completed, reason = await tournament_manager.is_tourn_task_completed(
            self._tournament_task(),
            self._task_obj(TaskStatus.SUCCESS.value, model_id="unsloth/Llama-3.2-3B"),
            self._config(),
        )
        assert completed is False
        assert reason == "More than half of the trainings failed"

    async def test_task_failure_without_recorded_trainings_still_stalls(self, monkeypatch):
        # Infra failure before any training was assigned: keep the investigate path, don't
        # silently hand the tournament to the boss.
        self._patch_trainings(monkeypatch, {})
        completed, _ = await tournament_manager.is_tourn_task_completed(
            self._tournament_task(), self._task_obj(TaskStatus.FAILURE.value), self._config()
        )
        assert completed is False

    async def test_single_failure_keeps_normal_completion(self, monkeypatch):
        # One survivor: round completes normally and the survivor challenges the boss.
        self._patch_trainings(monkeypatch, {"miner-a": "failure", "miner-b": "success"})
        completed, reason = await tournament_manager.is_tourn_task_completed(
            self._tournament_task(), self._task_obj(TaskStatus.SUCCESS.value), self._config()
        )
        assert completed is True
        assert reason == "Task completed successfully"
