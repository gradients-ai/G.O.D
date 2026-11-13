"""A text seed role that ran out of replacements (`failed`) never halts or blocks the text tournament.

The failed role is a boss result that does not exist: a round plays an ordinary task of its own (rounds 1-3 draw an instruct
task like the seed's, the boss round draws its own continuous task) and the crown rule gives a task with no valid boss score
to a challenger with a valid one. Same faking style as test_text_seed_rounds.py (its `world` fixture).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID
from uuid import uuid4

import pytest

import validator.tournament.round_results as rr
import validator.tournament.tournament_manager as tm
from core.constants.environments import TrainingStartPoint
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from tests.validator.tournament.test_seed_tasks import TID
from tests.validator.tournament.test_seed_tasks import _instruct
from tests.validator.tournament.test_seed_tasks import env  # noqa: F401  (fixture)
from tests.validator.tournament.test_text_seed_rounds import MINERS
from tests.validator.tournament.test_text_seed_rounds import ROUND_ID
from tests.validator.tournament.test_text_seed_rounds import TextWorld
from validator.db.sql import tasks as task_sql
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.tournament import constants as t_cst
from validator.tournament import seed_tasks
from validator.tournament import task_creator
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus
from validator.tournament.text_crown import text_head_to_head


@pytest.fixture
def world(env, monkeypatch) -> TextWorld:  # noqa: F811
    return TextWorld(env, monkeypatch)


def _snapshot(world: TextWorld) -> list:
    return [(row.role, row.attempt, row.task_id, row.attached_round_id) for row in world.env.store.rows]


async def _fail(world: TextWorld, role: SeedRole) -> TournamentSeedTask:
    """Create the text seeds and mark one role `failed` (out of replacements)."""
    seeds = await world.create_seeds()
    world.env.store.rows[[row.role for row in world.env.store.rows].index(role)].status = SeedStatus.FAILED
    return seeds[role]


class TestHalt:
    @pytest.mark.parametrize("role", [SeedRole.TEXT_CONTINUOUS, SeedRole.TEXT_R1, SeedRole.TEXT_R2, SeedRole.TEXT_R3])
    async def test_a_failed_text_role_does_not_halt(self, world, role):
        await _fail(world, role)

        assert await seed_tasks.seed_halt_reason(TID, None) is None

    async def test_a_failed_role_of_the_earlier_design_still_halts(self, world):
        await world.create_seeds()
        world.env.store.rows.append(
            TournamentSeedTask(
                tournament_id=TID, role=SeedRole.INPUT_SMALL, attempt=3, task_id=uuid4(), status=SeedStatus.FAILED
            )
        )

        assert "input_small" in await seed_tasks.seed_halt_reason(TID, None)


class TestAlert:
    def _out_of_replacements(self, world, role: SeedRole) -> TournamentSeedTask:
        """The role's row as its third attempt, after two charged replacements."""
        seed = next(row for row in world.env.store.rows if row.role == role)
        for attempt in range(1, t_cst.SEED_MAX_REPLACEMENTS + 1):
            world.env.store.rows.append(
                TournamentSeedTask(tournament_id=TID, role=role, attempt=attempt, task_id=uuid4(), status=SeedStatus.REPLACED)
            )
        seed.attempt = t_cst.SEED_MAX_REPLACEMENTS + 1
        return seed

    @pytest.mark.parametrize("role", [SeedRole.TEXT_CONTINUOUS, SeedRole.TEXT_R2])
    async def test_running_out_of_charged_replacements_alerts_once_without_halting(self, world, role):
        await world.create_seeds()
        seed = self._out_of_replacements(world, role)

        assert await seed_tasks.replace_seed(seed, "task ended failure", world.env.config) is None
        assert await seed_tasks.replace_seed(seed, "task ended failure", world.env.config) is None

        assert seed.status == SeedStatus.FAILED
        world.env.alerts.assert_awaited_once()
        message = world.env.alerts.await_args.kwargs["message"]
        assert role.value in message and "not halted" in message and "HALTED" not in message
        assert "base model" in message
        if role == SeedRole.TEXT_CONTINUOUS:
            assert "own continuous task" in message and "no upfront boss result" in message
        else:
            assert "plays a task" in message
        assert seed_tasks._SEED_HALT_ALERTED == set()
        assert await seed_tasks.seed_halt_reason(TID, None) is None

    async def test_the_prep_rejection_limit_alerts_once_without_halting(self, world, monkeypatch):
        seeds = await world.create_seeds()
        seed = seeds[SeedRole.TEXT_R3]
        world.env.tasks[seed.task_id].status = TaskStatus.PREP_TASK_FAILURE.value
        monkeypatch.setattr(t_cst, "SEED_MAX_PREP_REJECTIONS", 1)

        assert await seed_tasks.replace_seed(seed, "task ended prep", world.env.config, prep_rejected=True) is None

        stored = next(row for row in world.env.store.rows if row.role == SeedRole.TEXT_R3)
        assert stored.status == SeedStatus.FAILED  # `seed` is the caller's copy; the store holds the row
        world.env.alerts.assert_awaited_once()
        message = world.env.alerts.await_args.kwargs["message"]
        assert "not halted" in message and "HALTED" not in message and "prep-stage dataset rejections" in message
        assert seed_tasks._SEED_HALT_ALERTED == set()


@pytest.fixture
def drawn(world, monkeypatch) -> list:
    """The ordinary instruct draws of task_creator, recorded (models pool, kwargs)."""
    calls: list = []

    async def make_instruct(config, models, datasets, **kwargs):
        calls.append((models, kwargs))
        task = _instruct()
        world.env.tasks[task.task_id] = task
        return task

    monkeypatch.setattr(task_creator, "_get_text_models", lambda _keypair, low, high: ("pool", low, high))
    monkeypatch.setattr(task_creator, "_get_instruct_text_datasets", lambda _keypair: "datasets")
    monkeypatch.setattr(task_creator, "filter_quant_compatible", lambda pool: pool)
    monkeypatch.setattr(task_creator, "create_synthetic_instruct_text_task", make_instruct)
    return calls


class TestRoundsOneToThree:
    @pytest.mark.parametrize("number", [1, 2, 3])
    async def test_a_failed_role_draws_the_rounds_own_task_with_no_seed_and_no_boss(self, world, drawn, number):
        spec = t_cst.TEXT_ROUND_PLAN[number - 1]
        await _fail(world, spec.seed_role)
        rows = _snapshot(world)

        task_ids = await task_creator.create_text_tournament_tasks(world.round_of(number), TID, world.env.config)

        assert len(task_ids) == 1 and len(drawn) == 1
        assert drawn[0] == (
            ("pool", *spec.model_size_b),
            dict(enable_kl=False, allow_augmentation=True, carve_calibration=True),
        )
        (registered,) = world.registry.values()
        assert str(registered.task_id) == task_ids[0]
        assert (registered.round_id, registered.group_id) == (f"{ROUND_ID}_{number}", f"{ROUND_ID}_{number}_group_001")
        assert _snapshot(world) == rows  # no seed row
        assert task_ids[0] not in {str(row.task_id) for row in world.env.store.rows}
        assert task_ids[0] not in world.env.assigned  # the boss is not assigned

        again = await task_creator.create_text_tournament_tasks(world.round_of(number), TID, world.env.config)

        assert again == task_ids and len(world.registry) == 1 and len(drawn) == 1

    async def test_a_pending_replacement_still_keeps_the_round_pending(self, world, drawn):
        await _fail(world, SeedRole.TEXT_R2)
        world.env.store.rows[[row.role for row in world.env.store.rows].index(SeedRole.TEXT_R2)].status = SeedStatus.REPLACED

        with pytest.raises(task_creator.MissingBossRoundSeeds) as raised:
            await task_creator.create_text_tournament_tasks(world.round_of(2), TID, world.env.config)

        assert raised.value.roles == [SeedRole.TEXT_R2.value]
        assert world.registry == {} and drawn == []

    async def test_another_failed_role_does_not_change_a_live_roles_round(self, world, drawn):
        await _fail(world, SeedRole.TEXT_R2)

        task_ids = await task_creator.create_text_tournament_tasks(world.round_of(1), TID, world.env.config)

        assert task_ids == [str(next(r.task_id for r in world.env.store.rows if r.role == SeedRole.TEXT_R1))]
        assert drawn == []


class TestAttachedSeedThatFails:
    @staticmethod
    def _store_row(world, seed) -> TournamentSeedTask:
        return next(row for row in world.env.store.rows if row.task_id == seed.task_id)

    async def _attached_prep_failure(self, world, monkeypatch, drawn) -> tuple:
        seeds = await world.create_seeds()
        old = seeds[SeedRole.TEXT_R1]
        await task_creator.create_text_tournament_tasks(world.round_of(1), TID, world.env.config)
        world.env.assigned[str(old.task_id)] = [EMISSION_BURN_HOTKEY, "m1", "m2"]
        world.env.tasks[old.task_id].status = TaskStatus.PREP_TASK_FAILURE.value
        self._store_row(world, old).status = SeedStatus.FAILED  # the store holds copies of what create_seeds returned
        monkeypatch.setattr(task_creator, "get_tournament_rounds", AsyncMock(return_value=[
            SimpleNamespace(round_id=f"{ROUND_ID}_1", round_number=1)
        ]))
        # replace_tournament_task looks the original up by the registry row's string id; the fake's dict is keyed by UUID
        monkeypatch.setattr(
            task_sql, "get_task", AsyncMock(side_effect=lambda task_id, *_a, **_k: world.env.tasks.get(UUID(str(task_id))))
        )
        monkeypatch.setattr(task_sql, "get_expected_repo_name", AsyncMock(side_effect=lambda t, hk, _db: f"orig-{hk}"))
        return old, world.registry[old.task_id]

    async def test_a_failed_seed_whose_task_failed_prep_is_replaced_without_the_boss(self, world, monkeypatch, drawn):
        old, row = await self._attached_prep_failure(world, monkeypatch, drawn)
        task_obj = world.env.tasks[old.task_id]

        completed, reason = await tm.is_tourn_task_completed(row, task_obj, world.env.config)

        assert completed is False and "Replaced with a new task" in reason
        (registered,) = world.registry.values()
        assert registered.task_id != old.task_id
        assert (registered.round_id, registered.group_id) == (f"{ROUND_ID}_1", f"{ROUND_ID}_1_group_001")
        assert len(drawn) == 1 and drawn[0][0] == ("pool", *t_cst.TEXT_ROUND_PLAN[0].model_size_b)  # not the legacy small redraw
        assert world.env.assigned[str(registered.task_id)] == ["m1", "m2"]
        assert EMISSION_BURN_HOTKEY not in world.env.assigned[str(registered.task_id)]
        assert str(old.task_id) not in world.env.assigned or old.task_id not in world.registry

    async def test_a_seed_that_is_still_being_replaced_is_left_to_the_seed_loop(self, world, monkeypatch, drawn):
        old, row = await self._attached_prep_failure(world, monkeypatch, drawn)
        self._store_row(world, old).status = SeedStatus.TRAINING
        replace = AsyncMock()
        monkeypatch.setattr(tm, "replace_tournament_task", replace)

        completed = await tm.is_tourn_task_completed(row, world.env.tasks[old.task_id], world.env.config)

        assert completed == (False, "seed task failed prep; the seed is redrawn")
        replace.assert_not_awaited()


class TestBossRound:
    @pytest.fixture
    def boss_round(self, world, monkeypatch):
        calls = SimpleNamespace(continuous=[], other=[])

        async def make_continuous(config, lineage, seed_model):
            calls.continuous.append((lineage, seed_model))
            task = SimpleNamespace(
                task_id=uuid4(), task_type=TaskType.CHATTASK, training_start_point=TrainingStartPoint.CONTINUOUS_SFT
            )
            world.env.tasks[task.task_id] = task
            return task

        async def make_other(task_type, config, models, instruct_datasets, dpo_datasets, **kwargs):
            calls.other.append(task_type)
            task = SimpleNamespace(task_id=uuid4(), task_type=task_type, training_start_point=None)
            world.env.tasks[task.task_id] = task
            return task

        monkeypatch.setattr(task_creator, "create_continuous_sft_task", make_continuous)
        monkeypatch.setattr(task_creator, "_create_task_by_type", make_other)
        monkeypatch.setattr(task_creator, "_get_text_models", lambda keypair, *a, **k: "models")
        monkeypatch.setattr(task_creator, "_get_instruct_text_datasets", lambda keypair: "datasets")
        monkeypatch.setattr(task_creator, "_get_dpo_datasets", lambda keypair: "dpo")
        return calls

    async def test_a_failed_continuous_role_draws_the_rounds_own_continuous_task(self, world, boss_round):
        await _fail(world, SeedRole.TEXT_CONTINUOUS)
        rows = _snapshot(world)

        tasks = await task_creator._create_seeded_text_boss_round_tasks(TID, "round_boss", world.env.config)

        assert boss_round.continuous == [next(iter(t_cst.CONTINUOUS_SFT_LINEAGES.items()))]
        assert [task.task_type for task in tasks] == [TaskType.CHATTASK, TaskType.GRPOTASK, TaskType.DPOTASK]
        registered = {str(row.task_id): row for row in world.registry.values()}
        assert set(registered) == {str(task.task_id) for task in tasks}
        assert all(row.pair_id == "round_boss_pair_001" for row in registered.values())
        assert _snapshot(world) == rows  # no seed row, no marker
        assert str(tasks[0].task_id) not in world.env.assigned

    async def test_a_second_pass_creates_nothing_more(self, world, boss_round):
        await _fail(world, SeedRole.TEXT_CONTINUOUS)
        first = await task_creator._create_seeded_text_boss_round_tasks(TID, "round_boss", world.env.config)

        second = await task_creator._create_seeded_text_boss_round_tasks(TID, "round_boss", world.env.config)

        assert [t.task_id for t in second] == [t.task_id for t in first]
        assert len(boss_round.continuous) == 1 and boss_round.other == [TaskType.GRPOTASK, TaskType.DPOTASK]
        assert len(world.registry) == 3

    async def test_a_pending_replacement_still_raises_before_anything_is_drawn(self, world, boss_round):
        await _fail(world, SeedRole.TEXT_CONTINUOUS)
        next(row for row in world.env.store.rows if row.role == SeedRole.TEXT_CONTINUOUS).status = SeedStatus.REPLACED

        with pytest.raises(task_creator.MissingBossRoundSeeds):
            await task_creator._create_seeded_text_boss_round_tasks(TID, "round_boss", world.env.config)

        assert boss_round.continuous == [] and boss_round.other == [] and world.registry == {}

    async def test_a_live_continuous_seed_is_still_attached_and_marked(self, world, boss_round):
        seeds = await world.create_seeds()

        tasks = await task_creator._create_seeded_text_boss_round_tasks(TID, "round_boss", world.env.config)

        assert tasks[0].task_id == seeds[SeedRole.TEXT_CONTINUOUS].task_id and boss_round.continuous == []
        marked = next(row for row in world.env.store.rows if row.role == SeedRole.TEXT_CONTINUOUS)
        assert marked.attached_round_id == "round_boss"


class TestCrown:
    async def test_a_round_task_the_boss_has_no_result_on_goes_to_a_challenger_with_a_valid_score(self, monkeypatch):
        task_id = str(uuid4())
        challenger = MINERS[0]
        round_data = SimpleNamespace(round_id="r1", round_number=1)
        monkeypatch.setattr(rr, "get_tournament_rounds", AsyncMock(return_value=[
            SimpleNamespace(round_id=f"r{spec.round_number}", round_number=spec.round_number) for spec in t_cst.TEXT_ROUND_PLAN
        ]))
        row = TournamentTask(tournament_id=TID, round_id=round_data.round_id, task_id=task_id)
        monkeypatch.setattr(rr, "get_tournament_tasks", AsyncMock(return_value=[row]))
        monkeypatch.setattr(rr, "get_task", AsyncMock(return_value=SimpleNamespace(task_type=TaskType.INSTRUCTTEXTTASK)))
        results = [
            MinerResultsText(
                hotkey=challenger, test_loss=1.0, synth_loss=1.0, is_finetune=True, task_type=TaskType.INSTRUCTTEXTTASK
            )
        ]
        monkeypatch.setattr(rr, "get_task_results_for_ranking", AsyncMock(return_value=results))
        monkeypatch.setattr(
            rr,
            "get_training_status_for_task_and_hotkeys",
            AsyncMock(return_value={challenger: TrainingStatus.SUCCESS.value}),
        )

        results = await rr.seeded_text_round_head_to_heads(TID, challenger, None)

        assert len(results) == len(t_cst.TEXT_ROUND_PLAN)
        for result in results:
            assert result.boss_score is None and result.challenger_score is not None
            assert result.is_win and result.deficit == 0.0

    def test_no_valid_boss_score_with_a_valid_challenger_score_is_a_win(self):
        assert text_head_to_head("t", "round 2", TaskType.INSTRUCTTEXTTASK, None, 1.2).is_win


class TestFirstRoundGate:
    @pytest.fixture
    def first_round(self, world, monkeypatch):
        monkeypatch.setattr(tm, "get_tournament_rounds", AsyncMock(return_value=[]))
        text = TournamentData(tournament_id=TID, tournament_type=TournamentType.TEXT)
        monkeypatch.setattr(tm, "get_tournament", AsyncMock(return_value=text))
        monkeypatch.setattr(
            tm,
            "get_tournament_participants",
            AsyncMock(return_value=[SimpleNamespace(hotkey=EMISSION_BURN_HOTKEY), SimpleNamespace(hotkey="miner_a")]),
        )
        monkeypatch.setattr(tm, "get_node_by_hotkey", AsyncMock(return_value=SimpleNamespace(hotkey="miner_a")))
        monkeypatch.setattr(t_cst, "TOURN_DEDUP_ENABLED", False)
        create = AsyncMock()
        monkeypatch.setattr(tm, "_create_first_round", create)
        return create

    async def test_a_failed_role_does_not_hold_the_first_round(self, world, first_round):
        await _fail(world, SeedRole.TEXT_CONTINUOUS)
        world.continuous_error = RuntimeError("never retried: the role is failed")

        assert await tm.create_first_round_for_active_tournament(TID, world.env.config, None) is True

        first_round.assert_awaited_once()

    async def test_a_missing_role_still_holds_it(self, world, first_round):
        world.continuous_error = RuntimeError("content service down")

        assert await tm.create_first_round_for_active_tournament(TID, world.env.config, None) is False

        first_round.assert_not_awaited()

    async def test_a_quant_tournament_counts_a_failed_role_as_present(self, world, first_round, monkeypatch):
        order = seed_tasks.SEED_CREATION_ORDER
        monkeypatch.setattr(
            tm, "get_tournament", AsyncMock(return_value=TournamentData(tournament_id=TID, tournament_type=TournamentType.QUANT))
        )
        monkeypatch.setattr(tm, "bind_quant_source_tournament", AsyncMock(return_value=SimpleNamespace(text_tournament_id="t")))
        monkeypatch.setattr(
            tm, "create_seed_tasks", AsyncMock(return_value=[SimpleNamespace(role=role) for role in order[:-1]])
        )
        world.env.store.rows.append(
            TournamentSeedTask(tournament_id=TID, role=order[-1], attempt=3, task_id=uuid4(), status=SeedStatus.FAILED)
        )

        assert await tm.create_first_round_for_active_tournament(TID, world.env.config, None) is True

        first_round.assert_awaited_once()
