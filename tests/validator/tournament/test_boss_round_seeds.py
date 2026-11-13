"""The quant boss round on the upfront seeds (validator/tournament/README.md, "Compression tournament flow").

The boss round attaches the four compression seeds (the text boss's round-2 model freeform, its round-3 model fixed and its
continuous model freeform and fixed) and adds training rows for the challenger only; the challenger trains from the same
task.model_id as the boss. A role out of replacements plays a task of its own that the boss has no entry on. Two
compressions chained on a BOSS_INSTRUCT seed and released on the boss's output of it are also supported (the release tests
below). Nothing the boss trains is scheduled at or after the boss round.
"""

import logging
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.tournament.constants as t_cst
from core.constants.environments import TrainingStartPoint
from core.models.dataset_models import InstructTextDatasetType
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from validator.db.sql import tournament_seeds as seeds_sql
from validator.lifecycle import quant_reference as lifecycle_quant_reference
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tasks.synthetics.scheduler import quant_training_hours
from validator.tournament import orchestrator
from validator.tournament import seed_tasks
from validator.tournament import task_creator
from validator.tournament import tournament_manager as tm
from validator.tournament.models import BOSS_ROUND_SEED_ROLES
from validator.tournament.models import RETIRED_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import UNCREATED_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import KnockoutRound
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingRepoInfo
from validator.tournament.models import TrainingStatus


TID = "tourn_boss_seeds"
CHALLENGER = "5Challenger"
ROUND_ID = f"{TID}_round_004"
PAIR_ID = f"{ROUND_ID}_pair_001"
CONFIG = SimpleNamespace(psql_db=None, discord_url=None)
SUCCESS = TrainingStatus.SUCCESS.value
FAILURE = TrainingStatus.FAILURE.value
TRAINING = TrainingStatus.TRAINING.value
PINNED = TrainingRepoInfo(
    training_repo="https://github.com/org/repo", training_commit_hash="b" * 40, github_token=None, requested_datasets=None
)


def _fields(**overrides) -> dict:
    return {
        "is_organic": False,
        "task_id": uuid.uuid4(),
        "model_id": "org/base-model",
        "ds": "org/instruct-ds",
        "account_id": uuid.uuid4(),
        "hours_to_complete": 3.0,
        "created_at": datetime(2026, 11, 9),
        **overrides,
    }


def _instruct(**overrides) -> InstructTextRawTask:
    return InstructTextRawTask(
        **_fields(
            status=TaskStatus.TRAINING,
            training_data="s3://bucket/train.json",
            test_data="s3://bucket/test.json",
            calibration_data="s3://bucket/cal.json",
            model_params_count=3_000_000_000,
            **overrides,
        ),
        field_instruction="instruction",
        field_input="input",
        field_output="output",
        field_system="system",
        format="{instruction} {input}",
        no_input_format="{instruction}",
    )


def _own_upstream(upstream: InstructTextRawTask, freeform: bool = True, **overrides):
    fields = _fields(
        status=TaskStatus.AWAITING_UPSTREAM,
        upstream_task_id=upstream.task_id,
        model_id=upstream.model_id,
        ds=upstream.ds,
        hours_to_complete=1.0,
        training_start_point=TrainingStartPoint.OWN_UPSTREAM_MODEL,
        dataset_type=InstructTextDatasetType(field_instruction="question"),
        **overrides,
    )
    if freeform:
        return QuantFreeformRawTask(**fields)
    return QuantFixedRawTask(**fields, quant_spec=QuantFormat.W4A8_FP8_G128.spec())


def _chained(upstream: InstructTextRawTask, freeform: bool = True, **overrides):
    """A compression task of the boss round: chained on the BOSS_INSTRUCT seed, held until that upstream is SUCCESS."""
    fields = _fields(
        status=TaskStatus.AWAITING_UPSTREAM,
        upstream_task_id=upstream.task_id,
        model_id=upstream.model_id,
        ds=upstream.ds,
        hours_to_complete=1.0,
        dataset_type=InstructTextDatasetType(field_instruction="question"),
        **overrides,
    )
    if freeform:
        return QuantFreeformRawTask(**fields)
    return QuantFixedRawTask(**fields, quant_spec=QuantFormat.W4A8_FP8_G128.spec())


def _output(upstream: InstructTextRawTask, hotkey: str) -> str:
    return f"validator-org/tournament-{TID}-{upstream.task_id}-{hotkey[:8]}"


class World:
    """In-memory stand-in for the orchestrator's SQL seams."""

    def __init__(self, monkeypatch) -> None:
        self.tasks: dict[str, object] = {}
        self.assigned: dict[str, list[str]] = {}
        self.starting: dict[tuple[str, str], str] = {}
        self.training: dict[str, dict[str, str]] = {}
        self.failures: dict[tuple[str, str], str] = {}
        self.attached: set[str] = set()
        self.seeds: dict[SeedRole, TournamentSeedTask] = {}
        self.not_full: set[str] = set()
        self.rows: list = []
        self.updated: list = []
        mp = monkeypatch
        mp.setattr(orchestrator.task_sql, "get_task", AsyncMock(side_effect=lambda tid, db: self.tasks.get(str(tid))))
        mp.setattr(orchestrator.task_sql, "get_nodes_assigned_to_task", AsyncMock(side_effect=self._nodes))
        mp.setattr(orchestrator.task_sql, "get_starting_model_repo", AsyncMock(side_effect=self._get_starting))
        mp.setattr(orchestrator.task_sql, "set_starting_model_repo", AsyncMock(side_effect=self._set_starting))
        mp.setattr(orchestrator.task_sql, "update_task", AsyncMock(side_effect=self._update))
        mp.setattr(orchestrator.tournament_sql, "get_training_status_for_task", AsyncMock(side_effect=self._statuses))
        mp.setattr(
            orchestrator.tournament_sql, "get_training_status_for_task_and_hotkeys", AsyncMock(side_effect=self._statuses_for)
        )
        mp.setattr(orchestrator.tournament_sql, "get_tournament_id_by_task_id", AsyncMock(return_value=TID))
        mp.setattr(orchestrator.tournament_sql, "get_tournament_training_repo_and_commit", AsyncMock(return_value=PINNED))
        mp.setattr(
            orchestrator.tournament_sql, "add_tournament_task_hotkey_pairs_for_training", AsyncMock(side_effect=self._add_rows)
        )
        mp.setattr(
            orchestrator.tournament_sql,
            "get_tournaments_with_status",
            AsyncMock(
                return_value=[
                    TournamentData(tournament_id=TID, tournament_type=TournamentType.QUANT, status=TournamentStatus.ACTIVE)
                ]
            ),
        )
        mp.setattr(orchestrator.submissions_and_scoring_sql, "get_task_node_losses", AsyncMock(side_effect=self._losses))
        mp.setattr(
            orchestrator.submissions_and_scoring_sql, "set_task_node_quant_failure", AsyncMock(side_effect=self._set_failure)
        )
        mp.setattr(
            orchestrator.seeds_sql, "is_task_attached_to_round", AsyncMock(side_effect=lambda tid, db: str(tid) in self.attached)
        )
        mp.setattr(seeds_sql, "get_seed_task_by_task_id", AsyncMock(side_effect=self._seed_by_task))
        mp.setattr(orchestrator, "live_seed_tasks", AsyncMock(side_effect=lambda tid, db: dict(self.seeds)))
        mp.setattr(orchestrator, "_is_full_model_repo", AsyncMock(side_effect=lambda repo: repo not in self.not_full))
        mp.setattr(orchestrator.chaining, "chained_input_repo", AsyncMock(side_effect=self._input_repo))
        mp.setattr(orchestrator.chaining, "shared_input_candidates", AsyncMock(side_effect=lambda up, hotkeys, db: list(hotkeys)))

    async def _seed_by_task(self, task_id, db):
        return next((seed for seed in self.seeds.values() if str(seed.task_id) == str(task_id)), None)

    async def _nodes(self, task_id, db):
        return [SimpleNamespace(hotkey=hotkey) for hotkey in self.assigned.get(str(task_id), [])]

    async def _get_starting(self, task_id, hotkey, db):
        return self.starting.get((str(task_id), hotkey))

    async def _set_starting(self, task_id, hotkey, repo, db):
        self.starting[(str(task_id), hotkey)] = repo

    async def _update(self, task, db):
        self.updated.append(task)

    async def _statuses(self, task_id, db):
        return dict(self.training.get(str(task_id), {}))

    async def _statuses_for(self, task_id, hotkeys, db):
        return {hk: st for hk, st in self.training.get(str(task_id), {}).items() if hk in hotkeys}

    async def _losses(self, task_id, db):
        return [{"hotkey": hk, "quant_failure": reason} for (tid, hk), reason in self.failures.items() if tid == str(task_id)]

    async def _set_failure(self, task_id, hotkey, reason, db):
        self.failures[(str(task_id), hotkey)] = reason

    async def _add_rows(self, assignments, db):
        for assignment in assignments:
            self.training.setdefault(assignment.task_id, {})[assignment.hotkey] = TrainingStatus.PENDING.value
        self.rows += assignments

    async def _input_repo(self, upstream_id, hotkey, db):
        upstream = self.tasks[str(upstream_id)]
        return _output(upstream, hotkey)

    def add(self, task, hotkeys, role: SeedRole | None = None, attached: bool = False):
        self.tasks[str(task.task_id)] = task
        self.assigned[str(task.task_id)] = list(hotkeys)
        if role is not None:
            self.seeds[role] = TournamentSeedTask(tournament_id=TID, role=role, task_id=task.task_id, status=SeedStatus.TRAINED)
        if attached:
            self.attached.add(str(task.task_id))
        return task

    def boss_round(self, boss_status: str | None = SUCCESS):
        """The BOSS_INSTRUCT seed (SUCCESS, boss only) and the two attached compression seeds, boss and challenger assigned,
        the boss's run in boss_status. Returns (instruct, free, fixed)."""
        instruct = self.add(_instruct(), [EMISSION_BURN_HOTKEY], SeedRole.BOSS_INSTRUCT)
        instruct.status = TaskStatus.SUCCESS
        self.training[str(instruct.task_id)] = {EMISSION_BURN_HOTKEY: SUCCESS}
        free = self.add(_chained(instruct), [EMISSION_BURN_HOTKEY, CHALLENGER], SeedRole.BOSS_CHAINED_FREEFORM, attached=True)
        fixed = self.add(
            _chained(instruct, freeform=False), [EMISSION_BURN_HOTKEY, CHALLENGER], SeedRole.BOSS_CHAINED_FIXED, attached=True
        )
        for task in (free, fixed):
            task.status = TaskStatus.TRAINING
            task.model_id = _output(instruct, EMISSION_BURN_HOTKEY)
            self.training[str(task.task_id)] = {EMISSION_BURN_HOTKEY: boss_status} if boss_status else {}
        return instruct, free, fixed


@pytest.fixture
def world(monkeypatch):
    return World(monkeypatch)


class TestAttach:
    @pytest.fixture
    def creator(self, monkeypatch):
        seeds = {}
        registered: list[str] = []
        tasks = {}
        for role in SeedRole:
            if role in RETIRED_BOSS_ROUND_SEED_ROLES | UNCREATED_BOSS_ROUND_SEED_ROLES:
                continue  # a tournament has the four compression roles or the two chained ones, never both
            task = _instruct()
            tasks[str(task.task_id)] = task
            seeds[role] = TournamentSeedTask(tournament_id=TID, role=role, task_id=task.task_id)
        drawn = AsyncMock()
        marked = AsyncMock()
        monkeypatch.setattr(task_creator.seeds_sql, "mark_seed_attached", marked)

        async def register(task, tournament_id, round_id, config, group_id=None, pair_id=None):
            assert (round_id, pair_id) == (ROUND_ID, PAIR_ID)
            registered.append(str(task.task_id))

        async def existing(round_id, config, group_id=None, pair_id=None):
            return [SimpleNamespace(task_id=task_id) for task_id in registered]

        monkeypatch.setattr(
            task_creator.seeds_sql, "get_active_seed_tasks", AsyncMock(side_effect=lambda tid, db: list(seeds.values()))
        )
        monkeypatch.setattr(
            task_creator.seeds_sql, "get_latest_seed_tasks", AsyncMock(side_effect=lambda tid, db: list(seeds.values()))
        )
        monkeypatch.setattr(task_creator.task_sql, "get_task", AsyncMock(side_effect=lambda tid, db: tasks.get(str(tid))))
        monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", register)
        monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", existing)
        for name in (
            "create_synthetic_instruct_text_task",
            "create_chained_quant_task",
        ):
            monkeypatch.setattr(task_creator, name, drawn)
        return SimpleNamespace(seeds=seeds, tasks=tasks, registered=registered, drawn=drawn, marked=marked)

    async def test_the_boss_round_draws_nothing_and_attaches_exactly_the_four_compression_seeds(self, creator):
        tasks = await task_creator._create_quant_boss_round_tasks(TID, ROUND_ID, CONFIG)

        expected = {str(creator.seeds[role].task_id) for role in BOSS_ROUND_SEED_ROLES}
        assert {str(task.task_id) for task in tasks} == expected
        assert set(creator.registered) == expected and len(creator.registered) == 4
        assert str(creator.seeds[SeedRole.BOSS_INSTRUCT].task_id) not in creator.registered
        creator.drawn.assert_not_called()
        # The durable "ever attached" marker is written for every boss-round seed, with the round id.
        marked = {(str(call.args[0]), call.args[1]) for call in creator.marked.await_args_list}
        assert marked == {(task_id, ROUND_ID) for task_id in expected}

    async def test_attach_is_idempotent(self, creator):
        await task_creator._create_quant_boss_round_tasks(TID, ROUND_ID, CONFIG)
        await task_creator._create_quant_boss_round_tasks(TID, ROUND_ID, CONFIG)

        assert len(creator.registered) == 4

    async def test_a_missing_boss_seed_fails_the_attempt_instead_of_drawing_a_task(self, creator):
        del creator.seeds[SeedRole.BOSS_R3_FIXED]

        with pytest.raises(RuntimeError, match="boss_r3_fixed"):
            await task_creator._create_quant_boss_round_tasks(TID, ROUND_ID, CONFIG)

        assert creator.registered == []
        creator.drawn.assert_not_called()

    async def test_a_role_that_failed_plays_a_task_of_its_own_without_the_boss(self, creator, monkeypatch):
        failed = creator.seeds.pop(SeedRole.BOSS_R3_FIXED)
        failed_row = failed.model_copy(update={"status": SeedStatus.FAILED})
        monkeypatch.setattr(
            task_creator.seeds_sql,
            "get_latest_seed_tasks",
            AsyncMock(side_effect=lambda tid, db: [*creator.seeds.values(), failed_row]),
        )
        own_task = _instruct()
        without_boss = AsyncMock(return_value=own_task)
        monkeypatch.setattr(task_creator, "_create_boss_round_task_without_boss", without_boss)

        tasks = await task_creator._create_quant_boss_round_tasks(TID, ROUND_ID, CONFIG)

        assert len(tasks) == 4 and str(failed.task_id) not in creator.registered
        assert {str(task.task_id) for task in tasks} == {
            str(creator.seeds[role].task_id) for role in BOSS_ROUND_SEED_ROLES - {SeedRole.BOSS_R3_FIXED}
        } | {str(own_task.task_id)}
        without_boss.assert_awaited_once_with(SeedRole.BOSS_R3_FIXED, TID, ROUND_ID, PAIR_ID, CONFIG)

    async def test_the_task_of_a_failed_role_is_drawn_like_its_seed_and_registered_under_the_pair(self, creator, monkeypatch):
        own_task = _instruct()
        binding = SimpleNamespace(quant_tournament_id=TID, source_tournament_id="text")
        text_seed = TournamentSeedTask(tournament_id="text", role=SeedRole.TEXT_R3, task_id=uuid.uuid4())
        drawn = AsyncMock(return_value=own_task)
        monkeypatch.setattr(seed_tasks, "draw_compression_task", drawn)
        monkeypatch.setattr(task_creator.quant_sources, "bind_quant_source_tournament", AsyncMock(return_value=binding))
        monkeypatch.setattr(task_creator.quant_sources, "slot_text_seed", AsyncMock(return_value=text_seed))
        monkeypatch.setattr(
            task_creator.quant_sources,
            "record_new_quant_task",
            AsyncMock(side_effect=lambda source, config, replacing: source),
        )
        monkeypatch.setattr(task_creator.quant_sources_sql, "get_live_quant_task_source", AsyncMock(return_value=None))
        monkeypatch.setattr(task_creator.tournament_sql, "get_tournament_task_by_task_id", AsyncMock(return_value=None))

        task = await task_creator._create_boss_round_task_without_boss(SeedRole.BOSS_R3_FIXED, TID, ROUND_ID, PAIR_ID, CONFIG)

        assert task is own_task and creator.registered == [str(own_task.task_id)]
        drawn.assert_awaited_once_with(SeedRole.BOSS_R3_FIXED, TID, text_seed, CONFIG)
        creator.marked.assert_not_awaited()  # no seed row: it is the challenger's task alone

    async def test_the_task_of_a_failed_role_waits_while_its_text_seed_is_being_replaced(self, creator, monkeypatch):
        binding = SimpleNamespace(quant_tournament_id=TID, source_tournament_id="text")
        monkeypatch.setattr(task_creator.quant_sources, "bind_quant_source_tournament", AsyncMock(return_value=binding))
        monkeypatch.setattr(task_creator.quant_sources, "slot_text_seed", AsyncMock(return_value=None))
        monkeypatch.setattr(task_creator.quant_sources_sql, "get_live_quant_task_source", AsyncMock(return_value=None))

        with pytest.raises(task_creator.MissingBossRoundSeeds) as error:
            await task_creator._create_boss_round_task_without_boss(SeedRole.BOSS_R3_FIXED, TID, ROUND_ID, PAIR_ID, CONFIG)

        assert error.value.roles == ["text_r3"] and creator.registered == []

    async def test_a_role_with_a_pending_replacement_still_raises(self, creator, monkeypatch):
        replaced = creator.seeds.pop(SeedRole.BOSS_R2_FREEFORM).model_copy(update={"status": SeedStatus.REPLACED})
        monkeypatch.setattr(
            task_creator.seeds_sql,
            "get_latest_seed_tasks",
            AsyncMock(side_effect=lambda tid, db: [*creator.seeds.values(), replaced]),
        )

        with pytest.raises(task_creator.MissingBossRoundSeeds) as error:
            await task_creator._create_quant_boss_round_tasks(TID, ROUND_ID, CONFIG)

        assert error.value.roles == ["boss_r2_freeform"] and creator.registered == []

    async def test_a_tournament_of_the_earlier_design_attaches_its_two_chained_compressions(self, creator, monkeypatch):
        creator.seeds.clear()
        for role in RETIRED_BOSS_ROUND_SEED_ROLES:
            task = _instruct()
            creator.tasks[str(task.task_id)] = task
            creator.seeds[role] = TournamentSeedTask(tournament_id=TID, role=role, task_id=task.task_id)

        tasks = await task_creator._create_quant_boss_round_tasks(TID, ROUND_ID, CONFIG)

        assert {str(task.task_id) for task in tasks} == {str(seed.task_id) for seed in creator.seeds.values()}
        assert len(creator.registered) == 2


class TestChainedReleaseOnTheBossInstructSeed:
    async def test_the_boss_instruct_seed_is_an_input_seed_and_the_compressions_are_not(self, world):
        instruct, free, fixed = world.boss_round()
        small = world.add(_instruct(), [EMISSION_BURN_HOTKEY], SeedRole.INPUT_SMALL)

        assert await orchestrator._is_input_seed(instruct, CONFIG)
        assert await orchestrator._is_input_seed(small, CONFIG)
        assert not await orchestrator._is_input_seed(free, CONFIG)
        assert not await orchestrator._is_input_seed(fixed, CONFIG)

    @pytest.mark.parametrize("freeform", [True, False])
    async def test_a_compression_is_released_on_the_boss_output_alone_with_boss_hours(self, world, freeform):
        instruct = world.add(_instruct(), [EMISSION_BURN_HOTKEY], SeedRole.BOSS_INSTRUCT)
        instruct.status = TaskStatus.SUCCESS
        role = SeedRole.BOSS_CHAINED_FREEFORM if freeform else SeedRole.BOSS_CHAINED_FIXED
        task = world.add(_chained(instruct, freeform=freeform), [EMISSION_BURN_HOTKEY], role)

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.status == TaskStatus.PENDING
        assert task.model_id == _output(instruct, EMISSION_BURN_HOTKEY)
        orchestrator.chaining.shared_input_candidates.assert_awaited_once()
        assert orchestrator.chaining.shared_input_candidates.await_args.args[1] == [EMISSION_BURN_HOTKEY]
        assert task.hours_to_complete == quant_training_hours(
            task.task_type, instruct.model_params_count, boss=True, baseline_stats=instruct.baseline_stats
        )
        assert (str(task.task_id), EMISSION_BURN_HOTKEY) not in world.starting  # no per-player start model

    async def test_held_until_the_boss_instruct_seed_is_success(self, world):
        instruct = world.add(_instruct(), [EMISSION_BURN_HOTKEY], SeedRole.BOSS_INSTRUCT)
        task = world.add(_chained(instruct), [EMISSION_BURN_HOTKEY], SeedRole.BOSS_CHAINED_FREEFORM)

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.status == TaskStatus.AWAITING_UPSTREAM

    async def test_held_when_the_boss_output_is_not_a_full_model(self, world):
        instruct = world.add(_instruct(), [EMISSION_BURN_HOTKEY], SeedRole.BOSS_INSTRUCT)
        instruct.status = TaskStatus.SUCCESS
        task = world.add(_chained(instruct), [EMISSION_BURN_HOTKEY], SeedRole.BOSS_CHAINED_FREEFORM)
        world.not_full.add(_output(instruct, EMISSION_BURN_HOTKEY))

        await orchestrator._release_chained_task(task, CONFIG)

        assert task.status == TaskStatus.AWAITING_UPSTREAM  # an input seed is held for replacement, never failed


class TestChallengerRowsAfterTheBossRound:
    """Every training row added at or after the boss round is the challenger's."""

    async def test_challenger_rows_only_never_the_boss(self, world):
        _, free, fixed = world.boss_round(boss_status=TRAINING)

        await orchestrator._boss_round_seed_cycle(CONFIG)

        assert {row.hotkey for row in world.rows} == {CHALLENGER}
        assert {row.task_id for row in world.rows} == {str(free.task_id), str(fixed.task_id)}
        assert all(row.priority == 2 and row.training_repo == PINNED.training_repo for row in world.rows)

    async def test_the_challenger_trains_from_the_tasks_own_model_with_no_start_of_its_own(self, world):
        world.boss_round()

        await orchestrator._boss_round_seed_cycle(CONFIG)

        assert world.starting == {}

    async def test_the_challenger_plays_the_fixed_task_on_the_same_task_so_with_the_same_format(self, world):
        _, _, fixed = world.boss_round()

        await orchestrator._boss_round_seed_cycle(CONFIG)

        (row,) = [row for row in world.rows if row.task_id == str(fixed.task_id)]
        assert row.hotkey == CHALLENGER
        assert world.tasks[row.task_id].quant_spec == fixed.quant_spec

    async def test_a_task_still_before_training_gets_no_row_from_this_cycle(self, world):
        _, free, fixed = world.boss_round()
        free.status = TaskStatus.LOOKING_FOR_NODES
        fixed.status = TaskStatus.PREP_TASK_FAILURE

        await orchestrator._boss_round_seed_cycle(CONFIG)

        assert world.rows == []

    async def test_an_unattached_seed_gets_nothing(self, world):
        _, free, _ = world.boss_round()
        world.attached.discard(str(free.task_id))

        await orchestrator._boss_round_seed_cycle(CONFIG)

        assert str(free.task_id) not in {row.task_id for row in world.rows}

    async def test_the_cycle_is_idempotent(self, world):
        world.boss_round()

        await orchestrator._boss_round_seed_cycle(CONFIG)
        first = len(world.rows)
        await orchestrator._boss_round_seed_cycle(CONFIG)

        assert first == 2 and len(world.rows) == first

    async def test_only_quant_and_text_tournaments_are_walked(self, world, monkeypatch):
        world.boss_round()
        monkeypatch.setattr(
            orchestrator.tournament_sql,
            "get_tournaments_with_status",
            AsyncMock(
                return_value=[
                    TournamentData(tournament_id=TID, tournament_type=TournamentType.IMAGE, status=TournamentStatus.ACTIVE)
                ]
            ),
        )

        await orchestrator._boss_round_seed_cycle(CONFIG)

        assert world.rows == []


class TestSeedHold:
    @pytest.fixture
    def hold(self, world):
        _, free, _ = world.boss_round()
        return SimpleNamespace(world=world, task_id=free.task_id)

    async def _ready(self, hold) -> bool:
        return await seeds_sql.boss_round_seed_ready_to_finalise(hold.task_id, None)

    @pytest.mark.parametrize("challenger_status", [None, TrainingStatus.PENDING.value, TRAINING])
    async def test_held_until_the_challengers_row_is_terminal(self, hold, challenger_status):
        if challenger_status:
            hold.world.training[str(hold.task_id)][CHALLENGER] = challenger_status

        assert not await self._ready(hold)

    @pytest.mark.parametrize("terminal", [SUCCESS, FAILURE])
    async def test_released_once_both_rows_are_terminal(self, hold, terminal):
        hold.world.training[str(hold.task_id)][CHALLENGER] = terminal

        assert await self._ready(hold)

    async def test_the_instruct_seed_is_never_held(self, world):
        instruct, *_ = world.boss_round()

        assert await seeds_sql.boss_round_seed_ready_to_finalise(instruct.task_id, None)


class TestWarning:
    async def test_a_boss_row_still_training_is_logged_at_warning_and_does_not_block(self, world, caplog):
        instruct, free, fixed = world.boss_round(boss_status=SUCCESS)
        world.training[str(free.task_id)][EMISSION_BURN_HOTKEY] = TRAINING
        world.training[str(fixed.task_id)] = {}

        with caplog.at_level(logging.WARNING):
            await orchestrator._boss_round_seed_cycle(CONFIG)

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "still training" in r.getMessage()]
        assert f"boss entry {SeedRole.BOSS_CHAINED_FREEFORM.label} {free.task_id} still training" in warnings
        assert f"boss entry {SeedRole.BOSS_CHAINED_FIXED.label} {fixed.task_id} still training" in warnings
        assert not any(str(instruct.task_id) in message for message in warnings)
        # The challenger's rows still went out.
        assert {row.task_id for row in world.rows} >= {str(free.task_id), str(fixed.task_id)}


class TestOwnModelPredicates:
    def test_the_upstream_start_point_is_own_model_for_training_rows(self):
        task = _own_upstream(_instruct())
        assert orchestrator._is_own_model_quant(task)

    async def test_without_unusable_own_models_drops_a_scored_fail_on_an_upstream_task(self, world):
        instruct = world.add(_instruct(), [EMISSION_BURN_HOTKEY])
        task = world.add(_own_upstream(instruct), [EMISSION_BURN_HOTKEY, CHALLENGER])
        world.failures[(str(task.task_id), CHALLENGER)] = t_cst.UPSTREAM_FAILURE_REASON

        kept = await orchestrator._without_unusable_own_models(task, [EMISSION_BURN_HOTKEY, CHALLENGER], CONFIG)

        assert kept == [EMISSION_BURN_HOTKEY]

    async def test_ensure_quant_references_checks_each_players_own_reference(self, monkeypatch):
        task = _own_upstream(_instruct())
        failure = AsyncMock()

        async def reference_model(_task, hotkey, _db):
            return None if hotkey == "5NoModel" else f"repo-{hotkey}"

        monkeypatch.setattr(lifecycle_quant_reference, "quant_reference_model", AsyncMock(side_effect=reference_model))
        monkeypatch.setattr(lifecycle_quant_reference, "set_task_node_quant_failure", failure)
        monkeypatch.setattr(lifecycle_quant_reference, "rows_missing_a_reference", AsyncMock(return_value=[]))

        assert await lifecycle_quant_reference.ensure_quant_references(task, ["5NoModel", CHALLENGER], CONFIG)

        failure.assert_awaited_once()
        assert failure.await_args.args[1] == "5NoModel"


class TestSeedReplacementGuard:
    @pytest.fixture
    def processing(self, monkeypatch):
        seed = TournamentSeedTask(
            tournament_id=TID, role=SeedRole.BOSS_CHAINED_FREEFORM, task_id=uuid.uuid4(), status=SeedStatus.TRAINED
        )
        task = SimpleNamespace(task_id=seed.task_id, status=TaskStatus.FAILURE)
        draw = AsyncMock(return_value=None)
        update = AsyncMock()
        state = {"attached": True}
        monkeypatch.setattr(seed_tasks.seeds_sql, "get_active_seed_tasks", AsyncMock(return_value=[seed]))
        monkeypatch.setattr(seed_tasks.seeds_sql, "get_latest_seed_tasks", AsyncMock(return_value=[seed]))
        monkeypatch.setattr(
            seed_tasks.seeds_sql, "is_task_attached_to_round", AsyncMock(side_effect=lambda tid, db: state["attached"])
        )
        monkeypatch.setattr(seed_tasks.seeds_sql, "update_seed_status", update)
        monkeypatch.setattr(seed_tasks.seeds_sql, "get_seed_task_history", AsyncMock(return_value=[seed]))
        monkeypatch.setattr(seed_tasks.seeds_sql, "get_failed_seed_tasks", AsyncMock(return_value=[]))
        monkeypatch.setattr(seed_tasks.task_sql, "get_task", AsyncMock(return_value=task))
        monkeypatch.setattr(seed_tasks.tournament_sql, "get_training_status_for_task_and_hotkeys", AsyncMock(return_value={}))
        monkeypatch.setattr(seed_tasks, "live_seed_tasks", AsyncMock(return_value={seed.role: seed}))
        monkeypatch.setattr(seed_tasks, "_create_seed", draw)
        monkeypatch.setattr(seed_tasks, "_retry_missing_boss_round_seeds", AsyncMock())  # covered in test_seed_tasks
        monkeypatch.setattr(seed_tasks.seeds_sql, "tournament_is_decided", AsyncMock(return_value=False))
        monkeypatch.setattr(seed_tasks, "_SEED_ISSUE_ALERTED", set())
        return SimpleNamespace(seed=seed, task=task, replace=draw, update=update, state=state)

    async def test_an_attached_boss_round_seed_is_not_replaced_after_it_failed(self, processing, caplog):
        with caplog.at_level(logging.ERROR):
            await seed_tasks.process_tournament_seed_tasks(TID, CONFIG)

        processing.replace.assert_not_awaited()
        processing.update.assert_not_awaited()
        assert "is not replaced" in caplog.text

    async def test_the_durable_marker_counts_after_the_round_row_is_detached(self, processing):
        processing.state["attached"] = False  # a detached tournament_tasks row
        processing.seed.attached_round_id = ROUND_ID

        await seed_tasks.process_tournament_seed_tasks(TID, CONFIG)

        processing.replace.assert_not_awaited()

    async def test_an_unattached_boss_round_seed_is_still_replaced(self, processing):
        processing.state["attached"] = False

        await seed_tasks.process_tournament_seed_tasks(TID, CONFIG)

        processing.replace.assert_awaited_once()

    async def test_an_attached_seed_moves_to_ready_when_its_task_succeeds(self, processing):
        processing.task.status = TaskStatus.SUCCESS

        await seed_tasks.process_tournament_seed_tasks(TID, CONFIG)

        processing.replace.assert_not_awaited()
        processing.update.assert_awaited_once()
        assert processing.update.await_args.args[3] == SeedStatus.READY


class TestBossAssignmentOnTheBossRound:
    """The boss is assigned to every boss-round task except a quant task the round plays for a failed seed role."""

    @pytest.fixture
    def assigned(self, monkeypatch):
        seed_task, own_task = uuid.uuid4(), uuid.uuid4()
        assigned: list[tuple[str, str]] = []

        async def assign(task_id, node, _db):
            assigned.append((str(task_id), node.hotkey))

        async def seed_of(task_id, _db):
            if str(task_id) != str(seed_task):
                return None
            return TournamentSeedTask(tournament_id=TID, role=SeedRole.BOSS_R2_FREEFORM, task_id=seed_task)

        round_tasks = [
            SimpleNamespace(task_id=str(seed_task), pair_id=PAIR_ID),
            SimpleNamespace(task_id=str(own_task), pair_id=PAIR_ID),
        ]
        monkeypatch.setattr(tm, "get_tournament_tasks", AsyncMock(return_value=round_tasks))
        monkeypatch.setattr(tm.task_sql, "get_nodes_assigned_to_task", AsyncMock(return_value=[]))
        monkeypatch.setattr(tm.task_sql, "assign_node_to_task", assign)
        monkeypatch.setattr(tm.task_sql, "set_expected_repo_name", AsyncMock())
        monkeypatch.setattr(tm, "get_node_by_hotkey", AsyncMock(side_effect=lambda hotkey, _db: SimpleNamespace(hotkey=hotkey)))
        monkeypatch.setattr(tm.seeds_sql, "get_seed_task_by_task_id", seed_of)
        return SimpleNamespace(seed_task=str(seed_task), own_task=str(own_task), assigned=assigned)

    @pytest.mark.parametrize(
        ("tournament_type", "boss_on_own_task"), [(TournamentType.QUANT, False), (TournamentType.TEXT, True)]
    )
    async def test_the_boss_is_not_assigned_to_the_task_of_a_failed_quant_role(
        self, assigned, monkeypatch, tournament_type, boss_on_own_task
    ):
        monkeypatch.setattr(
            tm, "get_tournament", AsyncMock(return_value=TournamentData(tournament_id=TID, tournament_type=tournament_type))
        )
        challenger = "5Challenger"
        round_structure = KnockoutRound(pairs=[(challenger, EMISSION_BURN_HOTKEY)], round_id=ROUND_ID, round_number=4)

        await tm.assign_nodes_to_tournament_tasks(TID, round_structure, None, is_final_round=True)

        assert (assigned.seed_task, EMISSION_BURN_HOTKEY) in assigned.assigned
        assert (assigned.own_task, challenger) in assigned.assigned
        assert ((assigned.own_task, EMISSION_BURN_HOTKEY) in assigned.assigned) is boss_on_own_task
