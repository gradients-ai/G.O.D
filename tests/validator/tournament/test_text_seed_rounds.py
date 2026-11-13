"""Text tournament seeds and rounds 1-3: the boss trains four seed tasks at tournament start, each of rounds 1-3 attaches
its seed (the boss's run is the seed, the round's miners train the same task), standings ignore the boss, and a seed's
finalisation is held until it is attached and every player is done.

Same faking style as test_seed_tasks.py: an in-memory seed store and monkeypatched SQL functions.
"""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import UUID
from uuid import uuid4

import pytest

import validator.lifecycle.tasks as lifecycle_tasks
import validator.tournament.round_results as rr
import validator.tournament.tournament_manager as tm
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from tests.validator.tournament.quant_source_fakes import FakeQuantSourceStore
from tests.validator.tournament.test_compression_rounds import _SCHEMA
from tests.validator.tournament.test_compression_rounds import _hidden
from tests.validator.tournament.test_compression_rounds import cluster  # noqa: F401  (fixture)
from tests.validator.tournament.test_seed_adapter_merge import CONFIG as MERGE_CONFIG
from tests.validator.tournament.test_seed_adapter_merge import FULL_MODEL_FILES
from tests.validator.tournament.test_seed_adapter_merge import MERGED
from tests.validator.tournament.test_seed_adapter_merge import World
from tests.validator.tournament.test_seed_tasks import PINNED
from tests.validator.tournament.test_seed_tasks import TID
from tests.validator.tournament.test_seed_tasks import _instruct
from tests.validator.tournament.test_seed_tasks import _live
from tests.validator.tournament.test_seed_tasks import env  # noqa: F401  (fixture)
from validator.db.sql import submissions_and_scoring as submissions_and_scoring_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournament_seeds as seeds_sql
from validator.db.sql import tournaments as tournament_sql
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.tournament import constants as t_cst
from validator.tournament import orchestrator
from validator.tournament import seed_tasks
from validator.tournament import task_creator
from validator.tournament.models import ANY_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import ATTACHED_SEED_ROLES
from validator.tournament.models import BOSS_ROUND_SEED_ROLES
from validator.tournament.models import QUANT_ROUND_SEED_ROLES
from validator.tournament.models import TEXT_ROUND_SEED_ROLES
from validator.tournament.models import TEXT_SEED_ROLES
from validator.tournament.models import Group
from validator.tournament.models import GroupRound
from validator.tournament.models import KnockoutRound
from validator.tournament.models import QuantSourceSlot
from validator.tournament.models import QuantTaskSource
from validator.tournament.models import RoundType
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus


TEXT = TournamentType.TEXT
MINERS = [f"hk_{i:03d}" for i in range(12)]
ROUND_ID = "round_x"


class TextWorld:
    """The seed_tasks `env` fixture plus what a text tournament adds: its type, the continuous lineage and a round registry."""

    def __init__(self, seed_env, monkeypatch) -> None:
        self.env = seed_env
        self.monkeypatch = monkeypatch
        self.tournament_type = TEXT
        self.winner: str | None = None
        self.status = TournamentStatus.ACTIVE
        self.continuous_calls: list[tuple[str, str]] = []
        self.continuous_error: Exception | None = None
        self.registry: dict = {}  # task_id -> TournamentTask
        self.boss_status: dict[str, str] = {}  # task id -> the boss's training status on it
        self.trained: dict[str, str] = {}

        async def make_continuous(config, lineage, seed_model):
            self.continuous_calls.append((lineage, seed_model))
            if self.continuous_error is not None:
                raise self.continuous_error
            task = _instruct()
            seed_env.tasks[task.task_id] = task
            return task

        async def get_tournament(tournament_id, _db):
            return TournamentData(
                tournament_id=tournament_id, tournament_type=self.tournament_type, status=self.status, winner_hotkey=self.winner
            )

        async def training_status(task_id, hotkeys, _db):
            status = self.boss_status.get(str(task_id))
            return {EMISSION_BURN_HOTKEY: status} if status else {}

        async def all_training_status(task_id, _db):
            return self.trained

        monkeypatch.setattr(seed_tasks, "create_continuous_sft_task", make_continuous)
        monkeypatch.setattr(tournament_sql, "get_tournament", get_tournament)
        monkeypatch.setattr(tournament_sql, "get_training_status_for_task_and_hotkeys", training_status)
        monkeypatch.setattr(tournament_sql, "get_training_status_for_task", all_training_status)
        monkeypatch.setattr(submissions_and_scoring_sql, "get_task_node_losses", AsyncMock(return_value=[]))

        async def existing(round_id, config, group_id=None, pair_id=None):
            return [t for t in self.registry.values() if t.round_id == round_id and (group_id is None or t.group_id == group_id)]

        async def get_existing(tournament_tasks, config):
            return [seed_env.tasks[UUID(str(t.task_id))] for t in tournament_tasks]

        async def register(task, tournament_id, round_id, config, group_id=None, pair_id=None):
            self.registry[task.task_id] = TournamentTask(
                tournament_id=tournament_id, round_id=round_id, task_id=task.task_id, group_id=group_id, pair_id=pair_id
            )
            seed_env.store.attached.add(task.task_id)

        async def registered_row(task_id, _db):
            return next((t for t in self.registry.values() if str(t.task_id) == str(task_id)), None)

        async def detach(task_id, tournament_id, _db):
            self.registry = {k: v for k, v in self.registry.items() if str(k) != str(task_id)}
            seed_env.store.attached.discard(UUID(str(task_id)))
            seed_env.assigned.pop(str(task_id), None)

        monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", existing)
        monkeypatch.setattr(task_creator, "_get_existing_tasks", get_existing)
        monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", register)
        monkeypatch.setattr(tournament_sql, "get_tournament_task_by_task_id", registered_row)
        monkeypatch.setattr(task_sql, "detach_task_from_tournament", detach)

    async def create_seeds(self) -> dict[SeedRole, TournamentSeedTask]:
        seeds = await seed_tasks.create_seed_tasks(TID, self.env.config, TEXT)
        return {seed.role: seed for seed in seeds}

    def round_of(self, number: int) -> GroupRound:
        return GroupRound(groups=[Group(member_ids=MINERS[:4])], round_id=f"{ROUND_ID}_{number}", round_number=number)


@pytest.fixture
def world(env, monkeypatch) -> TextWorld:  # noqa: F811
    return TextWorld(env, monkeypatch)


def _seed_row(world: TextWorld, role: SeedRole) -> TournamentSeedTask:
    return _live(world.env)[role]


class TestSeedCreation:
    async def test_a_text_tournament_creates_exactly_the_four_text_roles_with_the_boss_assigned(self, world):
        seeds = await world.create_seeds()

        assert list(seeds) == list(seed_tasks.TEXT_SEED_CREATION_ORDER)
        assert set(seeds) == TEXT_SEED_ROLES
        assert {row.role for row in world.env.store.rows} == TEXT_SEED_ROLES
        for seed in seeds.values():
            assert world.env.assigned[str(seed.task_id)] == [EMISSION_BURN_HOTKEY]
            assert world.env.expected[str(seed.task_id)] == f"tournament-{TID}-{seed.task_id}-{EMISSION_BURN_HOTKEY[:8]}"
            assert (seed.boss_repo, seed.boss_commit) == (PINNED.training_repo, PINNED.training_commit_hash)

    async def test_the_round_seeds_are_quant_compatible_calibrated_instruct_tasks_in_their_bands(self, world):
        await world.create_seeds()

        assert world.env.pool_calls == [spec.model_size_b for spec in t_cst.TEXT_ROUND_PLAN]
        assert len(world.env.instruct_calls) == 3
        for _, kwargs in world.env.instruct_calls:
            assert kwargs == dict(enable_kl=False, allow_augmentation=True, carve_calibration=True)
        assert world.continuous_calls == [next(iter(t_cst.CONTINUOUS_SFT_LINEAGES.items()))]

    async def test_creating_again_creates_nothing_new(self, world):
        first = await world.create_seeds()
        second = await world.create_seeds()

        assert {role: s.task_id for role, s in first.items()} == {role: s.task_id for role, s in second.items()}
        assert len(world.env.store.rows) == 4 and len(world.env.instruct_calls) == 3 and len(world.continuous_calls) == 1

    def test_a_quant_tournament_creates_its_own_seven_compression_roles(self):
        assert seed_tasks.seed_creation_order(TournamentType.QUANT) == seed_tasks.SEED_CREATION_ORDER
        assert set(seed_tasks.SEED_CREATION_ORDER) == BOSS_ROUND_SEED_ROLES | QUANT_ROUND_SEED_ROLES
        assert len(seed_tasks.SEED_CREATION_ORDER) == 7
        assert not set(seed_tasks.SEED_CREATION_ORDER) & TEXT_SEED_ROLES
        assert seed_tasks.seed_creation_order(TEXT) == seed_tasks.TEXT_SEED_CREATION_ORDER

    @pytest.mark.parametrize("lineages", [{}, {"a": "org/a", "b": "org/b"}])
    async def test_the_continuous_seed_needs_exactly_one_lineage(self, world, monkeypatch, lineages):
        monkeypatch.setattr(t_cst, "CONTINUOUS_SFT_LINEAGES", lineages)

        with pytest.raises(ValueError, match="exactly one continuous-SFT lineage"):
            await seed_tasks._draw_text_continuous(world.env.config)
        seeds = await world.create_seeds()

        assert set(seeds) == TEXT_ROUND_SEED_ROLES  # the role is retried every cycle, the three others exist
        assert world.continuous_calls == []

    async def test_the_dataset_drawn_roles_are_the_round_seeds_not_the_continuous_one(self, world):
        assert TEXT_ROUND_SEED_ROLES <= seed_tasks.DATASET_DRAWN_ROLES
        assert SeedRole.TEXT_CONTINUOUS not in seed_tasks.DATASET_DRAWN_ROLES


class TestFirstRoundCreation:
    @pytest.fixture
    def first_round(self, world, monkeypatch):
        monkeypatch.setattr(tm, "get_tournament_rounds", AsyncMock(return_value=[]))
        monkeypatch.setattr(tm, "get_tournament", AsyncMock(return_value=TournamentData(tournament_id=TID, tournament_type=TEXT)))
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

    async def test_no_first_round_until_all_four_seeds_exist_and_a_retry_only_fills_the_gap(self, world, first_round):
        world.continuous_error = RuntimeError("content service down")

        assert await tm.create_first_round_for_active_tournament(TID, world.env.config, None) is False
        first_round.assert_not_awaited()
        assert {row.role for row in world.env.store.rows} == TEXT_ROUND_SEED_ROLES
        instruct_calls = len(world.env.instruct_calls)

        world.continuous_error = None
        assert await tm.create_first_round_for_active_tournament(TID, world.env.config, None) is True

        first_round.assert_awaited_once()
        assert {row.role for row in world.env.store.rows} == TEXT_SEED_ROLES
        assert len(world.env.instruct_calls) == instruct_calls  # the three existing seeds were kept

    async def test_a_restart_after_creation_creates_nothing_new(self, world, first_round):
        await tm.create_first_round_for_active_tournament(TID, world.env.config, None)
        rows = list(world.env.store.rows)

        await tm.create_first_round_for_active_tournament(TID, world.env.config, None)

        assert [(r.role, r.attempt, r.task_id) for r in world.env.store.rows] == [(r.role, r.attempt, r.task_id) for r in rows]


class TestAttachingASeedToItsRound:
    @pytest.mark.parametrize("number", [1, 2, 3])
    async def test_each_round_attaches_the_matching_seed_without_drawing_a_task(self, world, number):
        seeds = await world.create_seeds()
        spec = t_cst.TEXT_ROUND_PLAN[number - 1]
        drawn = len(world.env.tasks)

        task_ids = await task_creator.create_text_tournament_tasks(world.round_of(number), TID, world.env.config)

        assert task_ids == [str(seeds[spec.seed_role].task_id)]
        assert len(world.env.tasks) == drawn
        round_id = f"{ROUND_ID}_{number}"
        (registered,) = world.registry.values()
        assert (registered.round_id, registered.group_id) == (round_id, f"{round_id}_group_001")
        assert _seed_row(world, spec.seed_role).attached_round_id == round_id

        again = await task_creator.create_text_tournament_tasks(world.round_of(number), TID, world.env.config)

        assert again == task_ids and len(world.registry) == 1

    async def test_a_stop_between_registration_and_the_marker_is_repaired_by_the_next_pass(self, world, monkeypatch):
        seeds = await world.create_seeds()
        real_mark = seeds_sql.mark_seed_attached
        monkeypatch.setattr(task_creator.seeds_sql, "mark_seed_attached", AsyncMock(side_effect=RuntimeError("stopped")))

        with pytest.raises(RuntimeError, match="stopped"):
            await task_creator.create_text_tournament_tasks(world.round_of(1), TID, world.env.config)

        assert len(world.registry) == 1 and _seed_row(world, SeedRole.TEXT_R1).attached_round_id is None
        monkeypatch.setattr(task_creator.seeds_sql, "mark_seed_attached", real_mark)

        again = await task_creator.create_text_tournament_tasks(world.round_of(1), TID, world.env.config)

        assert again == [str(seeds[SeedRole.TEXT_R1].task_id)] and len(world.registry) == 1
        assert _seed_row(world, SeedRole.TEXT_R1).attached_round_id == f"{ROUND_ID}_1"

    @pytest.mark.parametrize("number", [1, 2, 3])
    async def test_a_role_without_a_live_seed_keeps_the_round_pending(self, world, number):
        await world.create_seeds()
        role = t_cst.TEXT_ROUND_PLAN[number - 1].seed_role
        world.env.store.rows = [row for row in world.env.store.rows if row.role != role]

        with pytest.raises(task_creator.MissingBossRoundSeeds) as raised:
            await task_creator.create_text_tournament_tasks(world.round_of(number), TID, world.env.config)

        assert raised.value.roles == [role.value] and "boss round" not in str(raised.value)
        assert world.registry == {}

    async def test_the_final_round_of_a_seeded_tournament_is_built_by_the_seeded_boss_round(self, world, monkeypatch):
        await world.create_seeds()
        seeded = AsyncMock(return_value=[SimpleNamespace(task_id=uuid4())])
        unseeded = AsyncMock()
        monkeypatch.setattr(task_creator, "_create_seeded_text_boss_round_tasks", seeded)
        monkeypatch.setattr(task_creator, "_create_new_text_boss_round_tasks", unseeded)
        knockout = KnockoutRound(pairs=[("a", EMISSION_BURN_HOTKEY)], round_id="round_final", round_number=4)

        await task_creator.create_text_tournament_tasks(knockout, TID, world.env.config, is_final_round=True)

        seeded.assert_awaited_once()
        unseeded.assert_not_awaited()

    async def test_an_unseeded_text_tournament_keeps_the_unseeded_group_round(self, world, monkeypatch):
        group_tasks = AsyncMock(return_value=[SimpleNamespace(task_id=uuid4())])
        monkeypatch.setattr(task_creator, "_create_group_text_tasks", group_tasks)

        await task_creator.create_text_tournament_tasks(world.round_of(2), TID, world.env.config)

        group_tasks.assert_awaited_once()
        assert world.registry == {}

    async def test_nodes_are_assigned_to_the_attached_seed_and_the_boss_is_left_alone(self, world, monkeypatch):
        seeds = await world.create_seeds()
        await task_creator.create_text_tournament_tasks(world.round_of(1), TID, world.env.config)
        task_id = str(seeds[SeedRole.TEXT_R1].task_id)
        monkeypatch.setattr(tm, "get_tournament_tasks", AsyncMock(return_value=list(world.registry.values())))
        monkeypatch.setattr(tm, "get_node_by_hotkey", AsyncMock(side_effect=lambda hk, _db: SimpleNamespace(hotkey=hk)))

        await tm.assign_nodes_to_tournament_tasks(TID, world.round_of(1), world.env.config.psql_db)

        assert world.env.assigned[task_id] == [EMISSION_BURN_HOTKEY, *MINERS[:4]]
        assert world.env.expected[task_id] == f"tournament-{TID}-{task_id}-{MINERS[3][:8]}"  # the last one assigned


@pytest.fixture
def inserted(monkeypatch) -> dict:
    captured: dict = {}

    async def fake_node(hotkey, psql_db):
        return SimpleNamespace(hotkey=hotkey)

    async def fake_round(round_data, psql_db):
        captured["round"] = round_data

    async def fake_groups(round_id, round_structure, psql_db):
        captured["structure"] = round_structure

    async def fake_pairs(round_id, pairs, psql_db):
        captured["pairs"] = pairs

    monkeypatch.setattr(tm, "get_node_by_hotkey", fake_node)
    monkeypatch.setattr(tm, "insert_tournament_round", fake_round)
    monkeypatch.setattr(tm, "insert_tournament_groups_with_members", fake_groups)
    monkeypatch.setattr(tm, "insert_tournament_pairs", fake_pairs)
    return captured


def _completed(number: int) -> TournamentRoundData:
    return TournamentRoundData(round_id="r", tournament_id=TID, round_number=number, round_type=RoundType.GROUP)


async def _next(winners: list[str], completed: TournamentRoundData, tournament_type: TournamentType = TEXT) -> None:
    tournament = TournamentData(tournament_id=TID, tournament_type=tournament_type)
    await tm.create_next_round(tournament, completed, list(winners), MagicMock(), psql_db=None)


class TestRoundStructure:
    async def test_seeded_rounds_two_and_three_are_single_group_rounds_without_the_boss(self, world, inserted):
        await world.create_seeds()

        await _next([*MINERS[:7], EMISSION_BURN_HOTKEY], _completed(1))
        assert inserted["round"].round_number == 2 and inserted["round"].is_final_round is False
        (group,) = inserted["structure"].groups
        assert sorted(group.member_ids) == MINERS[:7]

        await _next(MINERS[:4], _completed(2))
        assert inserted["round"].round_number == 3 and inserted["round"].round_type == RoundType.GROUP
        assert inserted["round"].is_final_round is False
        (group,) = inserted["structure"].groups
        assert sorted(group.member_ids) == MINERS[:4]

    async def test_a_lone_survivor_of_round_two_still_plays_round_three(self, world, inserted):
        await world.create_seeds()

        await _next(MINERS[:1], _completed(2))

        assert inserted["round"].round_number == 3 and inserted["round"].is_final_round is False
        assert isinstance(inserted["structure"], GroupRound)
        assert inserted["structure"].groups[0].member_ids == MINERS[:1]

    async def test_after_round_three_the_winner_meets_the_boss_in_the_final_knockout(self, world, inserted):
        await world.create_seeds()

        await _next(MINERS[:1], _completed(3))

        assert inserted["round"].round_number == 4 and inserted["round"].round_type == RoundType.KNOCKOUT
        assert inserted["round"].is_final_round is True
        assert [sorted(pair) for pair in inserted["pairs"]] == [sorted([MINERS[0], EMISSION_BURN_HOTKEY])]

    async def test_an_unseeded_text_tournament_keeps_its_knockout_after_round_one(self, world, inserted):
        await _next(MINERS[:2], _completed(1))  # no seed rows: an unseeded text tournament

        assert inserted["round"].round_type == RoundType.KNOCKOUT and inserted["round"].round_number == 2
        assert [sorted(pair) for pair in inserted["pairs"]] == [sorted(MINERS[:2])]

    async def test_an_unseeded_text_lone_survivor_goes_straight_to_the_final(self, world, inserted):
        await _next(MINERS[:1], _completed(2))

        assert inserted["round"].is_final_round is True and inserted["round"].round_type == RoundType.KNOCKOUT

    async def test_an_image_tournament_with_a_three_task_round_one_keeps_its_knockout_round_two(
        self, world, inserted, monkeypatch
    ):
        world.tournament_type = TournamentType.IMAGE
        await world.create_seeds()  # even stray seed rows never change an image tournament
        monkeypatch.setattr(rr, "is_planned_image_tournament", AsyncMock(return_value=False))

        await _next(MINERS[:2], _completed(1), TournamentType.IMAGE)

        assert inserted["round"].round_type == RoundType.KNOCKOUT

    @pytest.mark.parametrize("completed_round", [1, 2])
    async def test_a_planned_image_tournament_plays_rounds_two_and_three_as_single_group_rounds_without_the_boss(
        self, world, inserted, monkeypatch, completed_round
    ):
        monkeypatch.setattr(rr, "is_planned_image_tournament", AsyncMock(return_value=True))

        await _next([*MINERS[:1], EMISSION_BURN_HOTKEY], _completed(completed_round), TournamentType.IMAGE)

        assert inserted["round"].round_number == completed_round + 1
        assert inserted["round"].round_type == RoundType.GROUP and inserted["round"].is_final_round is False
        (group,) = inserted["structure"].groups
        assert group.member_ids == MINERS[:1]

    def test_the_organiser_flag_alone_makes_rounds_one_to_three_single_group(self):
        nodes = [SimpleNamespace(hotkey=hk) for hk in MINERS[:6]]
        for number in (1, 2, 3):
            result = tm.organise_tournament_round(nodes, MagicMock(), None, "r", number, False, single_group_rounds=True)
            assert isinstance(result, GroupRound) and len(result.groups) == 1
        final = tm.organise_tournament_round(nodes[:1], MagicMock(), None, "r", 4, True, single_group_rounds=True)
        assert isinstance(final, KnockoutRound)


def _text(hotkey: str, loss: float) -> MinerResultsText:
    return MinerResultsText(hotkey=hotkey, test_loss=loss, synth_loss=loss, is_finetune=True, task_type=TaskType.INSTRUCTTEXTTASK)


class TestStandings:
    @pytest.fixture
    def standings(self, world, monkeypatch):
        def wire(miners: list[str], boss_loss: float = 0.1):
            members = [SimpleNamespace(hotkey=hk) for hk in miners]
            results = [_text(hk, 1.0 + i / 100) for i, hk in enumerate(miners)] + [_text(EMISSION_BURN_HOTKEY, boss_loss)]

            async def fake_results(task_id, _db):
                return [r.model_copy() for r in results]

            monkeypatch.setattr(rr, "get_task_results_for_ranking", fake_results)
            monkeypatch.setattr(rr, "get_tournament_group_members", AsyncMock(return_value=members))
            monkeypatch.setattr(rr, "get_task", AsyncMock(return_value=SimpleNamespace(task_type=TaskType.INSTRUCTTEXTTASK)))
            monkeypatch.setattr(
                rr, "get_tournament", AsyncMock(return_value=TournamentData(tournament_id=TID, tournament_type=TEXT))
            )
            round_ids = {1: "r1", 2: "r2", 3: "r3"}

            async def fake_rounds(tournament_id, _db):
                return [
                    TournamentRoundData(round_id=rid, tournament_id=TID, round_number=n, round_type=RoundType.GROUP)
                    for n, rid in round_ids.items()
                ]

            async def fake_tasks(round_id, _db):
                return [TournamentTask(tournament_id=TID, round_id=round_id, task_id=uuid4(), group_id="g")]

            monkeypatch.setattr(rr, "get_tournament_rounds", fake_rounds)
            monkeypatch.setattr(rr, "get_tournament_tasks", fake_tasks)
            return [TournamentTask(tournament_id=TID, round_id="r", task_id=uuid4(), group_id="g")]

        return wire

    @pytest.mark.parametrize("number, advance", [(1, 8), (2, 4), (3, 1)])
    async def test_the_top_of_the_members_advance_and_the_boss_never_takes_a_slot(self, world, standings, number, advance):
        await world.create_seeds()
        tasks = standings(MINERS)  # the boss's loss (0.1) beats every miner on the same task

        winners = await rr.get_group_winners(_completed(number), tasks, psql_db=None)

        assert winners == MINERS[:advance]
        assert EMISSION_BURN_HOTKEY not in winners

    async def test_fewer_miners_than_the_advance_count_all_advance(self, world, standings):
        await world.create_seeds()
        tasks = standings(MINERS[:3])

        assert await rr.get_group_winners(_completed(1), tasks, psql_db=None) == MINERS[:3]
        assert await rr.get_group_winners(_completed(3), tasks, psql_db=None) == MINERS[:1]

    async def test_the_boss_result_does_not_change_the_miners_ranks(self, world, standings):
        await world.create_seeds()
        tasks = standings(MINERS)
        best = await rr.planned_round_ordered_hotkeys(_completed(1), tasks, None)
        standings(MINERS, boss_loss=50.0)

        assert await rr.planned_round_ordered_hotkeys(_completed(1), tasks, None) == best == MINERS

    async def test_the_round_two_backfill_uses_the_same_ordering(self, world, standings, monkeypatch):
        await world.create_seeds()
        tasks = standings(MINERS)
        monkeypatch.setattr(rr, "get_tournament_tasks", AsyncMock(return_value=tasks))
        tournament = TournamentData(tournament_id=TID, tournament_type=TEXT)

        selected = await rr.select_round2_entrants(tournament, _completed(1), {MINERS[1]}, psql_db=None)

        assert selected == [hk for hk in MINERS if hk != MINERS[1]][: t_cst.TEXT_ROUND_PLAN[0].advance]

    async def test_an_unseeded_text_round_one_keeps_the_top_two(self, world, monkeypatch):
        ordered = MINERS[:6]
        monkeypatch.setattr(rr, "get_tournament_tasks", AsyncMock(return_value=[]))
        monkeypatch.setattr(rr, "_small_tournament_ordered_hotkeys", AsyncMock(return_value=ordered))
        tournament = TournamentData(tournament_id=TID, tournament_type=TEXT)

        selected = await rr.select_round2_entrants(tournament, _completed(1), set(), psql_db=None)

        assert selected == ordered[: t_cst.SMALL_TOURNAMENT_ADVANCE]


class TestTheFinalisationHold:
    def _attach(self, world, role, attached: bool):
        seed = _seed_row(world, role)
        if attached:
            world.env.store.attached.add(seed.task_id)
            seed.attached_round_id = ROUND_ID
        return seed

    async def _ready(self, world, seed) -> bool:
        return await seeds_sql.boss_round_seed_ready_to_finalise(seed.task_id, world.env.config.psql_db)

    async def test_held_while_unattached(self, world):
        await world.create_seeds()
        seed = self._attach(world, SeedRole.TEXT_R1, attached=False)
        world.env.assigned[str(seed.task_id)] = [EMISSION_BURN_HOTKEY, "m1"]
        world.trained = {EMISSION_BURN_HOTKEY: "success", "m1": "success"}

        assert await self._ready(world, seed) is False

    async def test_held_without_a_miner_or_while_a_miner_is_unfinished_then_released(self, world):
        await world.create_seeds()
        seed = self._attach(world, SeedRole.TEXT_R2, attached=True)
        world.env.assigned[str(seed.task_id)] = [EMISSION_BURN_HOTKEY]
        world.trained = {EMISSION_BURN_HOTKEY: "success"}
        assert await self._ready(world, seed) is False

        world.env.assigned[str(seed.task_id)] = [EMISSION_BURN_HOTKEY, "m1", "m2"]
        world.trained = {EMISSION_BURN_HOTKEY: "success", "m1": "success", "m2": TrainingStatus.TRAINING.value}
        assert await self._ready(world, seed) is False

        world.trained["m2"] = TrainingStatus.FAILURE.value
        assert await self._ready(world, seed) is True

    @pytest.mark.parametrize("decided", ["winner", "completed", "cancelled"])
    async def test_released_when_the_tournament_was_decided_without_attaching_it(self, world, decided):
        await world.create_seeds()
        seed = self._attach(world, SeedRole.TEXT_R3, attached=False)
        if decided == "winner":
            world.winner = EMISSION_BURN_HOTKEY
        else:
            world.status = TournamentStatus(decided)

        assert await self._ready(world, seed) is True

    async def test_an_unattached_boss_round_seed_is_released_only_once_the_tournament_is_decided(self, world):
        row = TournamentSeedTask(tournament_id=TID, role=SeedRole.BOSS_CHAINED_FIXED, task_id=uuid4())
        world.env.store.rows.append(row)
        assert await self._ready(world, row) is False

        world.winner = EMISSION_BURN_HOTKEY  # no round will attach it now: it closes with the boss's stored result
        assert await self._ready(world, row) is True

    async def test_a_non_seed_task_is_never_held(self, world):
        assert await seeds_sql.boss_round_seed_ready_to_finalise(uuid4(), world.env.config.psql_db) is True

    async def test_the_lifecycle_hold_covers_text_seeds_only(self, world):
        await world.create_seeds()
        seed = self._attach(world, SeedRole.TEXT_R1, attached=False)
        text_task = SimpleNamespace(status=TaskStatus.TRAINING, task_id=seed.task_id, task_type=TaskType.INSTRUCTTEXTTASK)
        organic = SimpleNamespace(status=TaskStatus.TRAINING, task_id=uuid4(), task_type=TaskType.INSTRUCTTEXTTASK)

        assert await lifecycle_tasks._held_boss_round_seed(text_task, world.env.config) is True
        assert await lifecycle_tasks._held_boss_round_seed(organic, world.env.config) is False

    async def test_a_held_text_seed_is_not_moved_to_preevaluation(self, world, monkeypatch):
        await world.create_seeds()
        held = _seed_row(world, SeedRole.TEXT_R1)
        plain = _instruct(status=TaskStatus.TRAINING)
        world.env.tasks[plain.task_id] = plain
        held_task = world.env.tasks[held.task_id]
        held_task.status = TaskStatus.TRAINING
        update = AsyncMock()
        monkeypatch.setattr(
            tournament_sql, "get_tasks_with_all_training_completed", AsyncMock(return_value=[held.task_id, plain.task_id])
        )
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=TID))
        monkeypatch.setattr(task_sql, "update_task", update)

        await orchestrator._move_completed_tasks_to_preevaluation(world.env.config)

        assert [call.args[0].task_id for call in update.await_args_list] == [plain.task_id]
        assert plain.status == TaskStatus.PREEVALUATION and held_task.status == TaskStatus.TRAINING


class TestEvaluatingATextSeed:
    async def _seed(self, world, role=SeedRole.TEXT_R1, *, attached=False, task_status=TaskStatus.TRAINING.value, boss=None):
        await world.create_seeds()
        seed = _seed_row(world, role)
        if attached:
            world.env.store.attached.add(seed.task_id)
            seed.attached_round_id = ROUND_ID
        world.env.tasks[seed.task_id].status = task_status
        if boss:
            world.boss_status[str(seed.task_id)] = boss
        return seed, world.env.tasks[seed.task_id]

    async def _evaluate(self, world, seed, task):
        return await seed_tasks._evaluate_seed(seed, task, world.env.config)

    async def test_a_missing_task_is_replaced(self, world):
        seed, _ = await self._seed(world)

        assert await self._evaluate(world, seed, None) == (None, "task is gone")

    @pytest.mark.parametrize("attached", [False, True])
    async def test_a_prep_failure_is_replaced_even_when_attached(self, world, attached):
        seed, task = await self._seed(world, attached=attached, task_status=TaskStatus.PREP_TASK_FAILURE.value)

        status, reason = await self._evaluate(world, seed, task)

        assert status is None and reason == f"task ended {TaskStatus.PREP_TASK_FAILURE.value}"

    def test_a_prep_rejection_is_uncharged_for_the_round_roles_and_charged_for_the_continuous_seed(self):
        failed = SimpleNamespace(status=TaskStatus.PREP_TASK_FAILURE.value)
        for role in TEXT_ROUND_SEED_ROLES:
            assert seed_tasks.is_prep_rejection(TournamentSeedTask(tournament_id=TID, role=role, task_id=uuid4()), failed)
        continuous = TournamentSeedTask(tournament_id=TID, role=SeedRole.TEXT_CONTINUOUS, task_id=uuid4())
        assert not seed_tasks.is_prep_rejection(continuous, failed)

    async def test_a_failed_task_is_replaced_only_while_unattached(self, world):
        seed, task = await self._seed(world, task_status=TaskStatus.FAILURE.value)
        assert await self._evaluate(world, seed, task) == (None, f"task ended {TaskStatus.FAILURE.value}")

        attached, task = await self._seed(world, SeedRole.TEXT_R2, attached=True, task_status=TaskStatus.FAILURE.value)
        assert await self._evaluate(world, attached, task) == (None, None)

    async def test_a_boss_failure_on_an_unattached_seed_with_replacements_left_replaces_it(self, world):
        seed, task = await self._seed(world, boss=TrainingStatus.FAILURE.value)

        status, reason = await self._evaluate(world, seed, task)

        assert status is None and "the boss's run failed" in reason

    async def test_a_boss_failure_with_no_charged_replacement_left_is_the_bosss_failed_entry(self, world, caplog):
        seed, task = await self._seed(world, boss=TrainingStatus.FAILURE.value)
        for attempt in range(1, t_cst.SEED_MAX_REPLACEMENTS + 1):
            world.env.store.rows.append(
                TournamentSeedTask(
                    tournament_id=TID, role=seed.role, attempt=attempt, task_id=uuid4(), status=SeedStatus.REPLACED
                )
            )
        seed.attempt = t_cst.SEED_MAX_REPLACEMENTS + 1

        with caplog.at_level("WARNING"):
            assert await self._evaluate(world, seed, task) == (SeedStatus.TRAINED, None)
        assert "failed entry" in caplog.text

    async def test_a_boss_failure_on_an_attached_seed_is_never_a_replacement_or_a_halt(self, world, monkeypatch):
        seed, _ = await self._seed(world, attached=True, boss=TrainingStatus.FAILURE.value)
        replace = AsyncMock()
        monkeypatch.setattr(seed_tasks, "replace_seed", replace)

        for _ in range(3):
            await seed_tasks.process_tournament_seed_tasks(TID, world.env.config)

        replace.assert_not_awaited()
        live = _live(world.env)[seed.role]
        assert (live.task_id, live.status) == (seed.task_id, SeedStatus.TRAINED)
        assert await seed_tasks.seed_halt_reason(TID, None) is None

    async def test_the_bosss_success_while_training_makes_the_seed_trained(self, world):
        seed, task = await self._seed(world, boss=TrainingStatus.SUCCESS.value)

        assert await self._evaluate(world, seed, task) == (SeedStatus.TRAINED, None)

    async def test_a_successful_task_is_ready(self, world):
        seed, task = await self._seed(world, attached=True, task_status=TaskStatus.SUCCESS.value, boss="success")
        seed.status = SeedStatus.TRAINED

        assert await self._evaluate(world, seed, task) == (SeedStatus.READY, None)

    async def test_a_trained_entry_waits_for_its_round_to_finish(self, world):
        seed, task = await self._seed(world, attached=True, boss="failure")
        seed.status = SeedStatus.TRAINED

        assert await self._evaluate(world, seed, task) == (None, None)

    async def test_nothing_to_report_while_the_boss_is_still_training(self, world):
        seed, task = await self._seed(world, SeedRole.TEXT_CONTINUOUS, boss=TrainingStatus.TRAINING.value)

        assert await self._evaluate(world, seed, task) == (None, None)

    async def test_an_attached_failure_is_never_redrawn_even_when_asked_to(self, world):
        seed, _ = await self._seed(world, attached=True, task_status=TaskStatus.FAILURE.value)

        await seed_tasks.process_tournament_seed_tasks(TID, world.env.config)
        assert await seed_tasks.replace_seed(seed, "task ended failure", world.env.config) is None

        assert len(world.env.store.rows) == 4  # nothing was redrawn
        world.env.alerts.assert_awaited()


class TestReplacingAnAttachedSeed:
    async def _attached_prep_failure(self, world):
        seeds = await world.create_seeds()
        old = seeds[SeedRole.TEXT_R1]
        old_id = old.task_id
        await task_creator.create_text_tournament_tasks(world.round_of(1), TID, world.env.config)
        world.env.assigned[str(old_id)] = [EMISSION_BURN_HOTKEY, "m1", "m2"]
        world.env.tasks[old_id].status = TaskStatus.PREP_TASK_FAILURE.value
        return old_id

    async def test_the_replacement_takes_over_the_round_registration_and_every_node(self, world):
        old_id = await self._attached_prep_failure(world)

        await seed_tasks.process_tournament_seed_tasks(TID, world.env.config)

        new = _seed_row(world, SeedRole.TEXT_R1)
        assert new.attempt == 2 and new.task_id != old_id and new.attached_round_id == f"{ROUND_ID}_1"
        old = next(row for row in world.env.store.rows if row.task_id == old_id)
        assert old.status == SeedStatus.REPLACED and old.prep_rejected is True
        (registered,) = world.registry.values()
        assert str(registered.task_id) == str(new.task_id) and registered.group_id == f"{ROUND_ID}_1_group_001"
        assert world.env.assigned[str(new.task_id)] == [EMISSION_BURN_HOTKEY, "m1", "m2"]
        assert world.env.expected[str(new.task_id)] == f"tournament-{TID}-{new.task_id}-m2"
        assert str(old_id) not in world.env.assigned

    async def test_an_interrupted_swap_is_finished_by_the_next_cycle(self, world, monkeypatch):
        old_id = await self._attached_prep_failure(world)
        real_swap = seed_tasks.swap_seed_task_registration
        monkeypatch.setattr(seed_tasks, "swap_seed_task_registration", AsyncMock(side_effect=RuntimeError("db down")))

        await seed_tasks.process_tournament_seed_tasks(TID, world.env.config)  # the draw happened, the swap failed

        assert next(row for row in world.env.store.rows if row.task_id == old_id).status != SeedStatus.REPLACED
        monkeypatch.setattr(seed_tasks, "swap_seed_task_registration", real_swap)

        await seed_tasks.process_tournament_seed_tasks(TID, world.env.config)

        new = _seed_row(world, SeedRole.TEXT_R1)
        (registered,) = world.registry.values()
        assert str(registered.task_id) == str(new.task_id) and new.attached_round_id == f"{ROUND_ID}_1"
        assert next(row for row in world.env.store.rows if row.task_id == old_id).status == SeedStatus.REPLACED

    async def test_a_seed_replaced_while_a_round_registered_it_is_swapped_out_of_the_round(self, world, monkeypatch):
        old_id = await self._attached_prep_failure(world)
        real_swap = seed_tasks.swap_seed_task_registration
        monkeypatch.setattr(seed_tasks, "swap_seed_task_registration", AsyncMock(side_effect=RuntimeError("db down")))
        await seed_tasks.process_tournament_seed_tasks(TID, world.env.config)  # the draw happened, the swap failed
        # The round registered the seed just as the seed loop replaced it: the row says replaced, the round still holds it.
        old = next(row for row in world.env.store.rows if row.task_id == old_id)
        old.status, old.prep_rejected = SeedStatus.REPLACED, True
        monkeypatch.setattr(seed_tasks, "swap_seed_task_registration", real_swap)

        await seed_tasks.process_tournament_seed_tasks(TID, world.env.config)

        new = _seed_row(world, SeedRole.TEXT_R1)
        (registered,) = world.registry.values()
        assert new.task_id != old_id and str(registered.task_id) == str(new.task_id)
        assert new.attached_round_id == f"{ROUND_ID}_1"
        assert old.status == SeedStatus.REPLACED and old.prep_rejected is True  # the row is left as it was

    async def test_the_completion_check_leaves_the_replacement_to_the_seed_loop(self, world, monkeypatch):
        old_id = await self._attached_prep_failure(world)
        replace = AsyncMock()
        monkeypatch.setattr(tm, "replace_tournament_task", replace)
        row = world.registry[old_id]
        task_obj = SimpleNamespace(status=TaskStatus.PREP_TASK_FAILURE.value, task_type=TaskType.INSTRUCTTEXTTASK, task_id=old_id)

        completed = await tm.is_tourn_task_completed(row, task_obj, world.env.config)

        assert completed == (False, "seed task failed prep; the seed is redrawn")
        replace.assert_not_awaited()

    async def test_an_ordinary_text_task_prep_failure_is_still_replaced_at_once(self, world, monkeypatch):
        replace = AsyncMock(return_value="new")
        monkeypatch.setattr(tm, "replace_tournament_task", replace)
        row = TournamentTask(tournament_id=TID, round_id="r", task_id=uuid4(), group_id="g")
        task_obj = SimpleNamespace(
            status=TaskStatus.PREP_TASK_FAILURE.value, task_type=TaskType.INSTRUCTTEXTTASK, task_id=row.task_id
        )

        completed, _ = await tm.is_tourn_task_completed(row, task_obj, world.env.config)

        assert completed is False
        replace.assert_awaited_once()

    async def test_replacing_a_seed_repoints_a_quant_task_chained_on_it_even_when_it_is_a_seed_of_another_tournament(
        self, world, monkeypatch
    ):
        sources = FakeQuantSourceStore().install(monkeypatch)
        seeds = await world.create_seeds()
        old = seeds[SeedRole.TEXT_R2]
        old_task = world.env.tasks[old.task_id]
        other = SimpleNamespace(
            task_id=uuid4(),
            task_type=TaskType.QUANTFREEFORMTASK,
            upstream_task_id=old.task_id,
            ds="x",
            status=TaskStatus.AWAITING_UPSTREAM,
        )
        world.env.tasks[other.task_id] = other
        world.env.store.rows.append(
            TournamentSeedTask(tournament_id="quant_t", role=SeedRole.BOSS_R2_FREEFORM, task_id=other.task_id)
        )
        sources.sources.append(
            QuantTaskSource(
                quant_task_id=other.task_id,
                quant_tournament_id="quant_t",
                slot=QuantSourceSlot.BOSS_R2_FREEFORM,
                source_tournament_id=TID,
                source_role=SeedRole.TEXT_R2,
                source_task_id=old.task_id,
            )
        )
        own_compression = SimpleNamespace(
            task_id=uuid4(),
            task_type=TaskType.QUANTFIXEDTASK,
            upstream_task_id=old.task_id,
            ds="x",
            status=TaskStatus.AWAITING_UPSTREAM,
        )
        world.env.tasks[own_compression.task_id] = own_compression
        world.env.store.rows.append(
            TournamentSeedTask(tournament_id=TID, role=SeedRole.BOSS_CHAINED_FIXED, task_id=own_compression.task_id)
        )
        world.env.tasks[old.task_id].status = TaskStatus.PREP_TASK_FAILURE.value

        new = await seed_tasks.replace_seed(old, "prep", world.env.config, prep_rejected=True)

        assert new is not None
        repointed = [call.args[0].task_id for call in world.env.repoint.call_args_list]
        assert repointed == [other.task_id] and old_task.task_id != new.task_id
        # The quant task's persisted link moved with it, to the replacement text seed task.
        assert sources.sources[0].source_task_id == new.task_id


class TestTrainingRowsOfAnAttachedSeed:
    @pytest.fixture
    def rows(self, monkeypatch):
        added = []

        async def add(assignments, _db):
            added.extend(assignments)

        monkeypatch.setattr(
            orchestrator.task_sql,
            "get_nodes_assigned_to_task",
            AsyncMock(return_value=[SimpleNamespace(hotkey=h) for h in (EMISSION_BURN_HOTKEY, "m1", "m2")]),
        )
        monkeypatch.setattr(
            orchestrator.tournament_sql, "get_training_status_for_task", AsyncMock(return_value={EMISSION_BURN_HOTKEY: "success"})
        )
        monkeypatch.setattr(orchestrator.tournament_sql, "get_tournament_id_by_task_id", AsyncMock(return_value=TID))
        monkeypatch.setattr(
            orchestrator.tournament_sql, "get_tournament_training_repo_and_commit", AsyncMock(return_value=PINNED)
        )
        monkeypatch.setattr(orchestrator.tournament_sql, "add_tournament_task_hotkey_pairs_for_training", add)
        return added

    def _task(self, status=TaskStatus.TRAINING):
        return _instruct(status=status)

    async def test_the_miners_get_rows_on_a_training_seed_and_the_boss_never_a_second(self, rows):
        await orchestrator._add_challenger_rows(SeedRole.TEXT_R1, self._task(), SimpleNamespace(psql_db=None))

        assert [a.hotkey for a in rows] == ["m1", "m2"]

    async def test_a_task_not_yet_training_gets_its_rows_when_it_moves_there(self, rows):
        await orchestrator._add_challenger_rows(
            SeedRole.TEXT_R1, self._task(TaskStatus.LOOKING_FOR_NODES), SimpleNamespace(psql_db=None)
        )

        assert rows == []

    async def test_the_cycle_covers_attached_text_seeds_of_active_tournaments_only(self, world, monkeypatch):
        seeds = await world.create_seeds()
        for role in (SeedRole.TEXT_R1, SeedRole.TEXT_CONTINUOUS):
            world.env.tasks[seeds[role].task_id].status = TaskStatus.TRAINING
        world.env.store.attached.add(seeds[SeedRole.TEXT_R1].task_id)
        add = AsyncMock()
        alert = AsyncMock()
        monkeypatch.setattr(orchestrator, "_add_challenger_rows", add)
        monkeypatch.setattr(orchestrator.boss_seed_eval, "alert_if_boss_evaluation_missing", alert)
        monkeypatch.setattr(orchestrator, "_training_status_of", AsyncMock(return_value="success"))
        monkeypatch.setattr(
            tournament_sql,
            "get_tournaments_with_status",
            AsyncMock(return_value=[TournamentData(tournament_id=TID, tournament_type=TEXT, status=TournamentStatus.ACTIVE)]),
        )

        await orchestrator._boss_round_seed_cycle(world.env.config)

        assert [call.args[0] for call in add.await_args_list] == [SeedRole.TEXT_R1]  # attached; the other is not yet
        alert.assert_not_awaited()  # the quant-reference check is for quant compressions only

    def test_every_attached_role_is_a_boss_round_a_text_or_a_quant_round_seed(self):
        # Every boss-round role and every round seed is attached after it was trained.
        assert ATTACHED_SEED_ROLES == ANY_BOSS_ROUND_SEED_ROLES | TEXT_SEED_ROLES | QUANT_ROUND_SEED_ROLES


class TestTheAdapterMerge:
    async def test_a_text_seed_with_an_adapter_output_is_merged_and_the_repo_stored(self, monkeypatch):
        world = World(monkeypatch, role=SeedRole.TEXT_R1, tournament_type=TEXT)
        world.task.status = TaskStatus.TRAINING  # a text seed is not SUCCESS until its round is over

        await world.merge_cycle()

        assert world.seed.merged_output_repo == MERGED
        world.dispatch.assert_awaited_once()
        world.replace.assert_not_awaited()

    async def test_a_full_model_output_is_left_alone(self, monkeypatch):
        world = World(monkeypatch, role=SeedRole.TEXT_R2, tournament_type=TEXT)
        world.hub[world.raw_repo] = FULL_MODEL_FILES

        await world.merge_cycle()

        assert world.seed.merged_output_repo is None
        world.dispatch.assert_not_awaited()
        assert str(world.seed.task_id) in orchestrator._seed_output_known_full

    async def test_the_merge_waits_for_the_bosss_run_to_succeed(self, monkeypatch):
        world = World(monkeypatch, role=SeedRole.TEXT_R1, tournament_type=TEXT)
        world.trained = TrainingStatus.TRAINING.value

        await world.merge_cycle()

        world.dispatch.assert_not_awaited()

    async def test_the_text_status_does_not_wait_for_the_merge(self, monkeypatch):
        world = World(monkeypatch, role=SeedRole.TEXT_R1, tournament_type=TEXT)
        world.task.status = TaskStatus.TRAINING

        assert await seed_tasks._evaluate_seed(world.seed, world.task, MERGE_CONFIG) == (SeedStatus.TRAINED, None)

    async def test_an_image_tournament_is_never_merged(self, monkeypatch):
        world = World(monkeypatch, role=SeedRole.TEXT_R1, tournament_type=TournamentType.IMAGE)

        await world.merge_cycle()

        world.dispatch.assert_not_awaited()


class TestHours:
    async def test_text_round_seeds_are_non_boss_tasks_and_the_continuous_seed_is_not(self, world):
        seeds = await world.create_seeds()

        async def non_boss(role) -> bool:
            return await orchestrator._is_non_boss_tournament_task(world.env.tasks[seeds[role].task_id], world.env.config)

        assert [await non_boss(role) for role in seed_tasks.TEXT_SEED_CREATION_ORDER] == [True, True, True, False]
        task = world.env.tasks[seeds[SeedRole.TEXT_R1].task_id]
        assert await orchestrator._is_boss_round_task(task, world.env.config) is False


class TestHiddenSeedSql:
    """The generated SQL text and parameters of get_hidden_seed_task_ids for the text roles (the rule itself runs against a
    real Postgres in test_compression_rounds)."""

    async def _sql(self, monkeypatch):
        captured = {}

        class _Connection:
            async def fetch(self, query, *args):
                captured["query"], captured["args"] = query, args
                return []

        class _Db:
            async def connection(self):
                class _Ctx:
                    async def __aenter__(self_inner):
                        return _Connection()

                    async def __aexit__(self_inner, *exc):
                        return False

                return _Ctx()

        task_id = uuid4()
        await seeds_sql.get_hidden_seed_task_ids([str(task_id)], _Db())
        return captured, task_id

    async def test_the_text_roles_have_one_condition_that_tells_the_two_branches_apart(self, monkeypatch):
        captured, task_id = await self._sql(monkeypatch)
        query = " ".join(captured["query"].split())

        roles = ", ".join(f"'{role.value}'" for role in sorted(TEXT_SEED_ROLES, key=lambda role: role.value))
        assert f"CASE WHEN h.role IN ({roles}) THEN CASE WHEN h.via_quant_task THEN" in query
        assert "FALSE AS via_quant_task" in query and "TRUE AS via_quant_task" in query
        # The chained quant task follows its own quant tournament (its source row first); the text round releases nothing.
        assert "COALESCE(qs.quant_tournament_id, own.tournament_id, tt.tournament_id) AS owner_tournament_id" in query
        assert "COALESCE(o.status, t.status) NOT IN ('completed', 'cancelled')" in query
        assert "r.round_id = h.attached_round_id" not in query
        # The text seed follows its attachment and the live source rows of running quant tournaments.
        assert "h.attached_round_id IS NULL AND t.status NOT IN ('completed', 'cancelled')" in query
        assert "qs.source_task_id = h.task_id AND NOT qs.superseded AND qt_t.status NOT IN ('completed', 'cancelled')" in query
        assert captured["args"] == ([task_id],)

    async def test_the_arms_of_the_existing_roles_are_unchanged(self, monkeypatch):
        captured, _ = await self._sql(monkeypatch)
        query = captured["query"]

        assert "WHEN 'input_small' THEN NOT EXISTS" in query and "r.round_number = 2" in query
        assert "WHEN 'input_large' THEN NOT EXISTS" in query and "r.round_number = 3" in query
        assert "ELSE TRUE END" in query
        assert not any(f"WHEN '{role.value}'" in query for role in BOSS_ROUND_SEED_ROLES | TEXT_SEED_ROLES)

    async def test_no_task_ids_no_query(self):
        assert await seeds_sql.get_hidden_seed_task_ids([], None) == set()


class TestRoleHousekeeping:
    def test_the_labels_and_sets(self):
        assert [SeedRole(v).label for v in ("text_r1", "text_r2", "text_r3", "text_continuous")] == [
            "Text R1",
            "Text R2",
            "Text R3",
            "Text continuous",
        ]
        assert [SeedRole(v).label for v in ("boss_r2_freeform", "boss_r2_fixed", "boss_r3_freeform", "boss_r3_fixed")] == [
            "R2 model freeform",
            "R2 model fixed",
            "R3 model freeform",
            "R3 model fixed",
        ]
        assert [SeedRole(v).label for v in ("boss_continuous_freeform", "boss_continuous_fixed")] == [
            "Continuous model freeform",
            "Continuous model fixed",
        ]
        # The four quant compressions are the quant boss round's tasks: attached to the round after they were trained.
        assert BOSS_ROUND_SEED_ROLES <= ATTACHED_SEED_ROLES
        assert TEXT_SEED_ROLES == TEXT_ROUND_SEED_ROLES | {SeedRole.TEXT_CONTINUOUS}
        assert math.isclose(sum(spec.advance for spec in t_cst.TEXT_ROUND_PLAN), 13)
        assert [s.seed_role for s in t_cst.TEXT_ROUND_PLAN] == [SeedRole.TEXT_R1, SeedRole.TEXT_R2, SeedRole.TEXT_R3]


async def test_the_hiding_rule_of_the_text_roles_in_postgres(cluster):  # noqa: F811
    connection = await cluster.connect()
    try:
        await connection.execute(_SCHEMA)
        r1, r2, quant_on_r1, quant_on_r2, plain, orphan_quant = (uuid4() for _ in range(6))
        await connection.execute("INSERT INTO tournaments VALUES ('t', 'active'), ('q', 'active')")
        await connection.execute("INSERT INTO tournament_seed_tasks VALUES ('t', 'text_r1', 1, $1, 'training')", r1)
        await connection.execute(
            "INSERT INTO tournament_seed_tasks VALUES ('t', 'text_r2', 1, $1, 'training', 't_round_002')", r2
        )
        await connection.execute(
            "INSERT INTO quant_tasks VALUES ($1, $2), ($3, $4), ($5, $2)", quant_on_r1, r1, quant_on_r2, r2, orphan_quant
        )
        await connection.execute("INSERT INTO tournament_rounds VALUES ('t_round_002', 't', 2, 'active')")
        every = [r1, r2, quant_on_r1, quant_on_r2, plain, orphan_quant]

        # R1's seed is not attached yet: hidden. R2's seed is attached, but a running quant tournament holds a live source row
        # naming it: hidden. Each quant task is hidden until its own quant tournament is over; the text round's completion
        # releases nothing. The orphan has no source row, seed row or tournament_tasks row: it follows the text tournament.
        await connection.execute(
            "INSERT INTO quant_task_sources VALUES ($1, 'q', $2), ($3, 'q', $4)", quant_on_r1, r1, quant_on_r2, r2
        )
        everything = {str(r1), str(r2), str(quant_on_r1), str(quant_on_r2), str(orphan_quant)}
        assert await _hidden(connection, every) == everything
        await connection.execute("UPDATE tournament_rounds SET status = 'completed'")
        assert await _hidden(connection, every) == everything

        # A superseded source row no longer holds R2's seed (its quant task is gone from the slot); the seed is visible.
        await connection.execute("UPDATE quant_task_sources SET superseded = TRUE WHERE quant_task_id = $1", quant_on_r2)
        assert await _hidden(connection, every) == everything - {str(r2)}
        await connection.execute("UPDATE quant_task_sources SET superseded = FALSE")

        # The fallbacks that identify the owner of a quant task without a source row: its tournament_tasks row.
        await connection.execute("DELETE FROM quant_task_sources WHERE quant_task_id = $1", quant_on_r2)
        await connection.execute("INSERT INTO tournament_tasks VALUES ('q', $1)", quant_on_r2)
        assert str(quant_on_r2) in await _hidden(connection, every)
        await connection.execute("UPDATE tournaments SET status = 'completed' WHERE tournament_id = 'q'")
        # The quant tournament is over: its quant tasks are released and so is the seed it held.
        assert await _hidden(connection, every) == {str(r1), str(orphan_quant)}

        # The text tournament ending releases the unattached seed and (as the last resort) the orphan.
        await connection.execute("UPDATE tournaments SET status = 'cancelled' WHERE tournament_id = 't'")
        assert await _hidden(connection, every) == set()
    finally:
        await connection.close()


async def test_a_text_seed_stays_hidden_while_its_quant_tournament_runs_even_after_its_text_tournament_ends(cluster):  # noqa: F811
    connection = await cluster.connect()
    try:
        await connection.execute(_SCHEMA)
        r2, quant = uuid4(), uuid4()
        await connection.execute("INSERT INTO tournaments VALUES ('t', 'completed'), ('q', 'active')")
        await connection.execute(
            "INSERT INTO tournament_seed_tasks VALUES ('t', 'text_r2', 1, $1, 'ready', 't_round_002')", r2
        )
        await connection.execute("INSERT INTO quant_tasks VALUES ($1, $2)", quant, r2)
        await connection.execute("INSERT INTO quant_task_sources VALUES ($1, 'q', $2)", quant, r2)

        assert await _hidden(connection, [r2, quant]) == {str(r2), str(quant)}
        await connection.execute("UPDATE tournaments SET status = 'cancelled' WHERE tournament_id = 'q'")
        assert await _hidden(connection, [r2, quant]) == set()
    finally:
        await connection.close()
