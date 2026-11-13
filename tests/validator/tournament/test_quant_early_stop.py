"""Early stop of a quant tournament: once the round-3 cut leaves the last contender, a crown out of reach on its round 1-3
head-to-heads ends the tournament with the boss retained and no boss round. Rounds 1 and 2 always run in full.

Also the boss side: the quant seeds no round attached and the boss has not started are retired, a decided tournament
replaces or re-creates no quant seed, and its seeds no round attached no longer hold their finalisation.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

import validator.tournament.seed_tasks as seed_tasks
import validator.tournament.tournament_manager as tm
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournament_seeds as seeds_sql
from validator.db.sql import tournaments as tournament_sql
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.crown import HeadToHead
from validator.tournament.crown import HeadToHeadOutcome
from validator.tournament.models import RoundOutcome
from validator.tournament.models import RoundStatus
from validator.tournament.models import RoundType
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus


BOSS = EMISSION_BURN_HOTKEY
CHALLENGER = "5Challenger"
RUNNER_UP = "5RunnerUp"
TID = "quant_tournament"


def _head_to_head(round_number: int, outcome: HeadToHeadOutcome) -> HeadToHead:
    return HeadToHead(
        task_id=f"task-{round_number}",
        label=f"round {round_number}",
        task_type=TaskType.QUANTFREEFORMTASK,
        outcome=outcome,
        boss_score=100.0,
        challenger_score=96.0 if outcome == HeadToHeadOutcome.WIN else 103.0,
    )


WIN = HeadToHeadOutcome.WIN
BEYOND = HeadToHeadOutcome.BEYOND
SMALLER = HeadToHeadOutcome.WITHIN_SMALLER
LARGER = HeadToHeadOutcome.WITHIN_LARGER


def _round(number: int) -> TournamentRoundData:
    return TournamentRoundData(
        round_id=f"{TID}_round_{number:03d}",
        tournament_id=TID,
        round_number=number,
        round_type=RoundType.GROUP,
        status=RoundStatus.COMPLETED,
    )


class QuantCup:
    """advance_tournament for a quant tournament, with every database step faked at the tournament_manager boundary."""

    def __init__(self, monkeypatch) -> None:
        self.tournament = TournamentData(tournament_id=TID, tournament_type=TournamentType.QUANT, status=TournamentStatus.ACTIVE)
        self.config = SimpleNamespace(psql_db=None, discord_url=None)
        self.head_to_heads: list[HeadToHead] = [_head_to_head(n, WIN) for n in (1, 2, 3)]
        self.winners = [CHALLENGER]
        self.placements: list[tuple] = []
        self.eliminated: list[tuple[str, list[str]]] = []
        self.winners_set: list[str] = []
        self.create_next_round = AsyncMock()
        self.retire = AsyncMock(return_value=[SeedRole.BOSS_CONTINUOUS_FIXED])
        self.round_head_to_heads = AsyncMock(side_effect=lambda *_: list(self.head_to_heads))
        self.runner_up = AsyncMock(return_value=RUNNER_UP)
        self.publication = MagicMock()

        async def placements(tid, winner, second, third, _db):
            self.placements.append((tid, winner, second, third))

        async def eliminate(tid, round_id, hotkeys, _db):
            self.eliminated.append((round_id, list(hotkeys)))

        async def set_winner(_tid, winner, _db):
            self.winners_set.append(winner)

        async def participants(_tid, _db):
            return [SimpleNamespace(hotkey=hk, eliminated_in_round_id=None) for hk in (BOSS, CHALLENGER, RUNNER_UP)]

        mp = monkeypatch
        mp.setattr(tm, "find_groups_with_no_valid_scores", AsyncMock(return_value=[]))
        mp.setattr(tm, "get_round_winners", AsyncMock(side_effect=lambda *_: RoundOutcome(winners=list(self.winners))))
        mp.setattr(tm, "get_tournament_participants", participants)
        mp.setattr(tm, "eliminate_tournament_participants", eliminate)
        mp.setattr(tm, "update_tournament_placements", placements)
        mp.setattr(tm, "update_tournament_winner_hotkey", set_winner)
        mp.setattr(tm, "create_next_round", self.create_next_round)
        mp.setattr(tm, "notify_tournament_completed", AsyncMock())
        mp.setattr(tm, "upload_participant_repository", AsyncMock(return_value="uploaded/repo"))
        mp.setattr(tm, "schedule_boss_round_publication", self.publication)
        mp.setattr(tm, "quant_round_head_to_heads", self.round_head_to_heads)
        mp.setattr(tm, "seeded_text_round_head_to_heads", AsyncMock(side_effect=AssertionError("text only")))
        mp.setattr(tm, "get_quant_round_runner_up", self.runner_up)
        mp.setattr(tm, "retire_unattached_quant_seeds", self.retire)
        self.comparisons = AsyncMock(return_value=["compared"])
        self.stored: list[tuple] = []

        async def store(tid, comparisons, _db):
            self.stored.append((tid, comparisons, len(self.placements)))

        mp.setattr(tm, "boss_rank_comparisons", self.comparisons)
        mp.setattr(tm, "store_boss_rank_comparisons", store)
        mp.setattr(tm, "advance_continuous_sft_state", AsyncMock(side_effect=AssertionError("text only")))
        mp.setattr(
            tm,
            "_resolve_r2_entry_after_integrity_gates",
            AsyncMock(side_effect=lambda *_: (list(self.winners), set(), False)),
        )

    def advance(self, round_number: int):
        return tm.advance_tournament(self.tournament, _round(round_number), self.config, None)


@pytest.fixture
def cup(monkeypatch) -> QuantCup:
    return QuantCup(monkeypatch)


IMPOSSIBLE = [WIN, BEYOND, WIN]  # a non-win beyond the larger allowance rules the crown out
TWO_LARGER = [LARGER, LARGER, WIN]  # two non-wins and neither within the smaller allowance


class TestAfterRoundThree:
    @pytest.mark.parametrize("outcomes", [IMPOSSIBLE, TWO_LARGER, [BEYOND, BEYOND, BEYOND]])
    async def test_an_impossible_crown_skips_the_boss_round_and_retains_the_boss(self, cup, outcomes):
        cup.head_to_heads = [_head_to_head(n, outcome) for n, outcome in zip((1, 2, 3), outcomes)]

        await cup.advance(3)

        cup.create_next_round.assert_not_awaited()
        assert cup.placements == [(TID, BOSS, CHALLENGER, RUNNER_UP)]
        assert cup.winners_set == [BOSS]
        assert (_round(3).round_id, [CHALLENGER]) in cup.eliminated
        cup.retire.assert_awaited_once_with(TID, cup.config)

    async def test_third_place_is_the_runner_up_of_round_three(self, cup):
        cup.head_to_heads = [_head_to_head(n, outcome) for n, outcome in zip((1, 2, 3), IMPOSSIBLE)]

        await cup.advance(3)

        cup.runner_up.assert_awaited_once()
        tournament_id, pre_boss, challenger, _db = cup.runner_up.await_args.args
        assert (tournament_id, pre_boss.round_number, challenger) == (TID, 3, CHALLENGER)

    async def test_the_boss_rank_comparisons_are_stored_before_the_placements(self, cup):
        cup.head_to_heads = [_head_to_head(n, outcome) for n, outcome in zip((1, 2, 3), IMPOSSIBLE)]

        await cup.advance(3)

        cup.comparisons.assert_awaited_once_with(TID, CHALLENGER, RUNNER_UP, None)
        assert cup.stored == [(TID, ["compared"], 0)]

    async def test_a_failure_to_store_them_still_completes_the_tournament(self, cup, monkeypatch):
        cup.head_to_heads = [_head_to_head(n, outcome) for n, outcome in zip((1, 2, 3), IMPOSSIBLE)]
        monkeypatch.setattr(tm, "store_boss_rank_comparisons", AsyncMock(side_effect=RuntimeError("no table")))

        await cup.advance(3)

        assert cup.placements == [(TID, BOSS, CHALLENGER, RUNNER_UP)]

    async def test_the_completion_publishes_nothing_that_needs_a_boss_round(self, cup):
        cup.head_to_heads = [_head_to_head(n, outcome) for n, outcome in zip((1, 2, 3), IMPOSSIBLE)]

        await cup.advance(3)

        # The decoded publication only reads the final round's freeform tasks: with none it publishes nothing.
        cup.publication.assert_called_once()
        tm.upload_participant_repository.assert_awaited_once()
        assert tm.upload_participant_repository.await_args.args[2] == BOSS

    @pytest.mark.parametrize("outcomes", [[WIN, WIN, WIN], [SMALLER, LARGER, WIN], [WIN, WIN, LARGER]])
    async def test_a_possible_crown_creates_the_boss_round(self, cup, outcomes):
        cup.head_to_heads = [_head_to_head(n, outcome) for n, outcome in zip((1, 2, 3), outcomes)]

        await cup.advance(3)

        cup.create_next_round.assert_awaited_once()
        assert cup.create_next_round.await_args.args[2] == [CHALLENGER]
        assert cup.placements == [] and cup.winners_set == []
        cup.retire.assert_not_awaited()

    async def test_a_round_that_was_never_decided_rules_the_crown_out(self, cup):
        # A quant round holding no single task is left out of the head-to-heads; the boss round could never crown.
        cup.head_to_heads = [_head_to_head(1, WIN), _head_to_head(3, WIN)]

        await cup.advance(3)

        cup.create_next_round.assert_not_awaited()
        assert cup.placements == [(TID, BOSS, CHALLENGER, RUNNER_UP)]

    async def test_advancing_again_after_the_winner_is_set_changes_nothing(self, cup):
        cup.head_to_heads = [_head_to_head(n, outcome) for n, outcome in zip((1, 2, 3), IMPOSSIBLE)]
        await cup.advance(3)
        cup.tournament = cup.tournament.model_copy(update={"winner_hotkey": BOSS})

        await cup.advance(3)

        assert len(cup.placements) == 1 and cup.retire.await_count == 1


class TestAPlayedBossRound:
    @pytest.fixture
    def boss_round(self, cup, monkeypatch):
        challenger = SimpleNamespace(hotkey=CHALLENGER, training_repo="github/c", training_commit_hash="abc", github_token=None)
        monkeypatch.setattr(tm, "get_challenger_participant_for_retained_boss", AsyncMock(return_value=challenger))
        monkeypatch.setattr(
            tm,
            "resolve_stored_boss_code_review",
            AsyncMock(return_value=SimpleNamespace(halt=False, disqualified=False, replacement_hotkey=None)),
        )
        monkeypatch.setattr(tm, "get_tournament_tasks", AsyncMock(return_value=[]))
        monkeypatch.setattr(tm, "get_quant_pre_boss_runner_up", AsyncMock(return_value=RUNNER_UP))
        monkeypatch.setattr(tm, "generate_diff_report_and_notify_tournament_completed", AsyncMock())
        monkeypatch.setattr(tm, "create_benchmark_tasks_for_tournament_winner", AsyncMock(return_value=[]))
        final = TournamentRoundData(
            round_id=f"{TID}_round_004",
            tournament_id=TID,
            round_number=4,
            round_type=RoundType.KNOCKOUT,
            is_final_round=True,
            status=RoundStatus.COMPLETED,
        )
        return lambda: tm.advance_tournament(cup.tournament, final, cup.config, None)

    async def test_a_boss_that_holds_stores_the_comparisons_of_second_and_third(self, cup, boss_round):
        cup.winners = [BOSS]

        await boss_round()

        assert cup.placements == [(TID, BOSS, CHALLENGER, RUNNER_UP)]
        cup.comparisons.assert_awaited_once_with(TID, CHALLENGER, RUNNER_UP, None)
        assert cup.stored == [(TID, ["compared"], 0)]

    async def test_a_crowned_challenger_stores_none(self, cup, boss_round):
        cup.winners = [CHALLENGER]

        await boss_round()

        assert cup.placements == [(TID, CHALLENGER, BOSS, RUNNER_UP)]
        cup.comparisons.assert_not_awaited()


class TestRoundsOneAndTwoRunInFull:
    @pytest.mark.parametrize("round_number", [1, 2])
    async def test_no_early_stop_before_the_round_three_cut(self, cup, round_number):
        cup.head_to_heads = [_head_to_head(n, BEYOND) for n in (1, 2, 3)]
        cup.winners = [CHALLENGER, RUNNER_UP] if round_number == 1 else [CHALLENGER]

        await cup.advance(round_number)

        cup.round_head_to_heads.assert_not_awaited()
        cup.create_next_round.assert_awaited_once()
        assert cup.placements == [] and cup.winners_set == []


# --- the boss side ---------------------------------------------------------------------------------------------------


class SeedWorld:
    def __init__(self, monkeypatch, decided: bool = True) -> None:
        self.seeds: list[TournamentSeedTask] = []
        self.tasks: dict[str, SimpleNamespace] = {}
        self.attached: set[str] = set()
        self.status_updates: list[tuple[SeedRole, SeedStatus]] = []
        self.task_updates: list[tuple[str, TaskStatus]] = []
        self.boss_rows: dict[str, str] = {}
        self.row_updates: list[tuple[str, str, TrainingStatus]] = []
        self.decided = decided

        async def active(_tid, _db):
            return list(self.seeds)

        async def attached(task_id, _db):
            return str(task_id) in self.attached

        async def get_task(task_id, _db):
            return self.tasks.get(str(task_id))

        async def update_task(task, _db):
            self.task_updates.append((str(task.task_id), task.status))
            return task

        async def update_status(_tid, role, _attempt, status, _db):
            self.status_updates.append((role, status))

        async def decided(_tid, _db):
            return self.decided

        async def boss_row(task_id, hotkeys, _db):
            status = self.boss_rows.get(str(task_id))
            return {} if status is None else {BOSS: status}

        async def update_row(task_id, hotkey, status, _db):
            self.row_updates.append((str(task_id), hotkey, status))

        mp = monkeypatch
        mp.setattr(tournament_sql, "get_training_status_for_task_and_hotkeys", boss_row)
        mp.setattr(tournament_sql, "update_tournament_task_training_status", update_row)
        mp.setattr(seeds_sql, "get_active_seed_tasks", active)
        mp.setattr(seeds_sql, "is_task_attached_to_round", attached)
        mp.setattr(seeds_sql, "update_seed_status", update_status)
        mp.setattr(seeds_sql, "tournament_is_decided", decided)
        mp.setattr(task_sql, "get_task", get_task)
        mp.setattr(task_sql, "update_task", update_task)
        self.config = SimpleNamespace(psql_db=None, discord_url=None)

    def add(self, role: SeedRole, status: TaskStatus) -> TournamentSeedTask:
        seed = TournamentSeedTask(tournament_id=TID, role=role, task_id=uuid4())
        self.seeds.append(seed)
        self.tasks[str(seed.task_id)] = SimpleNamespace(task_id=seed.task_id, status=status)
        return seed


class TestBossSideAfterAnEarlyStop:
    async def test_every_unattached_quant_seed_the_boss_has_not_started_is_retired(self, monkeypatch):
        world = SeedWorld(monkeypatch)
        held = world.add(SeedRole.BOSS_CONTINUOUS_FIXED, TaskStatus.AWAITING_UPSTREAM)
        waiting = world.add(SeedRole.BOSS_R2_FREEFORM, TaskStatus.LOOKING_FOR_NODES)
        round_seed = world.add(SeedRole.QUANT_R3, TaskStatus.AWAITING_UPSTREAM)  # a round seed of a round never played
        queued = world.add(SeedRole.BOSS_CONTINUOUS_FREEFORM, TaskStatus.TRAINING)
        world.boss_rows[str(queued.task_id)] = TrainingStatus.PENDING.value
        running = world.add(SeedRole.BOSS_R3_FIXED, TaskStatus.TRAINING)
        world.boss_rows[str(running.task_id)] = TrainingStatus.TRAINING.value
        world.add(SeedRole.BOSS_R3_FREEFORM, TaskStatus.PREPARING_DATA)  # prep in flight: left to finish
        attached = world.add(SeedRole.QUANT_R2, TaskStatus.AWAITING_UPSTREAM)
        world.attached.add(str(attached.task_id))

        retired = await seed_tasks.retire_unattached_quant_seeds(TID, world.config)

        assert retired == [
            SeedRole.BOSS_CONTINUOUS_FIXED,
            SeedRole.BOSS_R2_FREEFORM,
            SeedRole.QUANT_R3,
            SeedRole.BOSS_CONTINUOUS_FREEFORM,
        ]
        assert world.task_updates == [
            (str(seed.task_id), TaskStatus.FAILURE) for seed in (held, waiting, round_seed, queued)
        ]
        assert world.status_updates == [(role, SeedStatus.FAILED) for role in retired]
        # The boss's queued run is never dispatched; the running one is left to finish.
        assert world.row_updates == [(str(queued.task_id), BOSS, TrainingStatus.FAILURE)]

    async def test_the_seed_pass_of_a_decided_quant_tournament_retires_them(self, monkeypatch):
        world = SeedWorld(monkeypatch)
        world.add(SeedRole.BOSS_CONTINUOUS_FIXED, TaskStatus.AWAITING_UPSTREAM)
        history = AsyncMock(return_value=list(world.seeds))
        monkeypatch.setattr(seeds_sql, "get_seed_task_history", history)
        retire = AsyncMock(return_value=[])
        monkeypatch.setattr(seed_tasks, "retire_unattached_quant_seeds", retire)
        monkeypatch.setattr(seed_tasks, "_retry_missing_boss_round_seeds", AsyncMock())
        monkeypatch.setattr(seed_tasks, "seed_halt_reason", AsyncMock(return_value=None))
        monkeypatch.setattr(seed_tasks, "_evaluate_seed", AsyncMock(return_value=(None, None)))

        await seed_tasks.process_tournament_seed_tasks(TID, world.config)
        world.decided = False
        await seed_tasks.process_tournament_seed_tasks(TID, world.config)

        retire.assert_awaited_once_with(TID, world.config)

    async def test_a_decided_tournament_does_not_replace_an_unattached_round_seed(self, monkeypatch):
        world = SeedWorld(monkeypatch)
        seed = world.add(SeedRole.QUANT_R3, TaskStatus.FAILURE)

        assert await seed_tasks._replacement_blocked(seed, "task ended failure", world.config)

    @pytest.mark.parametrize("role", [SeedRole.BOSS_R2_FREEFORM, SeedRole.BOSS_R2_FIXED, SeedRole.QUANT_R3])
    async def test_an_unattached_seed_of_a_decided_tournament_no_longer_holds_its_finalisation(self, monkeypatch, role):
        world = SeedWorld(monkeypatch)
        seed = world.add(role, TaskStatus.TRAINING)
        monkeypatch.setattr(seeds_sql, "get_seed_task_by_task_id", AsyncMock(return_value=seed))

        assert await seeds_sql.boss_round_seed_ready_to_finalise(seed.task_id, None)
        world.decided = False
        assert not await seeds_sql.boss_round_seed_ready_to_finalise(seed.task_id, None)

    async def test_a_decided_tournament_does_not_replace_an_unattached_boss_round_seed(self, monkeypatch):
        world = SeedWorld(monkeypatch)
        seed = world.add(SeedRole.BOSS_R2_FREEFORM, TaskStatus.FAILURE)

        assert await seed_tasks._replacement_blocked(seed, "task ended failure", world.config)

    async def test_an_undecided_tournament_still_replaces_an_unattached_boss_round_seed(self, monkeypatch):
        world = SeedWorld(monkeypatch, decided=False)
        seed = world.add(SeedRole.BOSS_R2_FREEFORM, TaskStatus.FAILURE)

        assert not await seed_tasks._replacement_blocked(seed, "task ended failure", world.config)

    async def test_a_decided_tournament_re_creates_no_missing_boss_round_seed(self, monkeypatch):
        world = SeedWorld(monkeypatch)
        create = AsyncMock()
        monkeypatch.setattr(seed_tasks, "_create_seed", create)
        monkeypatch.setattr(seeds_sql, "get_latest_seed_tasks", AsyncMock(return_value=[]))

        await seed_tasks._retry_missing_boss_round_seeds(TID, world.config)

        create.assert_not_awaited()

    async def test_no_round_seed_is_created_for_a_round_that_already_exists(self, monkeypatch):
        world = SeedWorld(monkeypatch, decided=False)
        created: list[SeedRole] = []

        async def create(role, _tid, _attempt, _live, _config):
            created.append(role)

        monkeypatch.setattr(seed_tasks, "_create_seed", create)
        monkeypatch.setattr(seeds_sql, "get_latest_seed_tasks", AsyncMock(return_value=[]))
        monkeypatch.setattr(seeds_sql, "get_seed_task_history", AsyncMock(return_value=[]))
        monkeypatch.setattr(seeds_sql, "role_ever_attached", AsyncMock(return_value=False))
        monkeypatch.setattr(tournament_sql, "get_tournament_rounds", AsyncMock(return_value=[_round(1), _round(2)]))

        await seed_tasks._retry_missing_boss_round_seeds(TID, world.config)

        assert SeedRole.QUANT_R1 not in created and SeedRole.QUANT_R2 not in created
        assert created == [role for role in seed_tasks.SEED_CREATION_ORDER if role not in (SeedRole.QUANT_R1, SeedRole.QUANT_R2)]
