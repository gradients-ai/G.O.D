"""Rounds 1-3 of a planned image tournament: the boss trains each round's task alongside the group, standings leave it out,
advancement is 8 / 4 / 1 (round 1 on its own task, rounds 2-3 on average rank over every round played), and the boss's
rank, third place and the empty-score check read the stored image results (weighted loss, lower is better).

The database layer is faked by monkeypatching the SQL functions the modules call.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

import validator.tournament.round_results as rr
import validator.tournament.tournament_manager as tm
from core.models.task_models import TaskType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsImage
from validator.tournament import constants as t_cst
from validator.tournament.models import Group
from validator.tournament.models import GroupRound
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType


TID = "imgtourn"
BOSS = EMISSION_BURN_HOTKEY
BEST = 0.01


def _round(number: int) -> TournamentRoundData:
    return TournamentRoundData(round_id=f"r{number}", tournament_id=TID, round_number=number, round_type=RoundType.GROUP)


def _result(hotkey: str, loss: float | None) -> MinerResultsImage:
    """A valid image result, or (loss None) a trained miner whose evaluation produced no usable result."""
    if loss is None:
        return MinerResultsImage(hotkey=hotkey, test_loss=0.0, synth_loss=0.0, is_finetune=False)
    return MinerResultsImage(hotkey=hotkey, test_loss=loss, synth_loss=loss, is_finetune=True)


class ImageWorld:
    """An in-memory image tournament: per round its tasks, the group's members and every task's results."""

    def __init__(self, monkeypatch, planned: bool = True) -> None:
        self.tasks: dict[int, list[UUID]] = {}
        self.members: dict[int, list[str]] = {}
        self.losses: dict[UUID, dict[str, float | None]] = {}
        self.planned = planned
        tournament = TournamentData(tournament_id=TID, tournament_type=TournamentType.IMAGE)

        async def fake_results(task_id, _db):
            return [_result(hotkey, loss) for hotkey, loss in self.losses[UUID(str(task_id))].items()]

        async def fake_rounds(_tid, _db):
            return [_round(number) for number in sorted(self.tasks)]

        async def fake_tasks(round_id, _db):
            number = int(round_id.removeprefix("r"))
            return [
                TournamentTask(tournament_id=TID, round_id=round_id, task_id=task_id, group_id=f"g{number}")
                for task_id in self.tasks[number]
            ]

        async def fake_members(group_id, _db):
            return [SimpleNamespace(hotkey=hotkey) for hotkey in self.members[int(group_id.removeprefix("g"))]]

        monkeypatch.setattr(rr, "get_task_results_for_ranking", fake_results)
        monkeypatch.setattr(rr, "get_tournament_rounds", fake_rounds)
        monkeypatch.setattr(rr, "get_tournament_tasks", fake_tasks)
        monkeypatch.setattr(rr, "get_tournament_group_members", fake_members)
        monkeypatch.setattr(rr, "get_tournament", AsyncMock(return_value=tournament))
        monkeypatch.setattr(rr, "is_planned_image_tournament", AsyncMock(side_effect=lambda *_: self.planned))
        self.tournament = tournament

    def play(self, number: int, members: list[str], *task_losses: dict[str, float | None]) -> None:
        """Round ``number`` played by ``members`` over one task per entry of ``task_losses`` (hotkey -> loss or None)."""
        self.members[number] = members
        self.tasks[number] = [UUID(int=number * 10 + index) for index in range(len(task_losses))]
        for task_id, losses in zip(self.tasks[number], task_losses, strict=True):
            self.losses[task_id] = losses

    def play_in_order(self, number: int, order: list[str], members: list[str] | None = None) -> None:
        """One task where ``order`` (best first) has distinct losses and the boss beats every miner."""
        self.play(number, members or order, {BOSS: BEST, **{hotkey: 1.0 + index / 10 for index, hotkey in enumerate(order)}})

    async def winners(self, number: int) -> list[str]:
        tasks = await rr.get_tournament_tasks(f"r{number}", None)
        return await rr.get_group_winners(_round(number), tasks, psql_db=None)

    async def ordered(self, number: int) -> list[str]:
        return await rr.planned_round_ordered_hotkeys(_round(number), await rr.get_tournament_tasks(f"r{number}", None), None)


@pytest.fixture
def world(monkeypatch) -> ImageWorld:
    return ImageWorld(monkeypatch)


MINERS = [f"m{i:02d}" for i in range(12)]
ABCDEF = list("ABCDEF")


class TestBossAssignment:
    ROUND_ID = "r1"
    TASK_ID = "11111111-2222-3333-4444-555555555555"

    @pytest.fixture
    def assignment(self, monkeypatch):
        state = SimpleNamespace(assigned=set(), assign_calls=[], repo_names={}, planned=True)

        async def get_nodes_assigned_to_task(task_id, _db):
            return [SimpleNamespace(hotkey=hotkey) for hotkey in state.assigned]

        async def assign_node_to_task(task_id, node, _db):
            state.assign_calls.append(node.hotkey)
            state.assigned.add(node.hotkey)

        async def set_expected_repo_name(task_id, node, _db, name):
            state.repo_names[node.hotkey] = name

        tasks = [
            TournamentTask(tournament_id=TID, round_id=self.ROUND_ID, task_id=self.TASK_ID, group_id=f"{self.ROUND_ID}_group_001")
        ]
        monkeypatch.setattr(tm, "get_tournament_tasks", AsyncMock(return_value=tasks))
        monkeypatch.setattr(tm, "get_node_by_hotkey", AsyncMock(side_effect=lambda hotkey, _db: SimpleNamespace(hotkey=hotkey)))
        monkeypatch.setattr(tm, "is_planned_image_tournament", AsyncMock(side_effect=lambda *_: state.planned))
        monkeypatch.setattr(tm.task_sql, "get_task", AsyncMock(return_value=SimpleNamespace(task_type=TaskType.IMAGETASK)))
        monkeypatch.setattr(tm.task_sql, "get_nodes_assigned_to_task", get_nodes_assigned_to_task)
        monkeypatch.setattr(tm.task_sql, "assign_node_to_task", assign_node_to_task)
        monkeypatch.setattr(tm.task_sql, "set_expected_repo_name", set_expected_repo_name)
        return state

    def _group_round(self, members: list[str]) -> GroupRound:
        return GroupRound(round_id=self.ROUND_ID, round_number=1, groups=[Group(member_ids=members)])

    async def test_the_boss_and_every_member_are_assigned_with_their_repo_names_once(self, assignment):
        members = MINERS[:4]
        group_round = self._group_round(members)

        await tm.assign_nodes_to_tournament_tasks(TID, group_round, None)

        assert set(assignment.assigned) == {*members, BOSS}
        assert assignment.repo_names == {hotkey: f"tournament-{TID}-{self.TASK_ID}-{hotkey[:8]}" for hotkey in {*members, BOSS}}
        assert group_round.groups[0].member_ids == members and BOSS not in group_round.groups[0].member_ids

        await tm.assign_nodes_to_tournament_tasks(TID, group_round, None)

        assert len(assignment.assign_calls) == len(members) + 1 == len(set(assignment.assign_calls))

    async def test_the_boss_is_not_assigned_to_a_task_of_the_three_task_round_one(self, assignment):
        assignment.planned = False
        members = MINERS[:4]

        await tm.assign_nodes_to_tournament_tasks(TID, self._group_round(members), None)

        assert set(assignment.assigned) == set(members)
        assert BOSS not in assignment.repo_names


class TestAdvancement:
    async def test_the_boss_takes_no_standing_so_round_one_is_the_miners_in_loss_order(self, world):
        world.play(1, MINERS, {BOSS: BEST, **{hotkey: 1.0 + index / 10 for index, hotkey in enumerate(reversed(MINERS))}})

        winners = await world.winners(1)

        assert winners == list(reversed(MINERS))[:8]
        assert BOSS not in winners and len(winners) == 8

    def test_the_advance_counts_are_eight_four_one(self):
        assert [spec.advance for spec in t_cst.IMAGE_ROUND_PLAN] == [8, 4, 1]

    @pytest.mark.parametrize(
        ("number", "field", "advances"),
        [(1, 12, 8), (2, 6, 4), (3, 4, 1), (1, 5, 5), (2, 3, 3), (3, 1, 1)],
    )
    async def test_each_round_advances_its_count_and_everyone_when_the_field_is_smaller(self, world, number, field, advances):
        miners = MINERS[:field]
        for played in range(1, number + 1):
            world.play_in_order(played, miners)

        winners = await world.winners(number)

        assert winners == miners[:advances]

    async def test_round_two_cuts_by_average_rank_not_by_the_round_two_task(self, world):
        # Round 2's task alone would advance E, F, A, B; average rank over rounds 1-2 advances A, B, E, C.
        world.play(1, ABCDEF, {BOSS: BEST, "A": 1.0, "B": 1.1, "C": 1.2, "D": 1.3, "E": 2.0, "F": 2.1})
        world.play(2, ABCDEF, {BOSS: BEST, "E": 1.0, "F": 1.1, "A": 1.2, "B": 1.3, "C": 1.4, "D": 1.5})

        # Average ranks: A 2, B 3, C 4, D 5, E 3, F 4. Ties go to the smaller summed gap to the best (both bests 1.0), not to
        # the round-2 winner: B (0.1+0.3)=0.4 beats E (1.0+0.0)=1.0 at 3, and C (0.2+0.4)=0.6 beats F (1.1+0.1)=1.2 at 4.
        assert await world.winners(2) == ["A", "B", "E", "C"]
        assert (await world.ordered(2))[:4] == ["A", "B", "E", "C"]

    async def test_round_three_picks_the_best_average_over_all_three_rounds(self, world):
        cohort = ["A", "B", "C", "D"]
        # Round 3's task alone favours D; averages: A 2.0, B 2.33, C 2.67, D 3.0.
        world.play_in_order(1, cohort)
        world.play_in_order(2, ["B", "C", "A", "D"], members=cohort)
        world.play_in_order(3, ["D", "A", "C", "B"], members=cohort)

        assert await world.winners(3) == ["A"]

    async def test_round_one_advances_on_its_own_task_alone(self, world):
        world.play_in_order(1, ABCDEF)
        world.play_in_order(2, list(reversed(ABCDEF)))

        assert await world.winners(1) == ABCDEF

    async def test_the_boss_is_never_ranked_into_a_slot_or_into_the_miners_ranks(self, world):
        miners = ["A", "B", "C", "D", "E"]
        world.play_in_order(1, miners)
        world.play_in_order(2, miners)

        winners = await world.winners(2)

        assert winners == ["A", "B", "C", "D"] and BOSS not in winners


class TestRoundTwoEntrants:
    async def test_an_excluded_miner_is_replaced_by_the_next_in_the_round_one_standings(self, world):
        world.play_in_order(1, MINERS)

        entrants = await rr.select_round2_entrants(world.tournament, _round(1), {MINERS[0], MINERS[3]}, None)

        assert entrants == [MINERS[i] for i in (1, 2, 4, 5, 6, 7, 8, 9)]

    async def test_the_boss_is_never_selected(self, world):
        world.play_in_order(1, MINERS)

        entrants = await rr.select_round2_entrants(world.tournament, _round(1), set(), None)

        assert entrants == MINERS[:8] and BOSS not in entrants


class TestBossRanks:
    async def test_the_boss_is_ranked_in_every_round_ties_share_a_rank_and_no_result_ranks_last(self, world):
        members = list("ABCDE")
        # Round 1: A 1, boss 2, B and C tied 3, D and E (no valid result) tied 5.
        world.play(1, members, {"A": 1.0, BOSS: 1.5, "B": 2.0, "C": 2.0, "D": None, "E": None})
        # Round 2: the boss leads. Round 3: the boss trails every miner.
        world.play(2, members[:4], {BOSS: 0.5, "A": 1.0, "B": 1.1, "C": 1.2, "D": 1.3})
        world.play(3, members[:2], {"A": 1.0, "B": 1.1, BOSS: 2.0})

        ranks = await rr.round_ranks_with_boss(TID, None)

        by_round = {entry.round_number: entry.ranks for entry in ranks.rounds}
        assert by_round[1] == {"A": 1, BOSS: 2, "B": 3, "C": 3, "D": 5, "E": 5}
        assert by_round[2] == {BOSS: 1, "A": 2, "B": 3, "C": 4, "D": 5}
        assert by_round[3] == {"A": 1, "B": 2, BOSS: 3}
        assert [entry.task_ids for entry in ranks.rounds] == [[str(world.tasks[n][0])] for n in (1, 2, 3)]


class TestRunnerUp:
    async def test_third_place_is_the_next_miner_after_the_contender_in_the_standings_after_round_three(self, world):
        cohort = ["A", "B", "C"]
        # Round 3's task alone ranks C, B, A; cumulative: A (1+1+3)/3, B (2+2+2)/3, C (3+3+1)/3. Contender A, third B.
        world.play_in_order(1, cohort)
        world.play_in_order(2, cohort)
        world.play_in_order(3, ["C", "B", "A"], members=cohort)

        assert await rr.get_planned_runner_up(world.tournament, "A", None) == "B"
        assert await rr.get_planned_runner_up(world.tournament, "B", None) == "A"

    async def test_there_is_none_when_round_three_was_not_played(self, world):
        world.play_in_order(1, ["A", "B", "C"])
        world.play_in_order(2, ["A", "B", "C"])

        assert await rr.get_planned_runner_up(world.tournament, "A", None) is None


class TestEmptyScoreGroups:
    async def test_a_round_where_only_the_boss_has_a_valid_score_is_an_empty_score_group(self, world):
        world.play(1, ["A", "B"], {BOSS: BEST, "A": None, "B": None})

        empty = await rr.find_groups_with_no_valid_scores(_round(1), None)

        found = [(group.group_id, str(group.task_id), group.participants) for group in empty]
        assert found == [("g1", str(world.tasks[1][0]), 2)]

    async def test_one_valid_miner_score_makes_the_group_valid(self, world):
        world.play(1, ["A", "B"], {BOSS: BEST, "A": None, "B": 1.0})

        assert await rr.find_groups_with_no_valid_scores(_round(1), None) == []


class TestRouting:
    async def test_a_three_task_round_one_advances_the_multi_match_top_two_not_the_plan(self, monkeypatch):
        world = ImageWorld(monkeypatch, planned=False)
        order = MINERS[:5]
        losses = {hotkey: 1.0 + index / 10 for index, hotkey in enumerate(order)}
        world.play(1, order, {BOSS: BEST, **losses}, {BOSS: BEST, **losses}, {BOSS: BEST, **losses})

        assert not await rr.plays_round_plan(world.tournament, None)
        winners = await world.winners(1)

        assert winners == order[: t_cst.SMALL_TOURNAMENT_ADVANCE] and len(winners) != t_cst.IMAGE_ROUND_PLAN[0].advance

    async def test_the_planned_structure_routes_to_the_plan(self, world):
        world.play_in_order(1, MINERS[:5])

        assert await rr.plays_round_plan(world.tournament, None)
        assert await world.winners(1) == MINERS[:5]
