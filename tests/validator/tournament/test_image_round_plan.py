"""The image round plan, the boss-round mix, round_plan per tournament type, and the data predicates that tell a planned
image tournament (one task per round, in one group) from one whose round 1 holds three tasks or is a knockout
round: is_planned_image_tournament and plays_round_plan.

The database layer is faked by monkeypatching the SQL functions the predicates call.
"""

import pytest

import validator.tournament.round_results as rr
from core.models.image_models import ImageModelType
from core.models.image_models import ImageTrainingMode
from validator.db.sql import tournaments as tournament_sql
from validator.tournament import constants as t_cst
from validator.tournament.models import ImageRoundSpec
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType


TID = "tournament-1"
ROUND_ID = "round-1"


class TestThePlan:
    def test_slots_caps_and_advances(self):
        assert t_cst.IMAGE_ROUND_PLAN == (
            ImageRoundSpec(
                round_number=1,
                task_specs=(
                    (ImageModelType.Z_IMAGE, ImageTrainingMode.TEXT_TO_IMAGE),
                    (ImageModelType.FLUX2_KLEIN_4B, ImageTrainingMode.TEXT_TO_IMAGE),
                    (ImageModelType.MOSS_IMAGE_BASE, ImageTrainingMode.TEXT_TO_IMAGE),
                ),
                max_num_prompts=t_cst.ROUND_ONE_IMAGE_MAX_SYNTH_PAIRS,
                advance=8,
            ),
            ImageRoundSpec(round_number=2, task_specs=t_cst.ROUND_TWO_IMAGE_TASK_SPECS, advance=4),
            ImageRoundSpec(round_number=3, task_specs=t_cst.FINAL_ROUND_IMAGE_TASK_SPECS, advance=1),
        )

    def test_the_advances_match_text(self):
        assert [spec.advance for spec in t_cst.IMAGE_ROUND_PLAN] == [spec.advance for spec in t_cst.TEXT_ROUND_PLAN]

    def test_the_boss_round_is_two_edit_and_two_text_to_image_tasks(self):
        assert t_cst.IMAGE_BOSS_ROUND_TASKS_PER_MODE == {ImageTrainingMode.EDIT: 2, ImageTrainingMode.TEXT_TO_IMAGE: 2}
        assert t_cst.IMAGE_BOSS_ROUND_TASKS == 4

    def test_the_final_slots_leave_room_for_round_three_and_the_boss_round(self):
        for mode, count in t_cst.IMAGE_BOSS_ROUND_TASKS_PER_MODE.items():
            assert sum(1 for _, slot_mode in t_cst.FINAL_ROUND_IMAGE_TASK_SPECS if slot_mode == mode) == count + 1

    def test_the_rounds_and_the_boss_round_are_the_crown_tasks(self):
        assert len(t_cst.IMAGE_ROUND_PLAN) + t_cst.IMAGE_BOSS_ROUND_TASKS == t_cst.CROWN_TASKS

    @pytest.mark.parametrize(
        "tournament_type, plan",
        [
            (TournamentType.QUANT, t_cst.QUANT_ROUND_PLAN),
            (TournamentType.TEXT, t_cst.TEXT_ROUND_PLAN),
            (TournamentType.IMAGE, t_cst.IMAGE_ROUND_PLAN),
        ],
    )
    def test_round_plan_per_type(self, tournament_type, plan):
        assert t_cst.round_plan(tournament_type) is plan
        assert [spec.round_number for spec in plan] == [1, 2, 3]

    def test_an_environment_tournament_has_no_plan(self):
        with pytest.raises(ValueError):
            t_cst.round_plan(TournamentType.ENVIRONMENT)


def _tournament(tournament_type: TournamentType) -> TournamentData:
    return TournamentData(tournament_id=TID, tournament_type=tournament_type)


def _round(round_number: int = 1, round_type: RoundType = RoundType.GROUP) -> TournamentRoundData:
    return TournamentRoundData(
        round_id=f"round-{round_number}", tournament_id=TID, round_number=round_number, round_type=round_type
    )


def _tasks(count: int) -> list[TournamentTask]:
    return [TournamentTask(tournament_id=TID, round_id=ROUND_ID, task_id=f"task-{i}") for i in range(count)]


@pytest.fixture
def fake_db(monkeypatch):
    """Tournaments, their rounds and the tasks of each round, served by the SQL functions the predicates call."""
    tournaments: dict[str, TournamentData] = {}
    rounds: dict[str, list[TournamentRoundData]] = {}
    tasks: dict[str, list[TournamentTask]] = {}

    async def get_tournament(tournament_id, psql_db):
        return tournaments.get(tournament_id)

    async def get_tournament_rounds(tournament_id, psql_db):
        return rounds.get(tournament_id, [])

    async def get_tournament_tasks(round_id, psql_db):
        return tasks.get(round_id, [])

    monkeypatch.setattr(tournament_sql, "get_tournament", get_tournament)
    monkeypatch.setattr(tournament_sql, "get_tournament_rounds", get_tournament_rounds)
    monkeypatch.setattr(tournament_sql, "get_tournament_tasks", get_tournament_tasks)
    return SimpleDb(tournaments, rounds, tasks)


class SimpleDb:
    def __init__(self, tournaments, rounds, tasks):
        self.tournaments, self.rounds, self.tasks = tournaments, rounds, tasks

    def add(self, tournament_type: TournamentType, first_round: TournamentRoundData | None = None, task_count: int = 0):
        self.tournaments[TID] = _tournament(tournament_type)
        if first_round is not None:
            self.rounds[TID] = [first_round]
            self.tasks[first_round.round_id] = _tasks(task_count)


class TestIsPlannedImageTournament:
    async def test_no_round_yet(self, fake_db):
        fake_db.add(TournamentType.IMAGE)

        assert await tournament_sql.is_planned_image_tournament(TID, None) is True

    @pytest.mark.parametrize("task_count", [0, 1])
    async def test_a_group_round_one_with_at_most_one_task(self, fake_db, task_count):
        fake_db.add(TournamentType.IMAGE, _round(1), task_count)

        assert await tournament_sql.is_planned_image_tournament(TID, None) is True

    async def test_a_group_round_one_with_three_tasks_is_not_planned(self, fake_db):
        fake_db.add(TournamentType.IMAGE, _round(1), 3)

        assert await tournament_sql.is_planned_image_tournament(TID, None) is False

    async def test_a_knockout_round_one(self, fake_db):
        fake_db.add(TournamentType.IMAGE, _round(1, RoundType.KNOCKOUT), 1)

        assert await tournament_sql.is_planned_image_tournament(TID, None) is False

    async def test_a_later_round_alone_does_not_make_a_round_one(self, fake_db):
        fake_db.add(TournamentType.IMAGE)
        fake_db.rounds[TID] = [_round(2)]

        assert await tournament_sql.is_planned_image_tournament(TID, None) is True

    @pytest.mark.parametrize("tournament_type", [TournamentType.TEXT, TournamentType.QUANT, TournamentType.ENVIRONMENT])
    async def test_other_types_are_not_image(self, fake_db, tournament_type):
        fake_db.add(tournament_type)

        assert await tournament_sql.is_planned_image_tournament(TID, None) is False

    async def test_an_unknown_tournament(self, fake_db):
        assert await tournament_sql.is_planned_image_tournament("missing", None) is False


class TestPlaysRoundPlan:
    @pytest.fixture(autouse=True)
    def predicates(self, monkeypatch):
        self.seeded, self.planned = False, False

        async def is_seeded_text_tournament(tournament_id, psql_db):
            return self.seeded

        async def is_planned_image_tournament(tournament_id, psql_db):
            return self.planned

        monkeypatch.setattr(rr, "is_seeded_text_tournament", is_seeded_text_tournament)
        monkeypatch.setattr(rr, "is_planned_image_tournament", is_planned_image_tournament)

    async def test_compression_always(self):
        assert await rr.plays_round_plan(_tournament(TournamentType.QUANT), None) is True

    @pytest.mark.parametrize("seeded", [True, False])
    async def test_text_when_seeded(self, seeded):
        self.seeded, self.planned = seeded, not seeded

        assert await rr.plays_round_plan(_tournament(TournamentType.TEXT), None) is seeded

    @pytest.mark.parametrize("planned", [True, False])
    async def test_image_when_planned(self, planned):
        self.seeded, self.planned = not planned, planned

        assert await rr.plays_round_plan(_tournament(TournamentType.IMAGE), None) is planned
