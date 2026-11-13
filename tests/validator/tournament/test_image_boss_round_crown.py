"""The crown of a planned image tournament: the boss round (three round tasks plus four boss-round tasks), the
early stop after round 3, the evaluation-fault hold and third place.

The score of a task is its weighted loss (lower is better). The tests drive the real round_results / tournament_manager code over
an in-memory tournament (`Cup`): only the database layer and the notifier are faked.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

import validator.tournament.round_results as rr
import validator.tournament.tournament_manager as tm
from core.models.image_models import ImageModelType
from core.models.image_models import ImageTrainingMode
from core.models.task_models import TaskType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsImage
from validator.tournament import constants as t_cst
from validator.tournament.models import RoundOutcome
from validator.tournament.models import RoundStatus
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus


TID = "tourn_image_crown"
BOSS = EMISSION_BURN_HOTKEY
CONTENDER = "contender_hk"
RUNNER_UP = "runner_up_hk"
OTHER = "other_hk"
BOSS_LOSS = 1.0

# The contender's loss against a boss loss of 1.0.
CLEAR_WIN = 0.9
WIN = 0.99  # exactly 1% lower: a win
SMALLER = 0.995  # lower, short of a win: within the smaller allowance
LEVEL = 1.0  # within the larger allowance
LIMIT = 1.01  # exactly 1% higher: the edge of the larger allowance
BEYOND = 1.02  # beyond both
NO_RESULT = None

ROUND_SLOTS = [
    (ImageModelType.FLUX2_KLEIN_4B, ImageTrainingMode.TEXT_TO_IMAGE),
    (ImageModelType.QWEN_IMAGE_EDIT_2511, ImageTrainingMode.EDIT),
    (ImageModelType.HIDREAM_O1, ImageTrainingMode.EDIT),
]
BOSS_ROUND_SLOTS = [
    (ImageModelType.FLUX2_KLEIN_4B, ImageTrainingMode.EDIT),
    (ImageModelType.MOSS_IMAGE_BASE, ImageTrainingMode.EDIT),
    (ImageModelType.KREA2, ImageTrainingMode.TEXT_TO_IMAGE),
    (ImageModelType.MOSS_IMAGE_BASE, ImageTrainingMode.TEXT_TO_IMAGE),
]
BOSS_ROUND_TASKS = len(BOSS_ROUND_SLOTS)
ROUND_TASKS = 3
ADD_BOSS_ROUND_DECIDERS = tm._add_boss_round_deciders  # the real function: the fixture replaces it on the module


def _result(hotkey: str, loss: float) -> MinerResultsImage:
    return MinerResultsImage(hotkey=hotkey, test_loss=loss, synth_loss=loss, is_finetune=True)


class Cup:
    """An in-memory planned image tournament: rounds 1-3 hold one task each, the boss round holds its tasks."""

    def __init__(self, monkeypatch, planned: bool = True) -> None:
        self.mp = monkeypatch
        self.planned = planned
        self.config = SimpleNamespace(psql_db=None, discord_url=None)
        self.tournament = TournamentData(tournament_id=TID, tournament_type=TournamentType.IMAGE, status=TournamentStatus.ACTIVE)
        self.rounds: list[TournamentRoundData] = []
        self.registry: list[TournamentTask] = []
        self.tasks: dict[str, SimpleNamespace] = {}
        self.losses: dict[str, dict[str, float | None]] = {}
        self.failed: set[tuple[str, str]] = set()  # (task id, hotkey) whose training failed
        self.ordering = [CONTENDER, RUNNER_UP, OTHER]  # the standings after round 3
        self.persisted: list[dict] = []
        self.status_updates: list[tuple[str, RoundStatus]] = []
        self.placements: list[tuple] = []
        self.eliminated: list[tuple[str, list[str]]] = []
        self.winners_set: list[str] = []
        self.create_next_round = AsyncMock()
        self.third_place_knockout = AsyncMock(return_value="knockout-third")
        self.compared = AsyncMock(return_value=[])
        self.stored = AsyncMock()
        self.discord = AsyncMock()
        self.deciders = AsyncMock()
        self.back_to_assignment = AsyncMock()
        self.upload = AsyncMock(return_value="uploaded/repo")
        self._wire()

    # ----- building the tournament -------------------------------------------------------------------------------------

    def _add_task(
        self, slot: tuple[ImageModelType, ImageTrainingMode], contender: float | None, boss: float | None = BOSS_LOSS
    ) -> str:
        task_id = str(UUID(int=len(self.tasks) + 1))
        model_type, training_mode = slot
        self.tasks[task_id] = SimpleNamespace(
            task_id=task_id, task_type=TaskType.IMAGETASK, model_type=model_type, training_mode=training_mode
        )
        self.losses[task_id] = {BOSS: boss, CONTENDER: contender}
        return task_id

    def _add_round(self, number: int, round_type: RoundType, final: bool = False) -> TournamentRoundData:
        round_data = TournamentRoundData(
            round_id=f"{TID}_round_{number:03d}",
            tournament_id=TID,
            round_number=number,
            round_type=round_type,
            is_final_round=final,
            status=RoundStatus.COMPLETED,
        )
        self.rounds.append(round_data)
        return round_data

    def play_rounds(self, contender_scores: list[float | None]) -> list[str]:
        """Rounds 1-3, one group and one task each, with the contender's loss on it."""
        task_ids = []
        for number, (slot, score) in enumerate(zip(ROUND_SLOTS, contender_scores, strict=True), start=1):
            round_data = self._add_round(number, RoundType.GROUP)
            task_id = self._add_task(slot, score)
            group_id = f"{round_data.round_id}_g"
            self.registry.append(
                TournamentTask(tournament_id=TID, round_id=round_data.round_id, task_id=task_id, group_id=group_id)
            )
            task_ids.append(task_id)
        return task_ids

    def play_boss_round(self, contender_scores: list[float | None]) -> list[str]:
        """The boss round: one task per slot (fewer when fewer scores are given), the contender against the boss."""
        round_data = self._add_round(4, RoundType.KNOCKOUT, final=True)
        task_ids = []
        for slot, score in zip(BOSS_ROUND_SLOTS, contender_scores, strict=False):
            task_id = self._add_task(slot, score)
            pair_id = f"{round_data.round_id}_p"
            self.registry.append(
                TournamentTask(tournament_id=TID, round_id=round_data.round_id, task_id=task_id, pair_id=pair_id)
            )
            task_ids.append(task_id)
        return task_ids

    def play(self, scores: list[float | None]) -> tuple[list[str], list[str]]:
        """All seven tasks from their seven contender scores: the first three are rounds 1-3, the rest the boss round."""
        return self.play_rounds(scores[:ROUND_TASKS]), self.play_boss_round(scores[ROUND_TASKS:])

    @property
    def round_three(self) -> TournamentRoundData:
        return self.rounds[2]

    @property
    def boss_round(self) -> TournamentRoundData:
        return self.rounds[3]

    def rows(self, round_data: TournamentRoundData) -> list[TournamentTask]:
        return [row for row in self.registry if row.round_id == round_data.round_id]

    async def knockout(self) -> rr.RoundOutcome:
        return await rr.get_knockout_winners(self.boss_round, self.rows(self.boss_round), None, self.config)

    def advance(self, completed: TournamentRoundData):
        return tm.advance_tournament(self.tournament, completed, self.config, None)

    # ----- the fakes ---------------------------------------------------------------------------------------------------

    def _wire(self) -> None:
        mp = self.mp

        async def results(task_id, _db):
            return [_result(hotkey, loss) for hotkey, loss in self.losses[str(task_id)].items() if loss is not None]

        async def training_status(task_id, hotkeys, _db):
            return {
                hotkey: TrainingStatus.FAILURE if (str(task_id), hotkey) in self.failed else TrainingStatus.SUCCESS
                for hotkey in hotkeys
            }

        async def get_task(task_id, _db):
            return self.tasks.get(str(task_id))

        async def rounds(_tid, _db):
            return list(self.rounds)

        async def tasks_of_round(round_id, _db):
            return [row for row in self.registry if row.round_id == round_id]

        async def pairs(_round_id, _db):
            return [SimpleNamespace(hotkey1=CONTENDER, hotkey2=BOSS)]

        async def persist(**kwargs):
            self.persisted.append(kwargs)

        async def planned(_tid, _db):
            return self.planned

        async def status(round_id, new_status, _db):
            self.status_updates.append((round_id, new_status))

        async def placements(tid, winner, second, third, _db):
            self.placements.append((tid, winner, second, third))

        async def eliminate(_tid, round_id, hotkeys, _db):
            self.eliminated.append((round_id, list(hotkeys)))

        async def set_winner(_tid, winner, _db):
            self.winners_set.append(winner)

        async def participants(_tid, _db):
            return [SimpleNamespace(hotkey=hotkey, eliminated_in_round_id=None) for hotkey in (BOSS, CONTENDER, RUNNER_UP, OTHER)]

        for name, fake in [
            ("get_task_results_for_ranking", results),
            ("get_training_status_for_task_and_hotkeys", training_status),
            ("get_task", get_task),
            ("get_tournament_rounds", rounds),
            ("get_tournament_tasks", tasks_of_round),
            ("get_tournament_pairs", pairs),
            ("count_hotkeys_assigned_to_task", AsyncMock(return_value=2)),
            ("update_threshold_adjusted_quality_scores_for_task", persist),
            ("get_tournament", AsyncMock(return_value=self.tournament)),
            ("is_planned_image_tournament", planned),
        ]:
            mp.setattr(rr, name, fake)
        mp.setattr(rr, "planned_round_ordered_hotkeys", AsyncMock(side_effect=lambda *_: list(self.ordering)))

        mp.setattr(tm, "find_groups_with_no_valid_scores", AsyncMock(return_value=[]))
        mp.setattr(tm, "get_tournament_participants", participants)
        mp.setattr(tm, "eliminate_tournament_participants", eliminate)
        mp.setattr(tm, "update_tournament_placements", placements)
        mp.setattr(tm, "update_tournament_winner_hotkey", set_winner)
        mp.setattr(tm, "update_round_status", status)
        mp.setattr(tm, "get_tournament_tasks", tasks_of_round)
        mp.setattr(tm, "create_next_round", self.create_next_round)
        mp.setattr(tm, "notify_tournament_completed", AsyncMock())
        mp.setattr(tm, "upload_participant_repository", self.upload)
        mp.setattr(tm, "_notify_discord", self.discord)
        mp.setattr(tm, "_add_boss_round_deciders", self.deciders)
        mp.setattr(tm, "_return_boss_round_for_node_assignment", self.back_to_assignment)
        mp.setattr(tm, "create_benchmark_tasks_for_tournament_winner", AsyncMock(return_value=[]))
        mp.setattr(tm, "generate_diff_report_and_notify_tournament_completed", AsyncMock())
        mp.setattr(tm, "get_pre_boss_knockout_loser", self.third_place_knockout)
        mp.setattr(tm, "boss_rank_comparisons", self.compared)
        mp.setattr(tm, "store_boss_rank_comparisons", self.stored)
        mp.setattr(tm, "_CROWN_EVALUATION_FAULT_ROUNDS_ALERTED", set())
        mp.setattr(
            tm,
            "get_challenger_participant_for_retained_boss",
            AsyncMock(
                return_value=SimpleNamespace(
                    hotkey=CONTENDER, training_repo="github/contender", training_commit_hash="abc", github_token=None
                )
            ),
        )
        mp.setattr(
            tm,
            "resolve_stored_boss_code_review",
            AsyncMock(return_value=SimpleNamespace(halt=False, disqualified=False, replacement_hotkey=None)),
        )

    def assert_nothing_changed(self) -> None:
        """A held round: no winner, placement, status, elimination, round, decider, rank comparison or persisted score."""
        assert self.winners_set == [] and self.placements == [] and self.status_updates == [] and self.persisted == []
        assert self.eliminated == []
        for seam in (self.create_next_round, self.deciders, self.back_to_assignment, self.compared, self.stored, self.upload):
            seam.assert_not_awaited()


@pytest.fixture
def cup(monkeypatch) -> Cup:
    return Cup(monkeypatch)


def _both_orders(name: str, scores: list[float | None], crowned: bool):
    """The same seven results with the non-wins on the boss-round tasks (the end) and on the round tasks (the start)."""
    return [
        pytest.param(scores, crowned, id=f"{name} / non-wins on boss-round tasks"),
        pytest.param(scores[::-1], crowned, id=f"{name} / non-wins on round tasks"),
    ]


VERDICT_CASES = [
    *_both_orders("seven wins", [WIN] * 7, True),
    *_both_orders("six wins and one smaller", [CLEAR_WIN] * 6 + [SMALLER], True),
    *_both_orders("six wins and one level", [CLEAR_WIN] * 6 + [LEVEL], True),
    *_both_orders("six wins and one at the larger limit", [CLEAR_WIN] * 6 + [LIMIT], True),
    *_both_orders("five wins, one smaller, one larger", [CLEAR_WIN] * 5 + [SMALLER, LIMIT], True),
    *_both_orders("five wins, two smaller", [CLEAR_WIN] * 5 + [SMALLER, SMALLER], True),
    *_both_orders("five wins, two larger", [CLEAR_WIN] * 5 + [LEVEL, LIMIT], False),
    *_both_orders("six wins and one beyond", [CLEAR_WIN] * 6 + [BEYOND], False),
    *_both_orders("five wins, one smaller, one beyond", [CLEAR_WIN] * 5 + [SMALLER, BEYOND], False),
    *_both_orders("four wins, three smaller", [CLEAR_WIN] * 4 + [SMALLER] * 3, False),
]


class TestBossRoundVerdict:
    @pytest.mark.parametrize("scores, crowned", VERDICT_CASES)
    async def test_the_crown_rule_over_the_seven_head_to_heads(self, cup, scores, crowned):
        cup.play(scores)

        outcome = await cup.knockout()

        assert not outcome.is_deferred
        assert outcome.winners == ([CONTENDER] if crowned else [BOSS])

    @pytest.mark.parametrize("on_round_task", [False, True], ids=["boss-round task", "round task"])
    async def test_a_failed_boss_training_against_a_valid_contender_is_a_win(self, cup, on_round_task):
        # The contender's loss is worse than the boss's own: only the failed boss training makes these three tasks wins.
        round_ids, boss_ids = cup.play([BEYOND] * 3 + [CLEAR_WIN] * 4 if on_round_task else [CLEAR_WIN] * 4 + [BEYOND] * 3)
        failed_ids = round_ids if on_round_task else boss_ids[1:]
        for task_id in failed_ids:
            cup.failed.add((task_id, BOSS))
            cup.losses[task_id][BOSS] = None

        outcome = await cup.knockout()

        assert outcome.winners == [CONTENDER]

    @pytest.mark.parametrize("on_round_task", [False, True], ids=["boss-round task", "round task"])
    async def test_a_failed_contender_training_is_beyond_and_the_boss_retains(self, cup, on_round_task):
        round_ids, boss_ids = cup.play([CLEAR_WIN] * 7)
        task_id = round_ids[1] if on_round_task else boss_ids[2]
        cup.failed.add((task_id, CONTENDER))

        outcome = await cup.knockout()

        assert outcome.winners == [BOSS]

    @pytest.mark.parametrize("on_round_task", [False, True], ids=["boss-round task", "round task"])
    async def test_no_contender_result_while_the_boss_has_one_is_beyond(self, cup, on_round_task):
        round_ids, boss_ids = cup.play([CLEAR_WIN] * 7)
        task_id = round_ids[0] if on_round_task else boss_ids[0]
        cup.losses[task_id][CONTENDER] = NO_RESULT

        outcome = await cup.knockout()

        assert not outcome.is_deferred and outcome.winners == [BOSS]

    async def test_a_boss_round_with_one_task_missing_never_crowns(self, cup):
        cup.play_rounds([CLEAR_WIN] * ROUND_TASKS)
        boss_ids = cup.play_boss_round([CLEAR_WIN] * (BOSS_ROUND_TASKS - 1))

        outcome = await cup.knockout()

        assert len(boss_ids) == 3 and outcome.winners == [BOSS]
        assert len(cup.persisted) == 3

    def test_the_boss_round_holds_four_tasks(self):
        assert t_cst.IMAGE_BOSS_ROUND_TASKS == BOSS_ROUND_TASKS


class TestPersistence:
    async def test_each_boss_round_task_is_stored_once_with_its_winner_and_no_round_task_is(self, cup):
        # Boss-round tasks: a win, a win, a task within the larger allowance, a task beyond.
        round_ids, boss_ids = cup.play([CLEAR_WIN] * 3 + [CLEAR_WIN, WIN, LEVEL, BEYOND])

        outcome = await cup.knockout()

        assert outcome.winners == [BOSS]
        assert [str(call["task_id"]) for call in cup.persisted] == boss_ids
        assert not {str(call["task_id"]) for call in cup.persisted} & set(round_ids)
        assert [call["winner_hotkey"] for call in cup.persisted] == [CONTENDER, CONTENDER, BOSS, BOSS]
        for call in cup.persisted:
            assert call["threshold_percentage"] == t_cst.IMAGE_CROWN_WIN_MARGIN
            assert call["compared_hotkeys"] == [BOSS, CONTENDER]

    async def test_a_crowned_contender_wins_every_stored_boss_round_task(self, cup):
        cup.play([WIN] * 7)

        outcome = await cup.knockout()

        assert outcome.winners == [CONTENDER]
        assert len(cup.persisted) == BOSS_ROUND_TASKS
        assert {call["winner_hotkey"] for call in cup.persisted} == {CONTENDER}


class TestEvaluationFault:
    """Both trainings succeeded and neither side has a valid result: the task is not decided and the round is held."""

    async def test_a_faulted_boss_round_task_defers_the_round_and_persists_nothing(self, cup):
        _, boss_ids = cup.play([CLEAR_WIN] * 7)
        cup.losses[boss_ids[1]] = {BOSS: None, CONTENDER: None}

        outcome = await cup.knockout()

        assert outcome.is_deferred and outcome.winners == []
        assert outcome.evaluation_fault_task_ids == [boss_ids[1]]
        assert cup.persisted == []

    async def test_a_round_task_and_a_boss_round_task_are_both_named(self, cup):
        round_ids, boss_ids = cup.play([CLEAR_WIN] * 7)
        cup.losses[round_ids[1]] = {BOSS: None, CONTENDER: None}
        cup.losses[boss_ids[3]] = {BOSS: None, CONTENDER: None}

        outcome = await cup.knockout()

        assert outcome.evaluation_fault_task_ids == [round_ids[1], boss_ids[3]]
        assert outcome.winners == [] and cup.persisted == []

    async def test_a_failed_training_is_a_decided_task_not_a_fault(self, cup):
        _, boss_ids = cup.play([CLEAR_WIN] * 7)
        cup.losses[boss_ids[0]] = {BOSS: None, CONTENDER: None}
        cup.failed.add((boss_ids[0], CONTENDER))

        outcome = await cup.knockout()

        assert not outcome.is_deferred and outcome.winners == [BOSS]

    async def test_the_held_boss_round_alerts_once_and_changes_nothing(self, cup):
        _, boss_ids = cup.play([CLEAR_WIN] * 7)
        cup.losses[boss_ids[2]] = {BOSS: None, CONTENDER: None}

        await cup.advance(cup.boss_round)
        await cup.advance(cup.boss_round)

        cup.discord.assert_awaited_once()
        assert boss_ids[2] in cup.discord.await_args.args[0]
        cup.assert_nothing_changed()

    async def test_the_held_round_three_check_alerts_once_and_creates_no_boss_round(self, cup):
        round_ids = cup.play_rounds([CLEAR_WIN] * ROUND_TASKS)
        cup.losses[round_ids[1]] = {BOSS: None, CONTENDER: None}

        await cup.advance(cup.round_three)
        await cup.advance(cup.round_three)

        cup.discord.assert_awaited_once()
        assert round_ids[1] in cup.discord.await_args.args[0]
        assert cup.winners_set == [] and cup.placements == [] and cup.persisted == []
        for seam in (cup.create_next_round, cup.compared, cup.stored, cup.upload):
            seam.assert_not_awaited()
        assert all(CONTENDER not in hotkeys for _, hotkeys in cup.eliminated)


class TestEarlyStopAfterRoundThree:
    @pytest.mark.parametrize(
        "scores",
        [
            pytest.param([CLEAR_WIN, CLEAR_WIN, BEYOND], id="one round task beyond"),
            pytest.param([CLEAR_WIN, LEVEL, LIMIT], id="two larger non-wins"),
            pytest.param([LEVEL, LEVEL, LEVEL], id="three non-wins"),
            pytest.param([CLEAR_WIN, CLEAR_WIN, NO_RESULT], id="no contender result"),
        ],
    )
    async def test_a_crown_out_of_reach_ends_the_tournament_with_the_boss_retained(self, cup, scores):
        cup.play_rounds(scores)

        await cup.advance(cup.round_three)

        cup.create_next_round.assert_not_awaited()
        assert cup.placements == [(TID, BOSS, CONTENDER, RUNNER_UP)]
        assert cup.winners_set == [BOSS]
        assert (cup.round_three.round_id, [CONTENDER]) in cup.eliminated
        cup.compared.assert_awaited_once_with(TID, CONTENDER, RUNNER_UP, None)
        cup.stored.assert_awaited_once()
        cup.upload.assert_awaited_once()
        assert cup.upload.await_args.args[2] == BOSS
        assert len(cup.rounds) == ROUND_TASKS

    @pytest.mark.parametrize(
        "scores",
        [
            pytest.param([CLEAR_WIN] * 3, id="three wins"),
            pytest.param([SMALLER, LIMIT, CLEAR_WIN], id="a smaller and a larger non-win"),
            pytest.param([SMALLER, SMALLER, CLEAR_WIN], id="two smaller non-wins"),
        ],
    )
    async def test_a_crown_still_possible_creates_the_boss_round_with_the_contender(self, cup, scores):
        cup.play_rounds(scores)

        await cup.advance(cup.round_three)

        cup.create_next_round.assert_awaited_once()
        assert cup.create_next_round.await_args.args[2] == [CONTENDER]
        assert cup.placements == [] and cup.winners_set == []
        cup.compared.assert_not_awaited()
        cup.upload.assert_not_awaited()

    async def test_the_verdict_counts_the_four_boss_round_tasks_as_wins(self, cup):
        cup.play_rounds([CLEAR_WIN, CLEAR_WIN, LIMIT])

        verdict = await tm._round_crown_verdict(cup.tournament, CONTENDER, None)

        assert verdict.can_still_be_crowned and not verdict.complete and not verdict.crowned

    async def test_a_missing_round_task_is_a_data_fault_not_a_loss(self, cup):
        cup.play_rounds([CLEAR_WIN] * ROUND_TASKS)
        cup.registry = [row for row in cup.registry if not row.round_id.endswith("_002")]

        with pytest.raises(ValueError, match="expected its one task"):
            await cup.advance(cup.round_three)

        assert cup.placements == [] and cup.winners_set == []
        cup.create_next_round.assert_not_awaited()


class TestThirdPlace:
    async def test_a_retained_boss_places_the_contender_second_and_the_runner_up_third(self, cup):
        cup.play([CLEAR_WIN] * 6 + [BEYOND])

        await cup.advance(cup.boss_round)

        assert cup.placements == [(TID, BOSS, CONTENDER, RUNNER_UP)]
        cup.third_place_knockout.assert_not_awaited()

    async def test_a_retained_boss_records_the_rank_comparisons_before_the_placements(self, cup):
        cup.play([CLEAR_WIN] * 6 + [BEYOND])
        order: list[str] = []
        cup.stored.side_effect = lambda *_: order.append("stored")
        original = tm.update_tournament_placements

        async def placements(*args):
            order.append("placements")
            await original(*args)

        cup.mp.setattr(tm, "update_tournament_placements", placements)

        await cup.advance(cup.boss_round)

        cup.compared.assert_awaited_once_with(TID, CONTENDER, RUNNER_UP, None)
        assert order == ["stored", "placements"]

    async def test_a_crowned_contender_places_the_boss_second_and_records_no_rank_comparisons(self, cup):
        cup.play([WIN] * 7)

        await cup.advance(cup.boss_round)

        assert cup.placements == [(TID, CONTENDER, BOSS, RUNNER_UP)]
        cup.compared.assert_not_awaited()
        cup.stored.assert_not_awaited()
        cup.third_place_knockout.assert_not_awaited()

    async def test_a_third_place_that_collides_with_a_placement_is_omitted(self, cup):
        cup.play([CLEAR_WIN] * 6 + [BEYOND])
        cup.ordering = [CONTENDER]

        await cup.advance(cup.boss_round)

        assert cup.placements == [(TID, BOSS, CONTENDER, None)]


class TestRouting:
    async def test_an_image_tournament_that_is_not_planned_resolves_its_boss_round_on_the_old_rule(self, cup, monkeypatch):
        cup.planned = False
        cup.play([CLEAR_WIN] * 7)
        planned_outcome = AsyncMock()
        monkeypatch.setattr(rr, "_get_image_boss_round_outcome", planned_outcome)
        calls: list[tuple] = []
        real = rr.determine_boss_round_winner

        def spy(*args, **kwargs):
            calls.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(rr, "determine_boss_round_winner", spy)

        outcome = await cup.knockout()

        planned_outcome.assert_not_awaited()
        assert len(calls) == 1 and not outcome.is_deferred and len(outcome.winners) == 1

    async def test_a_planned_image_boss_round_never_adds_deciders(self, cup, monkeypatch):
        cup.play([CLEAR_WIN] * 7)

        with pytest.raises(RuntimeError, match="never adds deciders"):
            await ADD_BOSS_ROUND_DECIDERS(cup.tournament, cup.boss_round, RoundOutcome(deferred_reason="draw"), cup.config, None)
