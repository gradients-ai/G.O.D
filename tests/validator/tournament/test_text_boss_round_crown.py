"""The boss round of a seeded text tournament: its tasks, the crown rule, the early stop and the flow around them.

The rule is tested as pure functions first. The flow tests drive the real advance_tournament / get_round_winners /
task-creation code over an in-memory tournament (`Cup`): only the database and the task factories are faked.
"""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import UUID
from uuid import uuid4

import pytest

import validator.tournament.round_results as rr
import validator.tournament.tournament_manager as tm
from core.constants.environments import TrainingStartPoint
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournament_seeds as seeds_sql
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.tournament import constants as t_cst
from validator.tournament import crown
from validator.tournament import task_creator
from validator.tournament.crown import HeadToHead
from validator.tournament.crown import HeadToHeadOutcome
from validator.tournament.crown import crown_verdict
from validator.tournament.models import GroupRound
from validator.tournament.models import KnockoutRound
from validator.tournament.models import PairedLossComparison
from validator.tournament.models import RoundOutcome
from validator.tournament.models import RoundStatus
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
from validator.tournament.thresholds import challenger_beats_boss
from validator.tournament.thresholds import quant_challenger_takes_fixed_boss_task


TID = "tourn_text_crown"
BOSS = EMISSION_BURN_HOTKEY
CHALLENGER = "challenger_hk"
RUNNER_UP = "runner_up_hk"
OTHER = "other_hk"
LINEAGE = "qwen3-8b"
BOSS_SCORE = 1000.0

# Challenger scores against a boss score of 1000 on a lower-is-better task.
WIN = 980.0  # 2% better
WIN_EXACT = 990.0  # exactly 1.00% better: a win
NEAR_WIN = 990.1  # 0.99% better: not a win, within the smaller allowance
AHEAD = 995.0  # 0.5% better: not a win, within the smaller allowance
BEHIND_SLIGHT = 1001.1  # 0.11% worse: within the larger allowance
BEHIND_LIMIT = 1010.0  # exactly 1% worse: the edge of the larger allowance
BEHIND_OVER = 1010.1  # 1.01% worse: beyond both


def _paired(challenger_won: bool = True, is_draw: bool = False) -> PairedLossComparison:
    return PairedLossComparison(
        n_examples=100, n_decided=0 if is_draw else 100, challenger_example_wins=90 if challenger_won else 10,
        boss_example_wins=10 if challenger_won else 90, win_rate=0.9 if challenger_won else 0.1,
        win_rate_lower_bound=0.8 if challenger_won else 0.05, mean_gap_nats=0.1 if challenger_won else -0.1,
        mean_gap_lower_bound=0.05 if challenger_won else -0.2, challenger_won=challenger_won, is_draw=is_draw,
        reason="won" if challenger_won else "lost", required_win_rate=0.6, required_mean_gap_nats=0.01,
    )  # fmt: skip


PAIRED_WIN = _paired()


def text_head_to_head(task_id, label, task_type, boss, challenger, paired=PAIRED_WIN):
    """The scalar rule under test: the paired test, where the type has one, is passed unless the test says otherwise."""
    return crown.text_head_to_head(task_id, label, task_type, boss, challenger, paired)


def _verdict(challenger_scores: list[float | None], boss: float | None = BOSS_SCORE, task_type=TaskType.INSTRUCTTEXTTASK):
    results = [text_head_to_head(f"t{i}", f"task {i}", task_type, boss, score) for i, score in enumerate(challenger_scores)]
    return crown_verdict(results)


def _crowned(challenger_scores) -> bool:
    return _verdict(challenger_scores).crowned


class TestExactDecimalBoundaries:
    """Scores exactly on a rule's decimal boundary meet it; scores just outside (1e-6 relative) do not."""

    @pytest.mark.parametrize("boss, challenger", [(1.15, 1.1385), (2.3, 2.277), (1000.0, 990.0), (0.7, 0.693)])
    def test_exactly_one_percent_better_is_a_win_lower_is_better(self, boss, challenger):
        assert text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, boss, challenger).is_win
        assert challenger_beats_boss(boss, challenger, False, t_cst.TEXT_CROWN_WIN_MARGIN)

    @pytest.mark.parametrize("boss, challenger", [(1.15, 1.1385), (2.3, 2.277), (1000.0, 990.0)])
    def test_just_short_of_one_percent_is_not_a_win(self, boss, challenger):
        short = boss - (boss - challenger) * (1 - 1e-6)
        assert not text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, boss, short).is_win
        assert not challenger_beats_boss(boss, short, False, t_cst.TEXT_CROWN_WIN_MARGIN)

    @pytest.mark.parametrize("boss, challenger", [(1.15, 1.1615), (2.3, 2.323), (100.0, 101.0), (0.7, 0.707)])
    def test_exactly_one_percent_better_is_a_win_higher_is_better(self, boss, challenger):
        assert text_head_to_head("t", "g", TaskType.GRPOTASK, boss, challenger).is_win
        assert challenger_beats_boss(boss, challenger, True, t_cst.TEXT_CROWN_WIN_MARGIN)
        short = boss + (challenger - boss) * (1 - 1e-6)
        assert not text_head_to_head("t", "g", TaskType.GRPOTASK, boss, short).is_win

    @pytest.mark.parametrize("boss, challenger", [(-1.15, -1.1385), (-2.3, -2.277), (-100.0, -99.0)])
    def test_a_negative_boss_score_at_exactly_one_percent_is_a_win(self, boss, challenger):
        assert text_head_to_head("t", "g", TaskType.GRPOTASK, boss, challenger).is_win
        assert challenger_beats_boss(boss, challenger, True, 0.01)
        short = boss + (challenger - boss) * (1 - 1e-6)
        assert not challenger_beats_boss(boss, short, True, 0.01)

    def test_against_a_boss_score_of_zero_level_is_still_not_a_win(self):
        assert not text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, 0.0, 0.0).is_win
        assert not text_head_to_head("t", "g", TaskType.GRPOTASK, 0.0, 0.0).is_win

    @pytest.mark.parametrize("boss", [0.4, 1.15, 2.3, 0.7])
    def test_a_deficit_of_exactly_one_percent_is_within_the_limit(self, boss):
        result = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, boss, boss * 1.01)

        assert result.outcome == HeadToHeadOutcome.WITHIN_LARGER
        assert crown_verdict([WIN_RESULT] * 6 + [result]).crowned_if_complete

    @pytest.mark.parametrize("boss", [0.4, 1.15])
    def test_a_deficit_just_above_one_percent_is_still_refused(self, boss):
        result = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, boss, boss * 1.01 * (1 + 1e-6))

        assert result.outcome == HeadToHeadOutcome.BEYOND
        assert not crown_verdict([WIN_RESULT] * 6 + [result]).crowned_if_complete

    def test_a_smaller_and_a_larger_allowance_non_win_are_crowned_two_larger_are_not(self):
        smaller = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, 0.4, 0.399)
        level = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, 0.7, 0.7)
        limit = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, 0.7, 0.7 * 1.01)

        assert (smaller.outcome, level.outcome) == (HeadToHeadOutcome.WITHIN_SMALLER, HeadToHeadOutcome.WITHIN_LARGER)
        assert crown_verdict([WIN_RESULT] * 5 + [smaller, limit]).crowned_if_complete
        assert not crown_verdict([WIN_RESULT] * 5 + [level, limit]).crowned_if_complete

    def test_an_infinite_deficit_stays_refused(self):
        unbounded = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, 0.0, 0.001)

        assert not crown_verdict([WIN_RESULT] * 6 + [unbounded]).crowned_if_complete

    @staticmethod
    def _win_paired() -> PairedLossComparison:
        return PairedLossComparison(
            n_examples=100, n_decided=100, challenger_example_wins=90, boss_example_wins=10, win_rate=0.9,
            win_rate_lower_bound=0.8, mean_gap_nats=0.1, mean_gap_lower_bound=0.05, challenger_won=True, reason="won",
            required_win_rate=0.6, required_mean_gap_nats=0.01,
        )  # fmt: skip

    @pytest.mark.parametrize("boss, challenger", [(1.15, 1.1385), (2.3, 2.277)])
    def test_the_quant_fixed_task_takes_exactly_one_percent_better(self, boss, challenger):
        won, _ = quant_challenger_takes_fixed_boss_task(self._win_paired(), boss, challenger)
        short = boss - (boss - challenger) * (1 - 1e-6)
        short_won, _ = quant_challenger_takes_fixed_boss_task(self._win_paired(), boss, short)

        assert won and not short_won


WIN_RESULT = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, BOSS_SCORE, WIN)


class TestTheCrownRule:
    @pytest.mark.parametrize(
        "name, scores, crowned",
        [
            ("seven wins", [WIN] * 7, True),
            ("six wins and a loss of exactly 1%", [WIN] * 6 + [BEHIND_LIMIT], True),
            ("six wins and a loss of 1.01%", [WIN] * 6 + [BEHIND_OVER], False),
            ("five wins, one ahead by 0.5% and one loss of exactly 1%", [WIN] * 5 + [AHEAD, BEHIND_LIMIT], True),
            ("five wins, losses of 0.11% and 1%: neither is smaller", [WIN] * 5 + [BEHIND_SLIGHT, BEHIND_LIMIT], False),
            ("five wins, one ahead by 0.5% and one loss of 1.01%", [WIN] * 5 + [AHEAD, BEHIND_OVER], False),
            ("four wins and three level tasks", [WIN] * 4 + [BOSS_SCORE] * 3, False),
            ("five wins and two tasks ahead by 0.5%", [WIN] * 5 + [AHEAD, AHEAD], True),
            ("five wins, a level task and a task ahead by 0.5%", [WIN] * 5 + [BOSS_SCORE, AHEAD], True),
            ("five wins and two level tasks: neither is smaller", [WIN] * 5 + [BOSS_SCORE, BOSS_SCORE], False),
            ("six wins and one level task", [WIN] * 6 + [BOSS_SCORE], True),
            ("exactly 1.00% counts as a win", [WIN_EXACT] * 7, True),
            ("0.99% is not a win: two of them are crowned, three are not", [WIN] * 5 + [NEAR_WIN, NEAR_WIN], True),
            ("three tasks short of the win margin", [WIN] * 4 + [NEAR_WIN] * 3, False),
        ],
    )
    def test_the_table(self, name, scores, crowned):
        assert _crowned(scores) is crowned, name

    def test_a_win_at_exactly_the_margin_and_a_hair_below(self):
        exact = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, BOSS_SCORE, WIN_EXACT)
        below = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, BOSS_SCORE, NEAR_WIN)

        assert exact.is_win and exact.deficit == 0.0
        assert not below.is_win and below.deficit == 0.0

    def test_a_non_win_where_the_challenger_is_ahead_is_within_the_smaller_allowance_with_no_deficit(self):
        ahead = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, BOSS_SCORE, AHEAD)

        assert (ahead.outcome, ahead.deficit) == (HeadToHeadOutcome.WITHIN_SMALLER, 0.0)
        assert ahead.improvement == pytest.approx(0.005)

    def test_the_deficit_is_relative_to_the_boss_score(self):
        behind = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, 2.0, 2.005)

        assert behind.deficit == pytest.approx(0.0025)
        assert behind.improvement == pytest.approx(-0.0025)

    @pytest.mark.parametrize(
        "boss, challenger, is_win, deficit",
        [
            (-100.0, -99.0, True, 0.0),  # +1% of the boss's magnitude
            (-100.0, -99.5, False, 0.0),  # ahead by 0.5%
            (-100.0, -100.25, False, 0.0025),
            (100.0, 102.0, True, 0.0),
            (100.0, 99.75, False, 0.0025),
            (100.0, 100.25, False, 0.0),  # ahead by 0.25%: not yet a win
            (100.0, 99.0, False, 0.01),  # 1% worse: higher is better here
        ],
    )
    def test_grpo_is_higher_is_better_including_negative_boss_scores(self, boss, challenger, is_win, deficit):
        result = text_head_to_head("t", "grpo", TaskType.GRPOTASK, boss, challenger)

        assert result.is_win is is_win
        assert result.deficit == pytest.approx(deficit)

    def test_everything_but_grpo_is_lower_is_better(self):
        for task_type in (TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.CHATTASK):
            assert text_head_to_head("t", "x", task_type, 1.0, 0.5).is_win
            assert not text_head_to_head("t", "x", task_type, 1.0, 1.5).is_win

    def test_a_boss_score_of_zero_needs_a_strictly_better_challenger(self):
        level = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, 0.0, 0.0)
        better = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, 0.0, -0.001)
        worse = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, 0.0, 0.001)
        grpo_level = text_head_to_head("t", "g", TaskType.GRPOTASK, 0.0, 0.0)
        grpo_better = text_head_to_head("t", "g", TaskType.GRPOTASK, 0.0, 0.001)
        grpo_worse = text_head_to_head("t", "g", TaskType.GRPOTASK, 0.0, -0.001)

        assert not level.is_win and level.deficit == 0.0
        assert not grpo_level.is_win and grpo_level.deficit == 0.0
        assert better.is_win and grpo_better.is_win
        assert not worse.is_win and math.isinf(worse.deficit)
        assert not grpo_worse.is_win and math.isinf(grpo_worse.deficit)

    @pytest.mark.parametrize("missing", [None, math.nan, math.inf])
    def test_a_challenger_without_a_valid_result_is_a_non_win_with_an_unbounded_deficit(self, missing):
        for boss in (BOSS_SCORE, None):
            result = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, boss, missing)
            assert not result.is_win and math.isinf(result.deficit)
        assert not _crowned([WIN] * 6 + [missing])

    @pytest.mark.parametrize("missing", [None, math.nan, -math.inf])
    def test_a_boss_without_a_valid_result_loses_to_a_challenger_that_has_one(self, missing):
        result = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, missing, 5.0)

        assert result.is_win and result.deficit == 0.0
        assert _verdict([5.0] * 7, boss=missing).crowned

    def test_neither_side_with_a_result_is_an_unbounded_non_win(self):
        result = text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, None, None)

        assert not result.is_win and math.isinf(result.deficit)

    @pytest.mark.parametrize("decided", [0, 1, 3, 6])
    def test_fewer_than_seven_tasks_never_crown(self, decided):
        verdict = _verdict([WIN] * decided)

        assert verdict.crowned_if_complete is True  # nothing so far rules it out
        assert verdict.complete is False and verdict.crowned is False

    def test_seven_tasks_are_complete(self):
        verdict = _verdict([WIN] * t_cst.CROWN_TASKS)

        assert verdict.complete and verdict.crowned and verdict.non_wins == 0 and verdict.decided == 7

    @pytest.mark.parametrize(
        "name, scores, possible",
        [
            ("three wins", [WIN] * 3, True),
            ("three non-wins", [BOSS_SCORE] * 3, False),
            ("a non-win within the smaller allowance and one within the larger", [WIN, AHEAD, BEHIND_LIMIT], True),
            ("two non-wins within the larger allowance", [WIN, BEHIND_SLIGHT, BEHIND_LIMIT], False),
            ("a non-win within the smaller allowance and one beyond", [WIN, AHEAD, BEHIND_OVER], False),
            ("one non-win at exactly 1% worse", [WIN, WIN, BEHIND_LIMIT], True),
            ("one non-win at 1.01% worse", [WIN, WIN, BEHIND_OVER], False),
            ("a level and an ahead non-win", [WIN, BOSS_SCORE, AHEAD], True),
            ("no result for one", [WIN, WIN, None], False),
        ],
    )
    def test_still_possible_on_the_three_tasks_of_rounds_one_to_three(self, name, scores, possible):
        assert _verdict(scores).crowned_if_complete is possible, name

    @pytest.mark.parametrize(
        "name, scores, possible",
        [
            ("six wins", [WIN] * 6, True),
            ("a non-win within the smaller allowance and one within the larger", [WIN] * 4 + [AHEAD, BEHIND_LIMIT], True),
            ("two non-wins within the larger allowance", [WIN] * 4 + [BEHIND_SLIGHT, BEHIND_LIMIT], False),
            ("three level non-wins", [WIN] * 3 + [BOSS_SCORE] * 3, False),
            ("one non-win at 1.01% worse", [WIN] * 5 + [BEHIND_OVER], False),
            ("one non-win at exactly 1% worse", [WIN] * 5 + [BEHIND_LIMIT], True),
            ("two ahead-by-0.5% non-wins", [WIN] * 4 + [AHEAD, AHEAD], True),
        ],
    )
    def test_still_possible_on_the_six_tasks_before_the_big_task(self, name, scores, possible):
        verdict = _verdict(scores)

        assert verdict.crowned_if_complete is possible, name
        assert verdict.crowned is False  # six tasks never crown

    def test_the_reason_says_what_failed(self):
        assert "3 non-wins" in _verdict([BOSS_SCORE] * 3 + [WIN] * 4).reason
        assert "task 6 beyond the larger allowance" in _verdict([WIN] * 6 + [BEHIND_OVER]).reason
        assert "neither within the smaller allowance" in _verdict([WIN] * 5 + [BEHIND_SLIGHT, BEHIND_LIMIT]).reason
        assert "task 6 beyond" in _verdict([WIN] * 6 + [None]).reason

    def test_the_constants(self):
        assert (t_cst.CROWN_TASKS, t_cst.TEXT_CROWN_WIN_MARGIN) == (7, 0.01)
        assert (t_cst.CROWN_MAX_NON_WINS, t_cst.CROWN_LARGER_ALLOWANCE) == (2, 0.01)
        assert t_cst.TEXT_BOSS_ROUND_STAGE_ONE_TASK_TYPES == (TaskType.GRPOTASK, TaskType.DPOTASK)
        # The unseeded text boss round keeps its own mix and its own expected size.
        assert t_cst.expected_boss_round_task_count(TournamentType.TEXT) == t_cst.FINAL_ROUND_TEXT_TASKS
        assert t_cst.FINAL_ROUND_TEXT_TASK_DISTRIBUTION == {
            TaskType.INSTRUCTTEXTTASK: 3,
            TaskType.DPOTASK: 1,
            TaskType.GRPOTASK: 1,
        }

    def test_the_head_to_head_model_is_strict(self):
        with pytest.raises(ValueError):
            HeadToHead(task_id="t", label="x", task_type=TaskType.DPOTASK, outcome=HeadToHeadOutcome.WIN, surprise=1)


class Cup:
    """An in-memory seeded text tournament, wired into the real round_results / task_creator / tournament_manager code."""

    def __init__(self, monkeypatch, seeded: bool = True) -> None:
        self.mp = monkeypatch
        self.seeded = seeded
        self.config = SimpleNamespace(keypair=None, psql_db=None, discord_url=None)
        self.tournament = TournamentData(tournament_id=TID, tournament_type=TournamentType.TEXT, status=TournamentStatus.ACTIVE)
        self.tasks: dict[str, SimpleNamespace] = {}
        self.losses: dict[str, dict[str, float | None]] = {}
        self.training: dict[str, dict[str, TrainingStatus]] = {}
        self.unassigned: set[str] = set()
        # (task_id, hotkey) pairs whose per-example vector is missing; every other scored entry has one (its score per row).
        self.missing_vectors: set[tuple[str, str]] = set()
        self.registry: list[TournamentTask] = []
        self.rounds: list[TournamentRoundData] = []
        self.seed = TournamentSeedTask(
            tournament_id=TID, role=SeedRole.TEXT_CONTINUOUS, task_id=uuid4(), status=SeedStatus.TRAINED
        )
        self.created: list[tuple[TaskType, tuple]] = []  # (type, models) of every task the factory was asked for
        self.persisted: list[dict] = []
        self.status_updates: list[tuple[str, RoundStatus]] = []
        self.placements: list[tuple] = []
        self.eliminated: list[tuple[str, list[str]]] = []
        self.carried: list[tuple] = []
        self.winners_set: list[str] = []
        self.boss_repo: str | None = "boss/continuous-output"
        self.ordering = [CHALLENGER, RUNNER_UP, OTHER]
        self.mark_attached = AsyncMock()
        self.create_next_round = AsyncMock()
        self.third_place_knockout = AsyncMock(return_value="knockout-third")
        self._wire()

    # ----- building the tournament -------------------------------------------------------------------------------------

    def add_task(
        self, task_type=TaskType.INSTRUCTTEXTTASK, *, continuous: bool = False, params: int = 4_000_000_000, task_id=None
    ):
        task_id = task_id or uuid4()
        task = SimpleNamespace(
            task_id=task_id,
            task_type=TaskType.CHATTASK if continuous else task_type,
            training_start_point=TrainingStartPoint.CONTINUOUS_SFT if continuous else TrainingStartPoint.DEFAULT,
            ds=t_cst.continuous_sft_ds(LINEAGE, "chunk-3") if continuous else "dataset",
            status=TaskStatus.SUCCESS,
            model_id="org/model",
            model_params_count=params,
        )
        self.tasks[str(task_id)] = task
        return task

    def score(self, task, boss: float | None, challenger: float | None) -> None:
        self.losses[str(task.task_id)] = {BOSS: boss, CHALLENGER: challenger}

    def earlier_rounds(self, scores: list[tuple[float | None, float | None]]) -> list:
        """Rounds 1-3, one group each with its seed task (the boss's and the challenger's score on it)."""
        tasks = []
        for number, (boss, challenger) in enumerate(scores, start=1):
            round_id = f"{TID}_round_{number:03d}"
            self.rounds.append(
                TournamentRoundData(
                    round_id=round_id,
                    tournament_id=TID,
                    round_number=number,
                    round_type=RoundType.GROUP,
                    status=RoundStatus.COMPLETED,
                )
            )
            task = self.add_task()
            self.score(task, boss, challenger)
            self.registry.append(
                TournamentTask(tournament_id=TID, round_id=round_id, task_id=task.task_id, group_id=f"{round_id}_group_001")
            )
            tasks.append(task)
        return tasks

    @property
    def final_round(self) -> TournamentRoundData:
        return TournamentRoundData(
            round_id=f"{TID}_round_004",
            tournament_id=TID,
            round_number=4,
            round_type=RoundType.KNOCKOUT,
            is_final_round=True,
            status=RoundStatus.COMPLETED,
        )

    async def start_boss_round(self) -> list[str]:
        """Stage 1 through the real creation entry point; the continuous seed is a task the boss already trained."""
        seed_task = self.add_task(continuous=True, task_id=self.seed.task_id)
        self.score(seed_task, None, None)
        knockout = KnockoutRound(pairs=[(CHALLENGER, BOSS)], round_id=self.final_round.round_id, round_number=4)
        return await task_creator.create_text_tournament_tasks(knockout, TID, self.config, is_final_round=True)

    def round_tasks(self, round_id: str) -> list[TournamentTask]:
        return [row for row in self.registry if row.round_id == round_id]

    def boss_round_task(self, task_type: TaskType = None, continuous: bool = False) -> SimpleNamespace:
        for row in self.round_tasks(self.final_round.round_id):
            task = self.tasks[str(row.task_id)]
            if (continuous and task.task_type == TaskType.CHATTASK) or (not continuous and task.task_type == task_type):
                return task
        raise LookupError((task_type, continuous))

    def assign_everything(self) -> None:
        self.unassigned.clear()

    def score_stage_one(self, continuous: tuple, grpo: tuple, dpo: tuple) -> None:
        """Scores in lower-is-better terms against a boss score of 1000; the GRPO one is mirrored (higher is better there)."""
        self.score(self.boss_round_task(continuous=True), *continuous)
        self.score(self.boss_round_task(TaskType.GRPOTASK), grpo[0], None if grpo[1] is None else 2 * BOSS_SCORE - grpo[1])
        self.score(self.boss_round_task(TaskType.DPOTASK), *dpo)

    # ----- the fakes ---------------------------------------------------------------------------------------------------

    def _wire(self) -> None:
        mp = self.mp

        async def results(task_id, _db):
            task = self.tasks[str(task_id)]
            return [
                MinerResultsText(hotkey=hk, test_loss=loss, synth_loss=loss, is_finetune=True, task_type=task.task_type)
                for hk, loss in self.losses.get(str(task_id), {}).items()
                if loss is not None
            ]

        async def training_status(task_id, hotkeys, _db):
            return self.training.get(str(task_id), {})

        async def per_example_losses(task_id, hotkey, _db):
            loss = self.losses.get(str(task_id), {}).get(hotkey)
            if loss is None or (str(task_id), hotkey) in self.missing_vectors:
                return None, None
            return [loss] * 50, "fingerprint"

        async def get_task(task_id, _db):
            return self.tasks.get(str(task_id))

        async def tournament_rounds(_tid, _db):
            return list(self.rounds)

        async def tasks_of_round(round_id, _db):
            return self.round_tasks(round_id)

        async def pairs(round_id, _db):
            return [SimpleNamespace(pair_id=f"{round_id}_pair_001", hotkey1=CHALLENGER, hotkey2=BOSS)]

        async def assigned_count(task_id, _db):
            return 0 if str(task_id) in self.unassigned else 2

        async def assigned_nodes(task_id, _db):
            return [] if str(task_id) in self.unassigned else [SimpleNamespace(hotkey=CHALLENGER), SimpleNamespace(hotkey=BOSS)]

        async def persist(**kwargs):
            self.persisted.append(kwargs)

        async def get_tournament(_tid, _db):
            return self.tournament

        async def seeded_flag(_tid, _db):
            return self.seeded

        async def active_seeds(_tid, _db):
            return [self.seed]

        async def seed_of_task(task_id, _db):
            return self.seed if str(task_id) == str(self.seed.task_id) else None

        async def status(round_id, new_status, _db):
            self.status_updates.append((round_id, new_status))

        async def placements(tid, winner, second, third, _db):
            self.placements.append((tid, winner, second, third))

        async def eliminate(tid, round_id, hotkeys, _db):
            self.eliminated.append((round_id, list(hotkeys)))

        async def advance_state(lineage, repo, source_round_id, _db, task_id=None):
            self.carried.append((lineage, repo, source_round_id, str(task_id)))

        async def set_winner(_tid, winner, _db):
            self.winners_set.append(winner)

        async def participants(_tid, _db):
            return [
                SimpleNamespace(hotkey=hk, eliminated_in_round_id=None) for hk in (BOSS, CHALLENGER, RUNNER_UP, OTHER)
            ]

        async def boss_repo(task_id, hotkey, _db, *, ranked=False):
            assert hotkey == BOSS
            return self.boss_repo

        for name, fake in [
            ("get_task_results_for_ranking", results),
            ("get_training_status_for_task_and_hotkeys", training_status),
            ("get_per_example_losses", per_example_losses),
            ("get_task", get_task),
            ("get_tournament_rounds", tournament_rounds),
            ("get_tournament_tasks", tasks_of_round),
            ("get_tournament_pairs", pairs),
            ("count_hotkeys_assigned_to_task", assigned_count),
            ("get_nodes_assigned_to_task", assigned_nodes),
            ("update_threshold_adjusted_quality_scores_for_task", persist),
            ("get_tournament", get_tournament),
            ("is_seeded_text_tournament", seeded_flag),
        ]:
            mp.setattr(rr, name, fake)
        mp.setattr(rr, "planned_round_ordered_hotkeys", AsyncMock(side_effect=lambda *_: list(self.ordering)))

        mp.setattr(seeds_sql, "is_seeded_text_tournament", seeded_flag)
        mp.setattr(seeds_sql, "get_active_seed_tasks", active_seeds)
        mp.setattr(seeds_sql, "get_latest_seed_tasks", active_seeds)
        mp.setattr(seeds_sql, "get_seed_task_by_task_id", seed_of_task)
        mp.setattr(seeds_sql, "mark_seed_attached", self.mark_attached)
        mp.setattr(task_sql, "get_task", get_task)
        mp.setattr(task_sql, "get_evaluated_repo_for_hotkey", boss_repo)
        mp.setattr(task_sql, "get_lowest_loss_repo_for_task", AsyncMock(return_value="lowest/loss-repo"))

        mp.setattr(task_creator, "_get_existing_tasks_by_identifier", self._existing_by_identifier)
        mp.setattr(task_creator, "_get_existing_tasks", self._existing_tasks)
        mp.setattr(task_creator, "_create_and_register_tournament_task", self._register)
        mp.setattr(task_creator, "_create_task_by_type", self._create_task_by_type)
        mp.setattr(task_creator, "_get_text_models", self._pool)
        mp.setattr(task_creator, "_get_instruct_text_datasets", lambda _keypair: ["instruct-datasets"])
        mp.setattr(task_creator, "_get_dpo_datasets", lambda _keypair: ["dpo-datasets"])

        mp.setattr(tm, "find_groups_with_no_valid_scores", AsyncMock(return_value=[]))
        mp.setattr(tm, "get_tournament_participants", participants)
        mp.setattr(tm, "eliminate_tournament_participants", eliminate)
        mp.setattr(tm, "update_tournament_placements", placements)
        mp.setattr(tm, "update_tournament_winner_hotkey", set_winner)
        mp.setattr(tm, "update_round_status", status)
        mp.setattr(tm, "get_tournament_tasks", tasks_of_round)
        mp.setattr(tm, "advance_continuous_sft_state", advance_state)
        mp.setattr(tm, "create_next_round", self.create_next_round)
        mp.setattr(tm, "notify_tournament_completed", AsyncMock())
        mp.setattr(tm, "upload_participant_repository", AsyncMock(return_value="uploaded/repo"))
        mp.setattr(tm, "_notify_discord", AsyncMock())
        mp.setattr(tm, "create_benchmark_tasks_for_tournament_winner", AsyncMock(return_value=[]))
        mp.setattr(tm, "generate_diff_report_and_notify_tournament_completed", AsyncMock())
        mp.setattr(tm, "get_pre_boss_knockout_loser", self.third_place_knockout)
        self.compared = AsyncMock(return_value=[])
        mp.setattr(tm, "boss_rank_comparisons", self.compared)
        mp.setattr(tm, "store_boss_rank_comparisons", AsyncMock())
        mp.setattr(
            tm,
            "get_challenger_participant_for_retained_boss",
            AsyncMock(
                return_value=SimpleNamespace(
                    hotkey=CHALLENGER, training_repo="github/challenger", training_commit_hash="abc", github_token=None
                )
            ),
        )
        mp.setattr(
            tm,
            "resolve_stored_boss_code_review",
            AsyncMock(return_value=SimpleNamespace(halt=False, disqualified=False, replacement_hotkey=None)),
        )

    async def _existing_by_identifier(self, round_id, config, group_id=None, pair_id=None):
        rows = self.round_tasks(round_id)
        if group_id:
            return [row for row in rows if row.group_id == group_id]
        if pair_id:
            return [row for row in rows if row.pair_id == pair_id]
        return rows

    async def _existing_tasks(self, rows, config):
        return [self.tasks[str(row.task_id)] for row in rows]

    async def _register(self, task, tournament_id, round_id, config, group_id=None, pair_id=None):
        self.registry.append(
            TournamentTask(
                tournament_id=tournament_id, round_id=round_id, task_id=task.task_id, group_id=group_id, pair_id=pair_id
            )
        )
        if str(task.task_id) != str(self.seed.task_id):
            self.unassigned.add(str(task.task_id))  # nodes are assigned by the pending pass, after the task exists

    def _pool(self, keypair, smallest_size_b=None, largest_size_b=None):
        return ("pool", smallest_size_b, largest_size_b)

    async def _create_task_by_type(
        self, task_type, config, models, instruct_datasets, dpo_datasets, model_id_override=None, allow_augmentation=True
    ):
        self.created.append((task_type, models, allow_augmentation))
        big = models is not None and models[1] == t_cst.BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B
        return self.add_task(task_type, params=40_000_000_000 if big else 4_000_000_000)

    def advance(self, tournament: TournamentData | None = None, completed: TournamentRoundData | None = None):
        return tm.advance_tournament(tournament or self.tournament, completed or self.final_round, self.config, None)


@pytest.fixture
def cup(monkeypatch) -> Cup:
    return Cup(monkeypatch)


def _types(created) -> list[TaskType]:
    return [entry[0] for entry in created]


class TestStageOneCreation:
    async def test_the_continuous_seed_is_attached_and_one_grpo_and_one_dpo_are_created(self, cup):
        task_ids = await cup.start_boss_round()

        pair_id = f"{cup.final_round.round_id}_pair_001"
        registered = {str(row.task_id): row for row in cup.round_tasks(cup.final_round.round_id)}
        assert str(cup.seed.task_id) in registered  # the seed itself, not a new continuous task
        assert set(task_ids) == set(registered) and len(registered) == 3
        assert all(row.pair_id == pair_id for row in registered.values())
        assert sorted(t.value for t in _types(cup.created)) == sorted([TaskType.GRPOTASK.value, TaskType.DPOTASK.value])
        cup.mark_attached.assert_awaited_once_with(cup.seed.task_id, cup.final_round.round_id, None)
        # the standard pool, not the large-instruct band, and no big task
        assert all(models == ("pool", None, None) for _, models, _ in cup.created)
        assert not any(task_creator.is_text_big_task(t.task_type) for t in cup.tasks.values())

    async def test_a_second_pass_creates_nothing(self, cup):
        first = await cup.start_boss_round()
        created = len(cup.created)

        second = await cup.start_boss_round()

        assert sorted(second) == sorted(first) and len(cup.created) == created
        assert len(cup.round_tasks(cup.final_round.round_id)) == 3

    async def test_a_restart_that_lost_the_dpo_task_creates_only_that_one(self, cup):
        await cup.start_boss_round()
        dpo = cup.boss_round_task(TaskType.DPOTASK)
        cup.registry = [row for row in cup.registry if str(row.task_id) != str(dpo.task_id)]
        before = len(cup.created)

        await cup.start_boss_round()

        assert _types(cup.created[before:]) == [TaskType.DPOTASK]

    async def test_a_failed_draw_raises_and_the_next_pass_draws_only_what_is_missing(self, cup, monkeypatch):
        real = cup._create_task_by_type
        state = {"fail": TaskType.DPOTASK}

        async def flaky(task_type, *args, **kwargs):
            if task_type == state["fail"]:
                raise RuntimeError("content service down")
            return await real(task_type, *args, **kwargs)

        monkeypatch.setattr(task_creator, "_create_task_by_type", flaky)
        with pytest.raises(RuntimeError, match="Dpo"):
            await cup.start_boss_round()
        state["fail"] = None

        await cup.start_boss_round()

        assert _types(cup.created).count(TaskType.GRPOTASK) == 1 and _types(cup.created).count(TaskType.DPOTASK) == 1

    async def test_a_missing_continuous_seed_raises_before_anything_is_drawn(self, cup, monkeypatch):
        monkeypatch.setattr(seeds_sql, "get_active_seed_tasks", AsyncMock(return_value=[]))
        monkeypatch.setattr(seeds_sql, "get_latest_seed_tasks", AsyncMock(return_value=[]))
        knockout = KnockoutRound(pairs=[(CHALLENGER, BOSS)], round_id=cup.final_round.round_id, round_number=4)

        with pytest.raises(task_creator.MissingBossRoundSeeds) as raised:
            await task_creator.create_text_tournament_tasks(knockout, TID, cup.config, is_final_round=True)

        assert raised.value.roles == [SeedRole.TEXT_CONTINUOUS.value]
        assert cup.created == [] and cup.registry == []

    async def test_the_big_task_is_an_instruct_task_from_the_large_pool_with_kl_on_and_no_augmentation(self, cup):
        await cup.start_boss_round()

        big = await task_creator.create_seeded_text_big_task(TID, cup.final_round.round_id, cup.config)

        task_type, models, allow_augmentation = cup.created[-1]
        assert task_type == TaskType.INSTRUCTTEXTTASK and allow_augmentation is False
        assert models == ("pool", t_cst.BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B, t_cst.BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B)
        assert big.task_id in {UUID(str(row.task_id)) for row in cup.round_tasks(cup.final_round.round_id)}
        again = await task_creator.create_seeded_text_big_task(TID, cup.final_round.round_id, cup.config)
        assert again.task_id == big.task_id and _types(cup.created).count(TaskType.INSTRUCTTEXTTASK) == 1

    async def test_stage_one_after_the_big_task_exists_creates_nothing(self, cup):
        await cup.start_boss_round()
        await task_creator.create_seeded_text_big_task(TID, cup.final_round.round_id, cup.config)
        created = len(cup.created)

        task_ids = await cup.start_boss_round()

        assert len(task_ids) == 3 and len(cup.created) == created
        assert len(cup.round_tasks(cup.final_round.round_id)) == 4


class TestEndingAfterRoundThree:
    @pytest.fixture
    def after_round_three(self, cup):
        def prepare(scores):
            cup.earlier_rounds(scores)
            round_three = TournamentRoundData(
                round_id=f"{TID}_round_003",
                tournament_id=TID,
                round_number=3,
                round_type=RoundType.GROUP,
                status=RoundStatus.COMPLETED,
            )
            cup.seed_task = cup.add_task(continuous=True, task_id=cup.seed.task_id)
            return round_three

        return prepare

    IMPOSSIBLE = [(BOSS_SCORE, BOSS_SCORE)] * 3  # three level tasks: three non-wins
    POSSIBLE = [(BOSS_SCORE, WIN)] * 3

    async def test_an_impossible_crown_ends_the_tournament_with_the_boss_retained(self, cup, after_round_three):
        round_three = after_round_three(self.IMPOSSIBLE)

        await cup.advance(completed=round_three)

        cup.compared.assert_awaited_once_with(TID, CHALLENGER, RUNNER_UP, None)
        assert cup.winners_set == [BOSS]
        assert cup.placements == [(TID, BOSS, CHALLENGER, RUNNER_UP)]
        assert (round_three.round_id, [CHALLENGER]) in cup.eliminated
        cup.create_next_round.assert_not_awaited()
        assert cup.created == []
        assert all(row.round_id != cup.final_round.round_id for row in cup.registry)

    async def test_the_continuous_lineage_is_carried_forward_once_with_the_bosss_own_output(self, cup, after_round_three):
        round_three = after_round_three(self.IMPOSSIBLE)

        await cup.advance(completed=round_three)

        assert cup.carried == [(LINEAGE, "boss/continuous-output", round_three.round_id, str(cup.seed.task_id))]

    async def test_without_a_valid_boss_result_the_lineage_keeps_its_model_but_the_cursor_advances(self, cup, after_round_three):
        round_three = after_round_three(self.IMPOSSIBLE)
        cup.boss_repo = None

        await cup.advance(completed=round_three)

        assert cup.carried == [(LINEAGE, None, round_three.round_id, str(cup.seed.task_id))]

    async def test_advancing_again_after_the_winner_is_set_changes_nothing(self, cup, after_round_three):
        round_three = after_round_three(self.IMPOSSIBLE)
        await cup.advance(completed=round_three)
        snapshot = (list(cup.carried), list(cup.placements), list(cup.eliminated), list(cup.winners_set))

        decided = cup.tournament.model_copy(update={"winner_hotkey": BOSS})
        await cup.advance(decided, round_three)

        assert (cup.carried, cup.placements, cup.eliminated, cup.winners_set) == snapshot
        cup.create_next_round.assert_not_awaited()

    async def test_a_possible_crown_creates_the_final_round_as_before(self, cup, after_round_three):
        round_three = after_round_three(self.POSSIBLE)

        await cup.advance(completed=round_three)

        cup.create_next_round.assert_awaited_once()
        assert cup.create_next_round.await_args.args[2] == [CHALLENGER]
        assert cup.placements == [] and cup.carried == [] and cup.winners_set == []

    async def test_round_two_never_stops_even_with_the_crown_out_of_reach(self, cup, after_round_three):
        after_round_three(self.IMPOSSIBLE)
        round_two = cup.rounds[1]

        await cup.advance(completed=round_two)

        cup.create_next_round.assert_awaited_once()
        assert cup.placements == [] and cup.carried == [] and cup.winners_set == []

    async def test_one_deficit_over_the_limit_is_enough_to_stop(self, cup, after_round_three):
        round_three = after_round_three([(BOSS_SCORE, WIN), (BOSS_SCORE, WIN), (BOSS_SCORE, BEHIND_OVER)])

        await cup.advance(completed=round_three)

        assert cup.placements == [(TID, BOSS, CHALLENGER, RUNNER_UP)]

    async def test_a_challenger_with_no_result_on_a_round_task_cannot_be_crowned(self, cup, after_round_three):
        round_three = after_round_three([(BOSS_SCORE, WIN), (BOSS_SCORE, WIN), (BOSS_SCORE, None)])

        await cup.advance(completed=round_three)

        assert cup.placements == [(TID, BOSS, CHALLENGER, RUNNER_UP)]

    async def test_no_round_three_winner_carries_the_lineage_forward_and_completes_with_the_boss(self, cup, after_round_three):
        round_three = after_round_three(self.POSSIBLE)
        cup.ordering = []  # nobody ranked

        await cup.advance(completed=round_three)

        assert cup.winners_set == [BOSS] and cup.placements == []
        assert cup.carried == [(LINEAGE, "boss/continuous-output", round_three.round_id, str(cup.seed.task_id))]

    async def test_a_missing_round_task_is_a_data_fault_not_a_loss(self, cup, after_round_three):
        round_three = after_round_three(self.POSSIBLE)
        cup.registry = [row for row in cup.registry if not row.round_id.endswith("_002")]

        with pytest.raises(ValueError, match="expected its one task"):
            await cup.advance(completed=round_three)

        assert cup.placements == [] and cup.winners_set == []

    async def test_an_unseeded_text_tournament_does_not_run_the_check(self, cup, after_round_three, monkeypatch):
        cup.seeded = False
        round_three = after_round_three(self.IMPOSSIBLE)
        monkeypatch.setattr(rr, "get_tournament_group_members", AsyncMock(return_value=[SimpleNamespace(hotkey=CHALLENGER)]))

        await cup.advance(completed=round_three)

        cup.create_next_round.assert_awaited_once()
        assert cup.placements == [] and cup.carried == []


class TestStageOneComplete:
    @pytest.fixture
    async def at_stage_one(self, cup):
        cup.earlier_rounds([(BOSS_SCORE, WIN)] * 3)
        await cup.start_boss_round()
        cup.assign_everything()
        return cup

    async def test_a_possible_crown_defers_for_the_big_task_creates_it_once_and_returns_the_round_to_pending(self, at_stage_one):
        cup = at_stage_one
        cup.score_stage_one((BOSS_SCORE, WIN), (BOSS_SCORE, WIN), (BOSS_SCORE, BEHIND_LIMIT))
        outcome = await rr.get_round_winners(cup.final_round, None, cup.config)
        assert outcome.is_deferred and outcome.needs_text_big_task and outcome.winners == []
        assert outcome.draw_resolution is None and cup.persisted == []

        await cup.advance()

        assert cup.status_updates == [(cup.final_round.round_id, RoundStatus.PENDING)]
        assert _types(cup.created).count(TaskType.INSTRUCTTEXTTASK) == 1
        assert cup.winners_set == [] and cup.placements == []
        stage_two = cup.round_tasks(cup.final_round.round_id)
        assert len(stage_two) == 4

        # The round comes back COMPLETED before the new task has nodes: only the status flip happens, no second task.
        await cup.advance()

        assert _types(cup.created).count(TaskType.INSTRUCTTEXTTASK) == 1 and len(cup.round_tasks(cup.final_round.round_id)) == 4
        assert cup.status_updates[-1] == (cup.final_round.round_id, RoundStatus.PENDING)

    async def test_creation_failure_leaves_the_round_completed_and_alerts_once(self, at_stage_one, monkeypatch):
        cup = at_stage_one
        cup.score_stage_one((BOSS_SCORE, WIN), (BOSS_SCORE, WIN), (BOSS_SCORE, WIN))
        monkeypatch.setattr(tm, "create_seeded_text_big_task", AsyncMock(side_effect=RuntimeError("no model")))
        tm._BIG_TASK_CREATION_FAILED_ROUNDS.discard(cup.final_round.round_id)

        await cup.advance()
        await cup.advance()

        assert cup.status_updates == []
        assert tm._notify_discord.await_count == 1
        tm._BIG_TASK_CREATION_FAILED_ROUNDS.discard(cup.final_round.round_id)

    async def test_an_impossible_crown_retains_the_boss_and_never_creates_the_big_task(self, at_stage_one):
        cup = at_stage_one
        # two non-wins, neither strictly better than the boss (0.11% and 1% worse), on top of the three round wins: out of reach
        cup.score_stage_one((BOSS_SCORE, WIN), (BOSS_SCORE, BEHIND_SLIGHT), (BOSS_SCORE, BEHIND_LIMIT))

        await cup.advance()

        assert _types(cup.created).count(TaskType.INSTRUCTTEXTTASK) == 0
        assert cup.status_updates == []
        assert cup.placements == [(TID, BOSS, CHALLENGER, RUNNER_UP)]  # runner-up of round 3, not a pre-boss knockout loser
        self.assert_no_knockout_third_place(cup)
        # the lineage is carried from the attached seed, in the normal final-round way
        assert cup.carried == [(LINEAGE, "lowest/loss-repo", cup.final_round.round_id, str(cup.seed.task_id))]

    @staticmethod
    def assert_no_knockout_third_place(cup) -> None:
        cup.third_place_knockout.assert_not_awaited()

    async def test_the_boss_round_winners_are_persisted_with_the_margin_and_never_for_the_earlier_rounds(self, at_stage_one):
        cup = at_stage_one
        cup.score_stage_one((BOSS_SCORE, BEHIND_SLIGHT), (BOSS_SCORE, WIN), (BOSS_SCORE, BEHIND_LIMIT))
        cup.persisted.clear()

        outcome = await rr.get_round_winners(cup.final_round, None, cup.config)

        # 0.11% and 1% worse, neither within the smaller allowance: not within reach, so the round resolves now and
        # persists its three tasks
        assert outcome.winners == [BOSS]
        persisted = {str(call["task_id"]): call for call in cup.persisted}
        stage_one = {str(row.task_id) for row in cup.round_tasks(cup.final_round.round_id)}
        assert set(persisted) == stage_one
        assert {str(row.task_id) for row in cup.registry if row.round_id != cup.final_round.round_id}.isdisjoint(persisted)
        grpo = persisted[str(cup.boss_round_task(TaskType.GRPOTASK).task_id)]
        continuous = persisted[str(cup.boss_round_task(continuous=True).task_id)]
        assert grpo["winner_hotkey"] == CHALLENGER and continuous["winner_hotkey"] == BOSS
        for call in persisted.values():
            assert call["threshold_percentage"] == t_cst.TEXT_CROWN_WIN_MARGIN
            assert call["compared_hotkeys"] == [BOSS, CHALLENGER]
            assert call["basis"].startswith("the text crown rule")
        assert "deficit 0.110%" in continuous["basis"] and "win, challenger +2.000%" in grpo["basis"]

    @pytest.mark.parametrize("side", [BOSS, CHALLENGER])
    async def test_a_missing_per_example_vector_leaves_the_task_with_the_boss(self, at_stage_one, side):
        cup = at_stage_one
        cup.score_stage_one((BOSS_SCORE, WIN), (BOSS_SCORE, WIN), (BOSS_SCORE, WIN))
        dpo = cup.boss_round_task(TaskType.DPOTASK)
        grpo = cup.boss_round_task(TaskType.GRPOTASK)
        cup.missing_vectors.update({(str(dpo.task_id), side), (str(grpo.task_id), side)})
        cup.persisted.clear()

        outcome = await rr.get_round_winners(cup.final_round, None, cup.config)

        # 2% better on DPO but no pairable vectors: a non-win with no deficit (still within reach). GRPO needs no vectors.
        assert outcome.needs_text_big_task
        assert cup.persisted == []
        head_to_head = await rr.text_task_head_to_head(dpo.task_id, "dpo", TaskType.DPOTASK, CHALLENGER, None)
        assert not head_to_head.is_win and head_to_head.deficit == 0.0
        assert (await rr.text_task_head_to_head(grpo.task_id, "grpo", TaskType.GRPOTASK, CHALLENGER, None)).is_win

    async def test_a_boss_without_a_valid_score_loses_even_without_vectors(self, at_stage_one):
        cup = at_stage_one
        cup.score_stage_one((BOSS_SCORE, WIN), (BOSS_SCORE, WIN), (None, WIN))

        dpo = cup.boss_round_task(TaskType.DPOTASK)

        assert (await rr.text_task_head_to_head(dpo.task_id, "dpo", TaskType.DPOTASK, CHALLENGER, None)).is_win

    async def test_rounds_one_to_three_use_the_paired_test_too(self, at_stage_one):
        cup = at_stage_one
        round_two = cup.round_tasks(f"{TID}_round_002")[0]
        cup.missing_vectors.add((str(round_two.task_id), BOSS))

        results = await rr.round_head_to_heads(
            TournamentData(tournament_id=TID, tournament_type=TournamentType.TEXT), CHALLENGER, None
        )

        assert [result.is_win for result in results] == [True, False, True]

    async def test_one_line_per_head_to_head_and_a_summary(self, at_stage_one, caplog):
        cup = at_stage_one
        cup.score_stage_one((BOSS_SCORE, WIN), (BOSS_SCORE, WIN), (BOSS_SCORE, WIN))

        with caplog.at_level("INFO"):
            await rr.get_round_winners(cup.final_round, None, cup.config)

        assert caplog.text.count("Text crown round ") == 3 and caplog.text.count("Text crown boss round ") >= 3
        assert "the challenger can still be crowned" in caplog.text

    async def test_grpo_is_read_higher_is_better_in_the_flow(self, at_stage_one):
        cup = at_stage_one
        grpo = cup.boss_round_task(TaskType.GRPOTASK)
        cup.score(cup.boss_round_task(continuous=True), BOSS_SCORE, WIN)
        cup.score(cup.boss_round_task(TaskType.DPOTASK), BOSS_SCORE, WIN)

        cup.score(grpo, BOSS_SCORE, 980.0)  # a lower reward is 2% worse: out of reach
        assert (await rr.get_round_winners(cup.final_round, None, cup.config)).winners == [BOSS]
        assert _types(cup.created).count(TaskType.INSTRUCTTEXTTASK) == 0

        cup.score(grpo, BOSS_SCORE, 1020.0)  # a higher reward is 2% better: a win
        assert (await rr.get_round_winners(cup.final_round, None, cup.config)).needs_text_big_task

    async def test_a_challenger_whose_training_failed_has_no_valid_result(self, at_stage_one):
        cup = at_stage_one
        cup.score_stage_one((BOSS_SCORE, WIN), (BOSS_SCORE, WIN), (BOSS_SCORE, WIN))
        cup.training[str(cup.boss_round_task(TaskType.DPOTASK).task_id)] = {CHALLENGER: TrainingStatus.FAILURE}

        outcome = await rr.get_round_winners(cup.final_round, None, cup.config)

        # one non-win with an unbounded deficit is above the limit: the crown is out of reach, the boss retains
        assert outcome.winners == [BOSS] and not outcome.is_deferred


class TestBigTaskComplete:
    @pytest.fixture
    async def at_stage_two(self, cup):
        cup.earlier_rounds([(BOSS_SCORE, WIN)] * 3)
        await cup.start_boss_round()
        await task_creator.create_seeded_text_big_task(TID, cup.final_round.round_id, cup.config)
        cup.assign_everything()
        return cup

    def _score(self, cup, continuous, grpo, dpo, big):
        cup.score_stage_one((BOSS_SCORE, continuous), (BOSS_SCORE, grpo), (BOSS_SCORE, dpo))
        cup.score(cup.boss_round_task(TaskType.INSTRUCTTEXTTASK), BOSS_SCORE, big)

    async def test_crowned(self, at_stage_two):
        cup = at_stage_two
        self._score(cup, WIN, WIN, WIN, BEHIND_LIMIT)  # seven tasks: six wins and a loss of exactly 1%

        outcome = await rr.get_round_winners(cup.final_round, None, cup.config)

        assert outcome.winners == [CHALLENGER] and not outcome.is_deferred
        assert len(cup.persisted) == 4

    async def test_not_crowned(self, at_stage_two):
        cup = at_stage_two
        self._score(cup, WIN, WIN, WIN, BEHIND_OVER)

        outcome = await rr.get_round_winners(cup.final_round, None, cup.config)

        assert outcome.winners == [BOSS]
        assert {call["winner_hotkey"] for call in cup.persisted} == {CHALLENGER, BOSS}

    async def test_a_crowned_challenger_completes_the_tournament_through_the_normal_path(self, at_stage_two):
        cup = at_stage_two
        self._score(cup, WIN, WIN, WIN, WIN)

        await cup.advance()

        assert cup.placements == [(TID, CHALLENGER, BOSS, RUNNER_UP)]
        assert cup.carried == [(LINEAGE, "lowest/loss-repo", cup.final_round.round_id, str(cup.seed.task_id))]
        cup.third_place_knockout.assert_not_awaited()
        assert cup.status_updates == []

    async def test_a_disqualified_challenger_still_advances_the_cursor_with_the_boss_model(self, at_stage_two, monkeypatch):
        cup = at_stage_two
        self._score(cup, WIN, WIN, WIN, WIN)
        monkeypatch.setattr(
            tm,
            "resolve_stored_boss_code_review",
            AsyncMock(return_value=SimpleNamespace(halt=False, disqualified=True, replacement_hotkey=RUNNER_UP)),
        )
        monkeypatch.setattr(tm, "get_tournament_participant", AsyncMock(return_value=None))

        await cup.advance()

        assert cup.carried == [(LINEAGE, "boss/continuous-output", cup.final_round.round_id, str(cup.seed.task_id))]
        assert cup.placements[0][1] == BOSS

    async def test_a_disqualified_challenger_with_no_boss_repo_advances_with_none(self, at_stage_two, monkeypatch):
        cup = at_stage_two
        cup.boss_repo = None
        self._score(cup, WIN, WIN, WIN, WIN)
        monkeypatch.setattr(
            tm,
            "resolve_stored_boss_code_review",
            AsyncMock(return_value=SimpleNamespace(halt=False, disqualified=True, replacement_hotkey=None)),
        )

        await cup.advance()

        assert cup.carried == [(LINEAGE, None, cup.final_round.round_id, str(cup.seed.task_id))]

    async def test_a_failed_advance_leaves_the_tournament_open_and_the_next_cycle_retries(self, at_stage_two, monkeypatch):
        cup = at_stage_two
        self._score(cup, WIN, WIN, WIN, WIN)
        done = tm.advance_continuous_sft_state
        monkeypatch.setattr(tm, "advance_continuous_sft_state", AsyncMock(side_effect=RuntimeError("db down")))

        with pytest.raises(RuntimeError):
            await cup.advance()
        assert cup.winners_set == [] and cup.placements == []

        monkeypatch.setattr(tm, "advance_continuous_sft_state", done)
        await cup.advance()

        assert len(cup.carried) == 1 and cup.placements == [(TID, CHALLENGER, BOSS, RUNNER_UP)]

    async def test_a_boss_that_holds_completes_through_the_normal_path(self, at_stage_two):
        cup = at_stage_two
        self._score(cup, WIN, WIN, WIN, BEHIND_OVER)

        await cup.advance()

        assert cup.placements == [(TID, BOSS, CHALLENGER, RUNNER_UP)]
        assert len(cup.carried) == 1
        cup.compared.assert_awaited_once_with(TID, CHALLENGER, RUNNER_UP, None)

    async def test_a_crowned_challenger_stores_no_boss_rank_comparison(self, at_stage_two):
        cup = at_stage_two
        self._score(cup, WIN, WIN, WIN, WIN)

        await cup.advance()

        cup.compared.assert_not_awaited()

    async def test_a_task_nobody_was_assigned_to_returns_the_round_to_pending_without_adding_anything(
        self, at_stage_two, monkeypatch
    ):
        cup = at_stage_two
        self._score(cup, WIN, WIN, WIN, WIN)
        cup.unassigned.add(str(cup.boss_round_task(TaskType.INSTRUCTTEXTTASK).task_id))
        decider = AsyncMock()
        monkeypatch.setattr(tm, "_add_boss_round_deciders", decider)

        await cup.advance()

        decider.assert_not_awaited()
        assert cup.status_updates == [(cup.final_round.round_id, RoundStatus.PENDING)]
        assert _types(cup.created).count(TaskType.INSTRUCTTEXTTASK) == 1


class TestSeededTextNeverDraws:
    async def test_the_seeded_boss_round_never_uses_the_unseeded_task_verdict_or_the_draw_tally(self, cup, monkeypatch):
        boom = MagicMock(side_effect=AssertionError("must not be reached for a seeded text tournament"))
        monkeypatch.setattr(rr, "_resolve_boss_round_draws", boom)
        monkeypatch.setattr(rr, "_resolve_boss_round_task_winner", boom)
        monkeypatch.setattr(rr, "determine_boss_round_winner", boom)
        cup.earlier_rounds([(BOSS_SCORE, WIN)] * 3)
        await cup.start_boss_round()
        cup.assign_everything()

        cup.score_stage_one((BOSS_SCORE, WIN), (BOSS_SCORE, WIN), (BOSS_SCORE, WIN))
        deferred = await rr.get_round_winners(cup.final_round, None, cup.config)
        await task_creator.create_seeded_text_big_task(TID, cup.final_round.round_id, cup.config)
        cup.assign_everything()
        cup.score(cup.boss_round_task(TaskType.INSTRUCTTEXTTASK), BOSS_SCORE, WIN)
        final = await rr.get_round_winners(cup.final_round, None, cup.config)

        assert deferred.draw_resolution is None and final.draw_resolution is None
        boom.assert_not_called()

    async def test_adding_deciders_to_a_seeded_text_round_is_refused(self, cup):
        outcome = RoundOutcome(deferred_reason="x", draw_resolution=None)

        with pytest.raises(RuntimeError, match="never adds deciders"):
            await tm._add_boss_round_deciders(cup.tournament, cup.final_round, outcome, cup.config, None)

        assert cup.created == [] and cup.status_updates == []

    async def test_a_deferred_seeded_round_is_never_routed_to_the_deciders(self, cup, monkeypatch):
        deciders = AsyncMock()
        returned = AsyncMock()
        monkeypatch.setattr(tm, "_add_boss_round_deciders", deciders)
        monkeypatch.setattr(tm, "_return_boss_round_for_node_assignment", returned)
        monkeypatch.setattr(
            tm,
            "get_round_winners",
            AsyncMock(return_value=RoundOutcome(deferred_reason="x", unassigned_task_ids=["a"])),
        )

        await cup.advance()

        deciders.assert_not_awaited()
        returned.assert_awaited_once()

    async def test_an_unseeded_text_deferral_still_goes_to_the_deciders(self, cup, monkeypatch):
        cup.seeded = False
        deciders = AsyncMock()
        monkeypatch.setattr(tm, "_add_boss_round_deciders", deciders)
        monkeypatch.setattr(
            tm, "get_round_winners", AsyncMock(return_value=RoundOutcome(deferred_reason="x", unassigned_task_ids=["a"]))
        )

        await cup.advance()

        deciders.assert_awaited_once()


class TestReplacingAPrepFailedBossRoundTask:
    @pytest.fixture
    def replacement(self, cup, monkeypatch):
        pools: list[tuple] = []
        instruct_calls: list[dict] = []

        def pool(keypair, smallest_size_b=None, largest_size_b=None):
            pools.append((smallest_size_b, largest_size_b))
            return ("pool", smallest_size_b, largest_size_b)

        async def instruct(config, models, datasets, **kwargs):
            instruct_calls.append({"models": models, **kwargs})
            return cup.add_task(TaskType.INSTRUCTTEXTTASK, params=40_000_000_000)

        monkeypatch.setattr(task_creator, "_get_text_models", pool)
        monkeypatch.setattr(task_creator, "create_synthetic_instruct_text_task", instruct)
        monkeypatch.setattr(task_creator.task_sql, "get_nodes_assigned_to_task", AsyncMock(return_value=[]))
        monkeypatch.setattr(task_creator.task_sql, "detach_task_from_tournament", AsyncMock())

        async def replace(original) -> str:
            return await task_creator.replace_tournament_task(
                str(original.task_id), TID, cup.final_round.round_id, None, f"{cup.final_round.round_id}_pair_001",
                cup.config, is_final_round=True,
            )

        return SimpleNamespace(replace=replace, pools=pools, instruct_calls=instruct_calls)

    @pytest.mark.parametrize("task_type", [TaskType.GRPOTASK, TaskType.DPOTASK])
    async def test_grpo_and_dpo_are_replaced_by_a_task_of_the_same_type(self, cup, replacement, task_type):
        original = cup.add_task(task_type, params=8_000_000_000)
        before = len(cup.created)

        new_id = await replacement.replace(original)

        assert _types(cup.created[before:]) == [task_type]
        assert cup.tasks[str(new_id)].task_type == task_type
        assert replacement.instruct_calls == []
        assert [row.task_id for row in cup.registry] == [str(new_id)]  # registered in the same pair

    async def test_the_big_task_is_redrawn_from_the_large_pool_with_kl_on_and_no_augmentation(self, cup, replacement):
        original = cup.add_task(TaskType.INSTRUCTTEXTTASK, params=40_000_000_000)
        before = len(cup.created)

        new_id = await replacement.replace(original)

        (call,) = replacement.instruct_calls
        assert call["models"] == ("pool", t_cst.BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B, t_cst.BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B)
        assert call["enable_kl"] is True and call["allow_augmentation"] is False
        assert len(cup.created) == before  # the same-type route was not used
        assert cup.tasks[str(new_id)].task_type == TaskType.INSTRUCTTEXTTASK

    async def test_the_completion_check_replaces_them_as_final_round_tasks(self, cup, monkeypatch):
        replace = AsyncMock(return_value="new")
        monkeypatch.setattr(tm, "replace_tournament_task", replace)
        monkeypatch.setattr(seeds_sql, "get_seed_task_by_task_id", AsyncMock(return_value=None))
        for task_type in (TaskType.GRPOTASK, TaskType.DPOTASK, TaskType.INSTRUCTTEXTTASK):
            task = cup.add_task(task_type)
            task.status = TaskStatus.PREP_TASK_FAILURE.value
            row = TournamentTask(tournament_id=TID, round_id="r", task_id=task.task_id, pair_id="r_pair_001")

            completed, _ = await tm.is_tourn_task_completed(row, task, cup.config, final_round=True)

            assert completed is False
            assert replace.await_args.kwargs["is_final_round"] is True


class TestOtherTournamentsKeepTheirBossRound:
    async def test_an_unseeded_text_tournament_creates_its_unseeded_boss_round(self, cup, monkeypatch):
        cup.seeded = False
        unseeded = AsyncMock(return_value=[SimpleNamespace(task_id=uuid4())])
        seeded = AsyncMock()
        monkeypatch.setattr(task_creator, "_create_new_text_boss_round_tasks", unseeded)
        monkeypatch.setattr(task_creator, "_create_seeded_text_boss_round_tasks", seeded)
        knockout = KnockoutRound(pairs=[(CHALLENGER, BOSS)], round_id=cup.final_round.round_id, round_number=4)

        await task_creator.create_text_tournament_tasks(knockout, TID, cup.config, is_final_round=True)

        unseeded.assert_awaited_once()
        seeded.assert_not_awaited()

    async def test_an_unseeded_text_boss_round_is_decided_by_the_unseeded_functions(self, cup, monkeypatch):
        cup.seeded = False
        task = cup.add_task(TaskType.INSTRUCTTEXTTASK)
        cup.score(task, 1.0, 0.5)
        cup.registry.append(
            TournamentTask(tournament_id=TID, round_id=cup.final_round.round_id, task_id=task.task_id, pair_id="p")
        )
        seeded_outcome = AsyncMock()
        monkeypatch.setattr(rr, "_get_seeded_text_boss_round_outcome", seeded_outcome)
        monkeypatch.setattr(rr, "get_per_example_losses", AsyncMock(return_value=(None, None)))
        resolve = AsyncMock(wraps=rr._resolve_boss_round_task_winner)
        decide = MagicMock(wraps=rr.determine_boss_round_winner)
        monkeypatch.setattr(rr, "_resolve_boss_round_task_winner", resolve)
        monkeypatch.setattr(rr, "determine_boss_round_winner", decide)

        outcome = await rr.get_round_winners(cup.final_round, None, cup.config)

        seeded_outcome.assert_not_awaited()
        resolve.assert_awaited_once()
        decide.assert_called_once()
        assert outcome.winners == [CHALLENGER]

    async def test_the_six_task_boss_round_of_an_image_tournament_with_a_three_task_round_one_is_untouched(
        self, cup, monkeypatch
    ):
        cup.tournament = cup.tournament.model_copy(update={"tournament_type": TournamentType.IMAGE})
        monkeypatch.setattr(rr, "is_planned_image_tournament", AsyncMock(return_value=False))
        task = cup.add_task(TaskType.IMAGETASK)
        cup.score(task, 1.0, 0.5)
        cup.registry.append(
            TournamentTask(tournament_id=TID, round_id=cup.final_round.round_id, task_id=task.task_id, pair_id="p")
        )
        seeded_outcome = AsyncMock()
        monkeypatch.setattr(rr, "_get_seeded_text_boss_round_outcome", seeded_outcome)
        monkeypatch.setattr(rr, "update_threshold_adjusted_quality_scores_for_task", AsyncMock())
        decide = MagicMock(wraps=rr.determine_boss_round_winner)
        monkeypatch.setattr(rr, "determine_boss_round_winner", decide)
        monkeypatch.setattr(
            rr,
            "get_task_results_for_ranking",
            AsyncMock(
                return_value=[
                    SimpleNamespace(hotkey=BOSS, test_loss=1.0, adjusted_loss=1.0),
                    SimpleNamespace(hotkey=CHALLENGER, test_loss=0.5, adjusted_loss=0.5),
                ]
            ),
        )
        monkeypatch.setattr(rr, "calculate_miner_ranking_and_scores", lambda results: results)

        outcome = await rr.get_round_winners(cup.final_round, None, cup.config)

        seeded_outcome.assert_not_awaited()
        decide.assert_called_once()
        assert outcome.winners == [CHALLENGER]

    async def test_round_one_to_three_of_an_unseeded_text_tournament_still_use_the_group_round(self, cup, monkeypatch):
        cup.seeded = False
        group_tasks = AsyncMock(return_value=[SimpleNamespace(task_id=uuid4())])
        monkeypatch.setattr(task_creator, "_create_group_text_tasks", group_tasks)
        group_round = GroupRound(groups=[], round_id="r2", round_number=2)

        await task_creator.create_text_tournament_tasks(group_round, TID, cup.config)

        group_tasks.assert_awaited_once()


class TestTheBossEvaluationQuery:
    class _Db:
        def __init__(self, row):
            self.row = row
            self.query = None

        async def connection(self):
            outer = self

            class _Connection:
                async def fetchrow(self, query, *args):
                    outer.query = (query, args)
                    return outer.row

            class _Ctx:
                async def __aenter__(self_inner):
                    return _Connection()

                async def __aexit__(self_inner, *exc):
                    return False

            return _Ctx()

    @pytest.mark.parametrize(
        "row, expected",
        [
            ({"repo": "boss/model", "test_loss": 1.2, "score_reason": None}, "boss/model"),
            ({"repo": "boss/model", "test_loss": 1.2, "score_reason": ""}, "boss/model"),
            ({"repo": "boss/model", "test_loss": math.nan, "score_reason": None}, None),
            ({"repo": "boss/model", "test_loss": math.inf, "score_reason": None}, None),
            ({"repo": "boss/model", "test_loss": 1.2, "score_reason": "evaluation failed"}, None),
            (None, None),
        ],
    )
    async def test_only_a_finite_loss_without_a_failure_reason_gives_the_repo(self, monkeypatch, row, expected):
        db = self._Db(row)
        task_id = uuid4()

        repo = await task_sql.get_evaluated_repo_for_hotkey(task_id, BOSS, db)

        assert repo == expected
        query, args = db.query
        assert "quality_score" not in query  # it needs no ranking
        assert args[0] == task_id and args[1] == BOSS

    @pytest.mark.parametrize(
        "row, expected",
        [
            ({"repo": "boss/model", "test_loss": 1.2, "score_reason": "Lost to winner X at 5% margin"}, "boss/model"),
            ({"repo": "boss/model", "test_loss": 1.2, "score_reason": None}, "boss/model"),
            ({"repo": "boss/model", "test_loss": math.nan, "score_reason": "Winner at 5% margin"}, None),
            ({"repo": "boss/model", "test_loss": math.inf, "score_reason": None}, None),
            (None, None),
        ],
    )
    async def test_a_ranked_task_ignores_the_ranking_verdict_but_needs_a_finite_loss(self, row, expected):
        assert await task_sql.get_evaluated_repo_for_hotkey(uuid4(), BOSS, self._Db(row), ranked=True) == expected
