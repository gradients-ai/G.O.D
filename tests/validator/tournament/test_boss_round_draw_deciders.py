"""Boss-round draws leave the dethrone tally and earn the round a replacement task.

A draw is zero decided examples: every held-out example landed inside the tie dead zone, so nothing
separated the two models. That is a property of the randomly drawn dataset, not of the challenger,
so it must not consume the one task the challenger is allowed to drop. Each drawn task is instead
excluded from the tally and replaced, so the round still resolves over a full set of decided results.
"""

import numpy as np
import pytest

from core.models.task_models import TaskType
from core.models.tournament_models import TournamentType
from validator.tournament import constants as t_cst
from validator.tournament.models import TournamentTask
from validator.tournament.round_results import _resolve_boss_round_draws
from validator.tournament.round_results import determine_boss_round_winner
from validator.tournament.thresholds import compare_paired_losses


BOSS = "boss_hotkey"
CHALLENGER = "challenger_hotkey"
TEXT_BOSS_ROUND_SIZE = t_cst.FINAL_ROUND_TEXT_TASKS


def _round_tasks(n: int) -> list[TournamentTask]:
    return [
        TournamentTask(
            tournament_id="tourn_1",
            round_id="tourn_1_round_004",
            task_id=f"00000000-0000-0000-0000-{i:012d}",
            pair_id="tourn_1_round_004_pair_001",
        )
        for i in range(n)
    ]


def test_all_examples_inside_dead_zone_is_a_draw():
    """The trigger condition: no example separated the two models by more than the dead zone."""
    boss = np.random.default_rng(7).uniform(0.02, 0.03, 800)
    challenger = boss - (t_cst.BOSS_ROUND_TIE_DEADZONE_NATS / 2)

    result = compare_paired_losses(list(boss), list(challenger))

    assert result.n_decided == 0
    assert result.is_draw is True
    assert result.challenger_won is False


def test_chris_scenario_challenger_takes_the_crown_after_winning_the_decider():
    """Boss 1, draw 1, challenger 3 -> add a task -> challenger wins it -> challenger is the boss.

    The draw never enters the tally, so the decider brings the decided count back to the boss round's
    full size and the usual "lose at most one" rule applies over those five.
    """
    # The four decided tasks from the original round, plus the decider the challenger then won.
    decided_winners = [BOSS, CHALLENGER, CHALLENGER, CHALLENGER, CHALLENGER]

    winner = determine_boss_round_winner(
        decided_winners,
        BOSS,
        TournamentType.TEXT,
        continuous_sft_winners=[CHALLENGER],
        num_continuous_sft_tasks=1,
    )

    assert len(decided_winners) == TEXT_BOSS_ROUND_SIZE
    assert winner == CHALLENGER


def test_a_dead_task_never_lowers_the_dethrone_bar():
    """Reached only when a decider itself draws, leaving four decided tasks instead of five.

    The bar stays pinned at the round's built size, so the challenger must sweep the four that
    worked rather than getting in on three. Excluding a draw takes it out of the numerator; it must
    not also shrink the denominator.
    """
    three_of_four = determine_boss_round_winner(
        [BOSS, CHALLENGER, CHALLENGER, CHALLENGER],
        BOSS,
        TournamentType.TEXT,
        continuous_sft_winners=[CHALLENGER],
        num_continuous_sft_tasks=1,
        expected_task_count=TEXT_BOSS_ROUND_SIZE,
    )
    assert three_of_four == BOSS

    swept_all_four = determine_boss_round_winner(
        [CHALLENGER, CHALLENGER, CHALLENGER, CHALLENGER],
        BOSS,
        TournamentType.TEXT,
        continuous_sft_winners=[CHALLENGER],
        num_continuous_sft_tasks=1,
        expected_task_count=TEXT_BOSS_ROUND_SIZE,
    )
    assert swept_all_four == CHALLENGER


def test_bar_is_not_pinned_when_nothing_drew():
    """No draws means no pin, so image and environment boss rounds are untouched by any of this.

    They cannot draw - the paired gate is text-only - and their bar keeps deriving from however many
    of their own tasks resolved, which is what a failed image task has always relied on.
    """
    winner = determine_boss_round_winner(
        [BOSS, CHALLENGER, CHALLENGER, CHALLENGER],
        BOSS,
        TournamentType.IMAGE,
        expected_task_count=None,
    )

    assert winner == CHALLENGER


def test_draws_earn_one_decider_each_mirroring_the_type_that_drew():
    resolution = _resolve_boss_round_draws(
        drawn_task_ids=["task_a", "task_b"],
        drawn_task_types=[TaskType.DPOTASK, TaskType.INSTRUCTTEXTTASK],
        drawn_continuous_sft_lineages=[None, None],
        round_tasks=_round_tasks(TEXT_BOSS_ROUND_SIZE),
        tournament_type=TournamentType.TEXT,
    )

    assert resolution.needs_deciders is True
    assert resolution.decider_task_types == [TaskType.DPOTASK, TaskType.INSTRUCTTEXTTASK]


def test_continuous_sft_draw_is_mirrored_by_lineage_not_downgraded():
    """The dethrone rule needs continuous-SFT tasks *won*, so a draw must be replaced in kind."""
    resolution = _resolve_boss_round_draws(
        drawn_task_ids=["task_a"],
        drawn_task_types=[TaskType.CHATTASK],
        drawn_continuous_sft_lineages=["qwen"],
        round_tasks=_round_tasks(TEXT_BOSS_ROUND_SIZE),
        tournament_type=TournamentType.TEXT,
    )

    assert resolution.needs_deciders is True
    assert resolution.decider_task_types == [TaskType.CHATTASK]
    assert resolution.continuous_sft_lineages == ["qwen"]


def test_deciders_are_capped_at_one_round_of_them():
    """A round already holding more tasks than its type calls for has had its deciders.

    So a decider that itself draws does not spawn another one - it is simply excluded and the round
    resolves on what was decided.
    """
    resolution = _resolve_boss_round_draws(
        drawn_task_ids=["task_a"],
        drawn_task_types=[TaskType.INSTRUCTTEXTTASK],
        drawn_continuous_sft_lineages=[None],
        round_tasks=_round_tasks(TEXT_BOSS_ROUND_SIZE + 1),
        tournament_type=TournamentType.TEXT,
    )

    assert resolution.needs_deciders is False
    assert resolution.can_add_deciders is False
    assert resolution.drawn_task_ids == ["task_a"]


def test_no_draws_means_nothing_to_do():
    resolution = _resolve_boss_round_draws(
        drawn_task_ids=[],
        drawn_task_types=[],
        drawn_continuous_sft_lineages=[],
        round_tasks=_round_tasks(TEXT_BOSS_ROUND_SIZE),
        tournament_type=TournamentType.TEXT,
    )

    assert resolution.needs_deciders is False
    assert resolution.drawn_task_ids == []


@pytest.mark.parametrize(
    "tournament_type,expected",
    [
        (TournamentType.TEXT, t_cst.FINAL_ROUND_TEXT_TASKS),
        # Image and environment cannot draw, so they get 0 and every draw rule stays off for them.
        (TournamentType.IMAGE, 0),
        (TournamentType.ENVIRONMENT, 0),
    ],
)
def test_expected_boss_round_task_count(tournament_type, expected):
    assert t_cst.expected_boss_round_task_count(tournament_type) == expected


def test_only_text_task_types_can_draw():
    """The premise the rest of this rests on: the paired gate, and so draws, are text-only."""
    assert set(t_cst.PAIRED_BOSS_ROUND_TASK_TYPES) == {
        TaskType.INSTRUCTTEXTTASK,
        TaskType.DPOTASK,
        TaskType.CHATTASK,
    }
