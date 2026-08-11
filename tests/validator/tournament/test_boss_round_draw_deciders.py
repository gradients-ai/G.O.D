"""Boss-round draws leave the dethrone tally and earn the round a replacement task.

A draw is zero decided examples: every held-out example landed inside the tie dead zone, so nothing
separated the two models. That is a property of the randomly drawn dataset, not of the challenger,
so it must not consume the one task the challenger is allowed to drop. Each drawn task is instead
excluded from the tally and replaced, so the round still resolves over a full set of decided results.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from core.models.task_models import TaskType
from core.models.tournament_models import TournamentType
from validator.tournament import constants as t_cst
from validator.tournament import round_results
from validator.tournament import tournament_manager
from validator.tournament.models import BossRoundDrawResolution
from validator.tournament.models import ContinuousSftLineageOutcome
from validator.tournament.models import RoundStatus
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
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
        continuous_sft_outcomes=[ContinuousSftLineageOutcome(lineage="qwen", winner_hotkey=CHALLENGER)],
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
        continuous_sft_outcomes=[ContinuousSftLineageOutcome(lineage="qwen", winner_hotkey=CHALLENGER)],
        expected_task_count=TEXT_BOSS_ROUND_SIZE,
    )
    assert three_of_four == BOSS

    swept_all_four = determine_boss_round_winner(
        [CHALLENGER, CHALLENGER, CHALLENGER, CHALLENGER],
        BOSS,
        TournamentType.TEXT,
        continuous_sft_outcomes=[ContinuousSftLineageOutcome(lineage="qwen", winner_hotkey=CHALLENGER)],
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
        decider_task_types=[TaskType.DPOTASK, TaskType.INSTRUCTTEXTTASK],
        round_tasks=_round_tasks(TEXT_BOSS_ROUND_SIZE),
        tournament_type=TournamentType.TEXT,
    )

    assert resolution.needs_deciders is True
    assert resolution.decider_task_types == [TaskType.DPOTASK, TaskType.INSTRUCTTEXTTASK]


def test_continuous_sft_draw_is_stood_in_for_by_an_instruct_task():
    """A drawn lineage gets a plain instruct task, not a second chunk of itself.

    The lineage has already cleared its own dethrone gate by drawing, so a second continuous-SFT
    task would buy nothing - while putting two tasks on one sequential train_index and leaving the
    chain's carry-forward to pick between them. The stand-in only tops the tally back up.
    """
    resolution = _resolve_boss_round_draws(
        drawn_task_ids=["task_a"],
        decider_task_types=[TaskType.INSTRUCTTEXTTASK],
        round_tasks=_round_tasks(TEXT_BOSS_ROUND_SIZE),
        tournament_type=TournamentType.TEXT,
    )

    assert resolution.needs_deciders is True
    assert resolution.decider_task_types == [TaskType.INSTRUCTTEXTTASK]


def test_a_drawn_lineage_satisfies_its_own_gate():
    """A draw is the task failing to separate the models, not the challenger failing to beat them.

    So it clears the lineage gate. The challenger still has to win the rest of the round - this only
    stops an undecidable continuous-SFT task costing them the crown on its own.
    """
    winner = determine_boss_round_winner(
        [BOSS, CHALLENGER, CHALLENGER, CHALLENGER, CHALLENGER],
        BOSS,
        TournamentType.TEXT,
        continuous_sft_outcomes=[ContinuousSftLineageOutcome(lineage="qwen", is_draw=True)],
        expected_task_count=TEXT_BOSS_ROUND_SIZE,
    )
    assert winner == CHALLENGER

    # But a draw is not a free pass on the rest: still only 3 of 5 decided won.
    short_elsewhere = determine_boss_round_winner(
        [BOSS, BOSS, CHALLENGER, CHALLENGER, CHALLENGER],
        BOSS,
        TournamentType.TEXT,
        continuous_sft_outcomes=[ContinuousSftLineageOutcome(lineage="qwen", is_draw=True)],
        expected_task_count=TEXT_BOSS_ROUND_SIZE,
    )
    assert short_elsewhere == BOSS


def test_a_lineage_the_boss_won_still_blocks():
    """A loss is not a draw: the boss demonstrably held that lineage, so the gate blocks."""
    winner = determine_boss_round_winner(
        [CHALLENGER, CHALLENGER, CHALLENGER, CHALLENGER],
        BOSS,
        TournamentType.TEXT,
        continuous_sft_outcomes=[ContinuousSftLineageOutcome(lineage="qwen", winner_hotkey=BOSS)],
        expected_task_count=TEXT_BOSS_ROUND_SIZE,
    )

    assert winner == BOSS


def test_deciders_are_capped_at_one_round_of_them():
    """A round already holding more tasks than its type calls for has had its deciders.

    So a decider that itself draws does not spawn another one - it is simply excluded and the round
    resolves on what was decided.
    """
    resolution = _resolve_boss_round_draws(
        drawn_task_ids=["task_a"],
        decider_task_types=[TaskType.INSTRUCTTEXTTASK],
        round_tasks=_round_tasks(TEXT_BOSS_ROUND_SIZE + 1),
        tournament_type=TournamentType.TEXT,
    )

    assert resolution.needs_deciders is False
    assert resolution.can_add_deciders is False
    assert resolution.drawn_task_ids == ["task_a"]


def test_no_draws_means_nothing_to_do():
    resolution = _resolve_boss_round_draws(
        drawn_task_ids=[],
        decider_task_types=[],
        round_tasks=_round_tasks(TEXT_BOSS_ROUND_SIZE),
        tournament_type=TournamentType.TEXT,
    )

    assert resolution.needs_deciders is False
    assert resolution.drawn_task_ids == []


def test_bar_is_clamped_so_it_can_never_be_unreachable():
    """Two draws outliving the single decider batch left 3 decided against a built size of 5.

    That asked for 4 wins out of 3 - a bar meant to stop draws helping the challenger had turned
    into draws guaranteeing the boss, the exact inversion this whole rule exists to prevent.
    Sweeping every task that measured anything is always enough.
    """
    winner = determine_boss_round_winner(
        [CHALLENGER, CHALLENGER, CHALLENGER],
        BOSS,
        TournamentType.TEXT,
        continuous_sft_outcomes=[ContinuousSftLineageOutcome(lineage="qwen", winner_hotkey=CHALLENGER)],
        expected_task_count=TEXT_BOSS_ROUND_SIZE,
    )
    assert winner == CHALLENGER

    # Clamping must not soften the ordinary case: one real loss out of three still fails.
    lost_one = determine_boss_round_winner(
        [BOSS, CHALLENGER, CHALLENGER],
        BOSS,
        TournamentType.TEXT,
        continuous_sft_outcomes=[ContinuousSftLineageOutcome(lineage="qwen", winner_hotkey=CHALLENGER)],
        expected_task_count=TEXT_BOSS_ROUND_SIZE,
    )
    assert lost_one == BOSS


def test_a_lineage_that_never_decides_blocks_the_dethrone():
    """A lineage whose task AND decider both drew must not remove itself from its own gate.

    Counting per task and subtracting the drawn ones took the total to zero, which switched the
    strictest rule in the tournament off exactly when it had the least evidence - the challenger was
    crowned having won no continuous-SFT task at all. Counted per lineage, an undecided lineage is a
    None entry that fails the "won every lineage" check.
    """
    winner = determine_boss_round_winner(
        [CHALLENGER, CHALLENGER, CHALLENGER, CHALLENGER],
        BOSS,
        TournamentType.TEXT,
        continuous_sft_outcomes=[ContinuousSftLineageOutcome(lineage="qwen")],  # no result at all
        expected_task_count=TEXT_BOSS_ROUND_SIZE,
    )

    assert winner == BOSS


def test_a_decider_settles_the_lineage_its_predecessor_drew():
    """The flip side: winning the lineage's decider does satisfy the gate."""
    winner = determine_boss_round_winner(
        [CHALLENGER, CHALLENGER, CHALLENGER, CHALLENGER, CHALLENGER],
        BOSS,
        TournamentType.TEXT,
        continuous_sft_outcomes=[ContinuousSftLineageOutcome(lineage="qwen", winner_hotkey=CHALLENGER)],
        expected_task_count=TEXT_BOSS_ROUND_SIZE,
    )

    assert winner == CHALLENGER


def test_draw_resolution_rejects_unknown_fields():
    """A mistyped field name silently dropped the decider types and stayed inert only by luck."""
    with pytest.raises(ValidationError):
        BossRoundDrawResolution(drawn_task_ids=["a"], drawn_task_types=[TaskType.DPOTASK])


def test_capped_resolution_still_reports_what_drew():
    """The capped path must carry its decider types, not silently blank them.

    It was passing them under the wrong keyword, which pydantic dropped - inert only because the
    capped path short-circuits before anything reads the field.
    """
    resolution = _resolve_boss_round_draws(
        drawn_task_ids=["task_a"],
        decider_task_types=[TaskType.INSTRUCTTEXTTASK],
        round_tasks=_round_tasks(TEXT_BOSS_ROUND_SIZE + 1),
        tournament_type=TournamentType.TEXT,
    )

    assert resolution.can_add_deciders is False
    assert resolution.decider_task_types == [TaskType.INSTRUCTTEXTTASK]


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


@pytest.mark.asyncio
async def test_orphaned_task_defers_the_round_instead_of_scoring_it(monkeypatch):
    """A boss-round task with no assigned nodes never ran, so it must not be scored against anyone.

    One gets here when a decider is created but the round never makes it back to PENDING - a crash,
    or a failure partway through creating a batch. The old path fell through to "no valid results ->
    winner is base contestant", charging the challenger a loss on a task that never happened, while
    the decider cap (round now holds more tasks than its built size) blocked any repair. Deferring
    sends the round back for node assignment instead.
    """
    round_tasks = _round_tasks(TEXT_BOSS_ROUND_SIZE + 1)
    orphan_id = round_tasks[-1].task_id

    async def fake_get_tournament(_tournament_id, _psql_db):
        return TournamentData(tournament_id="tourn_1", tournament_type=TournamentType.TEXT)

    async def fake_get_nodes_assigned_to_task(task_id, _psql_db):
        return [] if task_id == orphan_id else ["node_boss", "node_challenger"]

    monkeypatch.setattr(round_results, "get_tournament", fake_get_tournament)
    monkeypatch.setattr(round_results, "get_nodes_assigned_to_task", fake_get_nodes_assigned_to_task)

    completed_round = TournamentRoundData(
        round_id="tourn_1_round_004",
        tournament_id="tourn_1",
        round_number=4,
        round_type=RoundType.KNOCKOUT,
        is_final_round=True,
        status=RoundStatus.PENDING,
    )

    outcome = await round_results.get_knockout_winners(completed_round, round_tasks, psql_db=None, config=None)

    assert outcome.is_deferred is True
    assert outcome.unassigned_task_ids == [orphan_id]
    assert outcome.winners == []
