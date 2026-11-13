"""R2 dedup must always compare the advancing field against the boss code.

Regression test for the gap where a challenger that just re-submits the boss
(last round's winner) was only caught if the boss happened to advance into the
field, or if a second challenger copied it too. The boss is protected inside
run_pairwise_dedup, so including it can only flag copycats, never the boss.
"""
from unittest.mock import AsyncMock, patch

import pytest

from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament import dedup_gate
from validator.tournament.models import (
    DedupResult,
    TournamentData,
    TournamentParticipant,
)
from core.models.tournament_models import TournamentType


def _participant(hotkey: str) -> TournamentParticipant:
    return TournamentParticipant(
        tournament_id="tourn_x",
        hotkey=hotkey,
        training_repo=f"https://github.com/acme/{hotkey}",
        training_commit_hash="deadbeef",
        github_token=None,
    )


def _no_flag_result(cohort) -> DedupResult:
    return DedupResult(
        cohort=list(cohort),
        pair_verdicts=[],
        flagged_hotkeys=[],
        clusters=[],
        unresolved_pairs=[],
    )


def _tournament() -> TournamentData:
    return TournamentData(
        tournament_id="tourn_x",
        tournament_type=TournamentType.TEXT,
        status="active",
    )


@pytest.mark.asyncio
async def test_boss_is_included_even_when_not_in_the_field():
    """winners excludes the boss; the boss must still be in the compared refs."""
    boss = _participant(EMISSION_BURN_HOTKEY)
    c1 = _participant("challenger-1")
    c2 = _participant("challenger-2")  # not advancing -> must be excluded
    winners = ["challenger-1"]

    captured = {}

    async def fake_pairwise(refs, tournament_type, boss_hotkey=None):
        captured["hotkeys"] = {r.hotkey for r in refs}
        captured["boss_hotkey"] = boss_hotkey
        return _no_flag_result([r.hotkey for r in refs])

    with patch.object(dedup_gate, "get_tournament_participants", AsyncMock(return_value=[boss, c1, c2])), \
         patch.object(dedup_gate, "run_pairwise_dedup", side_effect=fake_pairwise), \
         patch.object(dedup_gate, "insert_dedup_review", AsyncMock()):
        decision = await dedup_gate._run_and_record_gate(
            _tournament(), "tourn_x_round_002", winners, config=object(), psql_db=None
        )

    # boss compared against the lone challenger; the non-advancer stays out
    assert captured["hotkeys"] == {"challenger-1", EMISSION_BURN_HOTKEY}
    assert captured["boss_hotkey"] == EMISSION_BURN_HOTKEY
    assert decision.halt is False


@pytest.mark.asyncio
async def test_single_challenger_still_gets_checked_against_boss():
    """One advancer alone used to be <2 refs and skip; with the boss it now runs."""
    boss = _participant(EMISSION_BURN_HOTKEY)
    c1 = _participant("challenger-1")

    ran = {"called": False}

    async def fake_pairwise(refs, tournament_type, boss_hotkey=None):
        ran["called"] = True
        return _no_flag_result([r.hotkey for r in refs])

    with patch.object(dedup_gate, "get_tournament_participants", AsyncMock(return_value=[boss, c1])), \
         patch.object(dedup_gate, "run_pairwise_dedup", side_effect=fake_pairwise), \
         patch.object(dedup_gate, "insert_dedup_review", AsyncMock()):
        await dedup_gate._run_and_record_gate(
            _tournament(), "tourn_x_round_002", ["challenger-1"], config=object(), psql_db=None
        )

    assert ran["called"] is True


@pytest.mark.asyncio
async def test_no_check_when_boss_has_no_repo_and_one_challenger():
    """If the boss has no repo this tournament, a lone challenger has no pair -> skip."""
    boss = TournamentParticipant(tournament_id="tourn_x", hotkey=EMISSION_BURN_HOTKEY, training_repo=None)
    c1 = _participant("challenger-1")

    with patch.object(dedup_gate, "get_tournament_participants", AsyncMock(return_value=[boss, c1])), \
         patch.object(dedup_gate, "run_pairwise_dedup", AsyncMock()) as pw, \
         patch.object(dedup_gate, "insert_dedup_review", AsyncMock()):
        decision = await dedup_gate._run_and_record_gate(
            _tournament(), "tourn_x_round_002", ["challenger-1"], config=object(), psql_db=None
        )

    pw.assert_not_called()
    assert decision.halt is False
