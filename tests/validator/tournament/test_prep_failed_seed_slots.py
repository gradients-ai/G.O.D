"""Both compression seeds failing prep must not crown the challenger: nobody produced anything on those slots.

boss_side_seed_failure returns None for a prep-failed seed without a usable challenger (see test_seed_tasks.py), so
the round applies the ordinary rule: no valid results, the boss keeps the task, and an empty slot is never a win.
"""

from tests.validator.tournament.test_boss_allowed_loss import BOSS
from tests.validator.tournament.test_boss_allowed_loss import CHALLENGER
from tests.validator.tournament.test_boss_allowed_loss import FIXED
from tests.validator.tournament.test_boss_allowed_loss import FREEFORM
from tests.validator.tournament.test_boss_allowed_loss import Scenario
from tests.validator.tournament.test_boss_allowed_loss import _add_wins
from tests.validator.tournament.test_boss_allowed_loss import _freeform


async def test_two_prep_failed_compression_seeds_do_not_crown_the_challenger(monkeypatch):
    scenario = Scenario(monkeypatch)  # no dead seeds: boss_side_seed_failure gave None for both slots
    scenario.add(FIXED, None)
    scenario.add(FREEFORM, None)

    outcome = await scenario.run()

    assert outcome.winners == [BOSS]


async def test_four_wins_and_two_prep_failed_slots_do_not_crown(monkeypatch):
    scenario = Scenario(monkeypatch)
    _add_wins(scenario, freeform=2, fixed=2)
    scenario.add(FIXED, None)
    scenario.add(FREEFORM, None)

    assert (await scenario.run()).winners == [BOSS]


async def test_one_prep_failed_slot_and_five_wins_still_crowns(monkeypatch):
    scenario = Scenario(monkeypatch)
    scenario.add(FIXED, None)
    _add_wins(scenario, freeform=3, fixed=2)

    outcome = await scenario.run()

    assert outcome.winners == [CHALLENGER]


async def test_one_prep_failed_slot_plus_one_decided_win_does_not_crown_on_the_empty_slot(monkeypatch):
    scenario = Scenario(monkeypatch)
    scenario.add(FIXED, None)
    scenario.add(FREEFORM, [_freeform(BOSS, 10_000), _freeform(CHALLENGER, 9_000)])

    outcome = await scenario.run()

    assert outcome.winners == [BOSS]
