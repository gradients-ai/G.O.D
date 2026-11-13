"""Compression seeds that failed prep must not crown the challenger: nobody produced anything on those slots.

boss_side_seed_failure returns None for a prep-failed seed without a usable challenger (see test_seed_tasks.py), so the
crown sees a task where the challenger has no valid entry: a non-win beyond both allowances, which rules the crown out.
"""

import pytest

from tests.validator.tournament.test_quant_boss_crown_rule import BOSS
from tests.validator.tournament.test_quant_boss_crown_rule import CHALLENGER
from tests.validator.tournament.test_quant_boss_crown_rule import NOBODY
from tests.validator.tournament.test_quant_boss_crown_rule import WIN
from tests.validator.tournament.test_quant_boss_crown_rule import Tournament


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcomes",
    [
        [WIN] * 5 + [NOBODY, NOBODY],
        [WIN] * 6 + [NOBODY],  # one empty slot is enough: it is beyond both allowances
        [WIN] * 3 + [NOBODY, WIN, WIN, WIN],
    ],
)
async def test_a_slot_nobody_has_a_result_on_never_crowns(monkeypatch, outcomes):
    assert await Tournament(monkeypatch, outcomes).run() == BOSS


@pytest.mark.asyncio
async def test_without_empty_slots_the_same_wins_crown(monkeypatch):
    assert await Tournament(monkeypatch, [WIN] * 7).run() == CHALLENGER
