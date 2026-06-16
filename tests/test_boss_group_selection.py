"""Tests for select_boss_group_index — the boss (champion) is injected into the smallest
group to minimise added PvP pairs and keep groups within the size cap.
"""

from core.models.tournament_models import Group
from validator.tournament.tournament_manager import select_boss_group_index


def _group(n: int) -> Group:
    return Group(member_ids=[f"hk_{i}" for i in range(n)], task_ids=[])


def test_picks_smallest_group():
    groups = [_group(4), _group(2), _group(3)]
    assert select_boss_group_index(groups) == 1


def test_tie_picks_first_smallest():
    groups = [_group(2), _group(2), _group(4)]
    assert select_boss_group_index(groups) == 0


def test_single_group():
    assert select_boss_group_index([_group(4)]) == 0
