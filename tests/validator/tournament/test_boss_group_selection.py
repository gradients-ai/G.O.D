from validator.tournament.models import Group
from validator.tournament.tournament_manager import select_boss_group_index


def _group(size: int) -> Group:
    return Group(member_ids=[f"hk_{index}" for index in range(size)], task_ids=[])


def test_select_boss_group_picks_smallest_group():
    assert select_boss_group_index([_group(4), _group(2), _group(3)]) == 1


def test_select_boss_group_tie_picks_first_smallest():
    assert select_boss_group_index([_group(2), _group(2), _group(4)]) == 0


def test_select_boss_group_single_group():
    assert select_boss_group_index([_group(4)]) == 0
