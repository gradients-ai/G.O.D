"""Text tournament round 1 is always a single group, regardless of field size.

Image tournaments keep the old participant-band-gated behavior (a single group only for
3..14 participants; a knockout or multi-group split otherwise) - only text changed.
"""

from unittest.mock import MagicMock

from validator.tournament.models import GroupRound
from validator.tournament.models import KnockoutRound
from validator.tournament.models import TournamentType
from validator.tournament.tournament_manager import organise_tournament_round


def _make_nodes(n: int) -> list:
    nodes = []
    for i in range(n):
        node = MagicMock()
        node.hotkey = f"hk_{i:03d}"
        nodes.append(node)
    return nodes


def _round(nodes, tournament_type, round_number=1) -> GroupRound | KnockoutRound:
    return organise_tournament_round(
        nodes,
        MagicMock(),
        tournament_type=tournament_type,
        round_id="test_round",
        round_number=round_number,
    )


class TestTextRoundOneAlwaysSingleGroup:
    def test_small_field_is_a_single_group(self):
        result = _round(_make_nodes(8), TournamentType.TEXT)
        assert isinstance(result, GroupRound)
        assert len(result.groups) == 1
        assert len(result.groups[0].member_ids) == 8

    def test_field_above_the_old_multi_group_threshold_is_still_a_single_group(self):
        result = _round(_make_nodes(50), TournamentType.TEXT)
        assert isinstance(result, GroupRound)
        assert len(result.groups) == 1
        assert len(result.groups[0].member_ids) == 50

    def test_very_large_field_is_still_a_single_group(self):
        result = _round(_make_nodes(200), TournamentType.TEXT)
        assert isinstance(result, GroupRound)
        assert len(result.groups) == 1
        assert len(result.groups[0].member_ids) == 200

    def test_only_applies_to_round_one(self):
        """A later round with a large surviving field still splits into multiple groups."""
        result = _round(_make_nodes(50), TournamentType.TEXT, round_number=2)
        assert isinstance(result, GroupRound)
        assert len(result.groups) > 1


class TestImageRoundOneUnchanged:
    def test_small_field_is_still_a_single_group(self):
        result = _round(_make_nodes(8), TournamentType.IMAGE)
        assert isinstance(result, GroupRound)
        assert len(result.groups) == 1

    def test_large_field_still_splits_into_multiple_groups(self):
        result = _round(_make_nodes(50), TournamentType.IMAGE)
        assert isinstance(result, GroupRound)
        assert len(result.groups) > 1
