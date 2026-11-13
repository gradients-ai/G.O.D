"""Every tournament type pays its places from one pool: its dynamic base share. A tournament won by the burn hotkey adds
nothing, and the pools and what they pay always add up."""

from itertools import product
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import validator.scoring.constants as cts
from core.models.tournament_models import TournamentType
from validator.scoring.emission_balance import compute_dynamic_base_weights
from validator.scoring.weights import apply_quant_tournament_weights
from validator.scoring.weights import apply_tournament_weights
from validator.scoring.weights import build_tournament_audit_data
from validator.scoring.weights import get_node_weights_from_tournament_audit_data
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentStatus


TEXT, QUANT, IMAGE = TournamentType.TEXT, TournamentType.QUANT, TournamentType.IMAGE
BOSS = cts.EMISSION_BURN_HOTKEY
CHAMPION = "5Champion"
SECOND = "5Second"
THIRD = "5Third"
BASE = {TEXT: 0.3, QUANT: 0.4, IMAGE: 0.3}
# Rank-weight ladder over three paid places.
FIRST_SHARE, SECOND_SHARE, THIRD_SHARE = 16 / 21, 4 / 21, 1 / 21


def _completed(tournament_type: TournamentType, winner: str, base_winner: str | None) -> TournamentData:
    return TournamentData(
        tournament_id=f"{tournament_type.value}_1",
        tournament_type=tournament_type,
        status=TournamentStatus.COMPLETED,
        base_winner_hotkey=base_winner,
        winner_hotkey=winner,
    )


async def _audit(winner: str, base_winner: str | None, base: dict[TournamentType, float] = BASE):
    """The audit data built from a fake database where every type's latest tournament was won by `winner`."""

    async def latest(_db, tournament_type, exclude_tournament_id=None):
        return _completed(tournament_type, winner, base_winner)

    full_results = AsyncMock(return_value=SimpleNamespace(tournament_id="t", rounds=[]))
    participants = [SimpleNamespace(hotkey=h, final_position=i) for i, h in enumerate((winner, SECOND, THIRD), start=1)]
    with (
        patch("validator.scoring.weights.get_latest_completed_tournament", latest),
        patch("validator.scoring.weights.get_tournament_full_results", full_results),
        patch("validator.scoring.weights.get_tournament_participants", AsyncMock(return_value=participants)),
        patch("validator.scoring.weights.get_active_tournament_participants", AsyncMock(return_value=[])),
        patch("validator.scoring.weights.get_weekly_task_participation_data", AsyncMock(return_value=[])),
        patch("validator.scoring.weights.get_boss_rank_comparisons", AsyncMock(return_value=[])),
        patch("validator.scoring.weights.get_dynamic_base_weights", AsyncMock(return_value=dict(base))),
    ):
        return await build_tournament_audit_data(MagicMock())


async def _node_weights(audit, hotkeys: list[str]) -> dict[str, float]:
    nodes = [MagicMock(hotkey=hotkey, node_id=index) for index, hotkey in enumerate(hotkeys)]
    with patch("validator.scoring.weights.fetch_nodes") as fetch:
        fetch._get_nodes_for_uid.return_value = nodes
        result = await get_node_weights_from_tournament_audit_data(MagicMock(), 1, audit)
    return dict(zip(hotkeys, result.node_weights))


@pytest.mark.asyncio
async def test_each_types_pool_is_its_dynamic_base_share_whoever_won():
    for winner, base_winner in ((CHAMPION, BOSS), (BOSS, CHAMPION), (BOSS, None)):
        audit = await _audit(winner, base_winner)

        assert audit.text_tournament_weight == pytest.approx(BASE[TEXT])
        assert audit.quant_tournament_weight == pytest.approx(BASE[QUANT])
        assert audit.image_tournament_weight == pytest.approx(BASE[IMAGE])
        assert audit.burn_weight == 0.0


@pytest.mark.asyncio
async def test_a_real_champion_is_paid_the_first_place_share_of_the_base_pool_in_every_type():
    audit = await _audit(CHAMPION, BOSS)

    weights = await _node_weights(audit, [CHAMPION, SECOND, THIRD, BOSS])

    # Every type has the same first-place hotkey, so its node collects first place of each pool.
    expected = FIRST_SHARE * (BASE[TEXT] + BASE[QUANT] + BASE[IMAGE])
    assert weights[CHAMPION] == pytest.approx(expected)
    assert weights[SECOND] == pytest.approx(SECOND_SHARE)
    assert weights[THIRD] == pytest.approx(THIRD_SHARE)
    assert weights[BOSS] == pytest.approx(0.0)
    assert sum(weights.values()) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_a_tournament_won_by_the_burn_hotkey_with_no_known_champion_gets_no_extra_pool():
    audit = await _audit(BOSS, None)

    weights = await _node_weights(audit, [SECOND, THIRD, BOSS])

    # The burn hotkey's first place is undistributed and reaches the burn node: pools are not enlarged.
    assert weights[SECOND] == pytest.approx(SECOND_SHARE)
    assert weights[THIRD] == pytest.approx(THIRD_SHARE)
    assert weights[BOSS] == pytest.approx(FIRST_SHARE)
    assert sum(weights.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("text,quant,image", list(product([None, 3, 40, 1000], repeat=3)))
def test_pools_sum_to_one_and_distributed_plus_undistributed_equal_the_total(text, quant, image):
    pools = compute_dynamic_base_weights({TEXT: text, QUANT: quant, IMAGE: image})
    assert sum(pools.values()) == pytest.approx(1.0, abs=1e-12)

    hotkey_to_node_id = {"a": 0, "b": 1, "c": 2}
    node_weights = [0.0] * 3
    full = {"a": FIRST_SHARE, "b": SECOND_SHARE, "c": THIRD_SHARE}
    first_two_only = {"a": FIRST_SHARE, "b": SECOND_SHARE}

    undistributed = apply_tournament_weights(first_two_only, full, hotkey_to_node_id, node_weights, pools[TEXT], pools[IMAGE])
    undistributed += apply_quant_tournament_weights(full, hotkey_to_node_id, node_weights, pools[QUANT])

    assert sum(node_weights) + undistributed == pytest.approx(1.0)
    assert undistributed == pytest.approx(pools[TEXT] * THIRD_SHARE)  # the unfilled third place of text
