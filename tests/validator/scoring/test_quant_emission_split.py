"""Three-way emission split (text / quant / image): guaranteed minimums plus a participation-driven flex pool."""

from itertools import product
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import validator.scoring.constants as cts
from core.models.tournament_models import TournamentType
from validator.endpoints.transfer_balances import get_tournament_fees
from validator.scoring.emission_balance import compute_dynamic_base_weights
from validator.scoring.emission_balance import get_dynamic_base_weights
from validator.scoring.emission_balance import max_weight_by_type
from validator.scoring.tournaments import first_place_share
from validator.scoring.tournaments import place_share
from validator.scoring.weights import apply_quant_tournament_weights
from validator.scoring.weights import build_tournament_audit_data
from validator.scoring.weights import get_node_weights_from_tournament_audit_data
from validator.scoring.weights import get_tournament_burn_details
from validator.tournament import constants as t_cst
from validator.tournament import cost_constants
from validator.tournament.models import TournamentAuditData
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentResults
from validator.tournament.models import TournamentResultsWithWinners
from validator.tournament.models import TournamentStatus


TEXT = TournamentType.TEXT
QUANT = TournamentType.QUANT
IMAGE = TournamentType.IMAGE

ENTRANTS = [None, 3, 10, 25, 35, 100, 1000]
BURN = "burn_hotkey"
CROWN = first_place_share(0)


def test_minimums_and_flex_constants():
    assert cts.TOURNAMENT_TEXT_WEIGHT == 0.25
    assert cts.TOURNAMENT_QUANT_WEIGHT == 0.15
    assert cts.TOURNAMENT_IMAGE_WEIGHT == 0.15
    assert cts.EMISSION_FLEX_POOL == pytest.approx(0.45)
    assert cts.MAX_TEXT_TOURNAMENT_WEIGHT == pytest.approx(0.25 + 0.45)
    assert cts.MAX_QUANT_TOURNAMENT_WEIGHT == pytest.approx(0.15 + 0.45)
    assert cts.MAX_IMAGE_TOURNAMENT_WEIGHT == pytest.approx(0.15 + 0.45)


@pytest.mark.parametrize("text,quant,image", list(product(ENTRANTS, repeat=3)))
def test_bases_sum_to_one_and_stay_within_minimum_plus_flex(text, quant, image):
    weights = compute_dynamic_base_weights({TEXT: text, QUANT: quant, IMAGE: image})

    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)
    assert cts.TOURNAMENT_TEXT_WEIGHT <= weights[TEXT] <= cts.MAX_TEXT_TOURNAMENT_WEIGHT
    assert cts.TOURNAMENT_QUANT_WEIGHT <= weights[QUANT] <= cts.MAX_QUANT_TOURNAMENT_WEIGHT
    assert cts.TOURNAMENT_IMAGE_WEIGHT <= weights[IMAGE] <= cts.MAX_IMAGE_TOURNAMENT_WEIGHT


def test_balanced_entrants_split_flex_at_the_minimums_ratio():
    weights = compute_dynamic_base_weights({TEXT: 100, QUANT: 60, IMAGE: 60})

    total_minimum = 0.25 + 0.15 + 0.15
    for tournament_type, minimum in ((TEXT, 0.25), (QUANT, 0.15), (IMAGE, 0.15)):
        assert weights[tournament_type] == pytest.approx(minimum + cts.EMISSION_FLEX_POOL * minimum / total_minimum)


def test_max_weight_by_type_returns_the_plain_caps():
    assert max_weight_by_type() == {
        TEXT: cts.MAX_TEXT_TOURNAMENT_WEIGHT,
        QUANT: cts.MAX_QUANT_TOURNAMENT_WEIGHT,
        IMAGE: cts.MAX_IMAGE_TOURNAMENT_WEIGHT,
    }


@pytest.mark.asyncio
async def test_minimums_apply_after_the_phase_in_with_no_completed_quant_tournament():
    with patch("validator.scoring.emission_balance.get_recent_contender_counts", AsyncMock(return_value=[])):
        weights = await get_dynamic_base_weights(MagicMock())

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights[QUANT] >= cts.TOURNAMENT_QUANT_WEIGHT
    assert weights[TEXT] < cts.TOURNAMENT_TEXT_WEIGHT + cts.EMISSION_FLEX_POOL + 1e-9
    assert weights[IMAGE] < cts.TOURNAMENT_IMAGE_WEIGHT + cts.EMISSION_FLEX_POOL + 1e-9


@pytest.mark.asyncio
async def test_burn_details_without_a_quant_tournament_still_reserve_quants_base():
    async def latest(psql_db, tournament_type, exclude_tournament_id=None):
        return None

    with (
        patch("validator.scoring.weights.get_latest_completed_tournament", latest),
        patch("validator.scoring.emission_balance.get_recent_contender_counts", AsyncMock(return_value=[])),
    ):
        burn = await get_tournament_burn_details(MagicMock())

    assert burn.quant_tournament_weight >= cts.TOURNAMENT_QUANT_WEIGHT
    assert burn.text_tournament_weight < cts.TOURNAMENT_TEXT_WEIGHT + cts.EMISSION_FLEX_POOL + 1e-9
    assert burn.text_tournament_weight + burn.quant_tournament_weight + burn.image_tournament_weight == pytest.approx(1.0)
    assert burn.burn_weight == 0.0


@pytest.mark.asyncio
async def test_audit_data_without_a_quant_tournament_keeps_quants_pool():
    async def latest(psql_db, tournament_type, exclude_tournament_id=None):
        return None

    with (
        patch("validator.scoring.weights.get_latest_completed_tournament", latest),
        patch("validator.scoring.weights.get_active_tournament_participants", AsyncMock(return_value=[])),
        patch("validator.scoring.weights.get_weekly_task_participation_data", AsyncMock(return_value=[])),
        patch("validator.scoring.emission_balance.get_recent_contender_counts", AsyncMock(return_value=[])),
    ):
        audit = await build_tournament_audit_data(MagicMock())

    assert audit.quant_tournament_data is None
    assert audit.quant_tournament_weight >= cts.TOURNAMENT_QUANT_WEIGHT
    assert audit.text_tournament_weight + audit.quant_tournament_weight + audit.image_tournament_weight == pytest.approx(1.0)


def test_old_audit_payloads_without_quant_fields_still_validate():
    audit = TournamentAuditData.model_validate({"text_tournament_weight": 0.6, "image_tournament_weight": 0.4})

    assert audit.quant_tournament_data is None
    assert audit.quant_tournament_weight == 0.0


@pytest.mark.asyncio
async def test_node_weights_without_a_quant_tournament_pay_no_quant_hotkey():
    audit = TournamentAuditData(text_tournament_weight=0.6, image_tournament_weight=0.4, burn_weight=0.0)
    nodes = [MagicMock(hotkey="text_winner", node_id=0), MagicMock(hotkey="quant_winner", node_id=1)]

    with (
        patch("validator.scoring.weights.fetch_nodes") as fetch,
        patch("validator.scoring.weights.cts.EMISSION_BURN_HOTKEY", "absent"),
    ):
        fetch._get_nodes_for_uid.return_value = nodes
        result = await get_node_weights_from_tournament_audit_data(MagicMock(), 1, audit)

    assert result.node_weights == [0.0, 0.0]


def test_quant_pool_pays_the_top_three_final_positions_once_completed():
    node_ids = {"a": 0, "b": 1, "c": 2, "d": 3}
    node_weights = [0.0] * 4

    undistributed = apply_quant_tournament_weights(
        quant_tournament_weights={"a": 0.76190476, "b": 0.19047619, "c": 0.04761905},
        hotkey_to_node_id=node_ids,
        all_node_weights=node_weights,
        scaled_quant_tournament_weight=0.35,
    )

    assert node_weights[3] == 0.0
    assert node_weights[0] == pytest.approx(0.76190476 * 0.35)
    assert node_weights[1] == pytest.approx(0.19047619 * 0.35)
    assert node_weights[2] == pytest.approx(0.04761905 * 0.35)
    assert undistributed == pytest.approx(0.35 - sum(node_weights), abs=1e-6)


@pytest.mark.asyncio
async def test_node_weights_pay_the_completed_quant_tournament_ranks():
    quant_data = TournamentResultsWithWinners(
        tournament_id="quant_1",
        rounds=[],
        base_winner_hotkey="q1",
        winner_hotkey="q1",
        final_positions={"q1": 1, "q2": 2, "q3": 3, "q4": 4},
    )
    audit = TournamentAuditData(
        quant_tournament_data=quant_data,
        text_tournament_weight=0.25,
        quant_tournament_weight=0.50,
        image_tournament_weight=0.25,
    )
    nodes = [MagicMock(hotkey=hotkey, node_id=idx) for idx, hotkey in enumerate(["q1", "q2", "q3", "q4", BURN])]

    with (
        patch("validator.scoring.weights.fetch_nodes") as fetch,
        patch("validator.scoring.weights.cts.EMISSION_BURN_HOTKEY", BURN),
    ):
        fetch._get_nodes_for_uid.return_value = nodes
        result = await get_node_weights_from_tournament_audit_data(MagicMock(), 1, audit)

    q1, q2, q3, q4, burn = result.node_weights
    assert q1 > q2 > q3 > 0.0
    assert q4 == 0.0
    assert q1 + q2 + q3 == pytest.approx(0.50)
    assert burn == pytest.approx(0.50)
    assert sum(result.node_weights) == pytest.approx(1.0)


def test_participation_fees_by_type():
    assert t_cst.TOURNAMENT_QUANT_PARTICIPATION_FEE_RAO == 500_000_000
    assert t_cst.TOURNAMENT_TEXT_PARTICIPATION_FEE_RAO == 600_000_000
    assert t_cst.TOURNAMENT_IMAGE_PARTICIPATION_FEE_RAO == 400_000_000
    by_type = cost_constants.TOURNAMENT_PARTICIPATION_FEE_RAO_BY_TYPE
    assert by_type["quant"] == 500_000_000
    assert by_type["text"] == 600_000_000
    assert by_type["image"] == 400_000_000


@pytest.mark.asyncio
async def test_fee_endpoint_reports_the_quant_fee():
    response = await get_tournament_fees()

    assert response.quant_tournament_fee_rao == 500_000_000
    assert response.text_tournament_fee_rao == 600_000_000
    assert response.image_tournament_fee_rao == 400_000_000


def _completed(tournament_id: str, tournament_type: TournamentType, base_winner: str, winner: str) -> TournamentData:
    return TournamentData(
        tournament_id=tournament_id,
        tournament_type=tournament_type,
        status=TournamentStatus.COMPLETED,
        base_winner_hotkey=base_winner,
        winner_hotkey=winner,
    )


# The text tournament was kept by its boss: 1st is the burn placeholder, paid to the defending champion "champ".
TEXT_DONE = _completed("text_1", TEXT, base_winner="champ", winner=BURN)
QUANT_DONE = _completed("quant_1", QUANT, base_winner="q_champ", winner="q1")
IMAGE_DONE = _completed("image_1", IMAGE, base_winner="i1", winner="i1")
PLACES = {
    "text_1": {BURN: 1, "t2": 2, "t3": 3, "t4": 4},
    "quant_1": {"q1": 1, "q2": 2, "q3": 3},
    "image_1": {"i1": 1, "i2": 2},
}


async def _audit_with(completed: list[TournamentData]) -> TournamentAuditData:
    by_type = {tournament.tournament_type: tournament for tournament in completed}

    async def latest(psql_db, tournament_type, exclude_tournament_id=None):
        return by_type.get(tournament_type)

    async def results(tournament_id, psql_db):
        return TournamentResults(tournament_id=tournament_id, rounds=[])

    async def participants(tournament_id, psql_db):
        return [
            TournamentParticipant(tournament_id=tournament_id, hotkey=hotkey, final_position=position)
            for hotkey, position in PLACES[tournament_id].items()
        ]

    with (
        patch("validator.scoring.weights.get_latest_completed_tournament", latest),
        patch("validator.scoring.weights.get_tournament_full_results", results),
        patch("validator.scoring.weights.get_tournament_participants", participants),
        patch("validator.scoring.weights.get_boss_rank_comparisons", AsyncMock(return_value=[])),
        patch("validator.scoring.weights.count_champion_defences", AsyncMock(return_value=0)),
        patch("validator.scoring.weights.get_active_tournament_participants", AsyncMock(return_value=[])),
        patch("validator.scoring.weights.get_weekly_task_participation_data", AsyncMock(return_value=[])),
        patch("validator.scoring.emission_balance.get_recent_contender_counts", AsyncMock(return_value=[])),
    ):
        return await build_tournament_audit_data(MagicMock())


async def _node_weights(audit: TournamentAuditData, hotkeys: list[str]) -> dict[str, float]:
    nodes = [MagicMock(hotkey=hotkey, node_id=idx) for idx, hotkey in enumerate(hotkeys)]
    with (
        patch("validator.scoring.weights.fetch_nodes") as fetch,
        patch("validator.scoring.weights.cts.EMISSION_BURN_HOTKEY", BURN),
    ):
        fetch._get_nodes_for_uid.return_value = nodes
        result = await get_node_weights_from_tournament_audit_data(MagicMock(), 1, audit)
    return dict(zip(hotkeys, result.node_weights))


@pytest.mark.asyncio
async def test_quant_pool_pays_text_placements_until_a_quant_tournament_completes():
    audit = await _audit_with([TEXT_DONE, IMAGE_DONE])

    assert audit.quant_tournament_data == audit.text_tournament_data
    assert audit.quant_tournament_data.tournament_id == "text_1"

    hotkeys = ["champ", "t2", "t3", "t4", "i1", "i2", BURN]
    weights = await _node_weights(audit, hotkeys)

    text_and_quant = audit.text_tournament_weight + audit.quant_tournament_weight
    assert weights["champ"] == pytest.approx(place_share(3, 1, CROWN) * text_and_quant)
    assert weights["t2"] == pytest.approx(place_share(3, 2, CROWN) * text_and_quant)
    assert weights["t3"] == pytest.approx(place_share(3, 3, CROWN) * text_and_quant)
    assert weights["t4"] == 0.0
    assert weights["i1"] == pytest.approx(place_share(2, 1, CROWN) * audit.image_tournament_weight)
    assert weights["i2"] == pytest.approx(place_share(2, 2, CROWN) * audit.image_tournament_weight)
    assert sum(weights.values()) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_quant_pool_pays_its_own_placements_once_a_quant_tournament_completes():
    audit = await _audit_with([TEXT_DONE, QUANT_DONE, IMAGE_DONE])

    assert audit.quant_tournament_data.tournament_id == "quant_1"
    assert audit.quant_tournament_data.final_positions == PLACES["quant_1"]

    hotkeys = ["champ", "t2", "t3", "q1", "q2", "q3", "i1", "i2", BURN]
    weights = await _node_weights(audit, hotkeys)

    for hotkey, position in (("champ", 1), ("t2", 2), ("t3", 3)):
        assert weights[hotkey] == pytest.approx(place_share(3, position, CROWN) * audit.text_tournament_weight)
    for hotkey, position in (("q1", 1), ("q2", 2), ("q3", 3)):
        assert weights[hotkey] == pytest.approx(place_share(3, position, CROWN) * audit.quant_tournament_weight)
    assert sum(weights.values()) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_text_and_image_data_do_not_depend_on_the_quant_fallback():
    without_quant = await _audit_with([TEXT_DONE, IMAGE_DONE])
    with_quant = await _audit_with([TEXT_DONE, QUANT_DONE, IMAGE_DONE])

    assert without_quant.text_tournament_data == with_quant.text_tournament_data
    assert without_quant.image_tournament_data == with_quant.image_tournament_data
    assert without_quant.text_tournament_data.tournament_id == "text_1"
    assert without_quant.image_tournament_data.tournament_id == "image_1"


@pytest.mark.asyncio
async def test_no_completed_tournament_of_any_type_leaves_the_quant_pool_unpaid():
    audit = await _audit_with([])

    assert audit.quant_tournament_data is None
    assert audit.text_tournament_data is None
