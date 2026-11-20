"""SQL behind the freeform reconcile: unpublished winner decoders and boss-round decoded models (mocked DB)."""

import re
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import validator.db.constants as cst
from core.constants.network import NETUID
from core.models.task_models import TaskType
from validator.db.sql import tournaments as tournament_sql
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.models import PendingDecoderRelease
from validator.tournament.models import TournamentType


def _mock_psql(fetch=None):
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=fetch or [])
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    psql = MagicMock()
    psql.connection = AsyncMock(return_value=cm)
    return psql, conn


def _flat(query: str) -> str:
    return re.sub(r"\s+", " ", query)


async def test_pending_decoder_releases_join_the_winner_and_skip_the_burn_hotkey():
    psql, conn = _mock_psql(fetch=[{cst.TOURNAMENT_ID: "tourn_1", cst.HOTKEY: "5Champion"}])

    result = await tournament_sql.get_pending_decoder_releases(EMISSION_BURN_HOTKEY, psql)

    assert result == [PendingDecoderRelease(tournament_id="tourn_1", hotkey="5Champion")]
    query, *args = conn.fetch.await_args.args
    flat = _flat(query)
    assert f"JOIN {cst.TOURNAMENTS_TABLE} t ON tp.{cst.TOURNAMENT_ID} = t.{cst.TOURNAMENT_ID}" in flat
    assert f"AND tp.{cst.HOTKEY} = t.{cst.WINNER_HOTKEY}" in flat
    assert f"tp.{cst.HOTKEY} <> $1" in flat
    assert f"t.{cst.TOURNAMENT_TYPE} = $2" in flat
    assert f"tp.{cst.DECODER_OBJECT_NAME} IS NOT NULL" in flat
    assert f"tp.{cst.DECODER_RELEASED_AT} IS NULL" in flat
    assert f"tp.{cst.BACKUP_REPO} IS NOT NULL" in flat
    assert "status" not in flat.lower()  # keyed on winner_hotkey, not on a tournament status value
    assert args == [EMISSION_BURN_HOTKEY, TournamentType.QUANT.value]


async def test_blocked_decoder_releases_are_the_same_query_without_a_backup_repo():
    psql, conn = _mock_psql(fetch=[{cst.TOURNAMENT_ID: "tourn_1", cst.HOTKEY: "5Champion"}])

    result = await tournament_sql.get_blocked_decoder_releases(EMISSION_BURN_HOTKEY, psql)

    assert [r.hotkey for r in result] == ["5Champion"]
    query, *args = conn.fetch.await_args.args
    flat = _flat(query)
    assert f"tp.{cst.BACKUP_REPO} IS NULL" in flat and "IS NOT NULL" in flat
    assert f"tp.{cst.BACKUP_REPO} IS NOT NULL" not in flat
    assert f"tp.{cst.HOTKEY} = t.{cst.WINNER_HOTKEY}" in flat and f"tp.{cst.DECODER_RELEASED_AT} IS NULL" in flat
    assert args == [EMISSION_BURN_HOTKEY, TournamentType.QUANT.value]


async def test_pending_decoded_tournaments_need_a_winner_and_unpublished_final_round_decoded_files():
    psql, conn = _mock_psql(fetch=[{cst.TOURNAMENT_ID: "tourn_1"}, {cst.TOURNAMENT_ID: "tourn_2"}])

    result = await tournament_sql.get_tournaments_with_pending_decoded_publication(psql)

    assert result == ["tourn_1", "tourn_2"]
    query, *args = conn.fetch.await_args.args
    flat = _flat(query)
    assert flat.count("DISTINCT") == 1
    assert f"t.{cst.WINNER_HOTKEY} IS NOT NULL" in flat
    assert f"t.{cst.TOURNAMENT_TYPE} = $1" in flat
    assert f"tr.{cst.IS_FINAL_ROUND} = true" in flat
    assert f"tk.{cst.TASK_TYPE} = $2" in flat
    assert f"tn.{cst.NETUID} = $3" in flat
    assert f"tn.{cst.DECODED_FILES} IS NOT NULL" in flat
    assert f"tn.{cst.DECODED_FILES} <> 'null'::jsonb" in flat
    assert f"tn.{cst.DECODED_FILES} <> '[]'::jsonb" in flat
    assert f"tn.{cst.DECODED_HF_REPO} IS NULL" in flat
    assert args == [TournamentType.QUANT.value, TaskType.QUANTFREEFORMTASK.value, NETUID]


async def test_empty_results_are_empty_lists():
    psql, _ = _mock_psql()

    assert await tournament_sql.get_pending_decoder_releases(EMISSION_BURN_HOTKEY, psql) == []
    assert await tournament_sql.get_blocked_decoder_releases(EMISSION_BURN_HOTKEY, psql) == []
    assert await tournament_sql.get_tournaments_with_pending_decoded_publication(psql) == []
