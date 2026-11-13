"""Decoder snapshot persistence: participant decoder columns and the task_nodes decode-result helpers (mocked DB)."""

import uuid
from datetime import datetime
from datetime import timezone
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

import validator.db.constants as cst
from core.constants.network import NETUID
from core.models.quant_artifact_models import QuantDecodedFile
from core.models.quant_models import QuantEvalGpu
from validator.db.sql import submissions_and_scoring as scoring_sql
from validator.db.sql import tournaments as tournament_sql
from validator.scoring.models import HubArtifactFile
from validator.tournament.models import TournamentParticipant


TASK_ID = uuid.uuid4()
TOURNAMENT_ID = "tourn_1"
HOTKEY = "5Miner"
SNAPSHOT_FIELDS = {
    "decoder_object_name": "quant-decoders/c/x.tar.xz",
    "decoder_xz_bytes": 2048,
    "decoder_sha256": "x" * 64,
    "decoder_tar_sha256": "t" * 64,
    "decoder_tree_sha": "a" * 40,
    "decoder_xz_version": "xz (XZ Utils) 5.4.5",
}
RELEASED_AT = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)
RELEASED_URL = "https://huggingface.co/datasets/org/freeform-releases/resolve/main/t/h/decoder.tar.xz"
EXPOSURE_COLUMNS = (cst.DECODER_TAR_SHA256, cst.DECODER_TREE_SHA, cst.DECODER_XZ_VERSION)


def _mock_psql(*, fetchrow=None, fetchval=None, fetch=None, execute="UPDATE 1"):
    conn = MagicMock()
    conn.fetchrow = AsyncMock(side_effect=fetchrow) if isinstance(fetchrow, list) else AsyncMock(return_value=fetchrow)
    conn.fetchval = AsyncMock(return_value=fetchval)
    conn.fetch = AsyncMock(return_value=fetch or [])
    conn.execute = AsyncMock(return_value=execute)
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=None)
    tx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    psql = MagicMock()
    psql.connection = AsyncMock(return_value=cm)
    return psql, conn


def _participant() -> TournamentParticipant:
    return TournamentParticipant(
        tournament_id=TOURNAMENT_ID,
        hotkey=HOTKEY,
        training_repo="https://github.com/miner/repo",
        training_commit_hash="c" * 40,
        **SNAPSHOT_FIELDS,
    )


def _participant_row(**decoder_columns) -> dict:
    return {
        cst.TOURNAMENT_ID: TOURNAMENT_ID,
        cst.HOTKEY: HOTKEY,
        cst.ELIMINATED_IN_ROUND_ID: None,
        cst.FINAL_POSITION: None,
        cst.TRAINING_REPO: "https://github.com/miner/repo",
        cst.TRAINING_COMMIT_HASH: "c" * 40,
        cst.GITHUB_TOKEN: None,
        cst.BACKUP_REPO: None,
        cst.REQUESTED_DATASETS: None,
        cst.CODE_REVIEW: None,
        cst.DECODER_OBJECT_NAME: None,
        cst.DECODER_XZ_BYTES: None,
        cst.DECODER_SHA256: None,
        cst.DECODER_TAR_SHA256: None,
        cst.DECODER_TREE_SHA: None,
        cst.DECODER_XZ_VERSION: None,
        cst.DECODER_RELEASED_URL: None,
        cst.DECODER_RELEASED_AT: None,
        **decoder_columns,
    }


def _snapshot_columns() -> dict:
    return {
        cst.DECODER_OBJECT_NAME: SNAPSHOT_FIELDS["decoder_object_name"],
        cst.DECODER_XZ_BYTES: SNAPSHOT_FIELDS["decoder_xz_bytes"],
        cst.DECODER_SHA256: SNAPSHOT_FIELDS["decoder_sha256"],
        cst.DECODER_TAR_SHA256: SNAPSHOT_FIELDS["decoder_tar_sha256"],
        cst.DECODER_TREE_SHA: SNAPSHOT_FIELDS["decoder_tree_sha"],
        cst.DECODER_XZ_VERSION: SNAPSHOT_FIELDS["decoder_xz_version"],
    }


def _bound_decoder_columns(query: str, args: tuple) -> dict:
    columns = query.split("(", 1)[1].split(")", 1)[0].replace("\n", " ").split(",")
    names = [column.strip() for column in columns]
    bound = dict(zip(names, args))
    wanted = (cst.DECODER_OBJECT_NAME, cst.DECODER_XZ_BYTES, cst.DECODER_SHA256, *EXPOSURE_COLUMNS)
    return {name: bound[name] for name in wanted}


async def test_add_tournament_participants_binds_decoder_columns():
    psql, conn = _mock_psql()
    await tournament_sql.add_tournament_participants([_participant()], psql)

    query, *args = conn.execute.await_args.args
    assert _bound_decoder_columns(query, tuple(args)) == _snapshot_columns()


async def test_enroll_insert_binds_decoder_columns():
    psql, conn = _mock_psql(fetchrow=[None, {cst.HOTKEY: HOTKEY}], fetchval=None)
    enrolled = await tournament_sql.enroll_tournament_participant_with_fee(_participant(), "5Cold", 10, psql)

    assert enrolled is True
    query, *args = conn.fetchrow.await_args_list[1].args
    assert f"INSERT INTO {cst.TOURNAMENT_PARTICIPANTS_TABLE}" in query
    assert _bound_decoder_columns(query, tuple(args)) == _snapshot_columns()


async def test_get_tournament_participant_maps_decoder_columns():
    psql, _ = _mock_psql(fetchrow=_participant_row(**_snapshot_columns()))
    participant = await tournament_sql.get_tournament_participant(TOURNAMENT_ID, HOTKEY, psql)

    assert participant.decoder_object_name == SNAPSHOT_FIELDS["decoder_object_name"]
    assert participant.decoder_xz_bytes == SNAPSHOT_FIELDS["decoder_xz_bytes"]
    assert participant.decoder_sha256 == SNAPSHOT_FIELDS["decoder_sha256"]
    assert participant.has_decoder_snapshot
    assert participant.decoder_tar_sha256 == SNAPSHOT_FIELDS["decoder_tar_sha256"]
    assert participant.decoder_tree_sha == SNAPSHOT_FIELDS["decoder_tree_sha"]
    assert participant.decoder_xz_version == SNAPSHOT_FIELDS["decoder_xz_version"]
    assert participant.decoder_released_url is None
    assert participant.decoder_released_at is None


async def test_get_tournament_participants_maps_release_columns():
    row = _participant_row(
        **_snapshot_columns(), **{cst.DECODER_RELEASED_URL: RELEASED_URL, cst.DECODER_RELEASED_AT: RELEASED_AT}
    )
    psql, conn = _mock_psql(fetch=[row])
    (participant,) = await tournament_sql.get_tournament_participants(TOURNAMENT_ID, psql)

    assert (participant.decoder_released_url, participant.decoder_released_at) == (RELEASED_URL, RELEASED_AT)
    query = conn.fetch.await_args.args[0]
    for column in (*EXPOSURE_COLUMNS, cst.DECODER_RELEASED_URL, cst.DECODER_RELEASED_AT):
        assert column in query


async def test_set_decoder_released_updates_only_the_release_columns():
    psql, conn = _mock_psql()
    await tournament_sql.set_decoder_released(TOURNAMENT_ID, HOTKEY, RELEASED_URL, RELEASED_AT, psql)

    query, *args = conn.execute.await_args.args
    set_clause = query.split("SET", 1)[1].split("WHERE", 1)[0]
    assert {a.split("=")[0].strip() for a in set_clause.split(",")} == {cst.DECODER_RELEASED_URL, cst.DECODER_RELEASED_AT}
    assert args == [RELEASED_URL, RELEASED_AT, TOURNAMENT_ID, HOTKEY]


async def test_set_decoded_hf_repo_updates_the_task_node():
    psql, conn = _mock_psql()
    await scoring_sql.set_decoded_hf_repo(TASK_ID, HOTKEY, "org/name-decoded", psql)

    query, *args = conn.execute.await_args.args
    assert query.strip().startswith(f"UPDATE {cst.TASK_NODES_TABLE}")
    assert cst.DECODED_HF_REPO in query
    assert args == [TASK_ID, HOTKEY, NETUID, "org/name-decoded"]


async def test_set_quant_decode_result_binds_file_lists_as_json_and_empty_as_null():
    psql, conn = _mock_psql()
    files = [QuantDecodedFile(name="a.safetensors", bytes=5, sha256="s" * 64)]
    await scoring_sql.set_quant_decode_result(
        TASK_ID,
        HOTKEY,
        artifact_bytes=1,
        decoder_bytes=2,
        decoder_sha256="s",
        quant_failure=None,
        psql_db=psql,
        artifact_files=files,
        decoded_files=[],
    )

    query, *args = conn.execute.await_args.args
    assert f"{cst.DECODE_ARTIFACT_FILES} = $9::jsonb" in query
    assert f"{cst.DECODED_FILES} = $10::jsonb" in query
    assert args[-2:] == [[{"name": "a.safetensors", "bytes": 5, "sha256": "s" * 64}], None]


async def test_set_task_node_losses_persists_hub_listing_without_overwriting_with_null():
    psql, conn = _mock_psql()
    files = [HubArtifactFile(name="m.bin", bytes=7, sha256="c" * 64, sha256_kind="sha256")]
    await scoring_sql.set_task_node_losses(TASK_ID, HOTKEY, 1.5, 1.5, psql, artifact_files=files, artifact_revision="rev")

    query, *args = conn.execute.await_args.args
    assert f"{cst.ARTIFACT_FILES} = COALESCE($13::jsonb" in query
    assert f"{cst.ARTIFACT_REVISION} = COALESCE($14" in query
    assert args[12:] == [[{"name": "m.bin", "bytes": 7, "sha256": "c" * 64, "sha256_kind": "sha256"}], "rev"]


async def test_get_quant_scorecard_rows_maps_columns():
    row = {
        cst.HOTKEY: HOTKEY,
        cst.TEST_LOSS: 2.0,
        cst.QUANT_FAILURE: None,
        cst.ARTIFACT_BYTES: 10,
        cst.DECODER_BYTES: 20,
        cst.PASSED_BAR: True,
        cst.FREEFORM_PENALTY: 0.1,
        cst.EVAL_GPU: {"name": "NVIDIA H100 PCIe", "sm_count": 114, "sku": "H100 PCIe", "override": False},
        cst.ARTIFACT_FILES: [{"name": "m.bin", "bytes": 10, "sha256": "c" * 64, "sha256_kind": "sha256"}],
        cst.ARTIFACT_REVISION: "rev",
        cst.DECODE_ARTIFACT_FILES: None,
        cst.DECODED_FILES: [{"name": "s.safetensors", "bytes": 3, "sha256": "e" * 64}],
        cst.DECODED_HF_REPO: None,
        cst.REPO: "org/repo",
    }
    psql, conn = _mock_psql(fetch=[row])
    rows = await scoring_sql.get_quant_scorecard_rows(TASK_ID, psql)

    assert rows[HOTKEY].artifact_files[0].name == "m.bin"
    assert rows[HOTKEY].decode_artifact_files is None
    assert rows[HOTKEY].decoded_files[0].sha256 == "e" * 64
    assert rows[HOTKEY].eval_gpu.sku == "H100 PCIe"
    assert conn.fetch.await_args.args[1:] == (TASK_ID, NETUID)


async def test_get_quant_scorecard_rows_joins_only_the_latest_submission_per_hotkey():
    psql, conn = _mock_psql(fetch=[])
    await scoring_sql.get_quant_scorecard_rows(TASK_ID, psql)

    query = " ".join(conn.fetch.await_args.args[0].split())
    # Not run on a real Postgres: the text is checked, not its semantics.
    assert (
        f"LEFT JOIN LATERAL ( SELECT sub.{cst.REPO} FROM {cst.SUBMISSIONS_TABLE} sub "
        f"WHERE sub.{cst.TASK_ID} = tn.{cst.TASK_ID} AND sub.{cst.HOTKEY} = tn.{cst.HOTKEY} "
        f"ORDER BY sub.{cst.CREATED_ON} DESC NULLS LAST, sub.{cst.SUBMISSION_ID} DESC LIMIT 1 ) s ON TRUE"
    ) in query
    assert f"LEFT JOIN {cst.SUBMISSIONS_TABLE}" not in query
    assert conn.fetch.await_args.args[1:] == (TASK_ID, NETUID)


async def test_get_tournament_participants_maps_missing_snapshot():
    psql, _ = _mock_psql(fetch=[_participant_row()])
    (participant,) = await tournament_sql.get_tournament_participants(TOURNAMENT_ID, psql)

    assert not participant.has_decoder_snapshot


async def test_set_quant_decode_result_updates_exactly_the_decode_columns():
    psql, conn = _mock_psql()
    await scoring_sql.set_quant_decode_result(
        TASK_ID,
        HOTKEY,
        artifact_bytes=10,
        decoder_bytes=20,
        decoder_sha256="s",
        quant_failure=None,
        psql_db=psql,
        build_warning="build.sh exited 0 but left /build empty",
    )

    query, *args = conn.execute.await_args.args
    set_clause = query.split("SET", 1)[1].split("WHERE", 1)[0]
    assigned = {assignment.split("=")[0].strip() for assignment in set_clause.split(",")}
    assert assigned == {
        cst.ARTIFACT_BYTES,
        cst.DECODER_BYTES,
        cst.DECODER_SHA256,
        cst.QUANT_FAILURE,
        cst.DECODER_BUILD_WARNING,
        cst.DECODE_ARTIFACT_FILES,
        cst.DECODED_FILES,
    }
    assert args == [TASK_ID, HOTKEY, NETUID, 10, 20, "s", None, "build.sh exited 0 but left /build empty", None, None]


@pytest.mark.parametrize(
    ("eval_gpu", "stored"),
    [
        (
            QuantEvalGpu(name="NVIDIA H100 PCIe", sm_count=114, sku="H100 PCIe"),
            {"name": "NVIDIA H100 PCIe", "sm_count": 114, "sku": "H100 PCIe", "override": False},
        ),
        (None, None),
    ],
)
async def test_set_task_node_losses_writes_the_eval_gpu(eval_gpu, stored):
    psql, conn = _mock_psql()
    await scoring_sql.set_task_node_losses(TASK_ID, HOTKEY, 1.5, 1.5, psql, eval_gpu=eval_gpu)

    query, *args = conn.execute.await_args.args
    assert f"{cst.EVAL_GPU} = $12::jsonb" in query
    assert args[11] == stored


async def test_set_quant_decode_result_tolerates_missing_row():
    psql, conn = _mock_psql(execute="UPDATE 0")
    await scoring_sql.set_quant_decode_result(
        TASK_ID, HOTKEY, artifact_bytes=None, decoder_bytes=None, decoder_sha256=None, quant_failure="x", psql_db=psql
    )

    conn.execute.assert_awaited_once()


def _task_node_row(hotkey: str, *, decoder_sha256=None, quant_failure=None, artifact_bytes=None) -> dict:
    return {
        cst.HOTKEY: hotkey,
        cst.ARTIFACT_BYTES: artifact_bytes,
        cst.DECODER_BYTES: None,
        cst.DECODER_SHA256: decoder_sha256,
        cst.QUANT_FAILURE: quant_failure,
    }


async def test_get_quant_decode_rows_maps_settled():
    rows = [
        _task_node_row("ok", decoder_sha256="s", artifact_bytes=10),
        _task_node_row("fail", quant_failure="bad decoder output"),
        # Evaluation writes artifact_bytes but never decoder_sha256: still unsettled.
        _task_node_row("untouched", artifact_bytes=10),
    ]
    psql, conn = _mock_psql(fetch=rows)
    decode_rows = await scoring_sql.get_quant_decode_rows(TASK_ID, psql)

    assert {hotkey: row.settled for hotkey, row in decode_rows.items()} == {"ok": True, "fail": True, "untouched": False}
    assert decode_rows["ok"].artifact_bytes == 10
    assert conn.fetch.await_args.args[1:] == (TASK_ID, NETUID)
