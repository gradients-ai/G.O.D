"""Freeform scorecard: the pure builder and the per-task assembly.

Covers scored, scored fail, unscored, without file data, completed or not.
"""

import math
import uuid
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.constants.environments import TrainingStartPoint
from core.models.quant_artifact_models import QuantDecodedFile
from core.models.quant_models import QuantEvalGpu
from core.models.task_models import TaskType
from validator.scoring import scorecard as scorecard_module
from validator.scoring.models import HubArtifactFile
from validator.scoring.models import QuantScorecardRow
from validator.scoring.models import QuantSkuReference
from validator.scoring.scorecard import build_quant_scorecard
from validator.scoring.scorecard import decode_matches_hub
from validator.scoring.scorecard import quant_scorecards_for_task
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType


TASK_ID = uuid.uuid4()
HOTKEY = "5Miner"
PCIE = QuantEvalGpu(name="NVIDIA H100 PCIe", sm_count=114, sku="H100 PCIe")
RELEASED_AT = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc) + timedelta(hours=1)

HUB_FILES = [
    HubArtifactFile(name=".gitattributes", bytes=1519, sha256="b" * 40, sha256_kind="git-blob-sha1"),
    HubArtifactFile(name="model.bin", bytes=4_000_000, sha256="c" * 64, sha256_kind="sha256"),
]
DECODE_INPUTS = [
    QuantDecodedFile(name=".gitattributes", bytes=1519, sha256="d" * 64),
    QuantDecodedFile(name="model.bin", bytes=4_000_000, sha256="c" * 64),
]
DECODED = [QuantDecodedFile(name="model-00001.safetensors", bytes=8_000_000, sha256="e" * 64)]


def _row(**overrides) -> QuantScorecardRow:
    fields = {
        "hotkey": HOTKEY,
        "test_loss": 2.04,
        "artifact_bytes": 4_001_519,
        "decoder_bytes": 2048,
        "passed_bar": True,
        "freeform_penalty": math.log(1.5),
        "eval_gpu": PCIE,
        "artifact_files": HUB_FILES,
        "artifact_revision": "r" * 40,
        "decode_artifact_files": DECODE_INPUTS,
        "decoded_files": DECODED,
        "repo": "gradients-io-tournaments/expected-name",
    }
    return QuantScorecardRow(**{**fields, **overrides})


def _participant(**overrides) -> TournamentParticipant:
    fields = {
        "tournament_id": "t1",
        "hotkey": HOTKEY,
        "training_repo": "https://github.com/private/repo",
        "training_commit_hash": "c" * 40,
        "github_token": "ghp_secret",
        "backup_repo": "https://github.com/public/backup",
        "decoder_object_name": "quant-decoders/c/x.tar.xz",
        "decoder_xz_bytes": 2048,
        "decoder_sha256": "x" * 64,
        "decoder_tar_sha256": "t" * 64,
        "decoder_tree_sha": "a" * 40,
        "decoder_xz_version": "xz (XZ Utils) 5.4.5",
    }
    return TournamentParticipant(**{**fields, **overrides})


class TestBuild:
    def test_scored_row(self) -> None:
        card = build_quant_scorecard(_row(), _participant(), reference_loss=2.0)

        assert card.bytes.total_bytes == 4_003_567
        assert card.bytes.effective_bytes == pytest.approx(4_003_567 * 1.5)
        assert card.bytes.ln_penalty == pytest.approx(math.log(1.5))
        assert card.quality.ratio == pytest.approx(1.02)
        assert (card.quality.reference_loss, card.quality.passed_bar, card.quality.quant_failure) == (2.0, True, None)
        assert card.artifact.revision == "r" * 40
        assert card.artifact.files == HUB_FILES
        assert card.artifact.decode_matches_hub is True
        assert card.decoder.model_dump() == {
            "xz_bytes": 2048,
            "xz_sha256": "x" * 64,
            "tar_sha256": "t" * 64,
            "tree_sha": "a" * 40,
            "xz_version": "xz (XZ Utils) 5.4.5",
        }
        assert card.decoded.files == DECODED
        assert card.decoded.urls is None
        assert card.release is None

    def test_scored_fail_has_no_ratio_but_keeps_bytes(self) -> None:
        row = _row(test_loss=None, quant_failure="decoder output does not match the declared shapes", passed_bar=None)
        card = build_quant_scorecard(row, _participant(), reference_loss=2.0)

        assert card.quality.quant_failure == "decoder output does not match the declared shapes"
        assert card.quality.ratio is None
        assert card.quality.test_loss is None
        assert card.bytes.artifact_bytes == 4_001_519

    def test_over_cap_row_has_no_penalty_and_plain_effective_bytes(self) -> None:
        card = build_quant_scorecard(_row(passed_bar=False, freeform_penalty=None), _participant(), reference_loss=1.0)

        assert card.bytes.ln_penalty is None
        assert card.bytes.effective_bytes == pytest.approx(4_003_567)
        assert card.quality.passed_bar is False

    def test_row_without_file_data_has_empty_file_lists_and_unknown_match(self) -> None:
        bare = _row(artifact_files=None, artifact_revision=None, decode_artifact_files=None, decoded_files=None)
        card = build_quant_scorecard(bare, None, reference_loss=None)

        assert card.artifact.files == []
        assert card.artifact.revision is None
        assert card.artifact.decode_matches_hub is None
        assert card.decoded.files == []
        assert card.decoder.xz_sha256 is None
        assert card.quality.reference_loss is None
        assert card.quality.ratio is None

    def test_no_bytes_gives_no_total(self) -> None:
        card = build_quant_scorecard(_row(artifact_bytes=None), _participant(), reference_loss=2.0)

        assert card.bytes.total_bytes is None
        assert card.bytes.effective_bytes is None

    def test_urls_are_passed_through(self) -> None:
        card = build_quant_scorecard(_row(), _participant(), 2.0, decoded_urls={"model-00001.safetensors": "https://s3/x"})

        assert card.decoded.urls == {"model-00001.safetensors": "https://s3/x"}

    def test_release_only_when_released_and_uses_the_public_repo(self) -> None:
        released = _participant(
            decoder_released_url="https://huggingface.co/datasets/org/freeform-releases/x", decoder_released_at=RELEASED_AT
        )
        card = build_quant_scorecard(_row(), released, 2.0)

        assert card.release is not None
        assert card.release.released_at == RELEASED_AT
        assert card.release.training_repo == "https://github.com/public/backup"
        assert card.release.training_commit_hash == "c" * 40
        assert card.release.decoder_snapshot_url == "https://huggingface.co/datasets/org/freeform-releases/x"

    def test_hf_repo_of_boss_round_copy(self) -> None:
        card = build_quant_scorecard(_row(decoded_hf_repo="org/expected-name-decoded"), _participant(), 2.0)

        assert card.decoded.hf_repo == "org/expected-name-decoded"


class TestDecodeMatchesHub:
    def test_unknown_when_either_side_is_empty(self) -> None:
        assert decode_matches_hub(None, DECODE_INPUTS) is None
        assert decode_matches_hub(HUB_FILES, []) is None

    def test_plain_file_compares_name_and_bytes_only(self) -> None:
        # .gitattributes is a plain git file: its hub id is a blob sha1, the trainer's is a content sha256.
        assert decode_matches_hub(HUB_FILES, DECODE_INPUTS) is True

    def test_lfs_sha_mismatch(self) -> None:
        tampered = [DECODE_INPUTS[0], DECODE_INPUTS[1].model_copy(update={"sha256": "f" * 64})]

        assert decode_matches_hub(HUB_FILES, tampered) is False

    def test_size_mismatch_missing_and_extra_files(self) -> None:
        resized = [DECODE_INPUTS[0].model_copy(update={"bytes": 1}), DECODE_INPUTS[1]]
        extra = [*DECODE_INPUTS, QuantDecodedFile(name="x", bytes=1, sha256="1" * 64)]

        assert decode_matches_hub(HUB_FILES, resized) is False
        assert decode_matches_hub(HUB_FILES, DECODE_INPUTS[:1]) is False
        assert decode_matches_hub(HUB_FILES, extra) is False


def _reference(loss: float, hotkey: str = "") -> QuantSkuReference:
    return QuantSkuReference(
        task_id=TASK_ID,
        hotkey=hotkey,
        sku=PCIE.reference_key,
        model="org/base",
        loss=loss,
        per_example_losses=[loss],
        eval_set_fingerprint="fp",
        eval_gpu=PCIE,
    )


@pytest.fixture
def assembly(monkeypatch):
    """Everything quant_scorecards_for_task reads; mutate .rows, .tournament_status and .participants per test."""
    state = SimpleNamespace(
        rows={HOTKEY: _row()},
        participants=[_participant()],
        tournament_id="t1",
        tournament_status=TournamentStatus.COMPLETED,
        references={("", PCIE.reference_key): _reference(2.0)},
        signed=[],
    )

    async def sign(bucket, object_name, expires=604800):
        state.signed.append(object_name)
        return f"https://signed/{object_name}"

    monkeypatch.setattr(scorecard_module.scoring_sql, "get_quant_scorecard_rows", AsyncMock(side_effect=lambda *_: state.rows))
    monkeypatch.setattr(scorecard_module, "get_quant_sku_references", AsyncMock(side_effect=lambda *_: state.references))
    monkeypatch.setattr(
        scorecard_module.tournament_sql, "get_tournament_id_by_task_id", AsyncMock(side_effect=lambda *_: state.tournament_id)
    )
    monkeypatch.setattr(
        scorecard_module.tournament_sql, "get_tournament_participants", AsyncMock(side_effect=lambda *_: state.participants)
    )
    monkeypatch.setattr(
        scorecard_module.tournament_sql,
        "get_tournament",
        AsyncMock(
            side_effect=lambda *_: TournamentData(
                tournament_id="t1", tournament_type=TournamentType.TEXT, status=state.tournament_status
            )
        ),
    )
    monkeypatch.setattr(scorecard_module.async_minio_client, "get_presigned_url", sign)
    return state


def _task(task_type=TaskType.QUANTFREEFORMTASK, start=TrainingStartPoint.DEFAULT):
    return SimpleNamespace(task_id=TASK_ID, task_type=task_type, training_start_point=start)


CONFIG = SimpleNamespace(psql_db="db")


class TestAssembly:
    async def test_completed_tournament_signs_decoded_shards_only(self, assembly) -> None:
        cards = await quant_scorecards_for_task(_task(), CONFIG)

        card = cards[HOTKEY]
        assert card.quality.reference_loss == 2.0
        assert card.decoded.urls == {
            "model-00001.safetensors": f"https://signed/quant-decoded/{TASK_ID}/{HOTKEY}/model-00001.safetensors"
        }
        assert assembly.signed == [f"quant-decoded/{TASK_ID}/{HOTKEY}/model-00001.safetensors"]

    async def test_active_tournament_signs_nothing(self, assembly) -> None:
        assembly.tournament_status = TournamentStatus.ACTIVE

        cards = await quant_scorecards_for_task(_task(), CONFIG)

        assert cards[HOTKEY].decoded.urls is None
        assert assembly.signed == []

    async def test_task_without_tournament_has_no_urls_and_no_decoder(self, assembly) -> None:
        assembly.tournament_id = None

        card = (await quant_scorecards_for_task(_task(), CONFIG))[HOTKEY]

        assert card.decoded.urls is None
        assert card.decoder.xz_sha256 is None

    async def test_unscored_row_gets_no_scorecard(self, assembly) -> None:
        assembly.rows = {HOTKEY: _row(test_loss=None, passed_bar=None, freeform_penalty=None), "5Scored": _row(hotkey="5Scored")}

        assert set(await quant_scorecards_for_task(_task(), CONFIG)) == {"5Scored"}

    async def test_scored_fail_gets_a_scorecard(self, assembly) -> None:
        assembly.rows = {HOTKEY: _row(test_loss=None, quant_failure="bad decode")}

        card = (await quant_scorecards_for_task(_task(), CONFIG))[HOTKEY]

        assert card.quality.quant_failure == "bad decode"

    async def test_own_model_task_reads_the_miners_own_reference(self, assembly) -> None:
        assembly.references = {(HOTKEY, PCIE.reference_key): _reference(1.0, HOTKEY)}

        card = (await quant_scorecards_for_task(_task(start=TrainingStartPoint.OWN_R1_MODEL), CONFIG))[HOTKEY]

        assert card.quality.reference_loss == 1.0
        assert card.quality.ratio == pytest.approx(2.04)

    async def test_fixed_task_has_no_scorecards(self, assembly) -> None:
        assert await quant_scorecards_for_task(_task(TaskType.QUANTFIXEDTASK), CONFIG) == {}

    async def test_signing_failure_degrades_to_no_urls(self, assembly, monkeypatch) -> None:
        monkeypatch.setattr(
            scorecard_module.async_minio_client, "get_presigned_url", AsyncMock(side_effect=RuntimeError("minio down"))
        )

        card = (await quant_scorecards_for_task(_task(), CONFIG))[HOTKEY]

        assert card.decoded.urls is None
