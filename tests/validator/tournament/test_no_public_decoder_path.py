"""No public response model serialises the decoder object name, a presigned decoder URL, the PAT or the private repo."""

import json
from datetime import datetime
from datetime import timezone

from core.models.quant_artifact_models import QuantDecodedFile
from validator.endpoints.tournament_analytics import _public_participants
from validator.scoring.models import HubArtifactFile
from validator.scoring.models import QuantScorecardRow
from validator.scoring.scorecard import build_quant_scorecard
from validator.tasks.models import HotkeyDetails
from validator.tournament.models import ActiveTournamentInfo
from validator.tournament.models import ActiveTournamentParticipant
from validator.tournament.models import ActiveTournamentsResponse
from validator.tournament.models import TournamentDetailsResponse
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType


PRIVATE_REPO = "https://github.com/miner-private/secret-decoder-repo"
PUBLIC_BACKUP = "https://github.com/gradients-opensource/backup-repo"
TOKEN = "github_pat_SECRET_TOKEN_VALUE"
OBJECT_NAME = "quant-decoders/" + "c" * 40 + "/" + "x" * 64 + ".tar.xz"
FORBIDDEN_VALUES = (OBJECT_NAME, "quant-decoders/", TOKEN, PRIVATE_REPO, "X-Amz-Signature")
FORBIDDEN_SCORECARD_KEYS = ("github_token", "decoder_object_name")


def _participant(released: bool) -> TournamentParticipant:
    return TournamentParticipant(
        tournament_id="t1",
        hotkey="5Miner",
        training_repo=PRIVATE_REPO,
        training_commit_hash="c" * 40,
        github_token=TOKEN,
        backup_repo=PUBLIC_BACKUP,
        decoder_object_name=OBJECT_NAME,
        decoder_xz_bytes=2048,
        decoder_sha256="x" * 64,
        decoder_tar_sha256="t" * 64,
        decoder_tree_sha="a" * 40,
        decoder_xz_version="xz (XZ Utils) 5.4.5",
        decoder_released_url="https://huggingface.co/datasets/org/freeform-releases/resolve/main/t1/5Miner/decoder.tar.xz"
        if released
        else None,
        decoder_released_at=datetime(2026, 11, 9, tzinfo=timezone.utc) if released else None,
    )


def _scorecard_json(released: bool) -> str:
    row = QuantScorecardRow(
        hotkey="5Miner",
        test_loss=2.04,
        artifact_bytes=100,
        decoder_bytes=2048,
        passed_bar=True,
        freeform_penalty=0.1,
        artifact_files=[HubArtifactFile(name="m.bin", bytes=100, sha256="c" * 64, sha256_kind="sha256")],
        artifact_revision="r" * 40,
        decoded_files=[QuantDecodedFile(name="s.safetensors", bytes=10, sha256="e" * 64)],
        repo="org/expected",
    )
    card = build_quant_scorecard(row, _participant(released), 2.0, decoded_urls={"s.safetensors": "https://s3/decoded/s?sig=1"})
    details = HotkeyDetails(
        hotkey="5Miner", quality_score=1.0, test_loss=2.04, synth_loss=2.04, repo="org/expected", rank=1, quant_scorecard=card
    )
    return details.model_dump_json()


def _assert_clean(serialised: str, keys: tuple[str, ...] = ()) -> None:
    for needle in (*FORBIDDEN_VALUES, *keys):
        assert needle not in serialised, needle


def test_scorecard_never_carries_decoder_object_private_repo_or_token() -> None:
    for released in (False, True):
        _assert_clean(_scorecard_json(released), FORBIDDEN_SCORECARD_KEYS)


def test_released_scorecard_shows_only_the_public_copy() -> None:
    serialised = _scorecard_json(True)

    assert PUBLIC_BACKUP in serialised
    assert "freeform-releases" in serialised


def test_tournament_details_participants_are_sanitised() -> None:
    for released in (False, True):
        response = TournamentDetailsResponse(
            tournament_id="t1",
            tournament_type=TournamentType.QUANT,
            status=TournamentStatus.COMPLETED,
            base_winner_hotkey=None,
            winner_hotkey="5Miner",
            participants=_public_participants([_participant(released)]),
            rounds=[],
            final_scores=[],
            text_tournament_weight=1.0,
            image_tournament_weight=0.0,
        )
        serialised = json.dumps(json.loads(response.model_dump_json()))

        # The repo URL is public metadata (results are labelled by it); its content stays private until handover.
        _assert_clean(serialised.replace(PRIVATE_REPO, ""))
        assert "x" * 64 in serialised  # the snapshot hash is public, the object is not
        assert PRIVATE_REPO in serialised


def test_participant_hash_fields_survive_sanitising() -> None:
    (public,) = _public_participants([_participant(True)])

    assert (public.decoder_tar_sha256, public.decoder_tree_sha, public.decoder_xz_version) == (
        "t" * 64,
        "a" * 40,
        "xz (XZ Utils) 5.4.5",
    )
    assert public.decoder_released_at is not None


def test_active_tournaments_response_lists_hotkeys_only() -> None:
    info = ActiveTournamentInfo(
        tournament_id="t1",
        tournament_type=TournamentType.QUANT,
        status=TournamentStatus.ACTIVE,
        participants=[ActiveTournamentParticipant(hotkey="5Miner")],
        created_at=datetime(2026, 11, 9),
    )

    _assert_clean(ActiveTournamentsResponse(text=info, image=None, environment=None).model_dump_json())
