"""Registration rejects a decoder the trainer's freeform decode would scored-fail on every task: no decode.py, or a tar
that extracts past the trainer's cap. Needs git, tar and xz on PATH, like test_decoder_snapshot.py."""

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fiber.chain.models import Node

from core.constants.quant import DECODER_ENTRYPOINT
from core.constants.quant import DECODER_MAX_EXTRACTED_BYTES
from core.models.payload_models import TrainingRepoResponse
from validator.tournament import decoder_snapshot
from validator.tournament import tournament_manager as tm
from validator.tournament.decoder_snapshot import DecoderNotRunnable
from validator.tournament.decoder_snapshot import snapshot_decoder
from validator.tournament.models import RespondingNode
from validator.tournament.models import TournamentParticipant


_MISSING = [binary for binary in ("git", "tar", "xz") if shutil.which(binary) is None]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=f"decoder snapshot tests need {', '.join(_MISSING)} on PATH")

COMMIT = "c" * 40

_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def _make_repo(path: Path, files: dict[str, bytes]) -> str:
    path.mkdir(parents=True)
    for name, content in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_bytes(content)
    for args in (["init", "--quiet", "--initial-branch=main"], ["add", "--all"], ["commit", "--quiet", "-m", "init"]):
        subprocess.run(["git", "-C", str(path), *args], env=_GIT_ENV, check=True, capture_output=True)
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], env=_GIT_ENV, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def uploads(monkeypatch) -> list[str]:
    calls: list[str] = []

    async def fake_upload(bucket, object_name, file_path):
        calls.append(object_name)
        return True

    monkeypatch.setattr(decoder_snapshot.async_minio_client, "upload_file", fake_upload)
    return calls


@pytest.mark.parametrize(
    "files",
    [
        {"decoder/main.py": b"print(1)\n"},  # no entrypoint at all
        {"decoder/sub/decode.py": b"print(1)\n"},  # nested: the trainer runs decoder/decode.py only
        {"decoder/decode.py/x.py": b"print(1)\n"},  # a directory, not a file
    ],
)
async def test_decoder_without_entrypoint_is_rejected_before_upload(tmp_path, uploads, files):
    origin = tmp_path / "origin"
    commit = _make_repo(origin, files)

    with pytest.raises(DecoderNotRunnable, match=DECODER_ENTRYPOINT):
        await snapshot_decoder(str(origin), commit, None, "bucket")
    assert uploads == []


async def test_decoder_extracting_past_the_trainer_cap_is_rejected_before_upload(tmp_path, uploads):
    # Zeros compress to a few KiB of xz, far under the 1 MiB xz cap, but extract past the trainer's 64 MiB cap.
    origin = tmp_path / "origin"
    commit = _make_repo(origin, {"decoder/decode.py": b"print(1)\n", "decoder/payload.bin": bytes(DECODER_MAX_EXTRACTED_BYTES)})

    with pytest.raises(DecoderNotRunnable, match="extracts to"):
        await snapshot_decoder(str(origin), commit, None, "bucket")
    assert uploads == []


async def test_decoder_at_the_cap_with_entrypoint_is_accepted(tmp_path, uploads):
    entrypoint = b"print(1)\n"
    origin = tmp_path / "origin"
    commit = _make_repo(
        origin,
        {"decoder/decode.py": entrypoint, "decoder/payload.bin": bytes(DECODER_MAX_EXTRACTED_BYTES - len(entrypoint))},
    )

    snapshot = await snapshot_decoder(str(origin), commit, None, "bucket")

    assert uploads == [snapshot.object_name]


def _responding_node() -> RespondingNode:
    node = Node(
        hotkey="5Miner1",
        coldkey="5Cold1",
        node_id=1,
        incentive=0.0,
        netuid=56,
        alpha_stake=0.0,
        tao_stake=0.0,
        stake=0.0,
        vtrust=0.0,
        last_updated=0.0,
        ip="10.0.0.1",
        ip_type=4,
        port=8091,
    )
    repo = TrainingRepoResponse(github_repo="https://github.com/miner/repo", commit_hash=COMMIT)
    return RespondingNode(node=node, training_repo_response=repo)


async def test_registration_excludes_an_unrunnable_decoder_with_a_warning(monkeypatch):
    monkeypatch.setattr(tm, "snapshot_decoder", AsyncMock(side_effect=DecoderNotRunnable("decoder/ has no decode.py")))
    warnings, errors = MagicMock(), MagicMock()
    monkeypatch.setattr(tm.logger, "warning", warnings)
    monkeypatch.setattr(tm.logger, "error", errors)

    assert await tm._snapshot_participant_decoder(_responding_node()) is False

    warnings.assert_called_once()
    assert "no decode.py" in warnings.call_args.args[0] and COMMIT in warnings.call_args.args[0]
    errors.assert_not_called()  # an expected rejection, not the "unexpected snapshot error" path


async def test_boss_with_an_unrunnable_decoder_keeps_empty_snapshot_fields(monkeypatch):
    monkeypatch.setattr(tm, "snapshot_decoder", AsyncMock(side_effect=DecoderNotRunnable("decoder/ has no decode.py")))
    boss = TournamentParticipant(
        tournament_id="t", hotkey="boss", training_repo="https://github.com/boss/repo", training_commit_hash=COMMIT
    )

    assert await tm._with_boss_decoder_snapshot(boss) == boss
