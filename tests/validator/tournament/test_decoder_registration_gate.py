"""Registration decoder gate: a QUANT entrant must carry a snapshot-able decoder/ at its pinned commit.

The same clone runs the build lint first (registration_lint), passed to snapshot_decoder as repo_check. A symlink
in decoder/ excludes the entrant like a missing decoder; so does an enforced lint failure. The lint is warn-only
by default (QUANT_REGISTRATION_LINT_ENFORCE False): a failing entrant is enrolled and its violations logged.

The snapshot taken here is what the trainer later decodes with, so its MinIO object, counted bytes
and sha land on the participant row. Text, image and environment registration never snapshot. The boss is
snapshotted too, but a missing decoder only leaves its fields empty (it then scored-fails freeform).
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fiber.chain.models import Node

from core.models.payload_models import TrainingRepoResponse
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament import registration_lint
from validator.tournament import tournament_manager as tm
from validator.tournament.decoder_snapshot import DECODER_SNAPSHOT_CONCURRENCY
from validator.tournament.decoder_snapshot import DecoderMissing
from validator.tournament.decoder_snapshot import DecoderSnapshot
from validator.tournament.decoder_snapshot import DecoderTooLarge
from validator.tournament.decoder_snapshot import DecoderUnsafeMembers
from validator.tournament.models import RespondingNode
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType
from validator.tournament.registration_lint import LintRule
from validator.tournament.registration_lint import LintViolation
from validator.tournament.registration_lint import RegistrationLintFailed
from validator.tournament.registration_lint import RegistrationLintReport
from validator.tournament.registration_lint import check_registration_lint


TOURNAMENT_ID = "tourn_gate"
BUCKET = "test-bucket"
COMMIT = "c" * 40
# A build with a tagged base image and unhashed requirements; it fails the lint.
TAGGED_UNHASHED_REPO = Path(__file__).parent / "fixtures" / "registration_lint" / "tagged_unhashed"


_FAILED_LINT = RegistrationLintReport(
    dockerfile="dockerfiles/standalone-text-trainer.dockerfile",
    violations=[
        LintViolation(
            rule=LintRule.UNPINNED_GIT,
            file="dockerfiles/standalone-text-trainer.dockerfile",
            line=3,
            detail="fiber @ git+https://github.com/rayonlabs/fiber.git@2.4.0 is not pinned to a 40-hex commit SHA",
        )
    ],
)


def _node(index: int) -> Node:
    return Node(
        hotkey=f"5Miner{index}",
        coldkey=f"5Cold{index}",
        node_id=index,
        incentive=0.0,
        netuid=56,
        alpha_stake=0.0,
        tao_stake=0.0,
        stake=0.0,
        vtrust=0.0,
        last_updated=0.0,
        ip=f"10.0.0.{index}",
        ip_type=4,
        port=8091,
    )


def _repo_for(node: Node) -> TrainingRepoResponse:
    return TrainingRepoResponse(github_repo=f"https://github.com/{node.hotkey}/repo", commit_hash=COMMIT)


def _snapshot(repo_url: str) -> DecoderSnapshot:
    return DecoderSnapshot(
        repo_url=repo_url,
        commit_hash=COMMIT,
        tar_sha256="t" * 64,
        xz_bytes=2048,
        xz_sha256="x" * 64,
        xz_version="xz (XZ Utils) 5.4.5",
        tree_sha="a" * 40,
        object_name=f"quant-decoders/{COMMIT}/{'x' * 64}.tar.xz",
    )


@pytest.fixture
def registration(monkeypatch):
    """Patch everything populate_tournament_participants touches; snapshot_decoder is set per test."""
    nodes = [_node(1), _node(2)]
    mocks = SimpleNamespace(
        nodes=nodes,
        enroll=AsyncMock(return_value=True),
        snapshot=AsyncMock(side_effect=lambda repo_url, *_, **__: _snapshot(repo_url)),
        tournament_type=TournamentType.QUANT,
    )

    async def get_tournament(tournament_id, psql_db):
        return TournamentData(tournament_id=tournament_id, tournament_type=mocks.tournament_type, status=TournamentStatus.PENDING)

    # Empty on the first read; one persisted miner on the re-read so the loop returns instead of sleeping.
    persisted = [[], [TournamentParticipant(tournament_id=TOURNAMENT_ID, hotkey="5Persisted")]]
    monkeypatch.setattr(tm.t_cst, "MIN_MINERS_FOR_TOURN", 1)
    monkeypatch.setattr(tm, "get_tournament", get_tournament)
    monkeypatch.setattr(tm, "get_tournament_participants", AsyncMock(side_effect=persisted))
    monkeypatch.setattr(tm, "get_all_nodes", AsyncMock(return_value=nodes))
    monkeypatch.setattr(tm, "_get_miner_training_repo", AsyncMock(side_effect=lambda node, *_: _repo_for(node)))
    monkeypatch.setattr(tm, "validate_github_tokens", AsyncMock())
    monkeypatch.setattr(tm, "validate_repo_obfuscation", AsyncMock(return_value=True))
    monkeypatch.setattr(tm, "validate_repo_license", AsyncMock(return_value=True))
    monkeypatch.setattr(tm, "snapshot_decoder", mocks.snapshot)
    monkeypatch.setattr(tm, "enroll_tournament_participant_with_fee", mocks.enroll)
    monkeypatch.setattr(tm.service_cst, "BUCKET_NAME", BUCKET)
    return mocks


def _enrolled(registration) -> dict[str, TournamentParticipant]:
    participants = [call.args[0] for call in registration.enroll.await_args_list]
    return {participant.hotkey: participant for participant in participants}


async def _populate():
    return await tm.populate_tournament_participants(TOURNAMENT_ID, SimpleNamespace(), psql_db=None)


def _failing_for_first_node(error: Exception):
    def snapshot(repo_url, *_, **__):
        if "5Miner1" in repo_url:
            raise error
        return _snapshot(repo_url)

    return snapshot


class TestRegistrationGate:
    @pytest.mark.parametrize(
        "error",
        [
            DecoderMissing("No decoder/ directory at commit"),
            DecoderTooLarge(2_000_000, 1_048_576),
            RuntimeError("'git' timed out after 600s"),
            DecoderUnsafeMembers(["decoder/alias.py"]),
            RegistrationLintFailed(_FAILED_LINT),
            # Unexpected: excluded too, rather than aborting the whole registration pass.
            OSError(28, "No space left on device"),
        ],
    )
    async def test_snapshot_failure_excludes_entrant(self, registration, error):
        registration.snapshot.side_effect = _failing_for_first_node(error)

        await _populate()

        assert set(_enrolled(registration)) == {"5Miner2"}

    async def test_unexpected_error_is_logged_without_the_token(self, monkeypatch):
        token = "ghp_secret"
        node = _node(1)
        repo = TrainingRepoResponse(github_repo=node.hotkey, commit_hash=COMMIT, github_token=token)
        monkeypatch.setattr(tm, "snapshot_decoder", AsyncMock(side_effect=OSError(f"https://x:{token}@github.com: disk full")))
        logged = []
        monkeypatch.setattr(tm.logger, "error", logged.append)

        passed = await tm._snapshot_participant_decoder(RespondingNode(node=node, training_repo_response=repo))

        assert passed is False
        assert len(logged) == 1 and "disk full" in logged[0] and token not in logged[0]

    async def test_lint_rejection_is_logged_with_its_reason(self, monkeypatch):
        node = _node(1)
        repo = _repo_for(node)
        monkeypatch.setattr(tm, "snapshot_decoder", AsyncMock(side_effect=RegistrationLintFailed(_FAILED_LINT)))
        logged = []
        monkeypatch.setattr(tm.logger, "warning", logged.append)

        passed = await tm._snapshot_participant_decoder(RespondingNode(node=node, training_repo_response=repo))

        assert passed is False
        assert len(logged) == 1 and "build lint failed" in logged[0] and "unpinned_git" in logged[0]

    async def test_passing_snapshot_lands_on_participant(self, registration):
        await _populate()

        enrolled = _enrolled(registration)
        assert set(enrolled) == {"5Miner1", "5Miner2"}
        participant = enrolled["5Miner1"]
        expected = _snapshot(participant.training_repo)
        assert participant.decoder_object_name == expected.object_name
        assert participant.decoder_xz_bytes == expected.xz_bytes
        assert participant.decoder_sha256 == expected.xz_sha256
        assert participant.has_decoder_snapshot
        registration.snapshot.assert_any_await(
            participant.training_repo, COMMIT, None, BUCKET, repo_check=check_registration_lint
        )

    @pytest.mark.parametrize("tournament_type", [TournamentType.TEXT, TournamentType.IMAGE])
    async def test_non_quant_never_snapshot(self, registration, tournament_type):
        registration.tournament_type = tournament_type

        await _populate()

        registration.snapshot.assert_not_awaited()
        enrolled = _enrolled(registration)
        assert set(enrolled) == {"5Miner1", "5Miner2"}
        assert not any(participant.has_decoder_snapshot for participant in enrolled.values())

    async def test_snapshots_run_in_parallel_under_the_cap(self, monkeypatch):
        in_flight = SimpleNamespace(now=0, peak=0)

        async def snapshot(repo_url, *_, **__):
            in_flight.now += 1
            in_flight.peak = max(in_flight.peak, in_flight.now)
            await asyncio.sleep(0.01)
            in_flight.now -= 1
            return _snapshot(repo_url)

        monkeypatch.setattr(tm, "snapshot_decoder", snapshot)
        nodes = [_node(index) for index in range(1, 2 * DECODER_SNAPSHOT_CONCURRENCY + 1)]
        responding = [RespondingNode(node=node, training_repo_response=_repo_for(node)) for node in nodes]

        passed = await tm._nodes_with_decoder_snapshots(responding)

        assert passed == responding
        assert in_flight.peak == DECODER_SNAPSHOT_CONCURRENCY


def _linting_tagged_unhashed_for_first_node(repo_url, *_, repo_check=None, **__):
    """snapshot_decoder stand-in that runs the real repo_check, on the tagged-unhashed fixture for 5Miner1."""
    if repo_check is not None and "5Miner1" in repo_url:
        repo_check(TAGGED_UNHASHED_REPO)
    return _snapshot(repo_url)


class TestRegistrationLintDefault:
    def test_lint_is_warn_only_by_default(self):
        assert tm.t_cst.QUANT_REGISTRATION_LINT_ENFORCE is False

    async def test_warn_only_enrols_a_failing_entrant_and_logs_its_violations(self, registration, monkeypatch):
        warnings: list[str] = []
        monkeypatch.setattr(registration_lint.logger, "warning", warnings.append)
        registration.snapshot.side_effect = _linting_tagged_unhashed_for_first_node

        await _populate()

        enrolled = _enrolled(registration)
        assert set(enrolled) == {"5Miner1", "5Miner2"}
        assert enrolled["5Miner1"].has_decoder_snapshot
        assert len(warnings) == 1
        assert warnings[0].startswith("Build lint (warn-only, entrant kept)")
        assert "unhashed_pip" in warnings[0] and "unpinned_git" in warnings[0]

    async def test_enforced_lint_excludes_the_failing_entrant(self, registration, monkeypatch):
        monkeypatch.setattr(tm.t_cst, "QUANT_REGISTRATION_LINT_ENFORCE", True)
        excluded: list[str] = []
        monkeypatch.setattr(tm.logger, "warning", excluded.append)
        registration.snapshot.side_effect = _linting_tagged_unhashed_for_first_node

        await _populate()

        assert set(_enrolled(registration)) == {"5Miner2"}
        assert any("Excluding 5Miner1: build lint failed" in message for message in excluded)


@pytest.fixture
def basic_tournament(monkeypatch):
    boss = TournamentParticipant(
        tournament_id="", hotkey=EMISSION_BURN_HOTKEY, training_repo="https://github.com/boss/repo", training_commit_hash=COMMIT
    )
    mocks = SimpleNamespace(
        add=AsyncMock(),
        snapshot=AsyncMock(side_effect=lambda repo_url, *_, **__: _snapshot(repo_url)),
    )
    monkeypatch.setattr(tm, "get_base_contestant", AsyncMock(return_value=boss))
    monkeypatch.setattr(tm, "get_latest_tournament_winner_participant", AsyncMock(return_value=None))
    monkeypatch.setattr(tm, "create_tournament", AsyncMock())
    monkeypatch.setattr(tm, "add_tournament_participants", mocks.add)
    monkeypatch.setattr(tm, "snapshot_decoder", mocks.snapshot)
    monkeypatch.setattr(tm.service_cst, "BUCKET_NAME", BUCKET)
    return mocks


def _added_boss(basic_tournament) -> TournamentParticipant:
    return basic_tournament.add.await_args.args[0][0]


class TestBossDecoderSnapshot:
    async def test_quant_boss_is_snapshotted(self, basic_tournament):
        await tm.create_basic_tournament(TournamentType.QUANT, psql_db=None, config=SimpleNamespace())

        boss = _added_boss(basic_tournament)
        basic_tournament.snapshot.assert_awaited_once_with("https://github.com/boss/repo", COMMIT, None, BUCKET)
        assert boss.decoder_sha256 == "x" * 64
        assert boss.has_decoder_snapshot

    async def test_boss_without_decoder_is_still_added_without_snapshot(self, basic_tournament):
        basic_tournament.snapshot.side_effect = DecoderMissing("No decoder/ directory at commit")

        await tm.create_basic_tournament(TournamentType.QUANT, psql_db=None, config=SimpleNamespace())

        boss = _added_boss(basic_tournament)
        assert boss.hotkey == EMISSION_BURN_HOTKEY
        assert (boss.decoder_object_name, boss.decoder_xz_bytes, boss.decoder_sha256) == (None, None, None)

    async def test_boss_with_symlinked_decoder_is_still_added_without_snapshot(self, basic_tournament):
        basic_tournament.snapshot.side_effect = DecoderUnsafeMembers(["decoder/alias.py"])

        await tm.create_basic_tournament(TournamentType.QUANT, psql_db=None, config=SimpleNamespace())

        boss = _added_boss(basic_tournament)
        assert not boss.has_decoder_snapshot
        # The boss is the pinned champion, never re-linted.
        assert "repo_check" not in basic_tournament.snapshot.await_args.kwargs

    @pytest.mark.parametrize("tournament_type", [TournamentType.TEXT, TournamentType.IMAGE])
    async def test_text_and_image_boss_are_not_snapshotted(self, basic_tournament, tournament_type):
        await tm.create_basic_tournament(tournament_type, psql_db=None, config=SimpleNamespace())

        basic_tournament.snapshot.assert_not_awaited()
        assert not _added_boss(basic_tournament).has_decoder_snapshot
