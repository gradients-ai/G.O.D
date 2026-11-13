"""Freeform exposure: handover release of the decoder snapshot and the boss-round decoded model publication.

HF is a fake HfApi injected through freeform_release._hf_api, MinIO a fake client, SQL helpers are patched.
"""

import asyncio
import hashlib
import json
import re
from datetime import datetime
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from core.models.quant_artifact_models import QuantDecodedFile
from core.models.task_models import TaskType
from ops.tools.quant import reconcile_freeform_publications as cli
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import HubArtifactFile
from validator.scoring.models import QuantScorecardRow
from validator.tournament import freeform_release as fr
from validator.tournament import tournament_manager as tm
from validator.tournament.models import PendingDecoderRelease
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentType


TOURNAMENT_ID = "tourn_x"
HOTKEY = "5Miner"
SNAPSHOT = b"fake xz bytes"
SNAPSHOT_SHA = hashlib.sha256(SNAPSHOT).hexdigest()
OBJECT_NAME = "quant-decoders/" + "c" * 40 + "/" + SNAPSHOT_SHA + ".tar.xz"
TASK_ID = "11111111-1111-1111-1111-111111111111"
SHARD = b"shard-bytes"


def _participant(**overrides) -> TournamentParticipant:
    fields = dict(
        tournament_id=TOURNAMENT_ID,
        hotkey=HOTKEY,
        training_repo="https://github.com/private/repo",
        training_commit_hash="c" * 40,
        backup_repo="https://github.com/gradients-opensource/backup",
        decoder_object_name=OBJECT_NAME,
        decoder_xz_bytes=len(SNAPSHOT),
        decoder_sha256=SNAPSHOT_SHA,
        decoder_tar_sha256="t" * 64,
        decoder_tree_sha="a" * 40,
        decoder_xz_version="xz 5.4.5",
    )
    fields.update(overrides)
    return TournamentParticipant(**fields)


def _row(**overrides) -> QuantScorecardRow:
    fields = dict(
        hotkey=HOTKEY,
        repo="miner/artifact",
        artifact_revision="r" * 40,
        artifact_files=[HubArtifactFile(name="m.bin", bytes=3, sha256="b" * 64, sha256_kind="sha256")],
        decoded_files=[
            QuantDecodedFile(name="model-00001.safetensors", bytes=len(SHARD), sha256=hashlib.sha256(SHARD).hexdigest())
        ],
    )
    fields.update(overrides)
    return QuantScorecardRow(**fields)


class FakeHf:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.repos: list[tuple[str, str, bool]] = []
        self.files: dict[str, bytes] = {}
        self.folders: list[tuple[str, str, list[str]]] = []

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        if self.fail:
            raise RuntimeError("hf down")
        self.repos.append((repo_id, repo_type, private))

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
        data = Path(path_or_fileobj).read_bytes() if isinstance(path_or_fileobj, str) else path_or_fileobj
        self.files[path_in_repo] = data

    def upload_folder(self, folder_path, repo_id, repo_type):
        names = sorted(p.name for p in Path(folder_path).iterdir())
        self.folders.append((repo_id, repo_type, names))


class FakeMinio:
    def __init__(self, payload: bytes = SNAPSHOT) -> None:
        self.payload = payload
        self.downloads: list[tuple[str, str]] = []

    async def download_file(self, bucket, object_name, file_path):
        self.downloads.append((bucket, object_name))
        Path(file_path).write_bytes(self.payload)

    async def get_presigned_url(self, bucket, object_name, expires=604800):
        return f"https://signed.example/{object_name}?X-Amz-Signature=secret"


@pytest.fixture
def env(monkeypatch):
    state = SimpleNamespace(
        hf=FakeHf(),
        minio=FakeMinio(),
        participant=_participant(),
        rows={HOTKEY: _row()},
        released=AsyncMock(),
        alerts=AsyncMock(),
        set_repo=AsyncMock(),
        materialised=[],
        fail_hotkeys=set(),
        tournament=TournamentData(tournament_id=TOURNAMENT_ID, tournament_type=TournamentType.QUANT, winner_hotkey=HOTKEY),
    )
    config = SimpleNamespace(psql_db="db", discord_url="https://discord.example/hook")
    state.config = config

    async def mark_released(tournament_id, hotkey, url, released_at, _db):
        state.participant = state.participant.model_copy(update={"decoder_released_url": url, "decoder_released_at": released_at})

    async def mark_decoded(task_id, hotkey, repo, _db):
        state.rows[hotkey] = state.rows[hotkey].model_copy(update={"decoded_hf_repo": repo})

    async def pending_releases(burn_hotkey, _db, *, backup: bool = True):
        p = state.participant
        winner = state.tournament.winner_hotkey
        if p.hotkey == winner and p.hotkey != burn_hotkey and p.has_decoder_snapshot and p.decoder_released_at is None:
            if bool(p.backup_repo) == backup:
                return [PendingDecoderRelease(tournament_id=p.tournament_id, hotkey=p.hotkey)]
        return []

    async def blocked_releases(burn_hotkey, _db):
        return await pending_releases(burn_hotkey, _db, backup=False)

    async def pending_decoded(_db):
        if state.tournament.winner_hotkey and any(r.decoded_files and not r.decoded_hf_repo for r in state.rows.values()):
            return [TOURNAMENT_ID]
        return []

    state.released.side_effect = mark_released
    state.set_repo.side_effect = mark_decoded
    monkeypatch.setattr(fr.tournament_sql, "get_tournament", AsyncMock(side_effect=lambda *_: state.tournament))
    monkeypatch.setattr(fr.tournament_sql, "get_pending_decoder_releases", pending_releases)
    monkeypatch.setattr(fr.tournament_sql, "get_blocked_decoder_releases", blocked_releases)
    monkeypatch.setattr(fr.tournament_sql, "get_tournaments_with_pending_decoded_publication", pending_decoded)

    monkeypatch.setattr(fr, "_hf_api", lambda: state.hf)
    monkeypatch.setattr(fr, "async_minio_client", state.minio)
    monkeypatch.setattr(fr, "notify_freeform_release_problem", state.alerts)
    monkeypatch.setattr(fr.tournament_sql, "get_tournament_participant", AsyncMock(side_effect=lambda *_: state.participant))
    monkeypatch.setattr(fr.tournament_sql, "set_decoder_released", state.released)
    monkeypatch.setattr(fr.tournament_sql, "get_tournament_task_ids_of_type", AsyncMock(return_value=[TASK_ID]))
    monkeypatch.setattr(fr.scoring_sql, "get_quant_scorecard_rows", AsyncMock(side_effect=lambda *_: state.rows))
    monkeypatch.setattr(fr.scoring_sql, "set_decoded_hf_repo", state.set_repo)
    monkeypatch.setattr(fr.task_sql, "get_task_by_id", AsyncMock(return_value=SimpleNamespace(task_id=TASK_ID)))
    monkeypatch.setattr(fr.task_sql, "get_expected_repo_name", AsyncMock(side_effect=lambda _t, hotkey, _db: f"repo-{hotkey}"))
    monkeypatch.setattr(fr, "quant_reference_model", AsyncMock(return_value="org/base-model"))

    def fake_materialise(original_model, urls, dest, hf_token):
        state.materialised.append((original_model, dict(urls)))
        if any(hotkey in next(iter(urls.values())) for hotkey in state.fail_hotkeys):
            raise RuntimeError("corrupt shard")
        dest.mkdir(parents=True)
        (dest / "config.json").write_text("{}")
        (dest / "model-00001.safetensors").write_bytes(SHARD)
        return dest

    monkeypatch.setattr(fr, "materialise_decoded_model", fake_materialise)
    return state


class TestReleaseDecoder:
    async def test_happy_path_publishes_snapshot_and_release_json(self, env) -> None:
        url = await fr.release_freeform_decoder(TOURNAMENT_ID, HOTKEY, env.config, "db")

        snapshot_path = f"{TOURNAMENT_ID}/{HOTKEY}/decoder-{SNAPSHOT_SHA}.tar.xz"
        assert url == f"https://huggingface.co/datasets/{fr._releases_repo()}/resolve/main/{snapshot_path}"
        assert env.hf.repos == [(fr._releases_repo(), "dataset", False)]
        assert env.hf.files[snapshot_path] == SNAPSHOT
        release = fr.FreeformRelease.model_validate(json.loads(env.hf.files[f"{TOURNAMENT_ID}/{HOTKEY}/release.json"]))
        assert release.tournament_id == TOURNAMENT_ID and release.hotkey == HOTKEY
        assert release.training_commit_hash == "c" * 40
        assert release.training_repo == "https://github.com/gradients-opensource/backup"
        assert release.decoder.xz_sha256 == SNAPSHOT_SHA and release.decoder.tree_sha == "a" * 40
        assert release.decoder.tar_sha256 == "t" * 64 and release.decoder.xz_version == "xz 5.4.5"
        assert release.decoder.xz_bytes == len(SNAPSHOT)
        [task] = release.tasks
        assert task.task_id == TASK_ID and task.repo == "miner/artifact" and task.revision == "r" * 40
        assert task.artifact_files[0].name == "m.bin" and task.decoded_files[0].name == "model-00001.safetensors"
        env.released.assert_awaited_once()
        args = env.released.await_args.args
        assert args[:3] == (TOURNAMENT_ID, HOTKEY, url) and isinstance(args[3], datetime) and args[4] == "db"
        assert "private/repo" not in json.dumps(json.loads(env.hf.files[f"{TOURNAMENT_ID}/{HOTKEY}/release.json"]))
        assert env.minio.downloads == [(fr.service_cst.BUCKET_NAME, OBJECT_NAME)]

    @pytest.mark.parametrize("payload", [b"tampered bytes", SNAPSHOT + b"x"])
    async def test_mismatch_refuses_and_publishes_nothing(self, env, payload) -> None:
        env.minio.payload = payload

        assert await fr.release_freeform_decoder(TOURNAMENT_ID, HOTKEY, env.config, "db") is None

        assert env.hf.repos == [] and env.hf.files == {}
        env.released.assert_not_awaited()
        env.alerts.assert_awaited_once()

    async def test_size_mismatch_alone_refuses(self, env) -> None:
        env.participant = _participant(decoder_xz_bytes=len(SNAPSHOT) + 1)

        assert await fr.release_freeform_decoder(TOURNAMENT_ID, HOTKEY, env.config, "db") is None
        assert env.hf.files == {}

    async def test_boss_is_skipped(self, env) -> None:
        assert await fr.release_freeform_decoder(TOURNAMENT_ID, EMISSION_BURN_HOTKEY, env.config, "db") is None
        assert env.minio.downloads == [] and env.hf.files == {}
        env.released.assert_not_awaited()

    async def test_boss_with_allow_boss_is_released(self, env) -> None:
        env.participant = _participant(hotkey=EMISSION_BURN_HOTKEY)
        env.tournament = env.tournament.model_copy(update={"winner_hotkey": EMISSION_BURN_HOTKEY})

        url = await fr.release_freeform_decoder(TOURNAMENT_ID, EMISSION_BURN_HOTKEY, env.config, "db", allow_boss=True)

        assert url is not None and f"{TOURNAMENT_ID}/{EMISSION_BURN_HOTKEY}/decoder-" in url
        env.released.assert_awaited_once()

    async def test_non_winner_with_a_valid_snapshot_is_refused(self, env) -> None:
        env.tournament = env.tournament.model_copy(update={"winner_hotkey": "5Champion"})

        assert await fr.release_freeform_decoder(TOURNAMENT_ID, HOTKEY, env.config, "db") is None

        assert env.minio.downloads == [] and env.hf.repos == [] and env.hf.files == {}
        env.released.assert_not_awaited()

    async def test_allow_boss_does_not_unlock_a_non_boss_non_winner(self, env) -> None:
        env.tournament = env.tournament.model_copy(update={"winner_hotkey": "5Champion"})

        assert await fr.release_freeform_decoder(TOURNAMENT_ID, HOTKEY, env.config, "db", allow_boss=True) is None
        assert env.hf.files == {}

    @pytest.mark.parametrize("winner", [None, ""])
    async def test_tournament_without_a_winner_refuses(self, env, winner) -> None:
        env.tournament = env.tournament.model_copy(update={"winner_hotkey": winner})

        assert await fr.release_freeform_decoder(TOURNAMENT_ID, HOTKEY, env.config, "db") is None
        assert env.hf.files == {}

    async def test_unknown_tournament_refuses(self, env) -> None:
        env.tournament = None

        assert await fr.release_freeform_decoder(TOURNAMENT_ID, HOTKEY, env.config, "db") is None
        assert env.hf.files == {}

    @pytest.mark.parametrize("backup_repo", [None, ""])
    async def test_missing_backup_repo_refuses(self, env, backup_repo) -> None:
        env.participant = _participant(backup_repo=backup_repo)

        assert await fr.release_freeform_decoder(TOURNAMENT_ID, HOTKEY, env.config, "db") is None

        assert env.minio.downloads == [] and env.hf.files == {}
        env.released.assert_not_awaited()

    async def test_cli_hand_release_of_a_non_winner_is_refused(self, env, capsys) -> None:
        env.tournament = env.tournament.model_copy(update={"winner_hotkey": "5Champion"})
        args = cli.build_parser().parse_args(["--release", "--tournament-id", TOURNAMENT_ID, "--hotkey", HOTKEY])

        assert await cli.run(args, env.config) == 1

        assert env.minio.downloads == [] and env.hf.files == {}
        env.released.assert_not_awaited()

    async def test_no_snapshot_is_skipped(self, env) -> None:
        env.participant = _participant(decoder_object_name=None, decoder_xz_bytes=None, decoder_sha256=None)

        assert await fr.release_freeform_decoder(TOURNAMENT_ID, HOTKEY, env.config, "db") is None
        assert env.minio.downloads == [] and env.hf.files == {}

    async def test_already_released_is_skipped(self, env) -> None:
        env.participant = _participant(
            decoder_released_url="https://huggingface.co/datasets/o/r/resolve/main/x.tar.xz",
            decoder_released_at=datetime(2026, 11, 9, tzinfo=timezone.utc),
        )

        url = await fr.release_freeform_decoder(TOURNAMENT_ID, HOTKEY, env.config, "db")

        assert url == "https://huggingface.co/datasets/o/r/resolve/main/x.tar.xz"
        assert env.minio.downloads == [] and env.hf.files == {}
        env.released.assert_not_awaited()

    async def test_hf_failure_returns_none_and_alerts(self, env) -> None:
        env.hf.fail = True

        assert await fr.release_freeform_decoder(TOURNAMENT_ID, HOTKEY, env.config, "db") is None

        env.released.assert_not_awaited()
        env.alerts.assert_awaited_once()

    async def test_minio_failure_returns_none(self, env, monkeypatch) -> None:
        monkeypatch.setattr(env.minio, "download_file", AsyncMock(side_effect=RuntimeError("minio down")))

        assert await fr.release_freeform_decoder(TOURNAMENT_ID, HOTKEY, env.config, "db") is None


class TestHooks:
    @pytest.fixture
    def hooks(self, monkeypatch):
        state = SimpleNamespace(schedule=MagicMock(), backup=AsyncMock(return_value="https://github.com/o/backup"))
        monkeypatch.setattr(tm, "schedule_freeform_release", state.schedule)
        monkeypatch.setattr(tm, "upload_tournament_participant_repository", state.backup)
        monkeypatch.setattr(tm, "get_tournament_participant", AsyncMock(return_value=_participant()))
        monkeypatch.setattr(tm, "update_tournament_participant_backup_repo", AsyncMock())
        return state

    async def test_release_is_scheduled_after_a_successful_push(self, hooks) -> None:
        config = SimpleNamespace()
        result = await tm.upload_participant_repository(TOURNAMENT_ID, TournamentType.QUANT, HOTKEY, 1, config, "db")

        assert result == "https://github.com/o/backup"
        hooks.schedule.assert_called_once_with(TOURNAMENT_ID, HOTKEY, config, "db")

    async def test_no_release_when_the_push_fails(self, hooks) -> None:
        hooks.backup.return_value = None

        await tm.upload_participant_repository(TOURNAMENT_ID, TournamentType.QUANT, HOTKEY, 1, SimpleNamespace(), "db")

        hooks.schedule.assert_not_called()

    @pytest.mark.parametrize("tournament_type", [TournamentType.TEXT, TournamentType.IMAGE, TournamentType.ENVIRONMENT])
    async def test_no_release_for_non_quant(self, hooks, tournament_type) -> None:
        await tm.upload_participant_repository(TOURNAMENT_ID, tournament_type, HOTKEY, 1, SimpleNamespace(), "db")

        hooks.schedule.assert_not_called()

    async def test_a_scheduling_failure_never_undoes_the_push(self, hooks) -> None:
        hooks.schedule.side_effect = RuntimeError("boom")

        result = await tm.upload_participant_repository(TOURNAMENT_ID, TournamentType.QUANT, HOTKEY, 1, SimpleNamespace(), "db")

        assert result == "https://github.com/o/backup"

    async def test_boss_round_publication_is_quant_only_and_never_raises(self, monkeypatch) -> None:
        schedule = MagicMock(side_effect=RuntimeError("boom"))
        monkeypatch.setattr(tm, "schedule_boss_round_publication", schedule)
        config = SimpleNamespace()

        await tm._publish_boss_round_decoded(
            SimpleNamespace(tournament_id=TOURNAMENT_ID, tournament_type=TournamentType.QUANT), config, "db"
        )
        schedule.assert_called_once_with(TOURNAMENT_ID, config, "db")

        schedule.reset_mock()
        await tm._publish_boss_round_decoded(
            SimpleNamespace(tournament_id=TOURNAMENT_ID, tournament_type=TournamentType.IMAGE), config, "db"
        )
        schedule.assert_not_called()

    def test_release_is_reachable_only_from_the_known_callers(self) -> None:
        validator_root = Path(fr.__file__).resolve().parents[1]
        repo_root = validator_root.parent
        sources = [*validator_root.rglob("*.py"), *(repo_root / "ops").rglob("*.py")]

        def callers(name: str) -> list[tuple[str, str]]:
            found = []
            for path in sources:
                if path.name == "freeform_release.py":
                    continue
                text = path.read_text()
                for match in re.finditer(rf"(?<!def ){name}\(", text):
                    function = re.findall(r"^(?:async )?def (\w+)", text[: match.start()], re.MULTILINE)[-1]
                    found.append((path.name, function))
            return found

        assert callers("schedule_freeform_release") == [("tournament_manager.py", "upload_participant_repository")]
        assert callers("release_freeform_decoder") == [("reconcile_freeform_publications.py", "run")]


class TestBackgroundExecution:
    @pytest.fixture
    def hooks(self, monkeypatch):
        state = SimpleNamespace(backup=AsyncMock(return_value="https://github.com/o/backup"))
        monkeypatch.setattr(tm, "upload_tournament_participant_repository", state.backup)
        monkeypatch.setattr(tm, "get_tournament_participant", AsyncMock(return_value=_participant()))
        monkeypatch.setattr(tm, "update_tournament_participant_backup_repo", AsyncMock())
        return state

    async def test_upload_returns_before_a_slow_publish_finishes(self, hooks, monkeypatch) -> None:
        gate, started, finished = asyncio.Event(), asyncio.Event(), []

        async def slow_release(*args, **kwargs):
            started.set()
            await gate.wait()
            finished.append(args)
            return "url"

        monkeypatch.setattr(fr, "release_freeform_decoder", slow_release)

        result = await asyncio.wait_for(
            tm.upload_participant_repository(TOURNAMENT_ID, TournamentType.QUANT, HOTKEY, 1, SimpleNamespace(), "db"), 5
        )

        assert result == "https://github.com/o/backup"
        await asyncio.wait_for(started.wait(), 5)
        assert finished == []
        gate.set()
        await fr.wait_for_background_publications()
        assert len(finished) == 1

    async def test_hanging_publication_does_not_block_the_next_cycle(self, monkeypatch) -> None:
        gate, started = asyncio.Event(), asyncio.Event()

        async def hanging_publish(tournament_id, config, psql_db):
            started.set()
            await gate.wait()
            return []

        monkeypatch.setattr(fr, "publish_boss_round_decoded_models", hanging_publish)
        monkeypatch.setattr(tm, "update_tournament_winner_hotkey", AsyncMock())
        monkeypatch.setattr(tm, "upload_participant_repository", AsyncMock())
        monkeypatch.setattr(tm, "notify_tournament_completed", AsyncMock())
        config = SimpleNamespace(discord_url="x")
        first = TournamentData(tournament_id="tourn_a", tournament_type=TournamentType.QUANT)
        second = TournamentData(tournament_id="tourn_b", tournament_type=TournamentType.QUANT)

        for tournament in (first, second):  # the serial active-tournament loop: the next tournament must be reached
            await asyncio.wait_for(
                tm._complete_with_defending_champion(tournament, SimpleNamespace(), config, "db", reason="defended"), 5
            )

        await asyncio.wait_for(started.wait(), 5)
        gate.set()
        await fr.wait_for_background_publications()

    async def test_task_failures_are_logged_by_class_only_and_dropped(self, monkeypatch, caplog) -> None:
        async def boom(*_args, **_kwargs):
            raise RuntimeError("token hf_secret in https://signed.example/x?X-Amz-Signature=secret")

        monkeypatch.setattr(fr, "publish_boss_round_decoded_models", boom)

        fr.schedule_boss_round_publication(TOURNAMENT_ID, SimpleNamespace(), "db")
        await fr.wait_for_background_publications()

        assert "RuntimeError" in caplog.text and "hf_secret" not in caplog.text and "X-Amz" not in caplog.text
        assert fr._background_tasks == set()

    def test_publication_lock_survives_separate_event_loops(self) -> None:
        async def contend() -> list[int]:
            order: list[int] = []

            async def worker(n: int) -> None:
                async with fr._publication_lock():
                    order.append(n)
                    await asyncio.sleep(0.01)

            await asyncio.gather(worker(1), worker(2))
            return order

        assert asyncio.run(contend()) == [1, 2]
        assert asyncio.run(contend()) == [1, 2]


class TestReconcile:
    async def test_hf_outage_then_reconcile_recovers_both_publications(self, env) -> None:
        env.hf.fail = True
        fr.schedule_freeform_release(TOURNAMENT_ID, HOTKEY, env.config, "db")
        fr.schedule_boss_round_publication(TOURNAMENT_ID, env.config, "db")
        await fr.wait_for_background_publications()
        assert env.hf.files == {} and env.hf.folders == []
        assert env.participant.decoder_released_at is None and env.rows[HOTKEY].decoded_hf_repo is None

        env.hf.fail = False
        report = await fr.reconcile_freeform_publications(env.config, "db")

        assert report.pending_releases == [PendingDecoderRelease(tournament_id=TOURNAMENT_ID, hotkey=HOTKEY)]
        assert report.pending_decoded_tournaments == [TOURNAMENT_ID]
        assert [r.hotkey for r in report.released] == [HOTKEY]
        assert report.published_models == [f"{fr.RAYONLABS_HF_USERNAME}/repo-{HOTKEY}-decoded"]
        assert env.participant.decoder_released_at is not None
        assert env.rows[HOTKEY].decoded_hf_repo is not None
        assert len(env.hf.files) == 2 and len(env.hf.folders) == 1

        again = await fr.reconcile_freeform_publications(env.config, "db")
        assert again.pending_releases == [] and again.pending_decoded_tournaments == [] and again.published_models == []
        assert len(env.hf.files) == 2 and len(env.hf.folders) == 1

    async def test_apply_false_only_reports(self, env) -> None:
        report = await fr.reconcile_freeform_publications(env.config, "db", apply=False)

        assert [r.hotkey for r in report.pending_releases] == [HOTKEY]
        assert report.pending_decoded_tournaments == [TOURNAMENT_ID]
        assert report.released == [] and report.published_models == [] and report.applied is False
        assert env.minio.downloads == [] and env.hf.repos == [] and env.hf.files == {} and env.hf.folders == []
        env.released.assert_not_awaited()
        env.set_repo.assert_not_awaited()

    async def test_blocked_champion_is_reported_and_not_released(self, env, caplog) -> None:
        env.participant = _participant(backup_repo=None)

        report = await fr.reconcile_freeform_publications(env.config, "db")

        assert report.pending_releases == []
        assert [r.hotkey for r in report.blocked_releases] == [HOTKEY]
        assert report.released == [] and env.hf.files == {}
        assert "no backup_repo" in caplog.text

    async def test_reconcile_during_an_in_flight_background_task_does_not_publish_twice(self, env) -> None:
        gate, started = asyncio.Event(), asyncio.Event()
        real_download = env.minio.download_file

        async def gated_download(bucket, object_name, file_path):
            started.set()
            await gate.wait()
            await real_download(bucket, object_name, file_path)

        env.minio.download_file = gated_download
        env.rows = {}
        fr.schedule_freeform_release(TOURNAMENT_ID, HOTKEY, env.config, "db")
        await asyncio.wait_for(started.wait(), 5)

        reconcile = asyncio.create_task(fr.reconcile_freeform_publications(env.config, "db"))
        await asyncio.sleep(0.05)
        assert not reconcile.done()  # waiting on the lock, not publishing alongside
        gate.set()
        report = await asyncio.wait_for(reconcile, 5)
        await fr.wait_for_background_publications()

        assert [r.hotkey for r in report.pending_releases] == [HOTKEY]  # it read the state before the task finished
        assert len(env.minio.downloads) == 1
        env.released.assert_awaited_once()
        assert len([n for n in env.hf.files if n.endswith(".tar.xz")]) == 1


class TestBossRoundDecodedModels:
    async def test_publishes_decoded_model_from_verified_shards(self, env) -> None:
        published = await fr.publish_boss_round_decoded_models(TOURNAMENT_ID, env.config, "db")

        repo = f"{fr.RAYONLABS_HF_USERNAME}/repo-{HOTKEY}-decoded"
        assert published == [repo]
        assert env.hf.repos == [(repo, "model", False)]
        assert env.hf.folders == [(repo, "model", ["config.json", "model-00001.safetensors"])]
        original_model, urls = env.materialised[0]
        assert original_model == "org/base-model"
        assert set(urls) == {"index.json", "model-00001.safetensors"}
        assert urls["index.json"].startswith(f"https://signed.example/quant-decoded/{TASK_ID}/{HOTKEY}/index.json")
        env.set_repo.assert_awaited_once_with(TASK_ID, HOTKEY, repo, "db")
        fr.tournament_sql.get_tournament_task_ids_of_type.assert_awaited_once_with(
            TOURNAMENT_ID, TaskType.QUANTFREEFORMTASK, "db", final_round_only=True
        )

    async def test_skips_already_published_and_undecoded(self, env) -> None:
        env.rows = {
            HOTKEY: _row(decoded_hf_repo="org/already-decoded"),
            "5NoDecode": _row(hotkey="5NoDecode", decoded_files=None),
        }

        assert await fr.publish_boss_round_decoded_models(TOURNAMENT_ID, env.config, "db") == []
        assert env.materialised == [] and env.hf.folders == []
        env.set_repo.assert_not_awaited()

    async def test_one_failing_submission_does_not_stop_the_next(self, env) -> None:
        env.rows = {"5Bad": _row(hotkey="5Bad"), "5Good": _row(hotkey="5Good")}
        env.fail_hotkeys = {"5Bad"}

        published = await fr.publish_boss_round_decoded_models(TOURNAMENT_ID, env.config, "db")

        assert published == [f"{fr.RAYONLABS_HF_USERNAME}/repo-5Good-decoded"]
        env.set_repo.assert_awaited_once()
        env.alerts.assert_awaited_once()

    async def test_listing_failure_never_raises(self, env, monkeypatch) -> None:
        monkeypatch.setattr(fr.tournament_sql, "get_tournament_task_ids_of_type", AsyncMock(side_effect=RuntimeError("db")))

        assert await fr.publish_boss_round_decoded_models(TOURNAMENT_ID, env.config, "db") == []
