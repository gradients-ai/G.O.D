"""Freeform exposure end to end (CPU, no network, a few seconds).

Real code at every step: git + xz decoder snapshot, check_quant_output, the trainer's run_freeform_decode (work dir,
permissions, download, extraction, output check, uploads), the orchestrator's _record_quant_decode, the hub listing
parser, the scorecard builder, get_task_with_hotkey_details, publish_boss_round_decoded_models (with the real
materialise_decoded_model), upload_participant_repository -> release_freeform_decoder, and finally the third-party
verifier in ops/tools/quant/verify_freeform_release.py.

Faked, because they are external: MinIO (a dict behind a local HTTP server for presigned reads), the HF hub (an in-memory
hub with LFS sha256 and git blob ids), the docker sandbox (the decoder runs on the host with the sandbox's argv/env), the
SQL layer (an in-memory store implementing exactly the functions the code calls) and the GitHub push (a bare local repo).

Run: pytest -o addopts='' -p no:randomly tests/e2e/test_freeform_exposure_e2e.py
"""

import fnmatch
import hashlib
import json
import lzma
import math
import os
import shutil
import stat
import struct
import subprocess
import sys
import threading
import urllib.request
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import numpy as np
import pytest

import trainer.constants as trainer_cst
import validator.infrastructure.service_constants as service_cst
from core.constants.quant import QUANT_MANIFEST_NAME
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestQuant
from core.models.quant_artifact_models import QuantDecodedIndex
from core.models.quant_artifact_models import QuantDecodeRequest
from core.models.quant_artifact_models import QuantDecodeResult
from core.models.quant_artifact_models import QuantManifest
from core.models.quant_artifact_models import QuantManifestFile
from core.models.quant_artifact_models import track_for
from core.models.quant_models import QuantCalibrationReference
from core.models.quant_models import QuantCalibrationRequest
from core.models.quant_models import QuantEvalGpu
from core.models.quant_models import QuantTarget
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from ops.tools.quant import verify_freeform_release as verifier
from trainer.quant_decode import job
from trainer.quant_decode.sandbox import sandbox_user
from trainer.quant_upload_check import check_quant_output
from trainer.quant_upload_check import quant_output_max_bytes
from validator.db import constants as db_cst
from validator.db.sql import auditing
from validator.db.sql import submissions_and_scoring as scoring_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournaments as tournament_sql
from validator.evaluation import quant_artifacts
from validator.evaluation.quant_artifacts import decoded_model_prefix
from validator.infrastructure.minio_client import async_minio_client
from validator.scoring import scorecard as scorecard_module
from validator.scoring import tasks as scoring_tasks
from validator.scoring.models import QuantScorecardRow
from validator.scoring.models import QuantSkuReference
from validator.tasks.models import QuantFreeformTask
from validator.tasks.models import QuantFreeformTaskWithHotkeyDetails
from validator.tournament import freeform_release
from validator.tournament import orchestrator
from validator.tournament import tournament_manager as tm
from validator.tournament.decoder_snapshot import snapshot_decoder
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentTaskTraining
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus


BUCKET = "e2e-bucket"
ORG = freeform_release.RAYONLABS_HF_USERNAME
TOURNAMENT_ID = "tourn_e2e"
TASK_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
CHAMPION = "5ChampionHotkeyAAAA"
RUNNER_UP = "5RunnerUpHotkeyBBBB"
BASE_MODEL = "org/base-model"
GITHUB_TOKEN = "ghp_e2e_super_secret_token"
SCALE = 0.01
TENSORS = {"embed.weight": [64, 32], "layers.0.mlp.weight": [32, 32], "norm.weight": [32]}
GPU = QuantEvalGpu(name="NVIDIA H100 PCIe", sm_count=114, sku="H100 PCIe")
REFERENCE_LOSS = 2.0
TEST_LOSS = 2.05
LN_PENALTY = math.log(1.1)
FORBIDDEN_FRAGMENTS = ("quant-decoders/", GITHUB_TOKEN, "X-Amz-Signature")

# The hub adds this file to every repo; it is counted in artifact_bytes like any other.
HUB_GITATTRIBUTES = (
    b"*.7z filter=lfs diff=lfs merge=lfs -text\n"
    b"*.bin filter=lfs diff=lfs merge=lfs -text\n"
    b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
)

DECODE_PY = '''"""Toy decoder __MARKER__: int8 payload x scale -> bf16 shards."""

import json
import struct
import sys
from pathlib import Path

import numpy as np


def to_bf16(values):
    return (values.astype(np.float32).view(np.uint32) >> 16).astype("<u2")


def write_shard(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        blob = array.tobytes()
        header[name] = {"dtype": "BF16", "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header, sort_keys=True).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def main(artifact, output):
    meta = json.loads((artifact / "quant_meta.json").read_text())
    payload = np.fromfile(artifact / "payload.bin", dtype=np.int8)
    decoded, position = {}, 0
    for spec in meta["tensors"]:
        count = int(np.prod(spec["shape"]))
        values = payload[position : position + count].astype(np.float32) * meta["scale"]
        decoded[spec["name"]] = to_bf16(values).reshape(spec["shape"])
        position += count
    names = list(decoded)
    groups = [names[:1], names[1:]]
    for index, group in enumerate(groups, 1):
        write_shard(output / f"model-{index:05d}-of-{len(groups):05d}.safetensors", {name: decoded[name] for name in group})


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
'''

GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "e2e",
    "GIT_AUTHOR_EMAIL": "e2e@example.com",
    "GIT_COMMITTER_NAME": "e2e",
    "GIT_COMMITTER_EMAIL": "e2e@example.com",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
}


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, check=True, capture_output=True, text=True).stdout.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def bf16_of(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.float32).view(np.uint32) >> 16).astype("<u2")


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        blob = array.tobytes()
        header[name] = {"dtype": "BF16", "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def make_base_model(path: Path) -> None:
    path.mkdir(parents=True)
    rng = np.random.RandomState(0)
    write_safetensors(path / "model.safetensors", {n: bf16_of(rng.randn(*s) * 0.02) for n, s in TENSORS.items()})
    (path / "config.json").write_text(json.dumps({"model_type": "toy", "hidden_size": 32}))
    (path / "tokenizer.json").write_text("{}")


def make_miner_repo(path: Path, marker: str) -> tuple[str, str]:
    """A private miner repo with two commits; returns (pinned commit, head commit). The second commit breaks the decoder,
    so only a snapshot taken at the pinned commit decodes correctly."""
    (path / "decoder").mkdir(parents=True)
    (path / "train").mkdir()
    git(path, "init", "--quiet", "--initial-branch=main")
    (path / "decoder" / "decode.py").write_text(DECODE_PY.replace("__MARKER__", marker))
    (path / "decoder" / "NOTES.md").write_text(f"decoder notes {marker}\n")
    (path / "train" / "pipeline.py").write_text(f"# private training pipeline {marker}\n")
    git(path, "add", "--all")
    git(path, "commit", "--quiet", "-m", "pinned")
    pinned = git(path, "rev-parse", "HEAD")
    (path / "decoder" / "decode.py").write_text("raise SystemExit(7)\n")
    (path / "train" / "pipeline.py").write_text(f"# private training pipeline v2 {marker}\n")
    git(path, "add", "--all")
    git(path, "commit", "--quiet", "-m", "later work")
    return pinned, git(path, "rev-parse", "HEAD")


def make_miner_output(out: Path, seed: int) -> None:
    """What the miner's pipeline leaves in its output dir: an opaque payload + config + manifest."""
    out.mkdir(parents=True)
    rng = np.random.RandomState(seed)
    payload = rng.randint(-100, 100, size=sum(math.prod(s) for s in TENSORS.values())).astype(np.int8)
    (out / "payload.bin").write_bytes(payload.tobytes())
    meta = {"format": "toy-int8-v1", "scale": SCALE, "tensors": [{"name": n, "shape": s} for n, s in TENSORS.items()]}
    (out / "quant_meta.json").write_text(json.dumps(meta))
    (out / "scratch.tmp").write_bytes(b"unlisted junk the upload check must prune")
    manifest = QuantManifest(
        version=1,
        task_id=str(TASK_ID),
        track=track_for(TaskType.QUANTFREEFORMTASK),
        files=[QuantManifestFile(name=n, bytes=(out / n).stat().st_size) for n in ("payload.bin", "quant_meta.json")],
    )
    (out / QUANT_MANIFEST_NAME).write_text(manifest.model_dump_json())


class FakeMinio:
    """Objects in a dict; presigned URLs point at a local HTTP server that serves them."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                key = self.path.split("?", 1)[0].lstrip("/").removeprefix(f"{BUCKET}/")
                data = outer.objects.get(key)
                self.send_response(200 if data is not None else 404)
                self.send_header("Content-Length", str(len(data or b"")))
                self.end_headers()
                self.wfile.write(data or b"")

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    async def upload_file(self, bucket, object_name, file_path, part_size=0, num_parallel_uploads=3):
        self.objects[object_name] = Path(file_path).read_bytes()
        return True

    async def download_file(self, bucket, object_name, file_path):
        Path(file_path).write_bytes(self.objects[object_name])

    async def get_presigned_url(self, bucket, object_name, expires=604800):
        return f"http://127.0.0.1:{self.server.server_port}/{bucket}/{object_name}?X-Amz-Signature=sig"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class FakeHub:
    """In-memory HF hub: repos of files, LFS sha256 for files matched by .gitattributes, git blob ids for the rest."""

    def __init__(self) -> None:
        self.repos: dict[tuple[str, str], dict[str, bytes]] = {}
        self.private: dict[tuple[str, str], bool] = {}

    def create_repo(self, repo_id, repo_type="model", private=False, exist_ok=False):
        key = (repo_type, repo_id)
        if key not in self.repos:
            self.repos[key] = {".gitattributes": HUB_GITATTRIBUTES}
            self.private[key] = private

    def put(self, repo_id: str, repo_type: str, name: str, data: bytes) -> None:
        self.repos[(repo_type, repo_id)][name] = data

    def upload_folder(self, folder_path, repo_id, repo_type="model"):
        for path in sorted(Path(folder_path).rglob("*")):
            if path.is_file():
                self.put(repo_id, repo_type, path.relative_to(folder_path).as_posix(), path.read_bytes())

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type="model"):
        data = Path(path_or_fileobj).read_bytes() if isinstance(path_or_fileobj, str) else path_or_fileobj
        self.put(repo_id, repo_type, path_in_repo, data)

    def is_lfs(self, repo_id: str, repo_type: str, name: str) -> bool:
        patterns = [
            line.split()[0]
            for line in self.repos[(repo_type, repo_id)][".gitattributes"].decode().splitlines()
            if "filter=lfs" in line
        ]
        return any(fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(Path(name).name, pattern) for pattern in patterns)

    def revision(self, repo_id: str) -> str:
        files = self.repos[("model", repo_id)]
        return hashlib.sha1(json.dumps({n: blob_sha1(d) for n, d in sorted(files.items())}).encode()).hexdigest()

    def model_info(self, repo_id: str):
        siblings = []
        for name, data in sorted(self.repos[("model", repo_id)].items()):
            lfs = SimpleNamespace(sha256=sha256_bytes(data), size=len(data)) if self.is_lfs(repo_id, "model", name) else None
            siblings.append(SimpleNamespace(rfilename=name, size=len(data), blob_id=blob_sha1(data), lfs=lfs))
        return SimpleNamespace(sha=self.revision(repo_id), siblings=siblings)

    def snapshot_download(self, repo_id, local_dir=None, token=None, ignore_patterns=None, revision=None):
        for name, data in self.repos[("model", repo_id)].items():
            if any(fnmatch.fnmatch(name, pattern) for pattern in ignore_patterns or []):
                continue
            path = Path(local_dir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (Path(local_dir) / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        return str(local_dir)

    def api(self):
        return self  # HfApi shape used by freeform_release: create_repo / upload_file / upload_folder


class FakeContainer:
    def __init__(self, kwargs: dict, observe, logs: list[bytes]) -> None:
        self.kwargs, self.observe, self.logs_out = kwargs, observe, logs
        self.name = kwargs["name"]

    def wait(self) -> dict:
        """Run the phase's command on the host: the sandbox's argv with each container path mapped to its host dir."""
        mounts = {mount["bind"]: host for host, mount in self.kwargs["volumes"].items()}
        command = self.kwargs["command"]
        if command == ["true"]:
            return {"StatusCode": 0}
        assert command[0] == "python" and command[2:] == ["/artifact", "/output"], command
        self.observe(mounts)

        def host_path(arg: str) -> str:
            for bind, host in mounts.items():
                if arg == bind or arg.startswith(bind + "/"):
                    return host + arg[len(bind) :]
            return arg

        argv = [sys.executable, *(host_path(a) for a in command[1:])]
        env = {**self.kwargs["environment"], "PATH": os.environ["PATH"]}
        proc = subprocess.run(argv, cwd=mounts["/decoder"], env=env, capture_output=True, check=False)
        self.logs_out.append(proc.stderr)
        return {"StatusCode": proc.returncode}

    def logs(self, stdout=True, stderr=True, tail="all") -> bytes:
        return b"".join(self.logs_out)

    def kill(self) -> None:
        pass

    def remove(self, force=False, v=False) -> None:
        pass


class FakeDockerClient:
    def __init__(self, observe) -> None:
        self.observe, self.runs, self._logs = observe, [], []
        self.containers = self

    def run(self, **kwargs) -> FakeContainer:
        self.runs.append(kwargs)
        return FakeContainer(kwargs, self.observe, self._logs)


class InMemoryStore:
    """The SQL layer, implementing exactly the functions the code under test calls."""

    def __init__(self, hub: FakeHub) -> None:
        self.hub = hub
        self.rows: dict[str, QuantScorecardRow] = {}
        self.participants: dict[str, TournamentParticipant] = {}
        self.status = TournamentStatus.ACTIVE
        self.final_round_task_ids: set[str] = set()
        self.task: QuantFreeformTask | None = None

    async def get_quant_scorecard_rows(self, task_id, psql_db):
        return {hotkey: row.model_copy(deep=True) for hotkey, row in self.rows.items()}

    async def set_quant_decode_result(
        self, task_id, hotkey, *, artifact_bytes, decoder_bytes, decoder_sha256, quant_failure, psql_db, build_warning=None,
        artifact_files=None, decoded_files=None,
    ):  # fmt: skip
        self.rows[hotkey] = self.rows[hotkey].model_copy(
            update={
                "artifact_bytes": artifact_bytes,
                "decoder_bytes": decoder_bytes,
                "quant_failure": quant_failure,
                "decode_artifact_files": artifact_files or None,
                "decoded_files": decoded_files or None,
            }
        )

    async def set_decoded_hf_repo(self, task_id, hotkey, repo, psql_db):
        self.rows[hotkey] = self.rows[hotkey].model_copy(update={"decoded_hf_repo": repo})

    async def get_tournament_participant(self, tournament_id, hotkey, psql_db):
        return self.participants.get(hotkey)

    async def get_tournament_participants(self, tournament_id, psql_db):
        return list(self.participants.values())

    async def get_tournament(self, tournament_id, psql_db):
        return TournamentData(
            tournament_id=tournament_id, tournament_type=TournamentType.QUANT, status=self.status, winner_hotkey=CHAMPION
        )

    async def get_tournament_task_ids_of_type(self, tournament_id, task_type, psql_db, final_round_only=False):
        ids = [str(TASK_ID)]
        return [i for i in ids if i in self.final_round_task_ids] if final_round_only else ids

    async def set_decoder_released(self, tournament_id, hotkey, url, released_at, psql_db):
        self.participants[hotkey] = self.participants[hotkey].model_copy(
            update={"decoder_released_url": url, "decoder_released_at": released_at}
        )

    async def update_backup_repo(self, tournament_id, hotkey, backup_repo, psql_db):
        self.participants[hotkey] = self.participants[hotkey].model_copy(update={"backup_repo": backup_repo})


class World:
    """The fakes, patched in, plus the per-stage drivers. One instance per test."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp = tmp_path
        self.minio = FakeMinio()
        self.hub = FakeHub()
        self.store = InMemoryStore(self.hub)
        self.config = SimpleNamespace(psql_db="db", discord_url="https://discord.example/hook")
        self.alerts = AsyncMock()
        self.public_remotes: dict[str, Path] = {}
        self.perm_observations: list[dict] = []
        self.cache_volume = tmp_path / "cache_volume"
        self.cache_volume.mkdir()
        self.work_root = tmp_path / "home" / ".local" / "share" / "god-trainer" / "quant_decode"
        self.base_dir = tmp_path / "base_model"
        make_base_model(self.base_dir)
        self.hub.create_repo(BASE_MODEL)
        self.hub.upload_folder(self.base_dir, BASE_MODEL)
        self.repos: dict[str, dict] = {}
        self._patch(monkeypatch)

    def _patch(self, mp: pytest.MonkeyPatch) -> None:
        store, hub = self.store, self.hub
        mp.setattr(service_cst, "BUCKET_NAME", BUCKET)
        for name in ("upload_file", "download_file", "get_presigned_url"):
            mp.setattr(async_minio_client, name, getattr(self.minio, name))
        mp.setattr(job, "snapshot_download", hub.snapshot_download)
        mp.setattr(quant_artifacts, "snapshot_download", hub.snapshot_download)
        mp.setattr(quant_artifacts, "_hub_model_info", lambda repo, token, timeout: hub.model_info(repo))
        mp.setattr(freeform_release, "_hf_api", hub.api)
        mp.setattr(freeform_release, "notify_freeform_release_problem", self.alerts)
        mp.setattr(tm, "get_tournament_participant", store.get_tournament_participant)
        mp.setattr(tm, "update_tournament_participant_backup_repo", store.update_backup_repo)
        mp.setattr(tm, "upload_tournament_participant_repository", self.squash_and_push)
        for module, names in (
            (scoring_sql, ("get_quant_scorecard_rows", "set_quant_decode_result", "set_decoded_hf_repo")),
            (
                tournament_sql,
                (
                    "get_tournament_participant",
                    "get_tournament_participants",
                    "get_tournament",
                    "get_tournament_task_ids_of_type",
                    "set_decoder_released",
                ),
            ),  # fmt: skip
        ):
            for name in names:
                mp.setattr(module, name, getattr(store, name))
        mp.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=TOURNAMENT_ID))
        mp.setattr(scorecard_module.tournament_sql, "get_tournament_id_by_task_id", AsyncMock(return_value=TOURNAMENT_ID))
        references = {
            ("", GPU.reference_key): QuantSkuReference(
                task_id=TASK_ID,
                sku=GPU.reference_key,
                model=BASE_MODEL,
                loss=REFERENCE_LOSS,
                per_example_losses=[REFERENCE_LOSS],
                eval_set_fingerprint="fp", eval_gpu=GPU,
            )
        }  # fmt: skip
        mp.setattr(scorecard_module, "get_quant_sku_references", AsyncMock(return_value=references))
        mp.setattr(task_sql, "get_task_by_id", AsyncMock(side_effect=lambda *_: store.task))
        mp.setattr(
            task_sql, "get_expected_repo_name", AsyncMock(side_effect=lambda _t, hotkey, _db: self.repos[hotkey]["artifact_name"])
        )

        conn = MagicMock()
        # The seed-hiding lookup (tournament_seed_tasks) finds no seed: this task is an ordinary quant task.
        conn.fetch = AsyncMock(side_effect=lambda query, *_: [] if "tournament_seed_tasks" in query else self.audit_rows())
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        self.audit_config = SimpleNamespace(psql_db=SimpleNamespace(connection=AsyncMock(return_value=cm)))

    def audit_rows(self) -> list[dict]:
        """The task_nodes x submissions rows get_task_with_hotkey_details selects."""
        return [
            {
                db_cst.HOTKEY: hotkey,
                db_cst.SUBMISSION_ID: None,
                db_cst.QUALITY_SCORE: 1.0,
                db_cst.TEST_LOSS: TEST_LOSS,
                db_cst.SYNTH_LOSS: TEST_LOSS,
                db_cst.SCORE_REASON: None,
                "rank": rank,
                db_cst.REPO: self.repos[hotkey]["artifact_repo"],
                db_cst.OFFER_RESPONSE: None,
            }
            for rank, hotkey in enumerate((CHAMPION, RUNNER_UP), 1)
        ]

    # ---- stage 1: the miner's private repo and registration snapshot --------------------------------------------------

    async def register(self, hotkey: str, marker: str) -> TournamentParticipant:
        repo_path = self.tmp / "private" / hotkey
        pinned, head = make_miner_repo(repo_path, marker)
        snapshot = await snapshot_decoder(str(repo_path), pinned, None, BUCKET)
        participant = tm._with_decoder_snapshot(
            TournamentParticipant(
                tournament_id=TOURNAMENT_ID,
                hotkey=hotkey,
                training_repo=str(repo_path),
                training_commit_hash=pinned,
                github_token=GITHUB_TOKEN,
            ),
            snapshot,
        )
        assert pinned != head
        self.store.participants[hotkey] = participant
        self.repos[hotkey] = {
            "path": repo_path,
            "pinned": pinned,
            "snapshot": snapshot,
            "artifact_name": f"expected-name-{hotkey[-4:].lower()}",
        }
        return participant

    # ---- stage 2/3: miner output, upload check, HF upload ---------------------------------------------------------------

    def miner_upload(self, hotkey: str, seed: int) -> Path:
        out = self.tmp / "miner_out" / hotkey
        make_miner_output(out, seed)
        verdict = check_quant_output(
            out, task_id=str(TASK_ID), task_type=TaskType.QUANTFREEFORMTASK, max_bytes=quant_output_max_bytes(self.base_dir)
        )
        assert verdict.ok, verdict.failure
        assert sorted(p.name for p in out.iterdir()) == [
            "payload.bin",
            QUANT_MANIFEST_NAME,
            "quant_meta.json",
        ]  # scratch.tmp pruned
        repo_id = f"{ORG}/{self.repos[hotkey]['artifact_name']}"
        self.hub.create_repo(repo_id, "model", private=False)
        self.hub.upload_folder(out, repo_id)
        self.repos[hotkey]["artifact_repo"] = repo_id
        self.repos[hotkey]["miner_out"] = out
        self.store.rows[hotkey] = QuantScorecardRow(hotkey=hotkey, repo=repo_id)
        return out

    # ---- stage 4: trainer decode -----------------------------------------------------------------------------------------

    def _observe_run_phase(self, mounts: dict[str, str]) -> None:
        def mode(path: Path) -> int:
            return stat.S_IMODE(path.stat().st_mode)

        artifact = Path(mounts["/artifact"])
        task_dir = artifact.parent
        self.perm_observations.append(
            {
                "work_root": mode(self.work_root),
                "task_dir": mode(task_dir),
                "xz": mode(task_dir / job.DECODER_XZ_NAME),
                "subdirs": {p.name: mode(p) for p in task_dir.iterdir() if p.is_dir()},
                "artifact_files": {p.name: mode(p) for p in artifact.rglob("*") if p.is_file()},
                "task_dir_path": task_dir,
            }
        )

    async def trainer_decode(self, hotkey: str) -> QuantDecodeResult:
        participant = self.store.participants[hotkey]
        request = QuantDecodeRequest(
            decoder_snapshot_url=await async_minio_client.get_presigned_url(BUCKET, participant.decoder_object_name),
            decoder_sha256=participant.decoder_sha256,
            bucket=BUCKET,
            object_prefix=decoded_model_prefix(TASK_ID, hotkey),
        )
        docker_client = FakeDockerClient(self._observe_run_phase)
        result = await job.run_freeform_decode(
            request,
            task_id=str(TASK_ID),
            hotkey=hotkey,
            artifact_repo=self.repos[hotkey]["artifact_repo"],
            base_model_dir=self.base_dir,
            hf_token=None,
            work_root=self.work_root,
            docker_client=docker_client,
        )
        self.repos[hotkey]["docker_runs"] = docker_client.runs
        return result

    # ---- stage 5: validator records the decode, scoring ----------------------------------------------------------------

    async def record_decode(self, hotkey: str, decode: QuantDecodeResult) -> bool:
        task = self.store.task
        training = TournamentTaskTraining(
            task=task, hotkey=hotkey, training_status=TrainingStatus.SUCCESS, n_training_attempts=1,
            created_at=datetime(2026, 11, 1), updated_at=datetime(2026, 11, 1), trainer_ip="10.0.0.1",
        )  # fmt: skip
        log = TrainerTaskLog(
            gpu_ids=[0],
            status=TaskStatus.SUCCESS,
            training_data=TrainRequestQuant(
                model=task.model_id,
                task_id=str(task.task_id),
                hours_to_complete=task.hours_to_complete,
                dataset=task.training_data,
                dataset_type=task.dataset_type,
                file_format="s3",
                task_type=task.task_type,
                quant_spec=None,
                quant_target=QuantTarget(
                    reference_model="m", reference_loss=2.0, reference_sku="s", bar_cap=0.04, bar=2.08, knee=1.04,
                    knee_loss=2.08, penalty_curvature=2500.0, cap=1.08, cap_loss=2.16,
                ),
                calibration=QuantCalibrationRequest(
                    examples_url="https://minio.example/cal.jsonl", eval_set_fingerprint="fp", n_examples=1,
                    reference=QuantCalibrationReference(model="m", loss=2.0, per_example_losses=[1.0], sku="s"),
                ),
            ),  # fmt: skip
            github_repo="https://github.com/miner/repo",
            hotkey=hotkey,
            quant_decode=decode,
        )
        return await orchestrator._record_quant_decode(training, log, self.config)

    def score(self, hotkey: str) -> quant_artifacts.HubArtifactListing:
        """Scoring's hub listing (the real wrapper), then the loss fields process_miners_pool would persist."""
        repo = self.repos[hotkey]["artifact_repo"]
        listing = scoring_tasks._hub_artifact_listing(repo)
        assert listing is not None
        self.store.rows[hotkey] = self.store.rows[hotkey].model_copy(
            update={
                "artifact_bytes": listing.bytes,
                "artifact_files": listing.files,
                "artifact_revision": listing.revision,
                "test_loss": TEST_LOSS,
                "passed_bar": True,
                "freeform_penalty": LN_PENALTY,
                "eval_gpu": GPU,
            }
        )
        return listing

    async def public_response(self) -> tuple[QuantFreeformTaskWithHotkeyDetails, str]:
        result = await auditing.get_task_with_hotkey_details(str(TASK_ID), self.audit_config)
        return result, result.model_dump_json()

    # ---- stage 7: completion + handover ----------------------------------------------------------------------------------

    async def squash_and_push(self, **kw) -> str:
        """repo_uploader.clone_and_push_repository's effect with real git: clone, check out the pinned commit, squash into an
        orphan commit, force-push to a bare 'GitHub' remote."""
        name = f"god-text-{kw['tournament_id']}-position-{kw['position']}".replace("_", "-")
        work = self.tmp / "squash" / name
        git(self.tmp, "clone", "--quiet", kw["training_repo"], str(work))
        git(work, "checkout", "--quiet", kw["commit_hash"])
        git(work, "remote", "remove", "origin")
        subprocess.run(["git", "branch", "-D", "main"], cwd=work, env=GIT_ENV, capture_output=True, check=False)
        git(work, "checkout", "--orphan", "main")
        git(work, "add", ".")
        git(work, "commit", "--quiet", "-m", f"Tournament winner repository - Commit: {kw['commit_hash'][:8]}")
        bare = self.tmp / "github" / f"{name}.git"
        bare.parent.mkdir(exist_ok=True)
        git(self.tmp, "init", "--quiet", "--bare", "--initial-branch=main", str(bare))
        git(work, "push", "--quiet", str(bare), "main", "--force")
        url = f"https://github.com/gradients-opensource/{name}.git"
        self.public_remotes[url] = bare
        return url

    def close(self) -> None:
        self.minio.close()


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(trainer_cst, "QUANT_DECODE_WORK_ROOT", tmp_path / "unused")
    w = World(tmp_path, monkeypatch)
    yield w
    w.close()


def _task() -> QuantFreeformTask:
    return QuantFreeformTask(
        task_id=TASK_ID,
        is_organic=False,
        status="success",
        model_id=BASE_MODEL,
        ds="org/dataset",
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 11, 1),
        test_data="https://bucket/test.jsonl",
        training_data="s3://bucket/train.json",
        dataset_type=InstructTextDatasetType(field_instruction="instruction", field_output="output"),
    )


def _third_party_inputs(world: World, scorecard: verifier.Scorecard, tmp: Path) -> verifier.VerifyInputs:
    """Only public things: the hub, the scorecard, the released dataset files and the public (squashed) repo."""
    release = scorecard.release
    assert release is not None and release.training_repo is not None
    dataset = world.hub.repos[("dataset", f"{ORG}/{freeform_release.RELEASES_DATASET_NAME}")]
    prefix = f"{TOURNAMENT_ID}/{CHAMPION}/"
    xz = tmp / "released.tar.xz"
    xz.write_bytes(dataset[f"{prefix}decoder-{scorecard.decoder.xz_sha256}.tar.xz"])
    repo_dir = tmp / "public_repo"
    git(tmp, "clone", "--quiet", str(world.public_remotes[release.training_repo]), str(repo_dir))
    artifact_dir = tmp / "artifact"
    world.hub.snapshot_download(scorecard.artifact.repo, artifact_dir)
    return verifier.VerifyInputs(
        scorecard=scorecard,
        decoder_xz=xz,
        repo_dir=repo_dir,
        artifact_dir=artifact_dir,
        release=verifier.load_release_json(dataset[f"{prefix}release.json"].decode()),
    )


async def test_freeform_exposure_end_to_end(world: World, tmp_path: Path) -> None:
    world.store.task = _task()

    # ---- 1. registration: real git archive + xz at the pinned (not HEAD) commit, uploaded to the fake MinIO ----
    champion = await world.register(CHAMPION, "CHAMPION_MARKER_8f31")
    await world.register(RUNNER_UP, "RUNNERUP_SECRET_MARKER_c0de")
    snap = world.repos[CHAMPION]["snapshot"]
    assert snap.tree_sha and snap.tar_sha256 and snap.xz_version and champion.decoder_xz_version == snap.xz_version
    assert (champion.decoder_tar_sha256, champion.decoder_tree_sha) == (snap.tar_sha256, snap.tree_sha)
    assert champion.decoder_sha256 == snap.xz_sha256 and champion.decoder_xz_bytes == snap.xz_bytes
    assert champion.decoder_object_name == f"quant-decoders/{world.repos[CHAMPION]['pinned']}/{snap.xz_sha256}.tar.xz"
    assert sha256_bytes(world.minio.objects[champion.decoder_object_name]) == snap.xz_sha256
    assert git(world.repos[CHAMPION]["path"], "rev-parse", f"{world.repos[CHAMPION]['pinned']}:decoder") == snap.tree_sha

    # ---- 2/3. miner output passes the real upload check (unlisted file pruned) and lands on the hub ----
    for seed, hotkey in enumerate((CHAMPION, RUNNER_UP), 1):
        world.miner_upload(hotkey, seed)
    champ_repo = world.repos[CHAMPION]["artifact_repo"]
    hub_files = world.hub.repos[("model", champ_repo)]
    assert set(hub_files) == {".gitattributes", "payload.bin", "quant_meta.json", QUANT_MANIFEST_NAME}
    plaintext_sum = sum(len(d) for d in hub_files.values())
    out = world.repos[CHAMPION]["miner_out"]
    assert plaintext_sum == len(HUB_GITATTRIBUTES) + sum(p.stat().st_size for p in out.iterdir())

    # ---- 4. trainer decode: private work dir, owner-only perms during the run, hashes of what it downloaded ----
    result = await world.trainer_decode(CHAMPION)
    assert result.ok, result
    run_obs = world.perm_observations[0]
    assert run_obs["work_root"] == 0o700 and run_obs["task_dir"] == 0o700
    assert run_obs["xz"] == 0o600 and set(run_obs["subdirs"].values()) == {0o700}
    assert run_obs["artifact_files"] and set(run_obs["artifact_files"].values()) == {0o600}
    assert not world.work_root.is_relative_to(world.cache_volume)
    assert list(world.cache_volume.iterdir()) == []  # nothing decode-related was ever put on the shared volume
    assert [p.name for p in world.work_root.iterdir()] == [job.WORK_ROOT_MARKER]  # work tree removed afterwards
    runs = world.repos[CHAMPION]["docker_runs"]
    assert [r["image"] for r in runs] == [trainer_cst.QUANT_DECODER_BUILD_IMAGE, trainer_cst.QUANT_DECODER_RUN_IMAGE]
    assert all(
        r["network_disabled"] and r["read_only"] and r["cap_drop"] == ["ALL"] and r["user"] == sandbox_user() for r in runs
    )
    assert runs[1]["command"] == ["python", "/decoder/decode.py", "/artifact", "/output"]
    assert {m["bind"]: m["mode"] for m in runs[1]["volumes"].values()} == {
        "/artifact": "ro", "/decoder": "ro", "/output": "rw", "/build": "ro",
    }  # fmt: skip

    hub_hashes = sorted((n, len(d), sha256_bytes(d)) for n, d in hub_files.items())
    assert [(f.name, f.bytes, f.sha256) for f in result.artifact_files] == hub_hashes  # sha256 of every file, .gitattributes too
    assert result.artifact_bytes == plaintext_sum
    prefix = decoded_model_prefix(TASK_ID, CHAMPION)
    index = QuantDecodedIndex.model_validate_json(world.minio.objects[f"{prefix}/index.json"])
    assert result.decoded_files == index.files and index.artifact_files == result.artifact_files
    assert len(result.decoded_files) == 2
    for shard in result.decoded_files:
        assert sha256_bytes(world.minio.objects[f"{prefix}/{shard.name}"]) == shard.sha256
    assert result.decoder_sha256 == champion.decoder_sha256 and result.decoder_bytes == champion.decoder_xz_bytes
    # the pinned commit's decoder ran, not HEAD's broken one
    runner_up_result = await world.trainer_decode(RUNNER_UP)
    assert runner_up_result.ok, runner_up_result

    # ---- 5. validator: record the decode, score from the hub listing, scorecard ----
    for hotkey, decode in ((CHAMPION, result), (RUNNER_UP, runner_up_result)):
        assert await world.record_decode(hotkey, decode) is True
        listing = world.score(hotkey)
        assert listing.bytes == sum(len(d) for d in world.hub.repos[("model", world.repos[hotkey]["artifact_repo"])].values())
    listing_files = {f.name: f for f in world.store.rows[CHAMPION].artifact_files}
    assert listing_files[".gitattributes"].sha256_kind == "git-blob-sha1" and listing_files[".gitattributes"].bytes == len(
        HUB_GITATTRIBUTES
    )
    assert listing_files["quant_meta.json"].sha256_kind == "git-blob-sha1"
    assert listing_files["payload.bin"].sha256_kind == "sha256"
    assert listing_files["payload.bin"].sha256 == sha256_bytes(hub_files["payload.bin"])
    assert world.store.rows[CHAMPION].artifact_revision == world.hub.revision(champ_repo)
    decoder_xz_bytes = champion.decoder_xz_bytes

    response, body = await world.public_response()
    assert isinstance(response, QuantFreeformTaskWithHotkeyDetails)
    card = {d.hotkey: d.quant_scorecard for d in response.hotkey_details}[CHAMPION]
    assert card is not None and card.artifact.decode_matches_hub is True
    assert card.bytes.artifact_bytes == plaintext_sum and card.bytes.decoder_xz_bytes == decoder_xz_bytes
    assert card.bytes.total_bytes == plaintext_sum + decoder_xz_bytes
    assert card.bytes.effective_bytes == pytest.approx(card.bytes.total_bytes * 1.1)
    assert card.quality.ratio == pytest.approx(TEST_LOSS / REFERENCE_LOSS)
    assert (card.decoder.xz_sha256, card.decoder.tree_sha, card.decoder.tar_sha256) == (
        champion.decoder_sha256, champion.decoder_tree_sha, champion.decoder_tar_sha256,
    )  # fmt: skip
    assert card.decoded.files == result.decoded_files and card.decoded.urls is None and card.release is None
    assert card.artifact.revision == world.hub.revision(champ_repo)
    for fragment in (*FORBIDDEN_FRAGMENTS, champion.decoder_object_name, str(world.repos[CHAMPION]["path"])):
        assert fragment not in body, fragment
    assert "release" in json.loads(body)["hotkey_details"][0]["quant_scorecard"]  # the key exists, its value is null
    assert json.loads(body)["hotkey_details"][0]["quant_scorecard"]["release"] is None

    # ---- 6. tournament completes: signed decoded links appear, boss-round HF copies are published ----
    world.store.status = TournamentStatus.COMPLETED
    world.store.final_round_task_ids = {str(TASK_ID)}
    response, body = await world.public_response()
    card = {d.hotkey: d.quant_scorecard for d in response.hotkey_details}[CHAMPION]
    assert card.decoded.urls is not None and set(card.decoded.urls) == {f.name for f in card.decoded.files}
    for shard in card.decoded.files:
        with urllib.request.urlopen(card.decoded.urls[shard.name], timeout=10) as r:
            assert sha256_bytes(r.read()) == shard.sha256
    for fragment in ("quant-decoders/", GITHUB_TOKEN, champion.decoder_object_name):
        assert fragment not in body, fragment

    published = await freeform_release.publish_boss_round_decoded_models(TOURNAMENT_ID, world.config, "db")
    assert sorted(published) == sorted(f"{ORG}/{world.repos[h]['artifact_name']}-decoded" for h in (CHAMPION, RUNNER_UP))
    decoded_repo = world.hub.repos[("model", f"{ORG}/{world.repos[CHAMPION]['artifact_name']}-decoded")]
    for shard in card.decoded.files:
        assert sha256_bytes(decoded_repo[shard.name]) == shard.sha256
    assert (
        "config.json" in decoded_repo
        and "index.json" not in decoded_repo
        and not any(n.endswith(".index.json") for n in decoded_repo)
    )
    assert await freeform_release.publish_boss_round_decoded_models(TOURNAMENT_ID, world.config, "db") == []  # idempotent
    assert world.store.rows[CHAMPION].decoded_hf_repo == f"{ORG}/{world.repos[CHAMPION]['artifact_name']}-decoded"

    # ---- 7. handover: squashed public repo + decoder snapshot and release.json in the public dataset ----
    backup = await tm.upload_participant_repository(TOURNAMENT_ID, TournamentType.QUANT, CHAMPION, 1, world.config, "db")
    assert backup in world.public_remotes
    await freeform_release.wait_for_background_publications()  # the release runs in the background
    releases = world.hub.repos[("dataset", f"{ORG}/{freeform_release.RELEASES_DATASET_NAME}")]
    release_prefix = f"{TOURNAMENT_ID}/{CHAMPION}/"
    assert set(releases) - {".gitattributes"} == {
        f"{release_prefix}decoder-{champion.decoder_sha256}.tar.xz",
        f"{release_prefix}release.json",
    }
    assert (
        releases[f"{release_prefix}decoder-{champion.decoder_sha256}.tar.xz"] == world.minio.objects[champion.decoder_object_name]
    )
    released_participant = world.store.participants[CHAMPION]
    assert released_participant.decoder_released_at is not None and released_participant.decoder_released_url.endswith(
        f"{release_prefix}decoder-{champion.decoder_sha256}.tar.xz"
    )
    assert not world.alerts.await_count
    assert not any(world.hub.private.values())  # every repo we created is public

    response, body = await world.public_response()
    cards = {d.hotkey: d.quant_scorecard for d in response.hotkey_details}
    assert cards[CHAMPION].release is not None and cards[CHAMPION].release.training_repo == backup
    assert cards[CHAMPION].release.training_commit_hash == world.repos[CHAMPION]["pinned"]
    assert cards[CHAMPION].release.decoder_snapshot_url == released_participant.decoder_released_url
    for fragment in ("quant-decoders/", GITHUB_TOKEN, champion.decoder_object_name, str(world.repos[CHAMPION]["path"])):
        assert fragment not in body, fragment

    # ---- 8. non-champion: scored, never released, its decoder is in no public store ----
    runner_up = world.store.participants[RUNNER_UP]
    assert cards[RUNNER_UP] is not None and cards[RUNNER_UP].release is None and cards[RUNNER_UP].decoder.xz_sha256
    assert runner_up.decoder_released_at is None and runner_up.backup_repo is None
    assert not any(name.startswith(f"{TOURNAMENT_ID}/{RUNNER_UP}") for name in releases)
    runner_up_xz = world.minio.objects[runner_up.decoder_object_name]
    for (repo_type, repo_id), files in world.hub.repos.items():
        for name, data in files.items():
            assert data != runner_up_xz and runner_up.decoder_sha256 not in name, (repo_id, name)
            if name.endswith(".tar.xz"):
                data = lzma.decompress(data)
            assert b"RUNNERUP_SECRET_MARKER" not in data, (repo_id, name)
    assert RUNNER_UP not in json.dumps({"remotes": sorted(world.public_remotes)})
    assert "RUNNERUP_SECRET_MARKER" not in body  # hashes are public by design (scorecard), the decoder is not

    # ---- 9. third party: public inputs only ----
    third = tmp_path / "third_party"
    third.mkdir()
    scorecard = verifier.load_scorecard(body, hotkey=CHAMPION)
    inputs = _third_party_inputs(world, scorecard, third)
    assert git(inputs.repo_dir, "rev-list", "--count", "HEAD") == "1"  # squashed: no private history
    assert git(inputs.repo_dir, "rev-parse", "HEAD:decoder") == scorecard.decoder.tree_sha
    report = verifier.verify_release(inputs, verifier.HostDecodeRunner())
    assert report.ok, report.failures()
    assert {c.name for c in report.checks} == {
        "decoder_xz_sha256", "decoder_xz_bytes", "release_json_decoder", "tree_sha", "tar_matches_repo", "artifact_file_hashes",
        "artifact_bytes", "total_bytes", "effective_bytes", "release_json_task", "decode_reproduces",
    }  # fmt: skip

    cli_args = [
        "--scorecard", str(_write(third / "task_response.json", body)),
        "--hotkey", CHAMPION, "--decoder-xz", str(inputs.decoder_xz),
        "--release-json", str(_write(third / "release.json", releases[f"{release_prefix}release.json"].decode())),
        "--repo-dir", str(inputs.repo_dir), "--artifact-dir", str(inputs.artifact_dir), "--decode", "--runner", "host",
    ]  # fmt: skip
    assert verifier.main(cli_args) == 0

    # ---- 10. negative checks: each tamper fails with a clear reason ----
    flipped = tmp_path / "tampered_artifact"
    shutil.copytree(inputs.artifact_dir, flipped)
    payload = flipped / "payload.bin"
    payload.write_bytes(bytes([payload.read_bytes()[0] ^ 1]) + payload.read_bytes()[1:])
    bad = verifier.verify_release(inputs.model_copy(update={"artifact_dir": flipped}), verifier.HostDecodeRunner())
    assert not bad.ok
    failed = {c.name: c.detail for c in bad.failures()}
    assert "payload.bin" in failed["artifact_file_hashes"] and "sha256" in failed["artifact_file_hashes"]
    assert "safetensors differs from the scorecard" in failed["decode_reproduces"]
    assert "tree_sha" not in failed and "decoder_xz_sha256" not in failed

    plain = tmp_path / "tampered_plain"
    shutil.copytree(inputs.artifact_dir, plain)
    meta_path = plain / "quant_meta.json"
    meta_path.write_text(meta_path.read_text().replace("0.01", "0.02"))  # same size, non-LFS file: git blob id must catch it
    assert (
        "quant_meta.json git-blob-sha1"
        in {c.name: c.detail for c in verifier.verify_release(inputs.model_copy(update={"artifact_dir": plain})).failures()}[
            "artifact_file_hashes"
        ]
    )

    wrong_tree = inputs.model_copy(
        update={
            "scorecard": scorecard.model_copy(
                deep=True, update={"decoder": scorecard.decoder.model_copy(update={"tree_sha": "0" * 40})}
            )
        }
    )
    failed = {c.name: c.detail for c in verifier.verify_release(wrong_tree).failures()}
    assert "tree_sha" in failed and "0" * 40 in failed["tree_sha"] and scorecard.decoder.tree_sha in failed["tree_sha"]

    forged = tmp_path / "forged.tar.xz"
    forged.write_bytes(
        lzma.compress(
            lzma.decompress(inputs.decoder_xz.read_bytes()).replace(b"CHAMPION_MARKER_8f31", b"CHAMPION_MARKER_ffff"),
            format=lzma.FORMAT_XZ,
        )
    )
    failed = {c.name: c.detail for c in verifier.verify_release(inputs.model_copy(update={"decoder_xz": forged})).failures()}
    assert {"decoder_xz_sha256", "tar_matches_repo"} <= set(failed) and "decode.py" in failed["tar_matches_repo"]
    garbage = tmp_path / "garbage.tar.xz"
    garbage.write_bytes(b"not an xz archive")
    failed = {
        c.name: c.detail
        for c in verifier.verify_release(
            inputs.model_copy(update={"decoder_xz": garbage}), verifier.HostDecodeRunner()
        ).failures()
    }
    assert "unreadable" in failed["tar_matches_repo"] and "decode failed" in failed["decode_reproduces"]

    arithmetic = scorecard.model_copy(
        deep=True,
        update={"bytes": scorecard.bytes.model_copy(update={"effective_bytes": scorecard.bytes.effective_bytes * 1.01})},
    )
    assert "effective_bytes" in {
        c.name for c in verifier.verify_release(inputs.model_copy(update={"scorecard": arithmetic})).failures()
    }


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path
