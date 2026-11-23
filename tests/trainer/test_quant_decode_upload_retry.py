"""Background retry of a decoded upload (trainer/quant_decode/upload_retry.py) against a fake storage and a fake clock."""

import asyncio
import json
import threading
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import pytest
from minio.error import S3Error

import trainer.constants as cst
from core.models.quant_artifact_models import QuantDecodedFile
from core.models.quant_artifact_models import QuantDecodedIndex
from core.models.quant_artifact_models import QuantDecodeResult
from tests.trainer.test_quant_decode_rerun import DECODE
from tests.trainer.test_quant_decode_rerun import _job
from trainer import job_state
from trainer import runtime
from trainer.quant_decode import job
from trainer.quant_decode import job_lock
from trainer.quant_decode import rerun
from trainer.quant_decode import upload
from trainer.quant_decode import upload_retry
from trainer.quant_decode.job import PendingUpload
from trainer.quant_decode.job import _DecodeProgress
from trainer.quant_decode.job import _WorkDirs
from trainer.quant_decode.job_lock import JobLock


TASK_ID = "task-1"
HOTKEY = "hk"
SHARDS = ["model-00001.safetensors", "model-00002.safetensors", "model-00003.safetensors"]
INDEX = "index.json"
WINDOW_SECONDS = 2 * 60 * 60
START = datetime(2030, 1, 1, tzinfo=timezone.utc)
FIRST_ERROR = "upload of model-00002.safetensors failed after 4 attempts (MaxRetryError)"


class Clock:
    """Stands in for the wall clock and for the waits between rounds: a wait only moves the clock."""

    def __init__(self) -> None:
        self.now = START
        self.pauses: list[float] = []
        self.during_pause = None

    def read(self) -> datetime:
        return self.now

    async def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)
        if self.during_pause is not None:
            await self.during_pause()
        self.now += timedelta(seconds=seconds)


class FakeStorage:
    """put_file and is_uploaded stand-ins. `errors` are raised by successive put_file calls, then puts succeed."""

    def __init__(self) -> None:
        self.errors: list[Exception] = []
        self.fail_always: Exception | None = None
        self.puts: list[str] = []
        self.stored: dict[str, bytes] = {}
        self.confirmed: set[str] = set()
        self.during_put = None

    def put_file(self, bucket: str, object_name: str, path: Path, sha256: str | None = None) -> None:
        name = object_name.rsplit("/", 1)[1]
        self.puts.append(name)
        if self.during_put is not None:
            self.during_put()
        if self.fail_always is not None:
            raise self.fail_always
        if self.errors:
            raise self.errors.pop(0)
        self.stored[name] = Path(path).read_bytes()

    def is_uploaded(self, bucket: str, object_name: str, size_bytes: int, sha256: str) -> bool:
        return object_name.rsplit("/", 1)[1] in self.confirmed


@pytest.fixture
def clock(monkeypatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(upload_retry, "_now", clock.read)
    monkeypatch.setattr(upload_retry, "_pause", clock.pause)
    monkeypatch.setattr(cst, "QUANT_DECODE_UPLOAD_RETRY_WINDOW_SECONDS", WINDOW_SECONDS)
    return clock


@pytest.fixture
def storage(monkeypatch) -> FakeStorage:
    storage = FakeStorage()
    monkeypatch.setattr(upload, "put_file", storage.put_file)
    monkeypatch.setattr(job, "is_uploaded", storage.is_uploaded)
    return storage


@pytest.fixture
def lines() -> list[str]:
    return []


@pytest.fixture
def report(lines: list[str]):
    async def record(message: str) -> None:
        lines.append(message)

    return record


@pytest.fixture
def work(tmp_path: Path) -> _WorkDirs:
    """A decode that passed its checks, uploaded its first shard and then lost the link."""
    work_root = tmp_path / "work_root"
    job.prepare_work_root(work_root)
    work = job.work_dirs(work_root, TASK_ID, HOTKEY)
    work.root.mkdir()
    work.output.mkdir()
    for name in SHARDS:
        (work.output / name).write_bytes(name.encode() * 10)
    first = work.output / SHARDS[0]
    progress = _DecodeProgress(
        artifact_bytes=100,
        decoder_bytes=10,
        decoder_sha256=DECODE.decoder_sha256,
        n_tensors=3,
        uploaded_files=[SHARDS[0]],
        decoded_files=[QuantDecodedFile(name=SHARDS[0], bytes=first.stat().st_size, sha256=job._sha256_file(first))],
    )
    pending = PendingUpload(
        token="token-1",
        task_id=TASK_ID,
        hotkey=HOTKEY,
        request=DECODE,
        progress=progress,
        retry_until=START + timedelta(seconds=WINDOW_SECONDS),
        settle_deadline=START + timedelta(seconds=WINDOW_SECONDS + 1200),
        last_error=FIRST_ERROR,
    )
    job.save_pending(work, pending)
    return work


@pytest.fixture
def history(monkeypatch, tmp_path: Path) -> Path:
    path = tmp_path / "task_history.json"
    monkeypatch.setattr(job_state, "TASK_HISTORY_FILE", path)
    monkeypatch.setattr(job_state, "task_history", [])
    return path


def write_history(path: Path, decode: QuantDecodeResult) -> None:
    path.write_text(json.dumps([_job(DECODE).model_copy(update={"quant_decode": decode}).model_dump(mode="json")]))


def stored_decode() -> QuantDecodeResult | None:
    job_state.load_task_history()
    return job_state.get_task(TASK_ID, HOTKEY).quant_decode


def dropped() -> Exception:
    return ConnectionResetError("reset")


def access_denied() -> S3Error:
    return S3Error("AccessDenied", "denied", "/bucket/key", "request-id", "host-id", None)


async def retry(work: _WorkDirs, report) -> QuantDecodeResult | None:
    return await upload_retry.retry_pending_upload(work.root.parent, TASK_ID, HOTKEY, report)


async def test_transient_failures_then_success_within_the_window_is_ok(work, clock, storage, report, lines):
    storage.errors = [dropped(), dropped()]

    result = await retry(work, report)

    assert result is not None and result.ok, result
    assert result.uploaded_files == [*SHARDS, INDEX]
    assert [f.name for f in result.decoded_files] == SHARDS
    assert result.index_object == f"{DECODE.object_prefix}/{INDEX}"
    assert result.upload_retry_until is None
    index = QuantDecodedIndex.model_validate_json(storage.stored[INDEX])
    assert index.files == result.decoded_files
    assert (index.task_id, index.hotkey, index.n_tensors) == (TASK_ID, HOTKEY, 3)
    assert len(clock.pauses) == 3  # one wait before each of the three rounds
    assert sum(clock.pauses) < WINDOW_SECONDS
    assert not work.root.exists()
    assert FIRST_ERROR in lines[0] and "giving up in 120 min" in lines[0]


async def test_files_already_uploaded_are_not_sent_again(work, clock, storage, report, monkeypatch):
    storage.errors = [dropped()]  # round 1: the second shard fails; round 2: it lands, then the third fails; round 3: rest
    real_put = storage.put_file

    def third_fails_once(bucket, object_name, path, sha256=None):
        if object_name.endswith(SHARDS[2]) and SHARDS[2] not in storage.puts:
            storage.puts.append(SHARDS[2])
            raise dropped()
        real_put(bucket, object_name, path, sha256)

    monkeypatch.setattr(upload, "put_file", third_fails_once)

    result = await retry(work, report)

    assert result is not None and result.ok, result
    assert storage.puts == [SHARDS[1], SHARDS[1], SHARDS[2], SHARDS[2], INDEX]
    assert SHARDS[0] not in storage.stored  # uploaded before the retry started; never sent again
    assert result.uploaded_files == [*SHARDS, INDEX]


async def test_a_file_found_in_storage_is_not_sent_again(work, clock, storage, report, lines):
    """The upload landed but its response was lost: storage has the object, the done set does not."""
    storage.confirmed = {SHARDS[1]}

    result = await retry(work, report)

    assert result is not None and result.ok, result
    assert storage.puts == [SHARDS[2], INDEX]
    assert result.uploaded_files == [*SHARDS, INDEX]
    assert any(f"{SHARDS[1]}" in line and "already in storage" in line for line in lines)


async def test_each_uploaded_file_is_saved_so_a_restart_skips_it(work, clock, storage, report, monkeypatch):
    seen: list[list[str]] = []

    def second_shard_then_fail(bucket, object_name, path, sha256=None):
        seen.append(job.load_pending(work).progress.uploaded_files)
        if object_name.endswith(SHARDS[2]):
            raise access_denied()

    monkeypatch.setattr(upload, "put_file", second_shard_then_fail)

    result = await retry(work, report)

    assert result is not None and result.infra_error is not None
    assert seen == [[SHARDS[0]], [SHARDS[0], SHARDS[1]]]  # the state on disk already had shard 2 when shard 3 started


async def test_window_exhausted_is_an_infra_error(work, clock, storage, report):
    storage.fail_always = dropped()

    result = await retry(work, report)

    assert result is not None and not result.ok and not result.upload_pending
    assert (
        result.infra_error
        == f"upload of {DECODE.object_prefix}/{SHARDS[1]} failed (ConnectionResetError); gave up after retrying for 120 min"
    )
    assert result.uploaded_files == [SHARDS[0]]
    assert sum(clock.pauses) == pytest.approx(WINDOW_SECONDS)  # the last round starts as the window closes
    assert set(storage.puts) == {SHARDS[1]}  # a round stops at its first failed file
    assert len(storage.puts) == len(clock.pauses)
    assert not work.root.exists()


async def test_waits_between_rounds_back_off_up_to_the_cap(work, clock, storage, report):
    storage.fail_always = dropped()

    await retry(work, report)

    base, cap = cst.QUANT_DECODE_UPLOAD_RETRY_ROUND_BASE_SECONDS, cst.QUANT_DECODE_UPLOAD_RETRY_ROUND_MAX_SECONDS
    for n, pause in enumerate(clock.pauses[:-1], start=1):  # the last one is cut to what is left of the window
        ceiling = min(cap, base * 2 ** (n - 1))
        assert ceiling / 2 <= pause <= ceiling
    assert max(clock.pauses) > base


async def test_a_permanent_error_ends_the_retry_at_once(work, clock, storage, report):
    storage.errors = [access_denied()]

    result = await retry(work, report)

    assert (
        result is not None and result.infra_error == f"upload of {DECODE.object_prefix}/{SHARDS[1]} failed (S3Error AccessDenied)"
    )
    assert len(clock.pauses) == 1
    assert not work.root.exists()


async def test_a_retry_whose_work_dir_was_taken_over_stores_nothing(work, clock, storage, report):
    async def another_decode_takes_over() -> None:
        job.save_pending(work, job.load_pending(work).model_copy(update={"token": "token-2"}))

    clock.during_pause = another_decode_takes_over

    assert await retry(work, report) is None

    assert storage.puts == []
    assert work.pending.exists()  # left for the new owner


async def test_decoded_files_removed_under_the_retry_store_nothing(work, clock, storage, report):
    """The re-run CLI starts the decode over: it empties the work dir while a round is uploading."""
    storage.during_put = lambda: job.remove_work_tree(work.root)

    assert await retry(work, report) is None


async def test_nothing_pending_returns_none(tmp_path: Path, report):
    assert await upload_retry.retry_pending_upload(tmp_path, TASK_ID, HOTKEY, report) is None


async def test_background_retry_stores_the_final_result_on_the_job(work, clock, storage, history):
    pending = job.load_pending(work)
    write_history(history, job.pending_result(pending))
    storage.errors = [dropped()]

    upload_retry.start(work.root.parent, TASK_ID, HOTKEY)
    await upload_retry._retries[(TASK_ID, HOTKEY)]

    decode = stored_decode()
    assert decode.ok and decode.uploaded_files == [*SHARDS, INDEX]
    assert (TASK_ID, HOTKEY) not in upload_retry._retries
    logs = job_state.get_task(TASK_ID, HOTKEY).logs
    assert any("next attempt in" in line for line in logs)
    assert "freeform decode ok" in logs[-1]


async def test_restart_resumes_a_pending_upload_under_a_new_token(work, clock, storage, history):
    write_history(history, job.pending_result(job.load_pending(work)))

    await upload_retry.resume_after_restart(work.root.parent)
    assert job.load_pending(work).token != "token-1"  # a retry from before the restart no longer owns the work dir
    assert stored_decode().upload_pending
    await upload_retry._retries[(TASK_ID, HOTKEY)]

    assert stored_decode().ok
    assert storage.puts == [SHARDS[1], SHARDS[2], INDEX]


async def test_restart_without_the_decoded_files_ends_in_an_infra_error(tmp_path, clock, storage, history):
    pending = QuantDecodeResult(upload_pending=True, upload_retry_until=START, uploaded_files=[SHARDS[0]], n_tensors=3)
    write_history(history, pending)

    await upload_retry.resume_after_restart(tmp_path / "work_root")

    decode = stored_decode()
    assert decode.infra_error == upload_retry.LOST_IN_RESTART_FAILURE
    assert not decode.upload_pending and decode.upload_retry_until is None
    assert decode.uploaded_files == [SHARDS[0]]
    assert upload_retry._retries == {}


async def test_restart_leaves_settled_jobs_alone(tmp_path, clock, storage, history):
    settled = QuantDecodeResult(ok=True, n_tensors=3)
    write_history(history, settled)

    await upload_retry.resume_after_restart(tmp_path / "work_root")

    assert stored_decode() == settled


@pytest.mark.parametrize(
    "result, retried", [(QuantDecodeResult(upload_pending=True, n_tensors=3), True), (QuantDecodeResult(ok=True), False)]
)
async def test_the_job_starts_a_background_retry_only_for_a_pending_upload(monkeypatch, history, tmp_path, result, retried):
    write_history(history, QuantDecodeResult(infra_error="earlier attempt"))
    started: list[tuple] = []
    cancelled: list[tuple] = []

    async def decode(task) -> QuantDecodeResult:
        assert cancelled == [(TASK_ID, HOTKEY)]  # an earlier retry of this job is stopped before the work dir is reused
        return result

    monkeypatch.setattr(runtime, "freeform_decode_result", decode)
    monkeypatch.setattr(runtime.upload_retry, "cancel", cancelled.append)
    monkeypatch.setattr(runtime.upload_retry, "start", lambda *args: started.append(args))
    monkeypatch.setattr(cst, "QUANT_DECODE_WORK_ROOT", tmp_path)

    await runtime._run_freeform_decode_step(rerun._proxy_request(_job(DECODE)), {})

    assert stored_decode() == result
    assert started == ([(tmp_path, TASK_ID, HOTKEY)] if retried else [])
    assert ("upload not complete" in job_state.get_task(TASK_ID, HOTKEY).logs[-1]) is retried


# --- ownership: the job's lock and the token


def job_lock_of(work: _WorkDirs) -> JobLock:
    return JobLock(work.root.parent, work.root.name)


class Stored:
    def __init__(self) -> None:
        self.results: list[QuantDecodeResult] = []
        self.work_existed: list[bool] = []
        self.error: Exception | None = None

    def into(self, work: _WorkDirs):
        async def store(result: QuantDecodeResult) -> None:
            self.work_existed.append(work.root.exists())
            if self.error is not None:
                raise self.error
            self.results.append(result)

        return store


async def test_a_round_holds_the_job_lock_while_it_uploads_and_not_while_it_waits(work, clock, storage, report):
    storage.errors = [dropped()]
    held_during_put: list[bool] = []
    free_during_pause: list[bool] = []

    def another_process_tries_the_lock() -> bool:
        other = job_lock_of(work)
        taken = other.try_acquire()
        other.release()
        return taken

    async def during_pause() -> None:
        free_during_pause.append(another_process_tries_the_lock())

    storage.during_put = lambda: held_during_put.append(not another_process_tries_the_lock())
    clock.during_pause = during_pause

    result = await retry(work, report)

    assert result is not None and result.ok
    assert held_during_put and all(held_during_put)
    assert free_during_pause and all(free_during_pause)
    assert another_process_tries_the_lock()  # released once settled


async def test_a_retry_that_finds_the_job_locked_by_a_decode_stops_and_stores_nothing(work, clock, storage, report):
    stored = Stored()
    decode_in_another_process = job_lock_of(work)
    assert decode_in_another_process.try_acquire()
    try:
        result = await upload_retry.retry_pending_upload(work.root.parent, TASK_ID, HOTKEY, report, stored.into(work))
    finally:
        decode_in_another_process.release()

    assert result is None
    assert storage.puts == [] and stored.results == []
    assert job.load_pending(work).token == "token-1"  # untouched


async def test_state_replaced_during_a_put_ends_the_retry_without_another_write(work, clock, storage, report):
    """Something that ignores the lock replaces the pending upload while a file is being sent. The retry must not put
    its own token back, send another file, publish the index, store a result or remove the directory."""
    stored = Stored()

    def replace_the_state() -> None:
        job.save_pending(work, job.load_pending(work).model_copy(update={"token": "token-2"}))

    storage.during_put = replace_the_state

    result = await upload_retry.retry_pending_upload(work.root.parent, TASK_ID, HOTKEY, report, stored.into(work))

    assert result is None
    assert storage.puts == [SHARDS[1]]  # the put in flight, and nothing after it
    assert INDEX not in storage.stored
    assert stored.results == []
    assert job.load_pending(work).token == "token-2"
    assert job.load_pending(work).progress.uploaded_files == [SHARDS[0]]
    assert sorted(p.name for p in work.output.iterdir()) == SHARDS


async def test_the_result_is_stored_before_the_work_dir_is_removed(work, clock, storage, report):
    stored = Stored()

    result = await upload_retry.retry_pending_upload(work.root.parent, TASK_ID, HOTKEY, report, stored.into(work))

    assert stored.results == [result] and result.ok
    assert stored.work_existed == [True]
    assert not work.root.exists()


async def test_a_result_that_cannot_be_stored_keeps_the_decoded_files_for_the_next_start(work, clock, storage, history):
    write_history(history, job.pending_result(job.load_pending(work)))
    stored = Stored()
    stored.error = OSError("disk full")

    with pytest.raises(OSError):
        await upload_retry.retry_pending_upload(work.root.parent, TASK_ID, HOTKEY, store=stored.into(work))

    assert work.pending.exists()
    assert job_lock_of(work).try_acquire()  # the lock is not left held


async def test_a_background_retry_that_crashes_does_not_raise_and_leaves_its_state(work, clock, storage, history, monkeypatch):
    write_history(history, job.pending_result(job.load_pending(work)))

    async def broken_store(task_id, hotkey, result) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(upload_retry, "update_quant_decode_result", broken_store)

    upload_retry.start(work.root.parent, TASK_ID, HOTKEY)
    await upload_retry._retries[(TASK_ID, HOTKEY)]

    assert stored_decode().upload_pending
    assert work.pending.exists()


async def test_progress_lines_that_cannot_be_written_do_not_end_the_retry(work, clock, storage):
    storage.errors = [dropped()]

    async def broken_report(message: str) -> None:
        raise OSError("task history unwritable")

    result = await upload_retry.retry_pending_upload(work.root.parent, TASK_ID, HOTKEY, broken_report)

    assert result is not None and result.ok


async def test_progress_that_cannot_be_saved_does_not_end_the_retry(work, clock, storage, report, monkeypatch):
    def unwritable(work_dirs, pending) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(job, "save_pending", unwritable)

    result = await retry(work, report)

    assert result is not None and result.ok
    assert result.uploaded_files == [*SHARDS, INDEX]


# --- startup sweep


async def test_restart_removes_work_dirs_with_nothing_to_resume_and_keeps_the_rest(work, clock, storage, history):
    write_history(history, job.pending_result(job.load_pending(work)))
    work_root = work.root.parent
    cut_short = work_root / "task-9_hk"  # a decode the restart interrupted
    (cut_short / "output").mkdir(parents=True)
    (cut_short / "output" / "model.safetensors").write_bytes(b"x" * 100)
    moved_aside = work_root / "task-9_hk.stale-abcd1234"
    moved_aside.mkdir()
    in_use = work_root / "task-8_hk"  # the re-run CLI is decoding in it
    in_use.mkdir()
    rerun_process = JobLock(work_root, in_use.name)
    assert rerun_process.try_acquire()
    unused_lock = JobLock(work_root, "task-7_hk")
    assert unused_lock.try_acquire()
    unused_lock.release()

    try:
        await upload_retry.resume_after_restart(work_root)
        await upload_retry._retries[(TASK_ID, HOTKEY)]
    finally:
        rerun_process.release()

    assert not cut_short.exists() and not moved_aside.exists()
    assert in_use.exists()
    assert (work_root / job.WORK_ROOT_MARKER).exists()
    assert not unused_lock.path.exists() and rerun_process.path.exists()
    assert stored_decode().ok


async def test_restart_never_sweeps_a_directory_without_the_decode_marker(tmp_path, clock, storage, history):
    write_history(history, QuantDecodeResult(ok=True))
    shared = tmp_path / "shared"
    (shared / "someone-elses-data").mkdir(parents=True)

    await upload_retry.resume_after_restart(shared)

    assert (shared / "someone-elses-data").exists()


async def test_restart_leaves_a_pending_job_alone_while_a_rerun_holds_its_lock(work, clock, storage, history):
    write_history(history, job.pending_result(job.load_pending(work)))
    rerun_process = job_lock_of(work)
    assert rerun_process.try_acquire()
    try:
        await upload_retry.resume_after_restart(work.root.parent)
    finally:
        rerun_process.release()

    assert stored_decode().upload_pending  # not marked lost: the re-run stores its own result
    assert upload_retry._retries == {}
    assert job.load_pending(work).token == "token-1"


def test_lock_dir_is_never_taken_for_a_work_dir(work):
    lock = job_lock_of(work)
    assert lock.try_acquire()
    lock.release()

    assert upload_retry._claim_work_dirs(work.root.parent)[0].task_id == TASK_ID
    assert (work.root.parent / job_lock.LOCK_DIR_NAME).is_dir()


# --- state files and results that predate the settle deadline, and state files that do not load


def test_a_state_file_without_a_settle_deadline_still_loads(work):
    state = json.loads(work.pending.read_text())
    del state["settle_deadline"]
    work.pending.write_text(json.dumps(state))

    pending = job.load_pending(work)

    assert pending is not None and pending.settle_deadline is None
    assert job.pending_result(pending).upload_settle_deadline == pending.retry_until


async def test_restart_resumes_a_state_file_without_a_settle_deadline(work, clock, storage, history):
    state = json.loads(work.pending.read_text())
    del state["settle_deadline"]
    work.pending.write_text(json.dumps(state))
    write_history(history, QuantDecodeResult(upload_pending=True, upload_retry_until=START, uploaded_files=[SHARDS[0]]))

    await upload_retry.resume_after_restart(work.root.parent)
    await upload_retry._retries[(TASK_ID, HOTKEY)]

    assert stored_decode().ok
    assert storage.puts == [SHARDS[1], SHARDS[2], INDEX]


@pytest.mark.parametrize("content", ["{not json", '{"token": "t"}', ""])
async def test_restart_keeps_decoded_files_whose_state_file_does_not_load(work, clock, storage, history, caplog, content):
    write_history(history, job.pending_result(job.load_pending(work)))
    work.pending.write_text(content)

    with caplog.at_level("ERROR"):
        await upload_retry.resume_after_restart(work.root.parent)

    assert sorted(p.name for p in work.output.iterdir()) == SHARDS  # never deleted on a state file we cannot read
    assert work.pending.read_text() == content
    assert any("state file cannot be read" in record.getMessage() for record in caplog.records)
    assert upload_retry._retries == {}
    assert stored_decode().infra_error == upload_retry.LOST_IN_RESTART_FAILURE  # nobody retries it: the validator hears


async def test_a_decode_that_never_got_the_job_lock_is_not_stored_over_the_result_on_the_job(monkeypatch, history, tmp_path):
    earlier = QuantDecodeResult(upload_pending=True, n_tensors=3)
    write_history(history, earlier)
    started: list[tuple] = []

    async def decode(task) -> QuantDecodeResult:
        return QuantDecodeResult(infra_error=job.JOB_BUSY_FAILURE)

    monkeypatch.setattr(runtime, "freeform_decode_result", decode)
    monkeypatch.setattr(runtime.upload_retry, "cancel", lambda key: None)
    monkeypatch.setattr(runtime.upload_retry, "start", lambda *args: started.append(args))

    await runtime._run_freeform_decode_step(rerun._proxy_request(_job(DECODE)), {})

    assert stored_decode() == earlier
    assert started == []
    assert job.JOB_BUSY_FAILURE in job_state.get_task(TASK_ID, HOTKEY).logs[-1]


async def test_the_job_lock_is_kept_until_the_work_dir_is_gone_even_when_cancelled(work, clock, storage, report, monkeypatch):
    removing, let_go = threading.Event(), threading.Event()
    locked_while_removing: list[bool] = []
    real_remove = job.remove_work_tree

    def slow_remove(root: Path) -> None:
        removing.set()
        let_go.wait(timeout=10)
        probe = job_lock_of(work)
        locked_while_removing.append(not probe.try_acquire())
        probe.release()
        real_remove(root)

    monkeypatch.setattr(job, "remove_work_tree", slow_remove)
    retrying = asyncio.create_task(retry(work, report))
    assert await asyncio.to_thread(removing.wait, 10)
    retrying.cancel()
    await asyncio.sleep(0.05)
    assert not retrying.done()  # still waiting for the removal
    let_go.set()

    with pytest.raises(asyncio.CancelledError):
        await retrying
    assert locked_while_removing == [True]
    assert not work.root.exists()
