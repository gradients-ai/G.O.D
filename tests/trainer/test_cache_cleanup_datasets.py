import os
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pytest

from core.models.dataset_models import FileFormat
from trainer.containers import cache_cleanup
from trainer.containers import downloader


OLD = (datetime.utcnow() - timedelta(hours=cache_cleanup.CUTOFF_HOURS + 1)).isoformat()
RECENT = (datetime.utcnow() - timedelta(minutes=5)).isoformat()


def _job(task_id: str, hotkey: str, status: str, started_at: str, finished_at: str | None) -> dict:
    return {
        "training_data": {"task_id": task_id},
        "hotkey": hotkey,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def _dataset_files(datasets: Path, task_id: str, age_hours: float) -> list[Path]:
    files = [datasets / f"{task_id}_train_data.json", datasets / f"{task_id}_tourn.zip"]
    stamp = time.time() - age_hours * 3600
    for path in files:
        path.write_text("[]")
        os.utime(path, (stamp, stamp))
    return files


@pytest.fixture
def datasets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "datasets"
    root.mkdir()
    monkeypatch.setattr(cache_cleanup, "CACHE_DATASETS_DIR", root)
    return root


def test_dataset_of_a_job_still_building_survives_an_old_finished_job_for_the_same_task(datasets: Path):
    # A job finished days ago and a job just accepted (dataset fetched, image still building) share the task's file.
    files = _dataset_files(datasets, "task", age_hours=cache_cleanup.CUTOFF_HOURS + 1)
    history = [
        _job("task", "old-hotkey", "success", OLD, OLD),
        _job("task", "new-hotkey", "training", RECENT, None),
    ]

    cache_cleanup.clean_datasets(history)

    assert all(path.exists() for path in files)


def test_dataset_written_after_the_history_was_read_survives(datasets: Path):
    # The history only shows the old job; the new job's download is younger than the cutoff.
    files = _dataset_files(datasets, "task", age_hours=0)

    cache_cleanup.clean_datasets([_job("task", "old-hotkey", "failure", OLD, OLD)])

    assert all(path.exists() for path in files)


@pytest.mark.asyncio
async def test_reusing_a_cached_dataset_marks_it_recent(monkeypatch: pytest.MonkeyPatch, datasets: Path):
    # A job that reuses an old task's cached file must not leave it looking stale to the cleanup.
    [cached, _] = _dataset_files(datasets, "task", age_hours=cache_cleanup.CUTOFF_HOURS + 1)
    monkeypatch.setattr(downloader.train_paths, "get_text_dataset_path", lambda task_id: str(cached))

    await downloader.download_text_dataset("task", "s3://unused", FileFormat.S3.value, str(datasets))
    cache_cleanup.clean_datasets([_job("task", "old-hotkey", "success", OLD, OLD)])

    assert cached.exists()


def test_dataset_of_jobs_that_all_finished_long_ago_is_deleted(datasets: Path):
    stale = _dataset_files(datasets, "stale", age_hours=cache_cleanup.CUTOFF_HOURS + 1)
    recent = _dataset_files(datasets, "recent", age_hours=cache_cleanup.CUTOFF_HOURS + 1)
    history = [
        _job("stale", "a", "success", OLD, OLD),
        _job("stale", "b", "failure", OLD, OLD),
        _job("recent", "a", "success", OLD, RECENT),
    ]

    cache_cleanup.clean_datasets(history)

    assert not any(path.exists() for path in stale)
    assert all(path.exists() for path in recent)


def test_training_job_with_a_stale_finished_at_keeps_its_dataset(datasets: Path):
    files = _dataset_files(datasets, "task", age_hours=cache_cleanup.CUTOFF_HOURS + 1)

    cache_cleanup.clean_datasets([_job("task", "a", "training", OLD, OLD)])

    assert all(path.exists() for path in files)
