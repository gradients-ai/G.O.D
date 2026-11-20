import json
import os
import re
import shutil
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from trainer import constants as cst
from trainer.model_artifacts import get_anonymous_model_dir


TASK_HISTORY_FILE = Path(cst.TASKS_FILE_PATH)
CHECKPOINTS_DIR = Path(cst.OUTPUT_CHECKPOINTS_PATH)
CACHE_MODELS_DIR = Path(cst.CACHE_MODELS_DIR)
CACHE_DATASETS_DIR = Path(cst.CACHE_DATASETS_DIR)
CUTOFF_HOURS = cst.CACHE_CLEANUP_CUTOFF_HOURS
TERMINAL_STATUSES = {"success", "failure"}
PARTIAL_DOWNLOAD_ROOTS = (
    CACHE_MODELS_DIR,
    CACHE_DATASETS_DIR,
    Path(cst.MINER_DATASETS_CACHE_DIR),
    Path(cst.HUGGINGFACE_CACHE_PATH),
)
# A download in progress (".<name>.download-<random>", plus the ".merged_tmp" a LoRA merge builds beside it), a cache
# entry mid-swap (".<name>.backup-<random>"), and the "<name>.tmp.<pid>" / "<name>.merged_tmp" older downloaders left.
PARTIAL_DOWNLOAD_NAME = re.compile(
    rf"^\..+(?:{re.escape(cst.CACHE_DOWNLOAD_TAG)}|\.backup-)|\.tmp\.\d+$|\.merged_tmp$"
)


def parse_time(dt_str: str | None) -> datetime | None:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


def is_older_than(dt_str: str | None, hours: int) -> bool:
    dt = parse_time(dt_str)
    if not dt:
        return False
    return datetime.utcnow() - dt > timedelta(hours=hours)


def get_model_folder(model_name: str) -> str:
    return get_anonymous_model_dir(model_name)


def load_task_history() -> list[dict]:
    if not TASK_HISTORY_FILE.exists():
        print(f"Task history file not found at {TASK_HISTORY_FILE}")
        return []
    with TASK_HISTORY_FILE.open("r") as f:
        return json.load(f)


def clean_checkpoints(task_history: list[dict]):
    task_id_to_times: dict[str, list[str | None]] = {}

    for task in task_history:
        task_id = task.get("training_data", {}).get("task_id")
        if not task_id:
            continue
        finished_at = task.get("finished_at")
        task_id_to_times.setdefault(task_id, []).append(finished_at)

    for task_id, finished_list in task_id_to_times.items():

        if all(is_older_than(finished_at, CUTOFF_HOURS) for finished_at in finished_list):
            target = Path(CHECKPOINTS_DIR, task_id)
            print(f"Deleting checkpoints for task {task_id} at {target}")
            shutil.rmtree(target, ignore_errors=True)


def _finished_before_cutoff(task: dict) -> bool:
    return task.get("status") in TERMINAL_STATUSES and is_older_than(task.get("finished_at"), CUTOFF_HOURS)


def clean_datasets(task_history: list[dict]):
    """Delete a task's dataset files once every job for that task id finished over CUTOFF_HOURS ago.

    Jobs of different hotkeys share one dataset file per task id, so any job that is still active, or finished
    recently, keeps it. A file written within CUTOFF_HOURS is kept too: it may belong to a job accepted after this run
    read the task history.
    """
    tasks_by_id: dict[str, list[dict]] = {}
    for task in task_history:
        task_id = task.get("training_data", {}).get("task_id")
        if task_id:
            tasks_by_id.setdefault(task_id, []).append(task)

    file_cutoff = time.time() - CUTOFF_HOURS * 3600
    for task_id, tasks in tasks_by_id.items():
        if not all(_finished_before_cutoff(task) for task in tasks):
            continue
        for dataset_file in (
            CACHE_DATASETS_DIR / f"{task_id}_train_data.json",
            CACHE_DATASETS_DIR / f"{task_id}_tourn.zip",
        ):
            try:
                if dataset_file.stat().st_mtime > file_cutoff:
                    continue
                print(f"Deleting dataset file: {dataset_file}")
                dataset_file.unlink()
            except FileNotFoundError:
                continue


def clean_models(task_history: list[dict]):
    recent_models = set()
    all_models = set()

    for task in task_history:
        model = task.get("training_data", {}).get("model")
        if not model:
            continue

        model_folder = get_model_folder(model)
        all_models.add(model_folder)

        status = task.get("status")
        started_at = task.get("started_at")
        finished_at = task.get("finished_at")

        if (
            status == "training"
            or not is_older_than(started_at, CUTOFF_HOURS)
            or not is_older_than(finished_at, CUTOFF_HOURS)
        ):
            recent_models.add(model_folder)

    if CACHE_MODELS_DIR.exists():
        for model_dir in CACHE_MODELS_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            if model_dir.name not in recent_models and model_dir.name in all_models:
                print(f"Deleting model folder: {model_dir}")
                shutil.rmtree(model_dir, ignore_errors=True)


def _last_write_time(path: Path) -> float:
    """Newest mtime anywhere under path: a running download keeps touching the file it is writing."""
    newest = path.lstat().st_mtime
    if path.is_dir() and not path.is_symlink():
        for root, dirs, files in os.walk(path):
            for name in dirs + files:
                try:
                    newest = max(newest, os.lstat(os.path.join(root, name)).st_mtime)
                except FileNotFoundError:
                    continue
    return newest


def clean_partial_downloads(roots=PARTIAL_DOWNLOAD_ROOTS, stale_hours: float = cst.CACHE_PARTIAL_DOWNLOAD_STALE_HOURS):
    """Remove partial downloads nothing has written to for stale_hours; a younger one may belong to a running download."""
    cutoff = time.time() - stale_hours * 3600
    for root in roots:
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            if not PARTIAL_DOWNLOAD_NAME.search(entry.name):
                continue
            try:
                if _last_write_time(entry) > cutoff:
                    continue
                print(f"Deleting stale partial download: {entry}")
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink()
            except FileNotFoundError:
                continue


def main():
    print(f"[{datetime.utcnow()}] Starting cleanup...")
    task_history = load_task_history()
    clean_checkpoints(task_history)
    clean_datasets(task_history)
    clean_models(task_history)
    clean_partial_downloads()
    print(f"[{datetime.utcnow()}] Cleanup complete.")


if __name__ == "__main__":
    main()
