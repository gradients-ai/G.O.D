import json
import shutil
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from trainer import constants as cst


TASK_HISTORY_FILE = Path(cst.TASKS_FILE_PATH)
CHECKPOINTS_DIR = Path(cst.OUTPUT_CHECKPOINTS_PATH)
CACHE_MODELS_DIR = Path(cst.CACHE_MODELS_DIR)
CACHE_DATASETS_DIR = Path(cst.CACHE_DATASETS_DIR)
CUTOFF_HOURS = cst.CACHE_CLEANUP_CUTOFF_HOURS


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
    return model_name.replace("/", "--")


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


def clean_datasets(task_history: list[dict]):
    for task in task_history:
        task_id = task.get("training_data", {}).get("task_id")
        finished_at = task.get("finished_at")
        if task_id and is_older_than(finished_at, CUTOFF_HOURS):
            candidate_files = [
                CACHE_DATASETS_DIR / f"{task_id}_train_data.json",
                CACHE_DATASETS_DIR / f"{task_id}_tourn.zip"
            ]
            for dataset_file in candidate_files:
                if dataset_file.exists():
                    print(f"Deleting dataset file: {dataset_file}")
                    dataset_file.unlink()


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


def main():
    print(f"[{datetime.utcnow()}] Starting cleanup...")
    task_history = load_task_history()
    clean_checkpoints(task_history)
    clean_datasets(task_history)
    clean_models(task_history)
    print(f"[{datetime.utcnow()}] Cleanup complete.")


if __name__ == "__main__":
    main()
