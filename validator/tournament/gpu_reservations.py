"""Job keys for trainer GPU reservations.

A reservation in `trainers_gpus` records the key of the job holding it (`reserved_by`), so the job's end releases exactly
the GPUs it still holds and never a GPU another job has taken since. The same key names the job's GPU cost run.
"""


def training_job_key(task_id: str, hotkey: str) -> str:
    return f"training:{task_id}:{hotkey}"


def prep_job_key(task_id: str, hotkey: str | None = None) -> str:
    identity = f"miner:{hotkey}" if hotkey is not None else "task"
    return f"prep:{task_id}:{identity}"
