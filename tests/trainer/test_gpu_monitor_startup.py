import threading

import pynvml


# Older nvidia-ml-py builds omit policies the monitor reads. The pinned package has them;
# define stand-ins only when missing so this module can import on either build.
for _name, _value in (
    ("NVML_PERF_POLICY_THERMAL", 0),
    ("NVML_PERF_POLICY_RELIABILITY", 1),
    ("NVML_PERF_POLICY_BOARD_LIMIT", 2),
):
    if not hasattr(pynvml, _name):
        setattr(pynvml, _name, _value)

import trainer.monitoring.monitor as gpu_monitor
from trainer import job_state
from trainer.monitoring.monitor import start_in_thread


def test_start_in_thread_runs_monitor(monkeypatch):
    started = threading.Event()
    monkeypatch.setattr(gpu_monitor, "main", started.set)
    start_in_thread()
    assert started.wait(2)


def test_monitor_failure_stays_in_the_thread(monkeypatch):
    finished = threading.Event()

    def boom():
        finished.set()
        raise RuntimeError("nvml down")

    monkeypatch.setattr(gpu_monitor, "main", boom)
    start_in_thread()
    assert finished.wait(2)


def test_read_persisted_jobs_does_not_replace_memory(tmp_path, monkeypatch):
    history = tmp_path / "task_history.json"
    history.write_text("[]")
    monkeypatch.setattr(job_state, "TASK_HISTORY_FILE", history)
    sentinel = object()
    job_state.task_history.append(sentinel)
    try:
        assert job_state.read_persisted_jobs() == []
        assert sentinel in job_state.task_history
    finally:
        job_state.task_history.remove(sentinel)
