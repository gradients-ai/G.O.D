from trainer import job_state
from trainer.host import are_gpus_available
from trainer.job_state import is_gpu_throttle_failure
from trainer.job_state import note_gpu_throttle_failure
from trainer.monitoring.blacklist import blacklist_gpu
from trainer.monitoring.blacklist import blacklisted_gpu_ids
from trainer.monitoring.blacklist import clear_blacklisted_gpus
from trainer.monitoring.throttle import GPU_THROTTLE_FAILURE_MARKER
from trainer.monitoring.throttle import logs_mark_gpu_throttle_failure


def setup_function():
    clear_blacklisted_gpus()
    job_state._gpu_throttle_failures.clear()


def teardown_function():
    clear_blacklisted_gpus()
    job_state._gpu_throttle_failures.clear()


def test_blacklist_sticks_for_the_process():
    assert blacklist_gpu(2) is True
    assert blacklist_gpu(2) is False
    assert blacklisted_gpu_ids() == {2}


def test_withdrawn_gpu_is_refused_for_new_training():
    blacklist_gpu(1)
    assert are_gpus_available([1]) is False
    assert are_gpus_available([0, 1]) is False


def test_throttle_log_marker_is_how_the_orchestrator_recognizes_the_failure():
    assert logs_mark_gpu_throttle_failure([f"{GPU_THROTTLE_FAILURE_MARKER} GPU 0 throttling"])
    assert logs_mark_gpu_throttle_failure(["download finished", "something else"]) is False


def test_throttle_failure_is_remembered_for_the_task():
    note_gpu_throttle_failure("task-1", "hk")
    assert is_gpu_throttle_failure("task-1", "hk")
    assert is_gpu_throttle_failure("task-1", "other") is False
