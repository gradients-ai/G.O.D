"""GPUs withdrawn from scheduling after a throttle fault.

The set lives for the life of the trainer process. Restarting the trainer after the
hardware is fixed puts the GPUs back in the pool; if the fault is still there, the
monitor withdraws them again.
"""

import threading


_blacklisted: set[int] = set()
_lock = threading.Lock()


def blacklist_gpu(index: int) -> bool:
    """Withdraw a GPU. Returns True the first time it is withdrawn."""
    with _lock:
        added = index not in _blacklisted
        _blacklisted.add(index)
        return added


def blacklisted_gpu_ids() -> set[int]:
    with _lock:
        return set(_blacklisted)


def clear_blacklisted_gpus() -> None:
    with _lock:
        _blacklisted.clear()
