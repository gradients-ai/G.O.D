"""Tests for the training container's resource limits."""

import pytest

from trainer import runtime
from trainer.constants import CPUS_PER_GPU


NANO = 1_000_000_000


@pytest.mark.parametrize(
    ("gpu_ids", "host_cpus", "expected_cpus"),
    [
        ([0], 40, CPUS_PER_GPU),
        ([0, 1], 40, 40),  # 2 x 24 is above this host's cores
        ([0, 1], 192, 2 * CPUS_PER_GPU),
        ([0, 1], None, 2 * CPUS_PER_GPU),  # unknown core count: the per-GPU limit
    ],
)
def test_cpu_limit_never_exceeds_the_host(monkeypatch: pytest.MonkeyPatch, gpu_ids, host_cpus, expected_cpus):
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: host_cpus)

    _, nano_cpus = runtime.calculate_container_resources(gpu_ids)

    assert nano_cpus == expected_cpus * NANO
