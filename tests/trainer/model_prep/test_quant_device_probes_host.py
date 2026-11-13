"""Host (torch-free) quant device probes: CPU and identity, and the DeviceProbeStats round trip."""

import math

from core.models.model_prep_models import DeviceProbeStats
from trainer.model_prep import device_probes


def test_cpu_probe_returns_positive_float():
    seconds = device_probes.measure_cpu_single_thread(elements=200_000, rounds=2, repeats=2)
    assert seconds is not None and seconds > 0 and math.isfinite(seconds)


def test_host_identity():
    identity = device_probes.host_identity()
    assert identity.cpu_count is not None and identity.cpu_count >= 1


def test_device_probe_stats_round_trip():
    identity = device_probes.host_identity()
    stats = DeviceProbeStats(
        cpu_single_thread_seconds=device_probes.measure_cpu_single_thread(elements=100_000, rounds=1, repeats=1),
        cpu_model=identity.cpu_model,
        cpu_count=identity.cpu_count,
        cpu_quota=identity.cpu_quota,
        ram_gb=identity.ram_gb,
        probe_seconds=1.0,
    )
    assert DeviceProbeStats.model_validate(stats.model_dump()) == stats
