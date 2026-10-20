import pytest

from core.models.trainer_contract_models import GPUInfo
from core.models.trainer_contract_models import GPUType
from validator.tournament.models import GpuRequirement
from validator.tournament.orchestrator import _select_paired_gpu_ids
from validator.tournament.orchestrator import _trainer_has_sufficient_gpus


@pytest.mark.parametrize(
    ("available", "count", "expected"),
    [
        # Whole pairs are preferred over first-fit for multi-GPU jobs.
        ([1, 2, 3, 4, 5, 6, 7], 2, [2, 3]),
        ([1, 2, 3, 4, 5, 6, 7], 4, [2, 3, 4, 5]),
        ([0, 1, 2, 3, 4, 5, 6, 7], 8, [0, 1, 2, 3, 4, 5, 6, 7]),
        # Single-GPU jobs take an orphan so free pairs stay intact.
        ([0, 1, 2, 4, 5], 1, [2]),
        ([0, 1, 2, 3], 1, [0]),
        # Not enough whole pairs: fill from orphans before splitting a pair.
        ([0, 1, 3, 5], 4, [0, 1, 3, 5]),
        ([1, 3], 2, [1, 3]),
        # Insufficient GPUs.
        ([0, 1, 2], 4, []),
        ([], 1, []),
    ],
)
def test_select_paired_gpu_ids(available: list[int], count: int, expected: list[int]) -> None:
    assert _select_paired_gpu_ids(available, count) == expected


def _gpu(gpu_id: int, available: bool, gpu_type: GPUType = GPUType.H100) -> GPUInfo:
    return GPUInfo(gpu_id=gpu_id, gpu_type=gpu_type, vram_gb=80, available=available)


def test_trainer_selection_skips_busy_and_other_types() -> None:
    gpus = [_gpu(0, False), _gpu(1, True), _gpu(2, True), _gpu(3, True), _gpu(4, True, GPUType.A100)]
    assert _trainer_has_sufficient_gpus(gpus, GpuRequirement.H100_2X) == [2, 3]
    assert _trainer_has_sufficient_gpus(gpus, GpuRequirement.H100_1X) == [1]
    assert _trainer_has_sufficient_gpus(gpus, GpuRequirement.A100) == [4]
    assert _trainer_has_sufficient_gpus(gpus, GpuRequirement.H100_4X) == []
