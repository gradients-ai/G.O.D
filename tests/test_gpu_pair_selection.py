import pytest

from core.models.trainer_contract_models import GPUInfo
from core.models.trainer_contract_models import GPUType
from validator.tournament.models import GpuRequirement
from validator.tournament.orchestrator import _nvlink_groups
from validator.tournament.orchestrator import _select_grouped_gpu_ids
from validator.tournament.orchestrator import _trainer_has_sufficient_gpus


PAIRS = [[0, 1], [2, 3], [4, 5], [6, 7]]


def _gpu(
    gpu_id: int,
    available: bool = True,
    nvlink_peers: list[int] | None = None,
    gpu_type: GPUType = GPUType.H100,
) -> GPUInfo:
    return GPUInfo(gpu_id=gpu_id, gpu_type=gpu_type, vram_gb=80, available=available, nvlink_peers=nvlink_peers)


def test_groups_from_reported_pairs() -> None:
    gpus = [_gpu(i, nvlink_peers=[i ^ 1]) for i in range(8)]
    assert _nvlink_groups(gpus) == PAIRS


def test_groups_fully_connected() -> None:
    gpus = [_gpu(i, nvlink_peers=[j for j in range(8) if j != i]) for i in range(8)]
    assert _nvlink_groups(gpus) == [list(range(8))]


def test_groups_no_nvlink() -> None:
    gpus = [_gpu(i, nvlink_peers=[]) for i in range(4)]
    assert _nvlink_groups(gpus) == [[0], [1], [2], [3]]


def test_groups_unknown_topology_assumes_adjacent_pairs() -> None:
    gpus = [_gpu(i) for i in range(8)]
    assert _nvlink_groups(gpus) == PAIRS


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
def test_select_with_pairs(available: list[int], count: int, expected: list[int]) -> None:
    assert _select_grouped_gpu_ids(available, PAIRS, count) == expected


@pytest.mark.parametrize("groups", [[list(range(8))], [[i] for i in range(8)]])
def test_select_first_fit_when_topology_is_uniform(groups: list[list[int]]) -> None:
    assert _select_grouped_gpu_ids([1, 2, 3, 4, 5, 6, 7], groups, 2) == [1, 2]


def test_trainer_selection_uses_reported_topology() -> None:
    # Bridged pairs 0-2 and 1-3: index pairing would pick [1, 2] across bridges.
    peers = {0: [2], 1: [3], 2: [0], 3: [1]}
    gpus = [_gpu(0, available=False, nvlink_peers=peers[0])] + [_gpu(i, nvlink_peers=peers[i]) for i in (1, 2, 3)]
    assert _trainer_has_sufficient_gpus(gpus, GpuRequirement.H100_2X) == [1, 3]
    assert _trainer_has_sufficient_gpus(gpus, GpuRequirement.H100_1X) == [2]


def test_trainer_selection_skips_busy_and_other_types() -> None:
    gpus = [_gpu(0, False), _gpu(1), _gpu(2), _gpu(3), _gpu(4, gpu_type=GPUType.A100)]
    assert _trainer_has_sufficient_gpus(gpus, GpuRequirement.H100_2X) == [2, 3]
    assert _trainer_has_sufficient_gpus(gpus, GpuRequirement.H100_1X) == [1]
    assert _trainer_has_sufficient_gpus(gpus, GpuRequirement.A100) == [4]
    assert _trainer_has_sufficient_gpus(gpus, GpuRequirement.H100_4X) == []
