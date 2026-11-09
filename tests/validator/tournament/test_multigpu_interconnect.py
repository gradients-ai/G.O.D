"""Multi-GPU (>1 GPU) tasks must be placed on SXM/NVLink trainers, not PCIe."""
import types
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from core.models.trainer_contract_models import GPUInfo
from core.models.trainer_contract_models import GPUInterconnect
from core.models.trainer_contract_models import GPUType
from validator.tournament import orchestrator
from validator.tournament.models import GpuRequirement
from validator.tournament.models import TrainerInfo


def _gpu(i, interconnect, nvlink=False, peers=None):
    return GPUInfo(
        gpu_id=i,
        gpu_type=GPUType.H100,
        vram_gb=80,
        available=True,
        interconnect=interconnect,
        nvlink=nvlink,
        nvlink_peers=peers,
    )


def _trainer(ip, interconnect, nvlink=False, n=4):
    return TrainerInfo(trainer_ip=ip, gpus=[_gpu(i, interconnect, nvlink) for i in range(n)])


# ---- helper ----
def test_is_high_interconnect_accepts_sxm_and_nvl():
    g = [_gpu(0, GPUInterconnect.SXM), _gpu(1, GPUInterconnect.SXM)]
    assert orchestrator._is_high_interconnect(g, [0, 1]) is True
    g = [_gpu(0, GPUInterconnect.NVL), _gpu(1, GPUInterconnect.NVL)]
    assert orchestrator._is_high_interconnect(g, [0, 1]) is True


def test_is_high_interconnect_rejects_pcie():
    g = [_gpu(0, GPUInterconnect.PCIE), _gpu(1, GPUInterconnect.PCIE)]
    assert orchestrator._is_high_interconnect(g, [0, 1]) is False


def _paired_box(n=8):
    """NVLink-bridged PCIe box: pairs 0-1, 2-3, 4-5, 6-7."""
    return [
        _gpu(i, GPUInterconnect.PCIE, nvlink=True, peers=[i ^ 1])
        for i in range(n)
    ]


def test_bonded_pair_passes_but_split_pair_fails():
    g = _paired_box(4)
    assert orchestrator._is_high_interconnect(g, [0, 1]) is True   # the bonded pair
    assert orchestrator._is_high_interconnect(g, [1, 2]) is False  # two halves of different pairs


def test_4x_and_8x_use_whole_pairs():
    g = _paired_box(8)
    assert orchestrator._is_high_interconnect(g, [0, 1, 2, 3]) is True        # 4x = two whole pairs
    assert orchestrator._is_high_interconnect(g, [0, 1, 4, 5]) is True        # 4x = two non-adjacent whole pairs
    assert orchestrator._is_high_interconnect(g, [0, 1, 2, 3, 4, 5, 6, 7]) is True  # 8x = four pairs


def test_partial_pair_in_a_multi_pair_selection_fails():
    g = _paired_box(8)
    # 4 GPUs but one pair is split (2 taken without 3) -> reject rather than straddle a pair
    assert orchestrator._is_high_interconnect(g, [0, 1, 2, 4]) is False
    assert orchestrator._is_high_interconnect(g, [0, 1, 2]) is False  # odd count can't be whole pairs


def test_is_high_interconnect_accepts_nvlink_bridged_pcie():
    g = [_gpu(0, GPUInterconnect.PCIE, nvlink=True), _gpu(1, GPUInterconnect.PCIE, nvlink=True)]
    assert orchestrator._is_high_interconnect(g, [0, 1]) is True


# ---- scheduler gate ----
@pytest.fixture
def _cfg():
    return types.SimpleNamespace(psql_db=None)


@pytest.mark.asyncio
async def test_multigpu_skips_pcie_picks_sxm(_cfg):
    trainers = [_trainer("pcie-box", GPUInterconnect.PCIE), _trainer("sxm-box", GPUInterconnect.SXM)]
    with patch.object(orchestrator.tournament_sql, "get_trainers", AsyncMock(return_value=trainers)):
        result = await orchestrator._check_suitable_gpus(_cfg, GpuRequirement.H100_2X)
    assert result is not None
    assert result[0] == "sxm-box"


@pytest.mark.asyncio
async def test_multigpu_waits_when_only_pcie(_cfg):
    trainers = [_trainer("pcie-box", GPUInterconnect.PCIE)]
    with patch.object(orchestrator.tournament_sql, "get_trainers", AsyncMock(return_value=trainers)):
        result = await orchestrator._check_suitable_gpus(_cfg, GpuRequirement.H100_2X)
    assert result is None  # holds for an SXM/NVLink box rather than running on PCIe


@pytest.mark.asyncio
async def test_single_gpu_still_allowed_on_pcie(_cfg):
    trainers = [_trainer("pcie-box", GPUInterconnect.PCIE)]
    with patch.object(orchestrator.tournament_sql, "get_trainers", AsyncMock(return_value=trainers)):
        result = await orchestrator._check_suitable_gpus(_cfg, GpuRequirement.H100_1X)
    assert result is not None
    assert result[0] == "pcie-box"
