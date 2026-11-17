"""Per-GPU CPU and memory shares of the trainer host.

Every GPU on the box owns an equal, fixed share of the host: a disjoint set of whole physical cores (NUMA-local to the
GPU when the box reports GPU affinity) and an equal slice of RAM. A job gets the shares of the GPUs it was given, so
two jobs on one box never contend for the same cores and a job's CPU budget does not depend on what else is running.

A small reserve of cores and memory is left to the host, the trainer and its helper containers.
"""

import math
import os
import subprocess
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field

import trainer.constants as cst
from trainer.telemetry import logger


NANO_CPUS_PER_CPU = 1_000_000_000
# Thread-pool sizes read by OpenMP (torch, numpy), MKL, OpenBLAS, numexpr and rayon (HF tokenizers, polars).
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "RAYON_NUM_THREADS",
)
SYS_CPU_ROOT = Path("/sys/devices/system/cpu")
SYS_PCI_DEVICES_ROOT = Path("/sys/bus/pci/devices")
# One directory per GPU, named by PCI address: the GPU count when nvidia-smi cannot answer.
NVIDIA_PROC_GPUS_ROOT = Path("/proc/driver/nvidia/gpus")
NVIDIA_SMI_TIMEOUT_SECONDS = 10


class HostTopology(BaseModel):
    """What the trainer can see of its host."""

    cpus: list[int] = Field(description="Logical CPUs the trainer may run on (its affinity mask).")
    cores: list[list[int]] = Field(description="Physical cores: hyperthread siblings within `cpus`, ordered by first CPU.")
    memory_bytes: int = Field(gt=0)
    gpu_local_cpus: list[list[int] | None] = Field(
        description="Per GPU index, the CPUs local to it (NUMA affinity); None when the host does not say."
    )


class HostPlan(BaseModel):
    """The fixed per-GPU split of one host."""

    gpu_count: int
    gpu_cpus: dict[int, list[int]] = Field(description="Per GPU index, its pinned CPUs. Empty when the host cannot pin.")
    cpus_per_gpu: int = Field(description="Logical CPUs per GPU share (the quota when unpinned).")
    memory_bytes_per_gpu: int
    numa_aware: bool = False
    host_cpus: list[int] = Field(description="CPUs left to the host, the trainer and helper containers.")


class ContainerResources(BaseModel):
    """The CPU and memory limits of one job container."""

    gpu_ids: list[int]
    cpus: list[int] = Field(description="Pinned CPUs; empty when the job is not pinned.")
    cpu_count: int = Field(gt=0, description="CPUs the job may use: the pinned set's size, or the quota.")
    memory_bytes: int = Field(gt=0)

    @property
    def cpuset_cpus(self) -> str | None:
        return format_cpu_list(self.cpus) if self.cpus else None

    def docker_kwargs(self) -> dict:
        """`mem_limit` plus either `cpuset_cpus` or `nano_cpus`.

        A pinned job gets no CFS quota: the cpuset already bounds it to `cpu_count` CPUs, so a quota of the same size
        can never bind and only adds per-period throttling on bursty thread pools.
        """
        kwargs: dict = {"mem_limit": self.memory_bytes}
        if self.cpus:
            kwargs["cpuset_cpus"] = self.cpuset_cpus
        else:
            kwargs["nano_cpus"] = self.cpu_count * NANO_CPUS_PER_CPU
        return kwargs

    @property
    def threads_per_gpu(self) -> int:
        return max(1, self.cpu_count // len(self.gpu_ids))

    def thread_env(self) -> dict[str, str]:
        """Sized to one GPU's share: multi-GPU jobs run one process per GPU (torchrun, accelerate), each of which
        would otherwise size its pools to the whole job."""
        return thread_env(self.threads_per_gpu)

    def describe(self) -> str:
        cpus = f"cpuset {self.cpuset_cpus}" if self.cpus else "unpinned quota"
        return f"GPUs {self.gpu_ids}: {self.cpu_count} CPUs ({cpus}), {self.memory_bytes / 1024**3:.0f} GiB memory"


def thread_env(threads: int) -> dict[str, str]:
    return {name: str(threads) for name in THREAD_ENV_VARS}


def parse_cpu_list(text: str) -> list[int]:
    """Parse a kernel CPU list ("0-3,8,10-11") into sorted CPU ids."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def format_cpu_list(cpus: list[int]) -> str:
    """Format CPU ids as a kernel/Docker CPU list ("0-3,8")."""
    ranges: list[str] = []
    ordered = sorted(set(cpus))
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        ranges.append(str(ordered[i]) if i == j else f"{ordered[i]}-{ordered[j]}")
        i = j + 1
    return ",".join(ranges)


def _physical_cores(cpus: list[int], cpu_root: Path) -> list[list[int]]:
    """Group `cpus` into physical cores by their hyperthread siblings; one core per CPU when the kernel does not say."""
    allowed = set(cpus)
    seen: set[int] = set()
    cores: list[list[int]] = []
    for cpu in cpus:
        if cpu in seen:
            continue
        try:
            siblings = parse_cpu_list((cpu_root / f"cpu{cpu}" / "topology" / "thread_siblings_list").read_text())
        except (OSError, ValueError):
            siblings = [cpu]
        core = sorted((set(siblings) & allowed) | {cpu})
        seen.update(core)
        cores.append(core)
    return sorted(cores, key=lambda core: core[0])


def _sysfs_pci_address(nvidia_bus_id: str) -> str:
    """nvidia-smi's "00000000:18:00.0" as sysfs names it: "0000:18:00.0"."""
    domain, rest = nvidia_bus_id.strip().split(":", 1)
    return f"{domain[-4:]}:{rest}".lower()


def _nvidia_smi_bus_ids() -> list[str]:
    """PCI bus id of every GPU, by GPU index (the index Docker device requests use)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
    ).stdout
    return parse_nvidia_smi_bus_ids(out)


def parse_nvidia_smi_bus_ids(out: str) -> list[str]:
    rows = [line.split(",") for line in out.splitlines() if line.strip()]
    by_index = {int(index): bus_id.strip() for index, bus_id in rows}
    return [by_index[i] for i in range(len(by_index))]


def _gpu_local_cpus_from_driver(pci_root: Path, proc_gpus_root: Path) -> list[list[int] | None]:
    """GPU affinity by GPU index, or (when nvidia-smi fails) only the GPU count from the driver: an even split."""
    try:
        return _gpu_local_cpus(_nvidia_smi_bus_ids(), pci_root)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as e:
        try:
            count = len([entry for entry in proc_gpus_root.iterdir() if entry.is_dir()])
        except OSError:
            count = 0
        if count == 0:
            raise RuntimeError(f"cannot list the host's GPUs: nvidia-smi failed ({e}) and {proc_gpus_root} is empty") from e
        logger.warning(f"nvidia-smi failed ({e}); splitting the host evenly across the driver's {count} GPUs")
        return [None] * count


def _gpu_local_cpus(bus_ids: list[str], pci_root: Path) -> list[list[int] | None]:
    local: list[list[int] | None] = []
    for bus_id in bus_ids:
        try:
            cpus = parse_cpu_list((pci_root / _sysfs_pci_address(bus_id) / "local_cpulist").read_text())
        except (OSError, ValueError):
            cpus = []
        local.append(cpus or None)
    return local


def read_host_topology(
    cpu_root: Path = SYS_CPU_ROOT,
    pci_root: Path = SYS_PCI_DEVICES_ROOT,
    proc_gpus_root: Path = NVIDIA_PROC_GPUS_ROOT,
    bus_ids: list[str] | None = None,
) -> HostTopology:
    cpus = sorted(os.sched_getaffinity(0))
    memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if bus_ids is None:
        gpu_local_cpus = _gpu_local_cpus_from_driver(pci_root, proc_gpus_root)
    else:
        gpu_local_cpus = _gpu_local_cpus(bus_ids, pci_root)
    return HostTopology(
        cpus=cpus,
        cores=_physical_cores(cpus, cpu_root),
        memory_bytes=memory_bytes,
        gpu_local_cpus=gpu_local_cpus,
    )


def _reserved_core_count(core_count: int, gpu_count: int) -> int:
    reserve = max(1, math.ceil(core_count * cst.HOST_CPU_RESERVE_FRACTION))
    # Never reserve a GPU's last core: on a box this small the jobs need the cores more than the host does.
    return reserve if core_count - reserve >= gpu_count else 0


def _memory_bytes_per_gpu(memory_bytes: int, gpu_count: int) -> int:
    reserve = max(cst.HOST_MEMORY_RESERVE_MIN_GB * 1024**3, math.ceil(memory_bytes * cst.HOST_MEMORY_RESERVE_FRACTION))
    if reserve >= memory_bytes:
        reserve = math.ceil(memory_bytes * cst.HOST_MEMORY_RESERVE_FRACTION)
    return (memory_bytes - reserve) // gpu_count


def _numa_split(
    cores: list[list[int]], gpu_local_cpus: list[list[int] | None], cores_per_gpu: int
) -> tuple[dict[int, list[list[int]]], int] | None:
    """Each GPU's cores from its own NUMA domain, or None when the affinity is unknown, overlapping or too uneven.

    GPUs that report the same local CPU set form one domain; domains must be disjoint. Each GPU takes `cores_per_gpu`
    whole cores from the top of its domain, so every GPU gets the same count and low CPUs stay with the host.
    Returns the split and the number of domains.
    """
    if any(local is None for local in gpu_local_cpus):
        return None
    domains: dict[frozenset[int], list[int]] = {}
    for gpu, local in enumerate(gpu_local_cpus):
        domains.setdefault(frozenset(local), []).append(gpu)
    keys = list(domains)
    if any(a & b for i, a in enumerate(keys) for b in keys[i + 1 :]):
        return None
    split: dict[int, list[list[int]]] = {}
    for local, gpus in domains.items():
        domain_cores = [core for core in cores if set(core) <= local]
        needed = cores_per_gpu * len(gpus)
        if len(domain_cores) < needed:
            return None
        taken = domain_cores[len(domain_cores) - needed :]
        for k, gpu in enumerate(gpus):
            split[gpu] = taken[k * cores_per_gpu : (k + 1) * cores_per_gpu]
    return split, len(domains)


def plan_host(topology: HostTopology) -> HostPlan:
    """Split the host into one equal share per GPU."""
    gpu_count = len(topology.gpu_local_cpus)
    if gpu_count == 0:
        raise ValueError("host has no GPUs to split")
    cores = topology.cores
    memory_per_gpu = _memory_bytes_per_gpu(topology.memory_bytes, gpu_count)

    if len(cores) < gpu_count:
        # Fewer physical cores than GPUs: no disjoint split exists, so share the CPUs by quota instead.
        return HostPlan(
            gpu_count=gpu_count,
            gpu_cpus={},
            cpus_per_gpu=max(1, len(topology.cpus) // gpu_count),
            memory_bytes_per_gpu=memory_per_gpu,
            host_cpus=topology.cpus,
        )

    cores_per_gpu = (len(cores) - _reserved_core_count(len(cores), gpu_count)) // gpu_count
    numa = _numa_split(cores, topology.gpu_local_cpus, cores_per_gpu)
    # One domain (a VM reporting every CPU local to every GPU) is the even split; only call it NUMA-aware with two.
    numa_aware = numa is not None and numa[1] > 1
    if numa is not None:
        by_gpu = numa[0]
    else:
        # Even contiguous split from the top; the spare cores (the reserve and the remainder) stay at the bottom.
        offset = len(cores) - cores_per_gpu * gpu_count
        by_gpu = {gpu: cores[offset + gpu * cores_per_gpu : offset + (gpu + 1) * cores_per_gpu] for gpu in range(gpu_count)}

    gpu_cpus = {gpu: sorted(cpu for core in gpu_cores for cpu in core) for gpu, gpu_cores in by_gpu.items()}
    assigned = {cpu for cpus in gpu_cpus.values() for cpu in cpus}
    cpus_per_gpu = min(len(cpus) for cpus in gpu_cpus.values())
    return HostPlan(
        gpu_count=gpu_count,
        gpu_cpus=gpu_cpus,
        cpus_per_gpu=cpus_per_gpu,
        memory_bytes_per_gpu=memory_per_gpu,
        numa_aware=numa_aware,
        host_cpus=[cpu for cpu in topology.cpus if cpu not in assigned],
    )


def resources_for(plan: HostPlan, gpu_ids: list[int]) -> ContainerResources:
    """The union of the shares of `gpu_ids`."""
    if not gpu_ids:
        raise ValueError("a job container needs at least one GPU")
    unknown = [gpu for gpu in gpu_ids if not 0 <= gpu < plan.gpu_count]
    if unknown:
        raise ValueError(f"GPU ids {unknown} are not on this host ({plan.gpu_count} GPUs)")
    shares = len(set(gpu_ids))
    cpus = sorted(cpu for gpu in set(gpu_ids) for cpu in plan.gpu_cpus.get(gpu, []))
    return ContainerResources(
        gpu_ids=sorted(set(gpu_ids)),
        cpus=cpus,
        cpu_count=len(cpus) if cpus else plan.cpus_per_gpu * shares,
        memory_bytes=plan.memory_bytes_per_gpu * shares,
    )


_host_plan: HostPlan | None = None


def host_plan() -> HostPlan:
    """The host's split, read once per trainer process (the host does not change under a running trainer)."""
    global _host_plan
    if _host_plan is None:
        topology = read_host_topology()
        plan = plan_host(topology)
        logger.info(
            f"Host split: {len(topology.cpus)} CPUs ({len(topology.cores)} cores), "
            f"{topology.memory_bytes / 1024**3:.0f} GiB, {plan.gpu_count} GPUs -> per GPU "
            f"{plan.cpus_per_gpu} CPUs, {plan.memory_bytes_per_gpu / 1024**3:.0f} GiB "
            f"({'NUMA-local' if plan.numa_aware else 'even split' if plan.gpu_cpus else 'unpinned quota'}); "
            f"per-GPU cpusets {{{', '.join(f'{g}: {format_cpu_list(c)}' for g, c in plan.gpu_cpus.items())}}}; "
            f"host keeps {format_cpu_list(plan.host_cpus) or 'none'}"
        )
        _host_plan = plan
    return _host_plan


def container_resources(gpu_ids: list[int]) -> ContainerResources:
    resources = resources_for(host_plan(), gpu_ids)
    logger.info(f"Allocating {resources.describe()}")
    return resources


def pinned_helper_kwargs(gpu_ids: list[int] | None) -> dict:
    """A per-job helper container (downloader, uploader, dataset download) runs on its job's cores, inside the job's
    GPU reservation, with no memory limit of its own."""
    if not gpu_ids:
        return {}
    resources = container_resources(gpu_ids)
    return {"cpuset_cpus": resources.cpuset_cpus} if resources.cpus else {}
