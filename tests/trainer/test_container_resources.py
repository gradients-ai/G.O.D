"""Per-GPU CPU and memory shares of the trainer host, against fake hosts."""

import math
from pathlib import Path

import pytest

import trainer.constants as cst
from trainer import host_resources
from trainer.host_resources import THREAD_ENV_VARS
from trainer.host_resources import HostPlan
from trainer.host_resources import HostTopology
from trainer.host_resources import format_cpu_list
from trainer.host_resources import parse_cpu_list
from trainer.host_resources import parse_nvidia_smi_bus_ids
from trainer.host_resources import plan_host
from trainer.host_resources import read_host_topology
from trainer.host_resources import resources_for


GIB = 1024**3


def _host(
    n_cpus: int,
    n_gpus: int,
    hyperthreads: str | None = None,
    memory_gib: int = 256,
    gpu_local_cpus: list[list[int] | None] | None = None,
) -> HostTopology:
    """A fake host. hyperthreads: None (one CPU per core), "split" (bare metal: i and i + n/2), "adjacent" (VM: 2i, 2i+1)."""
    cpus = list(range(n_cpus))
    if hyperthreads == "split":
        half = n_cpus // 2
        cores = [[i, i + half] for i in range(half)]
    elif hyperthreads == "adjacent":
        cores = [[2 * i, 2 * i + 1] for i in range(n_cpus // 2)]
    else:
        cores = [[cpu] for cpu in cpus]
    return HostTopology(
        cpus=cpus,
        cores=cores,
        memory_bytes=memory_gib * GIB,
        gpu_local_cpus=gpu_local_cpus if gpu_local_cpus is not None else [None] * n_gpus,
    )


def _core_of(topology: HostTopology) -> dict[int, int]:
    return {cpu: i for i, core in enumerate(topology.cores) for cpu in core}


def _assert_fair_and_disjoint(topology: HostTopology, plan: HostPlan) -> None:
    shares = list(plan.gpu_cpus.values())
    assert len(shares) == len(topology.gpu_local_cpus)
    assert len({len(share) for share in shares}) == 1, "every GPU gets the same number of CPUs"
    assert all(len(share) == plan.cpus_per_gpu > 0 for share in shares)
    seen: set[int] = set()
    for share in shares:
        assert not seen & set(share), "shares are disjoint"
        seen |= set(share)
    assert seen <= set(topology.cpus)
    assert set(plan.host_cpus) == set(topology.cpus) - seen
    assert plan.host_cpus, "the host keeps a reserve"
    # No physical core is split between two GPUs or between a GPU and the host.
    core_of = _core_of(topology)
    owners: dict[int, int] = {}
    for gpu, share in plan.gpu_cpus.items():
        for cpu in share:
            assert owners.setdefault(core_of[cpu], gpu) == gpu
    assert not {core_of[cpu] for cpu in plan.host_cpus} & set(owners)


@pytest.mark.parametrize("hyperthreads", [None, "split", "adjacent"])
@pytest.mark.parametrize(
    ("n_cpus", "n_gpus"),
    [(40, 2), (48, 2), (96, 4), (192, 8), (208, 8), (30, 4), (26, 3), (6, 2)],
)
def test_every_gpu_gets_an_equal_disjoint_share(n_cpus: int, n_gpus: int, hyperthreads: str | None):
    topology = _host(n_cpus, n_gpus, hyperthreads)
    plan = plan_host(topology)

    _assert_fair_and_disjoint(topology, plan)
    threads_per_core = 1 if hyperthreads is None else 2
    n_cores = n_cpus // threads_per_core
    reserve = max(1, math.ceil(n_cores * cst.HOST_CPU_RESERVE_FRACTION))
    assert plan.cpus_per_gpu == (n_cores - reserve) // n_gpus * threads_per_core


@pytest.mark.parametrize(("n_cpus", "expected_per_gpu"), [(41, 19), (40, 19), (48, 23), (13, 6)])
def test_odd_core_counts(n_cpus: int, expected_per_gpu: int):
    plan = plan_host(_host(n_cpus, 2))

    assert plan.cpus_per_gpu == expected_per_gpu
    _assert_fair_and_disjoint(_host(n_cpus, 2), plan)


def test_prod_box_two_one_gpu_jobs_split_the_box():
    """2x H100 PCIe, 40 vCPU (20 hyperthreaded cores): two 1-GPU jobs never share a core."""
    topology = _host(40, 2, "adjacent")
    plan = plan_host(topology)

    a, b = resources_for(plan, [0]), resources_for(plan, [1])

    assert a.cpu_count == b.cpu_count == 18
    assert not set(a.cpus) & set(b.cpus)
    assert a.cpu_count + b.cpu_count <= 40


@pytest.mark.parametrize("n_cpus", [40, 48])
def test_two_gpu_job_on_two_gpu_box_gets_the_whole_box(n_cpus: int):
    topology = _host(n_cpus, 2, "adjacent", memory_gib=480)
    plan = plan_host(topology)

    job = resources_for(plan, [0, 1])

    assert set(job.cpus) == set(topology.cpus) - set(plan.host_cpus)
    assert job.cpu_count == 2 * plan.cpus_per_gpu
    assert job.memory_bytes == 2 * plan.memory_bytes_per_gpu
    assert topology.memory_bytes - job.memory_bytes == pytest.approx(0.1 * topology.memory_bytes, abs=2)


def test_job_gets_the_union_of_its_gpus_shares():
    plan = plan_host(_host(192, 8, "split"))

    job = resources_for(plan, [2, 5])

    assert job.cpus == sorted(plan.gpu_cpus[2] + plan.gpu_cpus[5])
    assert job.cpu_count == 2 * plan.cpus_per_gpu


@pytest.mark.parametrize(
    ("memory_gib", "n_gpus", "expected_per_gpu_gib"),
    [
        (256, 2, (256 - 25.6) / 2),
        (1440, 8, (1440 - 144) / 8),
        (100, 2, (100 - cst.HOST_MEMORY_RESERVE_MIN_GB) / 2),  # the floor beats 10% on a small box
    ],
)
def test_memory_share(memory_gib: int, n_gpus: int, expected_per_gpu_gib: float):
    plan = plan_host(_host(48, n_gpus, memory_gib=memory_gib))

    assert plan.memory_bytes_per_gpu / GIB == pytest.approx(expected_per_gpu_gib, abs=1e-6)
    assert resources_for(plan, [0]).memory_bytes == plan.memory_bytes_per_gpu


def test_pinned_job_docker_kwargs_and_thread_env():
    job = resources_for(plan_host(_host(40, 2)), [1])

    kwargs = job.docker_kwargs()

    assert kwargs == {"mem_limit": job.memory_bytes, "cpuset_cpus": format_cpu_list(job.cpus)}
    assert "nano_cpus" not in kwargs, "a cpuset needs no CFS quota on top"
    env = job.thread_env()
    assert set(env) == set(THREAD_ENV_VARS)
    assert set(env.values()) == {str(job.cpu_count)}
    assert {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"} <= set(env)


def test_multi_gpu_job_sizes_thread_pools_per_gpu():
    """One process per GPU (torchrun, accelerate): each sized to one share, not the whole job."""
    plan = plan_host(_host(192, 8, "split"))

    job = resources_for(plan, [0, 1, 2, 3])

    assert job.cpu_count == 4 * plan.cpus_per_gpu
    assert set(job.thread_env().values()) == {str(plan.cpus_per_gpu)}


def test_numa_aware_split_keeps_each_gpu_on_its_socket():
    """2 sockets x 48 cores with hyperthreads (cpu i and i + 96 are siblings); GPUs 0-3 on socket 0, 4-7 on socket 1."""
    socket0 = list(range(0, 48)) + list(range(96, 144))
    socket1 = list(range(48, 96)) + list(range(144, 192))
    topology = _host(192, 8, "split", gpu_local_cpus=[socket0] * 4 + [socket1] * 4)

    plan = plan_host(topology)

    assert plan.numa_aware
    _assert_fair_and_disjoint(topology, plan)
    for gpu in range(4):
        assert set(plan.gpu_cpus[gpu]) <= set(socket0)
    for gpu in range(4, 8):
        assert set(plan.gpu_cpus[gpu]) <= set(socket1)


def test_uneven_numa_falls_back_to_an_even_split():
    """Three GPUs on a socket too small for three shares: fairness wins over locality."""
    socket0 = list(range(0, 20))
    socket1 = list(range(20, 40))
    topology = _host(40, 4, gpu_local_cpus=[socket0] * 3 + [socket1])

    plan = plan_host(topology)

    assert not plan.numa_aware
    _assert_fair_and_disjoint(topology, plan)


@pytest.mark.parametrize(
    "gpu_local_cpus",
    [
        [list(range(0, 24)), None],  # one GPU reports no affinity
        [list(range(0, 30)), list(range(10, 48))],  # overlapping, not identical
    ],
)
def test_unusable_affinity_falls_back_to_an_even_split(gpu_local_cpus):
    topology = _host(48, 2, gpu_local_cpus=gpu_local_cpus)

    plan = plan_host(topology)

    assert not plan.numa_aware
    _assert_fair_and_disjoint(topology, plan)


def test_single_numa_domain_matches_the_even_split():
    """A VM reports every CPU local to every GPU: the same split as with no affinity at all."""
    with_affinity = plan_host(_host(40, 2, "adjacent", gpu_local_cpus=[list(range(40))] * 2))
    without = plan_host(_host(40, 2, "adjacent"))

    assert with_affinity.gpu_cpus == without.gpu_cpus
    assert not with_affinity.numa_aware


def test_fewer_cores_than_gpus_shares_by_quota():
    plan = plan_host(_host(4, 8))

    assert plan.gpu_cpus == {}
    job = resources_for(plan, [3])
    assert job.cpus == []
    assert job.docker_kwargs() == {"mem_limit": job.memory_bytes, "nano_cpus": 1_000_000_000}
    assert job.thread_env()["OMP_NUM_THREADS"] == "1"


def test_tiny_box_keeps_no_cpu_reserve_rather_than_starve_a_gpu():
    plan = plan_host(_host(2, 2))

    assert [len(cpus) for cpus in plan.gpu_cpus.values()] == [1, 1]
    assert plan.host_cpus == []


def test_rejects_gpus_not_on_the_host():
    plan = plan_host(_host(40, 2))

    with pytest.raises(ValueError):
        resources_for(plan, [2])
    with pytest.raises(ValueError):
        resources_for(plan, [])


@pytest.mark.parametrize(
    ("text", "cpus"),
    [("0-3,8,10-11\n", [0, 1, 2, 3, 8, 10, 11]), ("5", [5]), ("0,20", [0, 20]), ("", [])],
)
def test_cpu_list_round_trip(text: str, cpus: list[int]):
    assert parse_cpu_list(text) == cpus
    assert parse_cpu_list(format_cpu_list(cpus)) == cpus


def test_read_host_topology_from_sysfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """8 CPUs, siblings (i, i + 4); GPU 0 local to 0-1,4-5 and GPU 1 to 2-3,6-7; the trainer may use all of them."""
    cpu_root, pci_root = tmp_path / "cpu", tmp_path / "pci"
    for cpu in range(8):
        topo = cpu_root / f"cpu{cpu}" / "topology"
        topo.mkdir(parents=True)
        core = cpu % 4
        (topo / "thread_siblings_list").write_text(f"{core},{core + 4}\n")
    for bus, local in (("0000:18:00.0", "0-1,4-5"), ("0000:af:00.0", "2-3,6-7")):
        (pci_root / bus).mkdir(parents=True)
        (pci_root / bus / "local_cpulist").write_text(local + "\n")
    monkeypatch.setattr(host_resources.os, "sched_getaffinity", lambda pid: set(range(8)))

    topology = read_host_topology(cpu_root, pci_root, bus_ids=["00000000:18:00.0", "00000000:AF:00.0"])

    assert topology.cores == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert topology.gpu_local_cpus == [[0, 1, 4, 5], [2, 3, 6, 7]]
    plan = plan_host(topology)
    assert plan.numa_aware
    assert plan.gpu_cpus == {0: [1, 5], 1: [3, 7]}
    assert plan.host_cpus == [0, 2, 4, 6]


def test_parse_nvidia_smi_bus_ids():
    out = "1, 00000000:AF:00.0\n0, 00000000:18:00.0\n\n"

    assert parse_nvidia_smi_bus_ids(out) == ["00000000:18:00.0", "00000000:AF:00.0"]


def test_nvidia_smi_failure_falls_back_to_the_driver_gpu_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fail():
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(host_resources, "_nvidia_smi_bus_ids", fail)
    monkeypatch.setattr(host_resources.os, "sched_getaffinity", lambda pid: set(range(40)))
    proc_gpus = tmp_path / "gpus"
    for bus in ("0000:01:00.0", "0000:02:00.0"):
        (proc_gpus / bus).mkdir(parents=True)

    topology = read_host_topology(tmp_path / "none", tmp_path / "none", proc_gpus)

    assert topology.gpu_local_cpus == [None, None]
    _assert_fair_and_disjoint(topology, plan_host(topology))


def test_no_gpus_visible_at_all_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fail():
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(host_resources, "_nvidia_smi_bus_ids", fail)

    with pytest.raises(RuntimeError):
        read_host_topology(tmp_path / "none", tmp_path / "none", tmp_path / "none")


def test_read_host_topology_without_sysfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(host_resources.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})

    topology = read_host_topology(tmp_path / "none", tmp_path / "none", bus_ids=["00000000:18:00.0"])

    assert topology.cores == [[0], [1], [2], [3]]
    assert topology.gpu_local_cpus == [None]
