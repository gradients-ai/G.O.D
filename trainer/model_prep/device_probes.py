"""
Device probes for quant model prep: small fixed workloads that give the unit times of the prep device.

Host probes (CPU, identity) need no torch. The GPU probes run on the loaded model. Every probe catches
its own exceptions and returns None on failure: a failed probe never fails prep.
"""

import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from core.models.model_prep_models import DeviceProbeStats
from core.models.model_prep_models import ThroughputStats


# Host probes are unit-tested without torch, so the import is optional.
try:
    import torch
except ImportError:
    torch = None


# The constants below define the fixed CPU workload. Changing any of them makes new readings incomparable with
# recorded ones.
CPU_PROBE_SEED = 1234
CPU_PROBE_ELEMENTS = 4_000_000
CPU_PROBE_ROUNDS = 14
CPU_PROBE_REPEATS = 1
CPU_PROBE_BINS = 4_096

TRAIN_MIN_SECONDS = 20.0
TRAIN_WINDOW_SECONDS = 4.0
TRAIN_WARMUP_STEPS = 3
TRAIN_MIN_WINDOWS = 5
TRAIN_HARD_CAP_FACTOR = 3.0


class HostIdentity(BaseModel):
    """CPU and RAM of the container."""

    cpu_model: str | None = None
    cpu_count: int | None = None
    cpu_quota: float | None = None
    ram_gb: float | None = None


def _log(name: str, value: str, seconds: float) -> None:
    print(f"[probe] {name}: {value} ({seconds:.1f}s)", flush=True)


def _fail(name: str, exc: BaseException) -> None:
    print(f"[probe] {name} failed: {type(exc).__name__}: {exc}", flush=True)


# --- Host probes (no torch) ---

def measure_cpu_single_thread(
    elements: int = CPU_PROBE_ELEMENTS, rounds: int = CPU_PROBE_ROUNDS, repeats: int = CPU_PROBE_REPEATS
) -> float | None:
    """Seconds for a fixed single-threaded numpy workload (sort + cumsum + bincount); min over repeats.

    Uses no BLAS, so BLAS thread settings do not affect it. Lower is faster.
    """
    t_start = time.time()
    try:
        data = np.random.default_rng(CPU_PROBE_SEED).integers(0, 1 << 30, size=elements, dtype=np.int64)
        best = math.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            work = data
            checksum = 0
            for _ in range(rounds):
                work = np.sort(work)[::-1].copy()
                checksum += int(np.cumsum(work % 1009)[-1])
                checksum += int(np.bincount(work % CPU_PROBE_BINS, minlength=CPU_PROBE_BINS).max())
                work = work ^ (checksum & 0xFFFF)
            best = min(best, time.perf_counter() - t0)
        _log("cpu_single_thread", f"{best:.3f}s", time.time() - t_start)
        return best
    except Exception as exc:
        _fail("cpu_single_thread", exc)
        return None


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cgroup_cpu_quota() -> float | None:
    cpu_max = _read_text("/sys/fs/cgroup/cpu.max")
    if cpu_max is not None:
        parts = cpu_max.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                return int(parts[0]) / int(parts[1])
            except (ValueError, ZeroDivisionError):
                return None
        return None
    quota = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        if quota is not None and period is not None and int(quota) > 0:
            return int(quota) / int(period)
    except (ValueError, ZeroDivisionError):
        pass
    return None


def _ram_gb() -> float | None:
    limit = _read_text("/sys/fs/cgroup/memory.max")
    if limit is not None and limit.isdigit():
        return int(limit) / 2**30
    meminfo = _read_text("/proc/meminfo")
    if meminfo is not None:
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2**20
    return None


def host_identity() -> HostIdentity:
    """CPU model, visible CPUs, cgroup CPU quota (cores) and RAM (cgroup limit, else host) of this container."""
    identity = HostIdentity()
    try:
        cpuinfo = _read_text("/proc/cpuinfo") or ""
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                identity.cpu_model = line.split(":", 1)[1].strip()
                break
        identity.cpu_count = len(os.sched_getaffinity(0))
        identity.cpu_quota = _cgroup_cpu_quota()
        identity.ram_gb = _ram_gb()
    except Exception as exc:
        _fail("host_identity", exc)
    return identity


# --- GPU probes (torch) ---

def _sync(device) -> None:
    """Wait for queued CUDA work; every GPU, since a multi-GPU model is split across them (device_map="auto")."""
    if torch.device(device).type == "cuda":
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


def _empty_cache() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def measure_train_throughput_windows(
    model,
    device,
    seq_len: int,
    vocab_size: int,
    micro_batch: int,
    min_seconds: float = TRAIN_MIN_SECONDS,
    window_seconds: float = TRAIN_WINDOW_SECONDS,
    warmup_steps: int = TRAIN_WARMUP_STEPS,
) -> tuple[float, list[float], float] | None:
    """Fwd+bwd tokens/s in timed windows on the loaded model: (median window rate, window rates, timed seconds).

    Same step and shape as stats.measure_training_throughput. Runs until min_seconds are timed over at least
    5 windows, and stops at 3x min_seconds of wall time regardless. Returns None on OOM or failure.
    """
    t_start = time.time()
    was_training = model.training
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    def _step() -> None:
        input_ids = torch.randint(0, vocab_size, (micro_batch, seq_len), device=device)
        outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids)
        outputs.loss.backward()
        model.zero_grad(set_to_none=True)

    result = None
    try:
        for _ in range(warmup_steps):
            _step()
        _sync(device)
        rates: list[float] = []
        timed = 0.0
        wall_start = time.perf_counter()
        window_start = wall_start
        steps = 0
        while True:
            _step()
            steps += 1
            now = time.perf_counter()
            if now - window_start >= window_seconds:
                _sync(device)
                now = time.perf_counter()
                seconds = now - window_start
                rates.append(steps * micro_batch * seq_len / max(seconds, 1e-9))
                timed += seconds
                window_start = now
                steps = 0
                if timed >= min_seconds and len(rates) >= TRAIN_MIN_WINDOWS:
                    break
            if now - wall_start >= TRAIN_HARD_CAP_FACTOR * min_seconds:
                break
        if rates:
            result = (statistics.median(rates), rates, timed)
            _log("train_windows", f"{result[0]:.0f} tok/s over {len(rates)} windows", time.time() - t_start)
    except torch.cuda.OutOfMemoryError:
        print("[probe] train_windows OOM, skipped", flush=True)
    except Exception as exc:
        _fail("train_windows", exc)
    finally:
        model.zero_grad(set_to_none=True)
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if not was_training:
            model.eval()
        _empty_cache()
    return result


def gpu_identity(device) -> tuple[str, int, float] | None:
    """(name, SM count, VRAM GB) of the CUDA device; None on CPU."""
    try:
        if torch.device(device).type != "cuda":
            return None
        index = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(index)
        return torch.cuda.get_device_name(index), props.multi_processor_count, props.total_memory / 2**30
    except Exception as exc:
        _fail("gpu_identity", exc)
        return None


# --- Orchestration ---

def run_quant_device_probes(
    model, device, throughput: ThroughputStats | None, vocab_size: int
) -> DeviceProbeStats:
    """Run every probe and fill DeviceProbeStats. Never raises; a failed probe leaves its fields None."""
    t_all = time.time()
    stats = DeviceProbeStats()
    try:
        if throughput is not None:
            train = measure_train_throughput_windows(
                model, device, throughput.seq_len, vocab_size, throughput.micro_batch_size
            )
            if train is not None:
                stats.train_tokens_per_sec, stats.train_window_tokens_per_sec, stats.train_timed_seconds = train
        gpu = gpu_identity(device)
        if gpu is not None:
            stats.gpu_name, stats.gpu_sm_count, stats.gpu_vram_gb = gpu

        stats.cpu_single_thread_seconds = measure_cpu_single_thread()
        identity = host_identity()
        stats.cpu_model = identity.cpu_model
        stats.cpu_count = identity.cpu_count
        stats.cpu_quota = identity.cpu_quota
        stats.ram_gb = identity.ram_gb
    except Exception as exc:
        _fail("run_quant_device_probes", exc)
    stats.probe_seconds = time.time() - t_all
    print(
        f"[probe] done in {stats.probe_seconds:.1f}s: train={stats.train_tokens_per_sec} "
        f"cpu={stats.cpu_single_thread_seconds}",
        flush=True,
    )
    return stats
