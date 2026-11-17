"""Trainer GPU monitor: clock throttle, ECC, NVLink, fabric manager, and hardware Xids.

Starts with the trainer process (`trainer.asgi` startup, which is what `task trainer` and the systemd
unit run). One loop ticks every GPU_MONITOR_POLL_SECONDS (default 5). Clock throttle alerts only when
a slowdown reason is set and the SM clock is below GPU_MONITOR_MAX_CLOCK_RATIO of max (default 0.9)
for GPU_MONITOR_TRIGGER_POLLS ticks (default 2, so 10 s). Temperature is reported in the message; it
is not a separate threshold, because a hot GPU at full clocks is normal and a cool GPU can still be
throttled. Tune the ratio from `gpu_throttle_observation` lines on a healthy loaded box.

`task trainer`, the trainer systemd unit, and `task bootstrap` install DCGM when it is missing.
The same tick asks DCGM (`dcgmi health -c a`) once that hostengine is up. A DCGM failure withdraws
that GPU. Warnings are logged. If the install fails, or the box has no nvidia-fabricmanager unit,
the NVML ECC, NVLink, and hardware-Xid checks run instead. Fabric manager is always a systemd check:
the unit has to be installed and inactive before every GPU on the box is withdrawn.
Env:
  DISCORD_WEBHOOK            webhook URL (without it: log-only)
  GPU_MONITOR_TASK_HISTORY   trainer task history file (default trainer/task_history.json, relative to cwd)
  GPU_MONITOR_HOST_LABEL     name shown in alerts (default: hostname)
  GPU_MONITOR_POLL_SECONDS   poll interval (default 5)
  GPU_MONITOR_TRIGGER_POLLS  consecutive throttled polls before alerting (default 2 = 10 s)
  GPU_MONITOR_RECOVER_POLLS  consecutive clean polls before a recovery message (default 60 = 5 min)
  GPU_MONITOR_MAX_CLOCK_RATIO  poll only counts if SM clock < ratio * max (default 0.9)
  GPU_MONITOR_INCLUDE_POWER_CAP  "1" to also alert on SW_POWER_CAP (default off)
  GPU_MONITOR_FAULT_WINDOW_SECONDS  window for throttle time and corrected ECC (default 300)
  GPU_MONITOR_FAULT_ALERT_SECONDS   fault-throttle seconds within the window that trigger an alert (default 10)
  GPU_MONITOR_ECC_SBE_THRESHOLD  new corrected ECC errors in the window that withdraw the GPU (default 10; 0 disables)
  GPU_MONITOR_DCGM           "0" to skip DCGM even when dcgmi is installed (default: use it when available)
"""

import asyncio
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pynvml

from core.logging import get_logger
from core.models.payload_models import TrainerTaskLog
from core.models.task_models import TaskStatus
from trainer import job_state
from trainer.gpu_blacklist import blacklist_gpu
from trainer.gpu_dcgm import query_dcgm_health
from trainer.gpu_fault import abort_trainings_on_gpu
from trainer.gpu_health import HARDWARE_XIDS
from trainer.gpu_health import HealthReading
from trainer.gpu_health import HealthState
from trainer.gpu_health import advance_health
from trainer.gpu_health import fabric_manager_is_down
from trainer.gpu_health import normalize_pci
from trainer.gpu_health import parse_xids
from trainer.gpu_throttle import DEFAULT_ALERT_REASONS
from trainer.gpu_throttle import ClockEventReason
from trainer.gpu_throttle import GpuJob
from trainer.gpu_throttle import GpuSample
from trainer.gpu_throttle import GpuThrottleState
from trainer.gpu_throttle import MonitorThresholds
from trainer.gpu_throttle import evaluate
from trainer.gpu_throttle import format_alert
from trainer.gpu_throttle import reason_names


logger = get_logger(__name__)

# Event loop of the trainer API, set when the monitor starts beside it. Standalone runs have none.
_trainer_loop: asyncio.AbstractEventLoop | None = None
# One announcement per GPU per fault kind, so a stuck counter cannot spam Discord every tick.
_announced: set[tuple[int, str]] = set()
_seen_xid_lines: set[str] = set()
_dcgm_mode: str | None = None

# Violation-counter policies that stay at zero on a healthy GPU. POWER is excluded: a loaded GPU
# accumulates power-cap time continuously (hours of it on a healthy H100).
FAULT_POLICIES = (pynvml.NVML_PERF_POLICY_THERMAL, pynvml.NVML_PERF_POLICY_RELIABILITY, pynvml.NVML_PERF_POLICY_BOARD_LIMIT)


def _str(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


def read_fault_throttle_ns(handle) -> int | None:
    total, supported = 0, False
    for policy in FAULT_POLICIES:
        try:
            total += pynvml.nvmlDeviceGetViolationStatus(handle, policy).violationTime
            supported = True
        except pynvml.NVMLError:
            continue
    return total if supported else None


def read_sample(index: int) -> GpuSample:
    h = pynvml.nvmlDeviceGetHandleByIndex(index)
    return GpuSample(
        timestamp_s=time.monotonic(),
        index=index,
        uuid=_str(pynvml.nvmlDeviceGetUUID(h)),
        name=_str(pynvml.nvmlDeviceGetName(h)),
        reasons=pynvml.nvmlDeviceGetCurrentClocksEventReasons(h),
        sm_clock_mhz=pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
        max_sm_clock_mhz=pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM),
        temperature_c=pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
        power_w=pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
        power_limit_w=pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000,
        utilization_pct=pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
        fault_throttle_ns=read_fault_throttle_ns(h),
    )


def _ecc_total(handle, error_type: int) -> int | None:
    try:
        return int(pynvml.nvmlDeviceGetTotalEccErrors(handle, error_type, pynvml.NVML_VOLATILE_ECC))
    except pynvml.NVMLError:
        return None


def _retirement_pending(handle) -> bool:
    fn = getattr(pynvml, "nvmlDeviceGetRetiredPagesPendingStatus", None)
    if fn is None:
        return False
    try:
        return bool(fn(handle))
    except pynvml.NVMLError:
        return False


def _nvlink_error_total(handle) -> int | None:
    get = getattr(pynvml, "nvmlDeviceGetNvLinkErrorCounter", None)
    counters = [
        getattr(pynvml, name, None)
        for name in (
            "NVML_NVLINK_ERROR_DL_CRC_DATA",
            "NVML_NVLINK_ERROR_DL_REPLAY",
            "NVML_NVLINK_ERROR_DL_RECOVERY",
        )
    ]
    counters = [c for c in counters if c is not None]
    if get is None or not counters:
        return None
    total, supported = 0, False
    for link in range(getattr(pynvml, "NVML_NVLINK_MAX_LINKS", 18)):
        for counter in counters:
            try:
                total += int(get(handle, link, counter))
                supported = True
            except pynvml.NVMLError:
                continue
    return total if supported else None


def _pci(handle) -> str:
    try:
        info = pynvml.nvmlDeviceGetPciInfo(handle)
    except pynvml.NVMLError:
        return ""
    bus = info.busId
    if isinstance(bus, bytes):
        bus = bus.decode()
    return normalize_pci(bus.split("\x00", 1)[0])


def read_health(index: int, xids: list[int]) -> HealthReading:
    h = pynvml.nvmlDeviceGetHandleByIndex(index)
    return HealthReading(
        timestamp_s=time.monotonic(),
        uncorrected=_ecc_total(h, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED),
        corrected=_ecc_total(h, pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED),
        retirement_pending=_retirement_pending(h),
        nvlink_errors=_nvlink_error_total(h),
        xids=xids,
    )


def _fabric_manager_state() -> tuple[str, str]:
    try:
        proc = subprocess.run(
            ["systemctl", "show", "nvidia-fabricmanager", "-p", "LoadState", "-p", "ActiveState"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "not-found", "inactive"
    load, active = "not-found", "inactive"
    for line in proc.stdout.splitlines():
        if line.startswith("LoadState="):
            load = line.split("=", 1)[1].strip()
        elif line.startswith("ActiveState="):
            active = line.split("=", 1)[1].strip()
    return load, active


def _recent_xid_lines() -> list[str]:
    try:
        proc = subprocess.run(
            ["journalctl", "-k", "-o", "cat", "--since", "10 seconds ago", "--no-pager"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    fresh = []
    for line in proc.stdout.splitlines():
        if "Xid" not in line or line in _seen_xid_lines:
            continue
        _seen_xid_lines.add(line)
        fresh.append(line)
    if len(_seen_xid_lines) > 500:
        _seen_xid_lines.clear()
    return fresh


def jobs_on_gpu(index: int) -> list[GpuJob]:
    """Re-read the trainer's task_history.json and return the running jobs assigned to this GPU index."""
    jobs = []
    for job in job_state.read_persisted_jobs():
        if job.status != TaskStatus.TRAINING or index not in job.gpu_ids:
            continue
        if isinstance(job, TrainerTaskLog):
            commit = (job.github_commit_hash or "")[:8]
            jobs.append(
                GpuJob(
                    job_type="training",
                    task_id=job.training_data.task_id,
                    hotkey=job.hotkey,
                    detail=f"{job.github_repo}@{commit}" if commit else job.github_repo,
                )
            )
        else:
            jobs.append(GpuJob(job_type="model_prep", task_id=job.task_id, hotkey=job.hotkey, detail=job.model_id))
    return jobs


def post_discord(client: httpx.Client, webhook: str | None, message: str) -> None:
    logger.warning(message)
    if not webhook:
        return
    try:
        client.post(webhook, json={"content": message}, timeout=10).raise_for_status()
    except httpx.HTTPError as e:
        logger.error(f"Discord post failed: {e}")


def main() -> None:
    webhook = os.getenv("DISCORD_WEBHOOK") or None
    host = os.getenv("GPU_MONITOR_HOST_LABEL") or socket.gethostname()
    poll_seconds = float(os.getenv("GPU_MONITOR_POLL_SECONDS", "5"))
    alert_reasons = DEFAULT_ALERT_REASONS
    if os.getenv("GPU_MONITOR_INCLUDE_POWER_CAP") == "1":
        alert_reasons |= ClockEventReason.SW_POWER_CAP
    thresholds = MonitorThresholds(
        alert_reasons=int(alert_reasons),
        max_clock_ratio=float(os.getenv("GPU_MONITOR_MAX_CLOCK_RATIO", "0.9")),
        trigger_polls=int(os.getenv("GPU_MONITOR_TRIGGER_POLLS", "2")),
        recover_polls=int(os.getenv("GPU_MONITOR_RECOVER_POLLS", "60")),
        fault_window_seconds=float(os.getenv("GPU_MONITOR_FAULT_WINDOW_SECONDS", "300")),
        fault_alert_seconds=float(os.getenv("GPU_MONITOR_FAULT_ALERT_SECONDS", "10")),
    )
    sbe_threshold = int(os.getenv("GPU_MONITOR_ECC_SBE_THRESHOLD", "10"))
    if history := os.getenv("GPU_MONITOR_TASK_HISTORY"):
        job_state.TASK_HISTORY_FILE = Path(history)
    if not webhook:
        logger.warning("DISCORD_WEBHOOK not set — throttle events will only be logged")

    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    states = {i: GpuThrottleState() for i in range(count)}
    health_states = {i: HealthState() for i in range(count)}
    logger.info(
        f"GPU monitor on {host}: {count} GPUs, poll {poll_seconds}s, sbe_threshold {sbe_threshold}, {thresholds}"
    )

    with httpx.Client() as client:
        post_discord(client, webhook, f"👀 GPU monitor started on **{host}** ({count} GPUs)")
        while True:
            dcgm = None if os.getenv("GPU_MONITOR_DCGM") == "0" else _poll_dcgm()
            xid_by_pci: dict[str, list[int]] = {}
            if dcgm is None:
                for pci, xid in parse_xids("\n".join(_recent_xid_lines())):
                    xid_by_pci.setdefault(pci, []).append(xid)
            pci_by_index: dict[int, str] = {}
            for i in range(count):
                try:
                    sample = read_sample(i)
                    if dcgm is None:
                        pci_by_index[i] = _pci(pynvml.nvmlDeviceGetHandleByIndex(i))
                except pynvml.NVMLError as e:
                    logger.error(f"NVML read failed for GPU {i}: {e}")
                    continue
                states[i], obs, alert = evaluate(states[i], sample, thresholds)
                if not obs.clean:
                    logger.info(
                        f"gpu_throttle_observation gpu={i} reasons={'|'.join(reason_names(obs.sampled_reasons)) or '-'} "
                        f"fault_delta_ms={obs.fault_delta_ns / 1e6:.1f} fault_window_s={obs.fault_window_seconds:.2f} "
                        f"sm={sample.sm_clock_mhz}/{sample.max_sm_clock_mhz} temp={sample.temperature_c} "
                        f"power={sample.power_w:.0f}/{sample.power_limit_w:.0f} util={sample.utilization_pct}"
                    )
                if alert and not alert.recovered:
                    blacklist_gpu(sample.index)
                    reason = ", ".join(reason_names(alert.reasons)) or "fault-throttle counter"
                    post_discord(client, webhook, format_alert(host, alert, jobs_on_gpu(i)))
                    _withdraw_gpu(sample.index, reason)
                elif alert and alert.recovered:
                    post_discord(client, webhook, format_alert(host, alert, jobs_on_gpu(i)))

                if dcgm is not None:
                    continue
                try:
                    reading = read_health(i, xid_by_pci.get(pci_by_index.get(i, ""), []))
                except pynvml.NVMLError as e:
                    logger.error(f"Health read failed for GPU {i}: {e}")
                    continue
                health_states[i], fault = advance_health(
                    health_states[i], reading, sbe_threshold, thresholds.fault_window_seconds
                )
                if fault:
                    _announce(client, webhook, host, i, fault.kind, fault.detail, jobs_on_gpu(i))
                window = health_states[i].corrected_window
                if not fault and window and window[-1][0] == reading.timestamp_s:
                    logger.info(
                        f"gpu_health_observation gpu={i} corrected_in_window="
                        f"{sum(d for _, d in window)} threshold={sbe_threshold}"
                    )
                ignored_xids = [xid for xid in reading.xids if xid not in HARDWARE_XIDS]
                if ignored_xids:
                    logger.info(f"gpu_health_observation gpu={i} xid={ignored_xids} (application xid, not withdrawn)")

            if dcgm is not None:
                for report in dcgm:
                    if report.gpu_index < 0 or report.gpu_index >= count:
                        logger.warning(f"DCGM GPU id {report.gpu_index} is outside 0..{count - 1}")
                        continue
                    for failure in report.failures:
                        _announce(
                            client,
                            webhook,
                            host,
                            report.gpu_index,
                            f"dcgm-{failure.name}",
                            f"DCGM {failure.name}: {failure.summary}",
                            jobs_on_gpu(report.gpu_index),
                        )
                    if report.warnings:
                        summary = "; ".join(item.summary for item in report.warnings)
                        logger.info(f"gpu_health_observation gpu={report.gpu_index} dcgm_warning={summary}")

            load_state, active_state = _fabric_manager_state()
            if fabric_manager_is_down(load_state, active_state):
                detail = f"nvidia-fabricmanager is {active_state}"
                for i in range(count):
                    _announce(client, webhook, host, i, "fabric", detail, jobs_on_gpu(i))
            unmatched = [
                xid
                for pci, xids in xid_by_pci.items()
                if pci not in pci_by_index.values()
                for xid in xids
                if xid in HARDWARE_XIDS
            ]
            if dcgm is None and unmatched:
                detail = "Xid " + ",".join(str(xid) for xid in unmatched) + " (no PCI match)"
                for i in range(count):
                    _announce(client, webhook, host, i, "xid-unmatched", detail, jobs_on_gpu(i))
            time.sleep(poll_seconds)


def _poll_dcgm():
    """DCGM health, or None so the caller uses the NVML ECC/Xid/NVLink path. Logged once per mode change."""
    global _dcgm_mode
    report = query_dcgm_health()
    mode = "on" if report is not None else "off"
    if mode != _dcgm_mode:
        if mode == "on":
            logger.info("DCGM health checks enabled")
        else:
            logger.warning("DCGM health unavailable; using NVML ECC, NVLink, and Xid checks")
        _dcgm_mode = mode
    return report


def _announce(client, webhook: str | None, host: str, gpu_index: int, kind: str, detail: str, jobs: list) -> None:
    blacklist_gpu(gpu_index)
    if (gpu_index, kind) in _announced:
        return
    _announced.add((gpu_index, kind))
    message = f"🔥 **{host}** GPU {gpu_index} fault: {detail}"
    if jobs:
        message += "\n" + "\n".join(f"job: {job.job_type} `{job.task_id}`" for job in jobs)
    post_discord(client, webhook, message)
    _withdraw_gpu(gpu_index, detail)


def _withdraw_gpu(gpu_index: int, reason: str) -> None:
    """Keep the GPU out of new work and fail trainings already on it."""
    coro = abort_trainings_on_gpu(gpu_index, reason)
    if _trainer_loop is not None and _trainer_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, _trainer_loop)
        future.add_done_callback(_log_abort_result)
        return
    try:
        asyncio.run(coro)
    except Exception:
        logger.exception(f"Failed to abort trainings on throttled GPU {gpu_index}")


def _log_abort_result(future: asyncio.Future) -> None:
    try:
        future.result()
    except Exception:
        logger.exception("Failed to abort trainings on a throttled GPU")


def _run_forever() -> None:
    try:
        main()
    except Exception:
        logger.exception("GPU throttle monitor exited")


def start_in_thread() -> None:
    """Run the monitor beside the trainer API. A failure here does not stop the API."""
    global _trainer_loop
    try:
        _trainer_loop = asyncio.get_running_loop()
    except RuntimeError:
        _trainer_loop = None
    threading.Thread(target=_run_forever, name="gpu-throttle-monitor", daemon=True).start()


if __name__ == "__main__":
    main()
