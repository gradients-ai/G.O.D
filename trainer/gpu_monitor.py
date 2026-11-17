"""Trainer GPU throttle monitor: polls NVML clock-event reasons and alerts Discord on transitions.

Run on the trainer host from the repo root (see `task trainer-gpu-monitor`) so it reads the same
trainer/task_history.json as the trainer API — that is how alerts name the job and hotkey on the GPU. Env:
  DISCORD_WEBHOOK            webhook URL (required to alert; without it, logs only)
  GPU_MONITOR_HOST_LABEL     name shown in alerts (default: hostname)
  GPU_MONITOR_POLL_SECONDS   poll interval (default 5)
  GPU_MONITOR_TRIGGER_POLLS  consecutive throttled polls before alerting (default 6 = 30 s)
  GPU_MONITOR_RECOVER_POLLS  consecutive clean polls before a recovery message (default 60 = 5 min)
  GPU_MONITOR_MAX_CLOCK_RATIO  poll only counts if SM clock < ratio * max (default 0.9)
  GPU_MONITOR_INCLUDE_POWER_CAP  "1" to also alert on SW_POWER_CAP (default off)
"""

import os
import socket
import time

import httpx
import pynvml

from core.logging import get_logger
from core.models.payload_models import TrainerTaskLog
from core.models.task_models import TaskStatus
from trainer import job_state
from trainer.gpu_throttle import DEFAULT_ALERT_REASONS
from trainer.gpu_throttle import ClockEventReason
from trainer.gpu_throttle import GpuJob
from trainer.gpu_throttle import GpuSample
from trainer.gpu_throttle import GpuThrottleState
from trainer.gpu_throttle import MonitorThresholds
from trainer.gpu_throttle import evaluate
from trainer.gpu_throttle import format_alert


logger = get_logger(__name__)


def _str(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


def read_sample(index: int) -> GpuSample:
    h = pynvml.nvmlDeviceGetHandleByIndex(index)
    return GpuSample(
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
    )


def jobs_on_gpu(index: int) -> list[GpuJob]:
    """Re-read the trainer's task_history.json and return the running jobs assigned to this GPU index."""
    job_state.load_task_history()
    jobs = []
    for job in job_state.task_history:
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
        trigger_polls=int(os.getenv("GPU_MONITOR_TRIGGER_POLLS", "6")),
        recover_polls=int(os.getenv("GPU_MONITOR_RECOVER_POLLS", "60")),
    )
    if not webhook:
        logger.warning("DISCORD_WEBHOOK not set — throttle events will only be logged")

    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    states = {i: GpuThrottleState() for i in range(count)}
    logger.info(f"GPU monitor on {host}: {count} GPUs, poll {poll_seconds}s, {thresholds}")

    with httpx.Client() as client:
        post_discord(client, webhook, f"👀 GPU throttle monitor started on **{host}** ({count} GPUs)")
        while True:
            for i in range(count):
                try:
                    sample = read_sample(i)
                except pynvml.NVMLError as e:
                    logger.error(f"NVML read failed for GPU {i}: {e}")
                    continue
                states[i], alert = evaluate(states[i], sample, thresholds)
                if alert:
                    post_discord(client, webhook, format_alert(host, alert, jobs_on_gpu(i)))
            time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
