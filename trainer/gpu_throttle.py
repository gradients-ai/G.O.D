"""Pure throttle-detection logic for the trainer GPU monitor (no NVML dependency)."""

from enum import IntFlag

from pydantic import BaseModel
from pydantic import Field


class ClockEventReason(IntFlag):
    """Mirror of NVML's nvmlClocksEventReason* bitmask."""

    GPU_IDLE = 0x1
    APPLICATIONS_CLOCKS_SETTING = 0x2
    SW_POWER_CAP = 0x4
    HW_SLOWDOWN = 0x8
    SYNC_BOOST = 0x10
    SW_THERMAL_SLOWDOWN = 0x20
    HW_THERMAL_SLOWDOWN = 0x40
    HW_POWER_BRAKE_SLOWDOWN = 0x80
    DISPLAY_CLOCK_SETTING = 0x100


# Reasons that mean the GPU is genuinely being slowed down. SW_POWER_CAP is excluded by
# default: a fully loaded H100 sits on its power cap routinely and that is not a fault.
DEFAULT_ALERT_REASONS = (
    ClockEventReason.HW_SLOWDOWN
    | ClockEventReason.SW_THERMAL_SLOWDOWN
    | ClockEventReason.HW_THERMAL_SLOWDOWN
    | ClockEventReason.HW_POWER_BRAKE_SLOWDOWN
)


class GpuSample(BaseModel):
    index: int
    uuid: str
    name: str
    reasons: int
    sm_clock_mhz: int
    max_sm_clock_mhz: int
    temperature_c: int
    power_w: float
    power_limit_w: float
    utilization_pct: int


class GpuThrottleState(BaseModel):
    throttling: bool = False
    consecutive_bad: int = 0
    consecutive_good: int = 0
    active_reasons: int = 0


class MonitorThresholds(BaseModel):
    alert_reasons: int = int(DEFAULT_ALERT_REASONS)
    # A poll only counts as throttled if a bad reason is set AND the SM clock is actually depressed;
    # a reason bit with clocks near max costs nothing and is the main false-positive source.
    max_clock_ratio: float = Field(default=0.9, gt=0, le=1)
    # At 5 s polls: alert after 10 s sustained, declare recovery only after 5 min clean, so a
    # flapping GPU produces one alert per episode rather than a stream.
    trigger_polls: int = Field(default=2, ge=1)
    recover_polls: int = Field(default=60, ge=1)


class GpuJob(BaseModel):
    """A trainer job (training or model prep) occupying a GPU, as recorded in task_history.json."""

    job_type: str
    task_id: str
    hotkey: str | None
    detail: str | None = None  # training: github_repo@commit; model_prep: model_id


class ThrottleAlert(BaseModel):
    recovered: bool
    sample: GpuSample
    reasons: int


def reason_names(mask: int) -> list[str]:
    return [r.name for r in ClockEventReason if r.name and mask & r.value]


def evaluate(
    state: GpuThrottleState, sample: GpuSample, thresholds: MonitorThresholds
) -> tuple[GpuThrottleState, ThrottleAlert | None]:
    """Advance one GPU's hysteresis state by one poll; return an alert only on a transition."""
    bad = sample.reasons & thresholds.alert_reasons
    clock_depressed = sample.max_sm_clock_mhz > 0 and sample.sm_clock_mhz < thresholds.max_clock_ratio * sample.max_sm_clock_mhz
    if not clock_depressed:
        bad = 0
    new = state.model_copy()

    if bad:
        new.consecutive_bad += 1
        new.consecutive_good = 0
        new.active_reasons |= bad
        if not new.throttling and new.consecutive_bad >= thresholds.trigger_polls:
            new.throttling = True
            return new, ThrottleAlert(recovered=False, sample=sample, reasons=new.active_reasons)
        return new, None

    new.consecutive_good += 1
    new.consecutive_bad = 0
    if not new.throttling:
        new.active_reasons = 0
        return new, None
    if new.consecutive_good >= thresholds.recover_polls:
        alert = ThrottleAlert(recovered=True, sample=sample, reasons=new.active_reasons)
        return GpuThrottleState(), alert
    return new, None


def _format_job(job: GpuJob) -> str:
    line = f"{job.job_type} task `{job.task_id}` hotkey `{job.hotkey or '-'}`"
    return f"{line} ({job.detail})" if job.detail else line


def format_alert(host: str, alert: ThrottleAlert, jobs: list[GpuJob]) -> str:
    s = alert.sample
    reasons = ", ".join(reason_names(alert.reasons)) or "none"
    clock_pct = 100 * s.sm_clock_mhz / s.max_sm_clock_mhz if s.max_sm_clock_mhz else 0.0
    head = (
        f"✅ **{host}** GPU {s.index} recovered from throttling"
        if alert.recovered
        else f"🔥 **{host}** GPU {s.index} is THROTTLING"
    )
    return (
        f"{head}\n"
        f"reasons: `{reasons}`\n"
        f"{s.name} · SM {s.sm_clock_mhz}/{s.max_sm_clock_mhz} MHz ({clock_pct:.0f}%) · "
        f"{s.temperature_c}°C · {s.power_w:.0f}/{s.power_limit_w:.0f} W · util {s.utilization_pct}%\n"
        f"uuid `{s.uuid}`\n" + ("\n".join(f"job: {_format_job(j)}" for j in jobs) if jobs else "job: none (GPU not assigned)")
    )
