"""Pure throttle-detection logic for the trainer GPU monitor (no NVML dependency)."""

from enum import IntFlag

from pydantic import BaseModel
from pydantic import Field


# Written into the trainer task log when a training is failed because its GPU faulted.
# The orchestrator sees this and requeues the training without charging another attempt.
GPU_THROTTLE_FAILURE_MARKER = "[GPU_THROTTLE]"


def logs_mark_gpu_throttle_failure(logs: list[str]) -> bool:
    return any(GPU_THROTTLE_FAILURE_MARKER in line for line in logs)


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
    timestamp_s: float  # monotonic clock
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
    # Cumulative ns throttled under NVML's thermal/reliability/board-limit policies (nvmlDeviceGetViolationStatus).
    # Zero on a healthy GPU; None if the driver doesn't expose it. Catches short bursts that fall between polls.
    fault_throttle_ns: int | None = None


class GpuThrottleState(BaseModel):
    throttling: bool = False
    consecutive_bad: int = 0
    consecutive_good: int = 0
    active_reasons: int = 0
    last_fault_throttle_ns: int | None = None
    fault_window: list[tuple[float, int]] = []  # (timestamp_s, ns throttled since previous poll)


class MonitorThresholds(BaseModel):
    alert_reasons: int = int(DEFAULT_ALERT_REASONS)
    # A poll only counts as throttled if a bad reason is set AND the SM clock is actually depressed;
    # a reason bit with clocks near max costs nothing and is the main false-positive source.
    max_clock_ratio: float = Field(default=0.9, gt=0, le=1)
    # At 5 s polls: alert after 10 s sustained, declare recovery only after 5 min clean, so a
    # flapping GPU produces one alert per episode rather than a stream.
    trigger_polls: int = Field(default=2, ge=1)
    recover_polls: int = Field(default=60, ge=1)
    # Second trigger from the cumulative counter: alert if this much fault-throttle time accumulates
    # within the window, even if no single poll ever sampled it.
    fault_window_seconds: float = Field(default=300, gt=0)
    fault_alert_seconds: float = Field(default=10, gt=0)


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
    fault_window_seconds: float  # fault-throttle time accumulated in the current window


class PollObservation(BaseModel):
    """What one poll saw, before hysteresis. Logged whenever non-clean to build a baseline."""

    sampled_reasons: int  # alert reasons set while the clock was actually depressed
    fault_delta_ns: int  # counter increase since the previous poll
    fault_window_seconds: float

    @property
    def clean(self) -> bool:
        return not self.sampled_reasons and not self.fault_delta_ns


def reason_names(mask: int) -> list[str]:
    return [r.name for r in ClockEventReason if r.name and mask & r.value]


def evaluate(
    state: GpuThrottleState, sample: GpuSample, thresholds: MonitorThresholds
) -> tuple[GpuThrottleState, PollObservation, ThrottleAlert | None]:
    """Advance one GPU's hysteresis state by one poll; return an alert only on a transition.

    Triggers on either `trigger_polls` consecutive sampled throttle polls, or `fault_alert_seconds` of
    fault-counter throttle time within `fault_window_seconds`. Recovers after `recover_polls` polls
    with neither a sampled reason nor any counter increase.
    """
    new = state.model_copy(deep=True)

    clock_depressed = sample.max_sm_clock_mhz > 0 and sample.sm_clock_mhz < thresholds.max_clock_ratio * sample.max_sm_clock_mhz
    sampled = sample.reasons & thresholds.alert_reasons if clock_depressed else 0

    delta = 0
    if sample.fault_throttle_ns is not None:
        last = state.last_fault_throttle_ns
        # A counter that went backwards (driver reload, GPU reset) re-baselines instead of counting.
        if last is not None and sample.fault_throttle_ns >= last:
            delta = sample.fault_throttle_ns - last
        new.last_fault_throttle_ns = sample.fault_throttle_ns
    cutoff = sample.timestamp_s - thresholds.fault_window_seconds
    new.fault_window = [(t, d) for t, d in new.fault_window if t > cutoff]
    if delta:
        new.fault_window.append((sample.timestamp_s, delta))
    window_s = sum(d for _, d in new.fault_window) / 1e9
    obs = PollObservation(sampled_reasons=sampled, fault_delta_ns=delta, fault_window_seconds=window_s)

    new.consecutive_bad = new.consecutive_bad + 1 if sampled else 0
    new.consecutive_good = new.consecutive_good + 1 if obs.clean else 0
    new.active_reasons |= sampled

    if not new.throttling:
        if new.consecutive_bad >= thresholds.trigger_polls or window_s >= thresholds.fault_alert_seconds:
            new.throttling = True
            return (
                new,
                obs,
                ThrottleAlert(recovered=False, sample=sample, reasons=new.active_reasons, fault_window_seconds=window_s),
            )
        if not sampled:
            new.active_reasons = 0
        return new, obs, None

    if new.consecutive_good >= thresholds.recover_polls:
        alert = ThrottleAlert(recovered=True, sample=sample, reasons=new.active_reasons, fault_window_seconds=window_s)
        return GpuThrottleState(last_fault_throttle_ns=new.last_fault_throttle_ns), obs, alert
    return new, obs, None


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
        f"reasons: `{reasons}` · fault-throttle time in window: {alert.fault_window_seconds:.1f} s\n"
        f"{s.name} · SM {s.sm_clock_mhz}/{s.max_sm_clock_mhz} MHz ({clock_pct:.0f}%) · "
        f"{s.temperature_c}°C · {s.power_w:.0f}/{s.power_limit_w:.0f} W · util {s.utilization_pct}%\n"
        f"uuid `{s.uuid}`\n" + ("\n".join(f"job: {_format_job(j)}" for j in jobs) if jobs else "job: none (GPU not assigned)")
    )
