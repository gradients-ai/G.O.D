from trainer.gpu_throttle import ClockEventReason
from trainer.gpu_throttle import GpuJob
from trainer.gpu_throttle import GpuSample
from trainer.gpu_throttle import GpuThrottleState
from trainer.gpu_throttle import MonitorThresholds
from trainer.gpu_throttle import evaluate
from trainer.gpu_throttle import format_alert


T = MonitorThresholds(trigger_polls=3, recover_polls=2)


def _sample(reasons: int, sm_clock_mhz: int = 1200) -> GpuSample:
    return GpuSample(
        index=0,
        uuid="GPU-x",
        name="H100",
        reasons=reasons,
        sm_clock_mhz=sm_clock_mhz,
        max_sm_clock_mhz=1980,
        temperature_c=88,
        power_w=650.0,
        power_limit_w=700.0,
        utilization_pct=100,
    )


def _run(seq: list[int]) -> list[tuple[int, bool]]:
    state, out = GpuThrottleState(), []
    for step, reasons in enumerate(seq):
        state, alert = evaluate(state, _sample(reasons), T)
        if alert:
            out.append((step, alert.recovered))
    return out


HW = int(ClockEventReason.HW_SLOWDOWN)
CAP = int(ClockEventReason.SW_POWER_CAP)


def test_alerts_after_sustained_throttle_and_recovers():
    assert _run([HW, HW, HW, HW, 0, 0]) == [(2, False), (5, True)]


def test_transient_blip_does_not_alert():
    assert _run([HW, HW, 0, HW, HW, 0]) == []


def test_power_cap_is_ignored_by_default():
    assert _run([CAP] * 10) == []


def test_flapping_during_throttle_does_not_spam_recovery():
    assert _run([HW, HW, HW, 0, HW, 0, HW]) == [(2, False)]


def test_reason_bit_with_near_max_clock_is_ignored():
    state = GpuThrottleState()
    for _ in range(10):
        state, alert = evaluate(state, _sample(HW, sm_clock_mhz=1900), T)
        assert alert is None


def test_format_names_reasons():
    state = GpuThrottleState()
    alerts = []
    for _ in range(3):
        state, alert = evaluate(state, _sample(HW | int(ClockEventReason.HW_THERMAL_SLOWDOWN)), T)
        alerts.append(alert)
    alert = alerts[-1]
    assert alert is not None
    job = GpuJob(job_type="training", task_id="t-123", hotkey="5Hk", detail="org/repo@abcd1234")
    msg = format_alert("trainer-1", alert, [job])
    assert "THROTTLING" in msg and "HW_SLOWDOWN" in msg and "HW_THERMAL_SLOWDOWN" in msg
    assert "t-123" in msg and "5Hk" in msg and "org/repo@abcd1234" in msg
    assert "GPU not assigned" in format_alert("trainer-1", alert, [])
