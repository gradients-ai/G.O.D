from trainer.gpu_throttle import ClockEventReason
from trainer.gpu_throttle import GpuJob
from trainer.gpu_throttle import GpuSample
from trainer.gpu_throttle import GpuThrottleState
from trainer.gpu_throttle import MonitorThresholds
from trainer.gpu_throttle import ThrottleAlert
from trainer.gpu_throttle import evaluate
from trainer.gpu_throttle import format_alert


T = MonitorThresholds(trigger_polls=3, recover_polls=2, fault_window_seconds=60, fault_alert_seconds=10)
HW = int(ClockEventReason.HW_SLOWDOWN)
CAP = int(ClockEventReason.SW_POWER_CAP)
S = 1_000_000_000  # ns per second


def _sample(t: float, reasons: int = 0, sm_clock_mhz: int = 1200, fault_ns: int | None = 0) -> GpuSample:
    return GpuSample(
        timestamp_s=t,
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
        fault_throttle_ns=fault_ns,
    )


def _run(samples: list[GpuSample]) -> list[tuple[int, bool]]:
    state, out = GpuThrottleState(), []
    for step, sample in enumerate(samples):
        state, _, alert = evaluate(state, sample, T)
        if alert:
            out.append((step, alert.recovered))
    return out


def _reasons(seq: list[int]) -> list[GpuSample]:
    return [_sample(5.0 * i, r) for i, r in enumerate(seq)]


def _counter(cumulative_s: list[float]) -> list[GpuSample]:
    return [_sample(5.0 * i, fault_ns=int(c * S)) for i, c in enumerate(cumulative_s)]


def test_alerts_after_sustained_throttle_and_recovers():
    assert _run(_reasons([HW, HW, HW, HW, 0, 0])) == [(2, False), (5, True)]


def test_transient_blip_does_not_alert():
    assert _run(_reasons([HW, HW, 0, HW, HW, 0])) == []


def test_power_cap_is_ignored_by_default():
    assert _run(_reasons([CAP] * 10)) == []


def test_flapping_during_throttle_does_not_spam_recovery():
    assert _run(_reasons([HW, HW, HW, 0, HW, 0, HW])) == [(2, False)]


def test_reason_bit_with_near_max_clock_is_ignored():
    assert _run([_sample(5.0 * i, HW, sm_clock_mhz=1900) for i in range(10)]) == []


def test_counter_bursts_between_polls_accumulate_to_alert():
    # 3 s of throttling per burst, never sampled by the bitmask; the 4th burst crosses 10 s in the window.
    assert _run(_counter([100, 103, 103, 106, 106, 109, 112, 112, 112])) == [(6, False), (8, True)]


def test_counter_baseline_and_slow_drip_do_not_alert():
    # A large pre-existing counter value is a baseline, not throttling; 2 s per minute never reaches 10 s / 60 s.
    assert _run(_counter([5000 + 2 * (i // 12) for i in range(60)])) == []


def test_counter_reset_rebaselines():
    assert _run(_counter([5000, 0, 0, 1, 1])) == []


def test_counter_increase_blocks_recovery():
    samples = _counter([0, 11, 11, 11.5, 11.5, 11.5])
    assert _run(samples) == [(1, False), (5, True)]


def test_unsupported_counter_is_ignored():
    assert _run([_sample(5.0 * i, fault_ns=None) for i in range(5)]) == []


def test_format_names_reasons_and_jobs():
    alert = ThrottleAlert(
        recovered=False,
        sample=_sample(0, HW | int(ClockEventReason.HW_THERMAL_SLOWDOWN)),
        reasons=HW | int(ClockEventReason.HW_THERMAL_SLOWDOWN),
        fault_window_seconds=4.2,
    )
    job = GpuJob(job_type="training", task_id="t-123", hotkey="5Hk", detail="org/repo@abcd1234")
    msg = format_alert("trainer-1", alert, [job])
    assert "THROTTLING" in msg and "HW_SLOWDOWN" in msg and "HW_THERMAL_SLOWDOWN" in msg and "4.2 s" in msg
    assert "t-123" in msg and "5Hk" in msg and "org/repo@abcd1234" in msg
    assert "GPU not assigned" in format_alert("trainer-1", alert, [])
