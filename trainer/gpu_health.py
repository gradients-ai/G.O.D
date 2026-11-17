"""ECC, NVLink, fabric-manager, and hardware Xid checks. No NVML dependency."""

import re

from pydantic import BaseModel


# Kernel Xids that mean the GPU or its fabric is broken. Application faults such as
# Xid 13 (graphics exception) and 31 (MMU fault) are omitted: a bad kernel raises those.
HARDWARE_XIDS = frozenset({48, 63, 64, 74, 79, 92, 94, 95, 119, 120, 123, 140, 154, 155})

_XID_RE = re.compile(r"Xid \(PCI:([0-9a-fA-F:\.]+)\):\s*(\d+)")


class HealthReading(BaseModel):
    timestamp_s: float
    uncorrected: int | None = None
    corrected: int | None = None
    retirement_pending: bool = False
    nvlink_errors: int | None = None
    xids: list[int] = []


class HealthState(BaseModel):
    uncorrected: int | None = None
    corrected: int | None = None
    corrected_window: list[tuple[float, int]] = []
    nvlink_errors: int | None = None


class HardwareFault(BaseModel):
    kind: str
    detail: str


def normalize_pci(bus_id: str) -> str:
    """Collapse NVML `00000000:81:00.0` and kernel `0000:81:00` to `81:00`."""
    body = bus_id.lower().split(".", 1)[0]
    parts = body.split(":")
    if len(parts) == 3 and set(parts[0]) <= {"0"}:
        return ":".join(parts[1:])
    return body


def parse_xids(text: str) -> list[tuple[str, int]]:
    """Pull (normalized pci, xid) pairs out of kernel log text."""
    found = []
    for pci, code in _XID_RE.findall(text):
        found.append((normalize_pci(pci), int(code)))
    return found


def fabric_manager_is_down(load_state: str, active_state: str) -> bool:
    """True only when the unit is installed and not active. A missing unit is a non-NVSwitch box."""
    if load_state.strip() == "not-found":
        return False
    return active_state.strip() != "active"


def _delta(previous: int | None, current: int | None) -> int:
    if previous is None or current is None or current < previous:
        return 0
    return current - previous


def advance_health(
    state: HealthState, reading: HealthReading, sbe_threshold: int, window_seconds: float
) -> tuple[HealthState, HardwareFault | None]:
    """Baseline counters on the first sample.

    Fault on a new DBE, pending retirement, NVLink errors, a hardware Xid, or too many new SBEs in the window.
    """
    new = state.model_copy(deep=True)
    uncorr_delta = _delta(state.uncorrected, reading.uncorrected)
    corr_delta = _delta(state.corrected, reading.corrected)
    nv_delta = _delta(state.nvlink_errors, reading.nvlink_errors)
    if reading.uncorrected is not None:
        new.uncorrected = reading.uncorrected
    if reading.corrected is not None:
        new.corrected = reading.corrected
    if reading.nvlink_errors is not None:
        new.nvlink_errors = reading.nvlink_errors

    cutoff = reading.timestamp_s - window_seconds
    new.corrected_window = [(t, d) for t, d in new.corrected_window if t > cutoff]
    if corr_delta:
        new.corrected_window.append((reading.timestamp_s, corr_delta))
    corrected_in_window = sum(d for _, d in new.corrected_window)

    fault = _fault(
        uncorrected_delta=uncorr_delta,
        corrected_in_window=corrected_in_window,
        sbe_threshold=sbe_threshold,
        retirement_pending=reading.retirement_pending,
        nvlink_error_delta=nv_delta,
        xids=reading.xids,
    )
    return new, fault


def _fault(
    *,
    uncorrected_delta: int,
    corrected_in_window: int,
    sbe_threshold: int,
    retirement_pending: bool,
    nvlink_error_delta: int,
    xids: list[int],
) -> HardwareFault | None:
    if uncorrected_delta > 0:
        return HardwareFault(kind="ecc-dbe", detail=f"uncorrected ECC +{uncorrected_delta}")
    if retirement_pending:
        return HardwareFault(kind="ecc-retire", detail="ECC page retirement pending")
    if sbe_threshold > 0 and corrected_in_window >= sbe_threshold:
        return HardwareFault(
            kind="ecc-sbe",
            detail=f"corrected ECC {corrected_in_window} in window (threshold {sbe_threshold})",
        )
    if nvlink_error_delta > 0:
        return HardwareFault(kind="nvlink", detail=f"NVLink errors +{nvlink_error_delta}")
    bad = [xid for xid in xids if xid in HARDWARE_XIDS]
    if bad:
        listed = ",".join(str(xid) for xid in bad)
        return HardwareFault(kind=f"xid-{listed}", detail=f"Xid {listed}")
    return None
