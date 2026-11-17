from trainer.gpu_health import HARDWARE_XIDS
from trainer.gpu_health import HealthReading
from trainer.gpu_health import HealthState
from trainer.gpu_health import advance_health
from trainer.gpu_health import fabric_manager_is_down
from trainer.gpu_health import normalize_pci
from trainer.gpu_health import parse_xids


def _reading(t: float, **kwargs) -> HealthReading:
    return HealthReading(timestamp_s=t, **kwargs)


def test_first_sample_baselines_existing_ecc():
    state, fault = advance_health(HealthState(), _reading(0, uncorrected=4, corrected=9), 10, 300)
    assert fault is None
    assert state.uncorrected == 4
    assert state.corrected == 9


def test_one_new_uncorrected_error_faults():
    state, _ = advance_health(HealthState(), _reading(0, uncorrected=0, corrected=0), 10, 300)
    _, fault = advance_health(state, _reading(5, uncorrected=1, corrected=0), 10, 300)
    assert fault is not None
    assert fault.kind == "ecc-dbe"
    assert "uncorrected ECC +1" in fault.detail


def test_corrected_errors_wait_for_the_window_threshold():
    state, fault = advance_health(HealthState(), _reading(0, corrected=0), 10, 300)
    state, fault = advance_health(state, _reading(5, corrected=3), 10, 300)
    assert fault is None
    _, fault = advance_health(state, _reading(10, corrected=10), 10, 300)
    assert fault is not None and fault.kind == "ecc-sbe"


def test_counter_reset_does_not_look_like_new_errors():
    state, _ = advance_health(HealthState(), _reading(0, uncorrected=20), 10, 300)
    _, fault = advance_health(state, _reading(5, uncorrected=0), 10, 300)
    assert fault is None


def test_pending_retirement_and_nvlink_and_hardware_xid():
    _, retire = advance_health(HealthState(), _reading(0, retirement_pending=True), 10, 300)
    assert retire is not None and retire.kind == "ecc-retire"
    state, _ = advance_health(HealthState(), _reading(0, nvlink_errors=1), 10, 300)
    _, nv = advance_health(state, _reading(5, nvlink_errors=2), 10, 300)
    assert nv is not None and nv.kind == "nvlink"
    _, xid = advance_health(HealthState(), _reading(0, xids=[79]), 10, 300)
    assert xid is not None and "79" in xid.detail
    _, app = advance_health(HealthState(), _reading(0, xids=[13, 31]), 10, 300)
    assert app is None
    assert 13 not in HARDWARE_XIDS


def test_fabric_manager_only_when_the_unit_exists():
    assert fabric_manager_is_down("not-found", "inactive") is False
    assert fabric_manager_is_down("loaded", "active") is False
    assert fabric_manager_is_down("loaded", "failed") is True
    assert fabric_manager_is_down("loaded", "inactive") is True


def test_xid_pci_from_nvml_and_the_kernel_log_match():
    assert normalize_pci("00000000:81:00.0") == normalize_pci("0000:81:00")
    text = "NVRM: Xid (PCI:0000:81:00): 79, GPU has fallen off the bus."
    assert parse_xids(text) == [("81:00", 79)]
