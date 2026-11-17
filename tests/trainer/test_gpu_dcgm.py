import json
import subprocess

from trainer.gpu_dcgm import parse_dcgm_health
from trainer.gpu_dcgm import query_dcgm_health


def test_parses_failure_and_ignores_skipped_nvlink_on_pcie():
    payload = {
        "body": {
            "Overall Health": {"status": "Failure"},
            "GPUs": [
                {
                    "GPU ID": "0",
                    "Overall Health": {"status": "Failure"},
                    "Memory system": {
                        "status": "Failure",
                        "errors": ["GPU 0 has uncorrectable ECC errors."],
                    },
                    "NVLink system": {"status": "Skipped"},
                    "Thermal system": {"status": "Healthy"},
                },
                {
                    "GPU ID": "1",
                    "Memory system": {"status": "Warning", "errors": ["12 volatile correctable errors"]},
                    "NVLink system": {"status": "Healthy"},
                },
            ],
        }
    }
    report = parse_dcgm_health(json.dumps(payload))
    assert [(item.gpu_index, [f.name for f in item.failures]) for item in report] == [
        (0, ["Memory system"]),
        (1, []),
    ]
    assert report[0].failures[0].summary == "GPU 0 has uncorrectable ECC errors."
    assert report[1].warnings[0].name == "Memory system"


def test_parses_flat_status_strings():
    payload = {"GPU": [{"GPU": "2", "PCIe": "Fail", "Power": "Pass"}]}
    report = parse_dcgm_health(json.dumps(payload))
    assert report[0].gpu_index == 2
    assert report[0].failures[0].name == "PCIe"


def test_query_returns_none_when_dcgmi_is_missing(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("dcgmi")

    monkeypatch.setattr(subprocess, "run", missing)
    assert query_dcgm_health() is None


def test_query_returns_none_when_hostengine_is_down(monkeypatch):
    def down(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="Unable to connect")

    monkeypatch.setattr(subprocess, "run", down)
    assert query_dcgm_health() is None
