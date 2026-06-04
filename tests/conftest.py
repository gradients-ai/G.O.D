import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests marked gpu; skipped by default for local CPU-only development",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        return

    skip_gpu = pytest.mark.skip(reason="need --run-gpu to run GPU/model dependency tests")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
