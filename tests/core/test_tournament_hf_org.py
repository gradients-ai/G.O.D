import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.constants import credentials
from validator.infrastructure import service_constants


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_service_constants_reexports_the_single_definition():
    assert service_constants.RAYONLABS_HF_USERNAME is credentials.RAYONLABS_HF_USERNAME


@pytest.mark.parametrize("value, expected", [(None, "gradients-io-tournaments"), ("scratch-org", "scratch-org")])
def test_tournament_hf_org_from_env(value, expected):
    # A fresh interpreter, so the import-time read is exercised without reloading modules other tests hold.
    env = {k: v for k, v in os.environ.items() if k != "TOURNAMENT_HF_USERNAME"}
    if value is not None:
        env["TOURNAMENT_HF_USERNAME"] = value
    code = (
        "from core.constants.credentials import RAYONLABS_HF_USERNAME as a;"
        "from validator.infrastructure.service_constants import RAYONLABS_HF_USERNAME as b;"
        "print(a, b)"
    )
    # cwd outside the repo so a stray .env there cannot set the variable.
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**env, "PYTHONPATH": str(REPO_ROOT)},
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split()[-2:] == [expected, expected]
