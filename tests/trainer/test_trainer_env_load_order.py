import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE = "import trainer.asgi; from validator.infrastructure import minio_client as m; print(m.S3_COMPATIBLE_ENDPOINT)"


def _endpoint_seen_by_trainer(cwd: Path, environ_endpoint: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("S3_")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    if environ_endpoint is not None:
        env["S3_COMPATIBLE_ENDPOINT"] = environ_endpoint
    out = subprocess.run([sys.executable, "-c", CODE], cwd=cwd, env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


@pytest.fixture
def trainer_dir(tmp_path: Path) -> Path:
    (tmp_path / ".trainer.env").write_text("S3_COMPATIBLE_ENDPOINT=from-trainer-env:9000\n")
    return tmp_path


def test_trainer_env_file_is_loaded_before_s3_settings_are_read(trainer_dir):
    # `python -m trainer.asgi` with no --env-file: the S3 client must still see .trainer.env.
    assert _endpoint_seen_by_trainer(trainer_dir, None) == "from-trainer-env:9000"


def test_process_environment_still_wins_over_the_env_file(trainer_dir):
    # `uvicorn --env-file` (the pm2 launch) sets the environment first; load_dotenv never overrides it.
    assert _endpoint_seen_by_trainer(trainer_dir, "from-environ:9000") == "from-environ:9000"
