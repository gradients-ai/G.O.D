from unittest.mock import AsyncMock

import pytest

from validator.tournament import orchestrator
from validator.tournament.image_builds import ImageGate


PREBUILT_IMAGE = "registry.example:5000/god/trainers@sha256:abc"


@pytest.fixture(autouse=True)
def every_training_image_is_ready(monkeypatch: pytest.MonkeyPatch):
    """Scheduler tests run without an image build box: every row's image is ready (validator/tournament/image_builds.py)."""
    monkeypatch.setattr(orchestrator, "image_for_training", AsyncMock(return_value=(ImageGate.READY, PREBUILT_IMAGE)))
