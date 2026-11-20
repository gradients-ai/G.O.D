from datetime import datetime
from datetime import timezone

import pytest

import validator.scoring.constants as cts


@pytest.fixture
def quant_ramp_complete(monkeypatch):
    """Compression's emission phase-in long finished, so the split is the full participation-driven one."""
    monkeypatch.setattr(cts, "QUANT_EMISSION_RAMP_START", datetime(2000, 1, 1, tzinfo=timezone.utc))
