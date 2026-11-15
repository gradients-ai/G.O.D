"""Shared fixtures for the tournament tests.

Unless a test is marked real_compression_assignment, the text tournament's compression source assignment is the recorded
legacy mapping (text R1/R2/R3 -> compression R1/R2/R3, boss freeform on text R2, boss fixed on text R3) and no database is
read for it. test_compression_assignment.py covers the drawn assignment and the table.
"""

import pytest

from validator.tournament import compression_assignment as compression_assignment_module
from validator.tournament import quant_sources
from validator.tournament import seed_tasks
from validator.tournament import task_creator


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_compression_assignment: read the compression source assignment through its real lookup"
    )


@pytest.fixture(autouse=True)
def legacy_compression_assignment(request, monkeypatch):
    if request.node.get_closest_marker("real_compression_assignment"):
        return

    async def recorded(text_tournament_id, _psql_db):
        return compression_assignment_module.legacy_compression_assignment(text_tournament_id)

    for module in (seed_tasks, task_creator, quant_sources):
        monkeypatch.setattr(module, "compression_assignment", recorded)
