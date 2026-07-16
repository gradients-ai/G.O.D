from datetime import datetime
from datetime import timezone
from decimal import Decimal
from uuid import uuid4

from validator.tournament import cost_constants
from validator.tournament.cost_calculator import calculate_weekly_costs
from validator.tournament.cost_calculator import get_week_window


UTC = timezone.utc


def test_current_week_starts_monday_at_11_utc():
    now = datetime(2026, 7, 16, 18, 0, tzinfo=UTC)

    start, end = get_week_window(now=now)

    assert start == datetime(2026, 7, 13, 11, 0, tzinfo=UTC)
    assert end == now


def test_before_monday_pivot_belongs_to_previous_week():
    now = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)

    start, end = get_week_window(now=now)

    assert start == datetime(2026, 7, 6, 11, 0, tzinfo=UTC)
    assert end == now


def test_previous_week_is_a_complete_monday_window():
    now = datetime(2026, 7, 16, 18, 0, tzinfo=UTC)

    start, end = get_week_window(now=now, week_offset=-1)

    assert start == datetime(2026, 7, 6, 11, 0, tzinfo=UTC)
    assert end == datetime(2026, 7, 13, 11, 0, tzinfo=UTC)


def test_weekly_costs_separate_provisioned_attributed_and_idle(monkeypatch):
    monkeypatch.setattr(cost_constants, "H100_8X_HOURLY_USD", Decimal("80"))
    monkeypatch.setattr(cost_constants, "H100_HOURLY_USD", Decimal("10"))
    monkeypatch.setattr(cost_constants, "A100_HOURLY_USD", Decimal("5"))
    task_id = uuid4()
    start = datetime(2026, 7, 13, 11, 0, tzinfo=UTC)
    end = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    zero = Decimal(0)
    rows = {
        "tournaments": [{
            "tournament_id": "tourn-1",
            "tournament_type": "text",
            "status": "active",
            "created_at": start,
            "updated_at": start,
        }],
        "tasks": [{"task_id": task_id, "tournament_id": "tourn-1"}],
        "costs": [{
            "task_id": task_id,
            "tournament_id": "tourn-1",
            "training_wall_seconds": Decimal(3600),
            "training_gpu_seconds": Decimal(7200),
            "training_cost_usd": Decimal(20),
            "training_success_count": 1,
            "training_failure_count": 0,
            "prep_wall_seconds": zero,
            "prep_gpu_seconds": zero,
            "prep_cost_usd": zero,
            "prep_success_count": 0,
            "prep_failure_count": 0,
            "evaluation_wall_seconds": Decimal(3600),
            "evaluation_gpu_seconds": Decimal(3600),
            "evaluation_cost_usd": Decimal(5),
            "evaluation_success_count": 1,
            "evaluation_failure_count": 0,
        }],
        "active": [],
        "hotkeys": [{"task_id": task_id, "hotkey_count": 4}],
        "capacity": [
            {
                "trainer_ip": "trainer-1",
                "gpu_id": gpu_id,
                "gpu_type": "H100",
                "vram_gb": 80,
                "started_at": start,
                "ended_at": end,
            }
            for gpu_id in range(8)
        ],
    }

    result = calculate_weekly_costs(
        rows=rows,
        window_start=start,
        window_end=end,
        week_offset=0,
    )

    assert result.totals.provisioned_h100_gpu_hours == 8
    assert result.totals.provisioned_h100_cost_usd == 80
    assert result.totals.attributed_h100_gpu_hours == 2
    assert result.totals.idle_h100_gpu_hours == 6
    assert result.totals.idle_h100_cost_usd == 60
    assert result.totals.evaluation_a100_cost_usd == 5
    assert result.totals.total_bill_usd == 85
    assert result.tasks[0].hotkey_count == 4


def test_active_run_is_accrued_through_window_end(monkeypatch):
    monkeypatch.setattr(cost_constants, "H100_8X_HOURLY_USD", Decimal("80"))
    monkeypatch.setattr(cost_constants, "H100_HOURLY_USD", Decimal("10"))
    task_id = uuid4()
    start = datetime(2026, 7, 13, 11, 0, tzinfo=UTC)
    end = datetime(2026, 7, 13, 11, 30, tzinfo=UTC)
    rows = {
        "tournaments": [{
            "tournament_id": "tourn-1",
            "tournament_type": "text",
            "status": "active",
            "created_at": start,
            "updated_at": start,
        }],
        "tasks": [{"task_id": task_id, "tournament_id": "tourn-1"}],
        "costs": [],
        "active": [{
            "run_key": "training-run",
            "task_id": task_id,
            "tournament_id": "tourn-1",
            "category": "training",
            "gpu_type": "H100",
            "gpu_count": 2,
            "hourly_rate_per_gpu_usd": Decimal("10"),
            "started_at": start,
            "metadata": {},
        }],
        "hotkeys": [],
        "capacity": [],
    }

    result = calculate_weekly_costs(
        rows=rows,
        window_start=start,
        window_end=end,
        week_offset=0,
    )

    assert result.totals.training_wall_hours == 0.5
    assert result.totals.training_gpu_hours == 1
    assert result.totals.training_cost_usd == 10
