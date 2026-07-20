from datetime import datetime
from datetime import timedelta
from datetime import timezone
from decimal import Decimal
from typing import Any

from validator.tournament import cost_constants
from validator.tournament.models import GpuCostCategoryBreakdown
from validator.tournament.models import TaskGpuCostBreakdown
from validator.tournament.models import TournamentGpuCostBreakdown
from validator.tournament.models import TournamentGpuCostTotals
from validator.tournament.models import WeeklyTournamentGpuCostsResponse


_CATEGORIES = ("training", "prep", "evaluation")


def get_week_window(
    *,
    now: datetime | None = None,
    week_offset: int = 0,
    explicit_start: datetime | None = None,
) -> tuple[datetime, datetime]:
    if week_offset > 0:
        raise ValueError("week_offset must be zero or negative")
    if explicit_start is not None and week_offset != 0:
        raise ValueError("week_start and week_offset cannot be used together")
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if explicit_start is not None:
        start = (
            explicit_start.replace(tzinfo=timezone.utc)
            if explicit_start.tzinfo is None
            else explicit_start.astimezone(timezone.utc)
        )
    else:
        start = (now - timedelta(days=now.weekday())).replace(
            hour=cost_constants.WEEK_PIVOT_HOUR_UTC,
            minute=0,
            second=0,
            microsecond=0,
        )
        if now < start:
            start -= timedelta(days=7)
        start += timedelta(weeks=week_offset)
    natural_end = start + timedelta(days=7)
    return start, min(now, natural_end) if start <= now < natural_end else natural_end


def _zero_category() -> dict[str, Decimal | int]:
    return {
        "wall_seconds": Decimal(0),
        "gpu_seconds": Decimal(0),
        "cost_usd": Decimal(0),
        "success_count": 0,
        "failure_count": 0,
    }


def _category_model(values: dict[str, Decimal | int]) -> GpuCostCategoryBreakdown:
    return GpuCostCategoryBreakdown(
        wall_hours=float(Decimal(values["wall_seconds"]) / Decimal(3600)),
        gpu_hours=float(Decimal(values["gpu_seconds"]) / Decimal(3600)),
        cost_usd=float(values["cost_usd"]),
        success_count=int(values["success_count"]),
        failure_count=int(values["failure_count"]),
    )


def calculate_weekly_costs(
    *,
    rows: dict[str, list[dict[str, Any]]],
    window_start: datetime,
    window_end: datetime,
    week_offset: int,
    is_current_window: bool | None = None,
    tao_price_usd: float | None = None,
) -> WeeklyTournamentGpuCostsResponse:
    tournament_rows = {row["tournament_id"]: row for row in rows["tournaments"]}
    hotkey_counts = {row["task_id"]: row["hotkey_count"] for row in rows["hotkeys"]}
    task_totals: dict[Any, dict[str, Any]] = {}

    for task in rows["tasks"]:
        task_totals[task["task_id"]] = {
            "tournament_id": task["tournament_id"],
            "task_type": task.get("task_type"),
            "base_model": task.get("base_model"),
            "round_number": task.get("round_number"),
            "round_type": task.get("round_type"),
            **{category: _zero_category() for category in _CATEGORIES},
        }

    for row in rows["runs"]:
        effective_start = max(row["started_at"], window_start)
        effective_end = min(row["ended_at"] or window_end, window_end)
        if effective_start >= effective_end:
            continue
        wall_seconds = Decimal(str((effective_end - effective_start).total_seconds()))
        gpu_seconds = wall_seconds * Decimal(row["gpu_count"])
        task = task_totals.setdefault(
            row["task_id"],
            {
                "tournament_id": row["tournament_id"],
                "task_type": None,
                "base_model": None,
                "round_number": None,
                "round_type": None,
                **{category: _zero_category() for category in _CATEGORIES},
            },
        )
        values = task[row["category"]]
        values["wall_seconds"] += wall_seconds
        values["gpu_seconds"] += gpu_seconds
        values["cost_usd"] += gpu_seconds * row["hourly_rate_per_gpu_usd"] / Decimal(3600)
        if row["ended_at"] is not None and row["ended_at"] <= window_end:
            if row["outcome"] == "success":
                values["success_count"] += 1
            elif row["outcome"] == "failure":
                values["failure_count"] += 1

    tournament_totals: dict[str, dict[str, dict[str, Decimal | int]]] = {
        tournament_id: {category: _zero_category() for category in _CATEGORIES}
        for tournament_id in tournament_rows
    }
    task_models = []
    for task_id, task in task_totals.items():
        category_models = {category: _category_model(task[category]) for category in _CATEGORIES}
        tournament = tournament_totals[task["tournament_id"]]
        for category in _CATEGORIES:
            for key in tournament[category]:
                tournament[category][key] += task[category][key]
        task_models.append(
            TaskGpuCostBreakdown(
                task_id=task_id,
                tournament_id=task["tournament_id"],
                task_type=task.get("task_type"),
                base_model=task.get("base_model"),
                round_number=task.get("round_number"),
                round_type=task.get("round_type"),
                hotkey_count=hotkey_counts.get(task_id, 0),
                **category_models,
                total_cost_usd=sum(model.cost_usd for model in category_models.values()),
            )
        )
    task_models.sort(
        key=lambda task: (task.tournament_id, task.round_number if task.round_number is not None else -1, str(task.task_id))
    )

    participant_counts = {
        row["tournament_id"]: int(row["participant_count"]) for row in rows.get("participants", [])
    }

    tournament_models = []
    for tournament_id, row in tournament_rows.items():
        categories = {
            category: _category_model(tournament_totals[tournament_id][category])
            for category in _CATEGORIES
        }
        participant_count = participant_counts.get(tournament_id, 0)
        fee_rao = cost_constants.TOURNAMENT_PARTICIPATION_FEE_RAO_BY_TYPE.get(
            str(row["tournament_type"]), Decimal(0)
        )
        fee_collected_tao = float(fee_rao * Decimal(participant_count) / cost_constants.RAO_PER_TAO)
        fee_collected_usd = fee_collected_tao * tao_price_usd if tao_price_usd else 0.0
        tournament_models.append(
            TournamentGpuCostBreakdown(
                tournament_id=tournament_id,
                tournament_type=str(row["tournament_type"]),
                status=str(row["status"]),
                started_at=row["created_at"],
                completed_at=row["updated_at"] if str(row["status"]) == "completed" else None,
                participant_count=participant_count,
                fee_collected_tao=fee_collected_tao,
                fee_collected_usd=fee_collected_usd,
                **categories,
                total_attributed_cost_usd=sum(model.cost_usd for model in categories.values()),
            )
        )

    total_fees_collected_tao = sum(model.fee_collected_tao for model in tournament_models)
    total_fees_collected_usd = total_fees_collected_tao * tao_price_usd if tao_price_usd else 0.0

    first_tournament_started_at = min(
        (row["created_at"] for row in tournament_rows.values()),
        default=None,
    )
    billing_start = first_tournament_started_at or window_start
    provisioned_gpu_seconds = Decimal(0)
    for row in rows["capacity"]:
        if "H100" not in row["gpu_type"].upper():
            continue
        start = max(row["started_at"], billing_start)
        end = min(row["ended_at"] or window_end, window_end)
        if end > start:
            provisioned_gpu_seconds += Decimal(str((end - start).total_seconds()))

    attributed_h100_gpu_seconds = sum(
        Decimal(task[category]["gpu_seconds"])
        for task in task_totals.values()
        for category in ("training", "prep")
    )
    training_wall_seconds = sum(
        Decimal(task["training"]["wall_seconds"]) for task in task_totals.values()
    )
    training_gpu_seconds = sum(
        Decimal(task["training"]["gpu_seconds"]) for task in task_totals.values()
    )
    training_cost = sum(
        Decimal(task["training"]["cost_usd"]) for task in task_totals.values()
    )
    prep_wall_seconds = sum(
        Decimal(task["prep"]["wall_seconds"]) for task in task_totals.values()
    )
    prep_gpu_seconds = sum(
        Decimal(task["prep"]["gpu_seconds"]) for task in task_totals.values()
    )
    prep_cost = sum(
        Decimal(task["prep"]["cost_usd"]) for task in task_totals.values()
    )
    attributed_h100_cost = sum(
        Decimal(task[category]["cost_usd"])
        for task in task_totals.values()
        for category in ("training", "prep")
    )
    evaluation_gpu_seconds = sum(
        Decimal(task["evaluation"]["gpu_seconds"]) for task in task_totals.values()
    )
    evaluation_wall_seconds = sum(
        Decimal(task["evaluation"]["wall_seconds"]) for task in task_totals.values()
    )
    evaluation_cost = sum(Decimal(task["evaluation"]["cost_usd"]) for task in task_totals.values())
    idle_gpu_seconds = max(Decimal(0), provisioned_gpu_seconds - attributed_h100_gpu_seconds)
    provisioned_cost = (
        provisioned_gpu_seconds * cost_constants.H100_HOURLY_USD / Decimal(3600)
    )
    idle_cost = idle_gpu_seconds * cost_constants.H100_HOURLY_USD / Decimal(3600)

    completed_times = [
        row["updated_at"] for row in tournament_rows.values() if str(row["status"]) == "completed"
    ]
    return WeeklyTournamentGpuCostsResponse(
        week_offset=week_offset,
        window_start=window_start,
        window_end=window_end,
        is_current_window=week_offset == 0 if is_current_window is None else is_current_window,
        first_tournament_started_at=first_tournament_started_at,
        last_tournament_completed_at=max(completed_times, default=None),
        h100_8x_hourly_rate_usd=float(cost_constants.H100_8X_HOURLY_USD),
        a100_hourly_rate_usd=float(cost_constants.A100_HOURLY_USD),
        tao_price_usd=tao_price_usd,
        totals=TournamentGpuCostTotals(
            training_wall_hours=float(training_wall_seconds / Decimal(3600)),
            training_gpu_hours=float(training_gpu_seconds / Decimal(3600)),
            training_cost_usd=float(training_cost),
            prep_wall_hours=float(prep_wall_seconds / Decimal(3600)),
            prep_gpu_hours=float(prep_gpu_seconds / Decimal(3600)),
            prep_cost_usd=float(prep_cost),
            evaluation_wall_hours=float(evaluation_wall_seconds / Decimal(3600)),
            provisioned_h100_gpu_hours=float(provisioned_gpu_seconds / Decimal(3600)),
            provisioned_h100_cost_usd=float(provisioned_cost),
            attributed_h100_gpu_hours=float(attributed_h100_gpu_seconds / Decimal(3600)),
            attributed_h100_cost_usd=float(attributed_h100_cost),
            idle_h100_gpu_hours=float(idle_gpu_seconds / Decimal(3600)),
            idle_h100_cost_usd=float(idle_cost),
            evaluation_a100_gpu_hours=float(evaluation_gpu_seconds / Decimal(3600)),
            evaluation_a100_cost_usd=float(evaluation_cost),
            total_bill_usd=float(provisioned_cost + evaluation_cost),
            total_fees_collected_tao=total_fees_collected_tao,
            total_fees_collected_usd=total_fees_collected_usd,
        ),
        tournaments=tournament_models,
        tasks=task_models,
    )
