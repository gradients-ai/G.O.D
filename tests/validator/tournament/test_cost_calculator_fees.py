"""Cost calculator: per-tournament fee collection (participants x per-type fee) and the
task-level metadata (task type, base model, round) that feed the cost dashboard.

Fees are derived, not measured: we charge every participant the flat per-type fee, so the
collected amount is simply participant_count * fee. The two things worth pinning are that the
fee uses the right per-type rate (text/image differ) converted RAO -> TAO, and that task
metadata + round-aware sorting survive the aggregation into the response.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from decimal import Decimal

from validator.tournament import cost_constants
from validator.tournament.cost_calculator import calculate_weekly_costs


WINDOW_START = datetime(2026, 7, 13, 11, 0, 0, tzinfo=timezone.utc)
WINDOW_END = WINDOW_START + timedelta(days=7)


def _uuid(n: int) -> str:
    return f"{n:08d}-1111-1111-1111-111111111111"


def _rows(tournaments, tasks, participants):
    return {
        "tournaments": tournaments,
        "tasks": tasks,
        "runs": [],
        "hotkeys": [],
        "participants": participants,
        "capacity": [
            {
                "gpu_type": "H100",
                "started_at": WINDOW_START,
                "ended_at": WINDOW_END,
                "trainer_ip": "x",
                "gpu_id": 0,
                "vram_gb": 80,
            }
        ],
    }


def _calc(rows):
    return calculate_weekly_costs(
        rows=rows,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        week_offset=0,
        is_current_window=True,
    )


def _tournament_row(tournament_id: str, tournament_type: str):
    return {
        "tournament_id": tournament_id,
        "tournament_type": tournament_type,
        "status": "completed",
        "created_at": WINDOW_START + timedelta(hours=1),
        "updated_at": WINDOW_START + timedelta(hours=5),
    }


def test_fee_uses_per_type_rate_and_sums_across_tournaments():
    rows = _rows(
        tournaments=[
            _tournament_row("t_text", "text"),
            _tournament_row("t_image", "image"),
        ],
        tasks=[],
        participants=[
            {"tournament_id": "t_text", "participant_count": 4},
            {"tournament_id": "t_image", "participant_count": 5},
        ],
    )

    result = _calc(rows)
    fees = {t.tournament_id: t.fee_collected_tao for t in result.tournaments}

    text_fee = float(
        cost_constants.TOURNAMENT_PARTICIPATION_FEE_RAO_BY_TYPE["text"]
        * Decimal(4)
        / cost_constants.RAO_PER_TAO
    )
    image_fee = float(
        cost_constants.TOURNAMENT_PARTICIPATION_FEE_RAO_BY_TYPE["image"]
        * Decimal(5)
        / cost_constants.RAO_PER_TAO
    )

    assert fees["t_text"] == text_fee
    assert fees["t_image"] == image_fee
    assert result.totals.total_fees_collected_tao == text_fee + image_fee


def test_missing_participant_rows_yield_zero_fee():
    rows = _rows(
        tournaments=[_tournament_row("t_text", "text")],
        tasks=[],
        participants=[],
    )

    result = _calc(rows)

    assert result.tournaments[0].participant_count == 0
    assert result.tournaments[0].fee_collected_tao == 0.0
    assert result.totals.total_fees_collected_tao == 0.0


def test_task_metadata_flows_through_and_sorts_by_round():
    rows = _rows(
        tournaments=[_tournament_row("t_text", "text")],
        tasks=[
            {
                "task_id": _uuid(2),
                "tournament_id": "t_text",
                "task_type": "InstructTextTask",
                "base_model": "Qwen/Qwen2.5-7B",
                "round_number": 3,
                "round_type": "knockout",
            },
            {
                "task_id": _uuid(1),
                "tournament_id": "t_text",
                "task_type": "DpoTask",
                "base_model": "meta-llama/Llama-3.1-8B",
                "round_number": 1,
                "round_type": "group",
            },
        ],
        participants=[{"tournament_id": "t_text", "participant_count": 2}],
    )

    result = _calc(rows)

    assert [task.round_number for task in result.tasks] == [1, 3]
    first = result.tasks[0]
    assert first.task_type == "DpoTask"
    assert first.base_model == "meta-llama/Llama-3.1-8B"
    assert first.round_type == "group"
