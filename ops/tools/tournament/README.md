# Tournament Tools

Manual tournament control and inspection tools.

## Contents

- `backfill_third_place.py`: backfill missing 3rd-place placements on tournaments that already have positions 1 and 2.
- `complete_tournament.py`: manually complete a tournament.
- `manage_tournament_task.py`: manage tournament task state.
- `reverse_tournament_round.py`: reverse a tournament round. A seed task of the round (a `tournament_seed_tasks` row: text rounds 1-3, the
  continuous task, the quant boss-round compressions) is never deleted: it is detached from the round, the round miners' training rows,
  assignments and submissions on it are removed, and the task, the boss's row and the boss's results are kept; a text seed's attached
  marker is cleared when that is safe, and its task status is never reset (a warning says when it is past `training`). The other tasks of
  the round are deleted. The tool refuses, changing nothing, while a live `quant_task_sources` row names a task of the round as its
  source (a bound quant tournament depends on it; delete the dependent quant tasks first, cancelling or completing the quant tournament does not lift it).
- `start_tournament.py`: manually create a basic tournament of the given type: `python -m ops.tools.tournament.start_tournament <text|image|quant>`.
- `tournament_status_summary.py`: summarize tournament status.
- `__init__.py`: package marker.
