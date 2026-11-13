# Tournament tools

Manual tournament control and inspection. Each tool states its usage in its module docstring, in `--help` or in its usage message.

## Contents

| Path | Purpose |
| --- | --- |
| `approve_task_failure_review.py` | Lists and approves tasks held for review because more than half their trainings failed. |
| `backfill_third_place.py` | Sets third place on completed tournaments that record only first and second. |
| `code_review.py` | Shows or resolves a round 2 challenger code review (`show`, `agree`, `skip`). |
| `complete_tournament.py` | Marks a finished tournament `completed`. |
| `dedup_check.py` | Runs the duplicate-submission check on two repositories locally. Writes nothing. |
| `dedup_dryrun.py` | Runs the duplicate-submission check over a tournament's participants. Read-only. |
| `manage_tournament_task.py` | Replaces, retries or allows a tournament task. |
| `re_evaluate_tournament_group_task.py` | Resets hotkeys on a completed group task for retraining and full re-evaluation. |
| `restamp_adapter_base.py` | Rewrites `base_model_name_or_path` in a LoRA adapter's `adapter_config.json`. |
| `reverse_tournament_round.py` | Deletes a round and reactivates the one before it. |
| `start_tournament.py` | Creates a tournament: `python -m ops.tools.tournament.start_tournament <text\|image\|quant>`. |
| `tournament_status_summary.py` | Prints a status summary of the active text and image tournaments (`--text`, `--image`). It has no compression option. |

## Reversing a round

`reverse_tournament_round.py` treats the round's seed tasks differently from its other tasks.

| Task | What the tool does |
| --- | --- |
| Seed task (a `tournament_seed_tasks` row) | Detached from the round and kept, with the boss's row and results. Round miners' rows on it are deleted. |
| Any other task of the round | Deleted. |

The tool refuses and changes nothing while a live `quant_task_sources` row names a task of the round as its source.

## See also

- [Validator tournament](../../../validator/tournament/README.md): crown rule, seeds, orchestration.
- [Miner hub](../../../docs/miner.md#what-miners-do): each guide it links has a local testing section.
