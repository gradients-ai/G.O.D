# Validator

The validator runtime: task creation, tournament orchestration, evaluation, scoring and weight setting.

## Contents

| Path | Purpose |
| --- | --- |
| `app/` | Configuration loading and FastAPI dependencies. |
| `assets/` | Prompts, bundled chat templates and a test training config. |
| `db/` | Database connection, migrations and SQL access. |
| `endpoints/` | FastAPI routes of the validator service. |
| `evaluation/` | Evaluation dispatch, evaluators, model checks and result processing. |
| `infrastructure/` | Clients for external services, and repository review and duplicate detection. |
| `lifecycle/` | The loops that move a task from creation to a finalised result. |
| `nodes/` | Node refresh from the metagraph. |
| `scoring/` | Task scoring, tournament scoring and weights. |
| `tasks/` | Task models, dataset and model prep, reward functions and synthetic task creation. |
| `tournament/` | Tournament orchestration: participants, rounds, seed tasks, the crown and reports. |
| `transfers/` | Balance and transfer models, and the TAO price lookup. |
| `asgi.py` | FastAPI application factory and service entrypoint. |
| `constants.py` | Re-exports the lifecycle, scoring, evaluation, infrastructure, tasks and tournament constants. `db/constants.py` is imported by name. |
| `runtime.py` | `TaskRuntimeDeps`, the narrow dependencies of validator functions that run without the full config. |

## See also

- [Developer guide](../docs/developer.md): setup, configuration and running the services.
- [Validator tournament](tournament/README.md): crown rule, seeds, orchestration.
- [Evaluation internals](evaluation/README.md): evaluation internals and failure classes.
- [Scoring](scoring/README.md): scoring and weights internals.
