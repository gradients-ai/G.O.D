# Validator

Validator runtime for task creation, tournament orchestration, evaluation, scoring, storage, and weight setting.

## Contents

- `app/`: configuration loading and FastAPI dependencies.
- `assets/`: validator runtime assets such as prompts and test training configs.
- `db/`: database connection, migrations, constants, and SQL access modules.
- `endpoints/`: FastAPI routes exposed by the validator service.
- `evaluation/`: evaluation runtimes, model checks, evaluator entrypoints, PvP, and result processing.
- `infrastructure/`: external service wrappers and infrastructure clients.
- `lifecycle/`: validator task lifecycle loops.
- `nodes/`: metagraph/node refresh logic.
- `scoring/`: task scoring, tournament scoring, and weight setting.
- `tasks/`: task schemas, dataset prep, model prep, rewards, requests, and synthetic task creation.
- `tournament/`: tournament state machine, participants, rounds, task creation, and reports.
- `transfers/`: balance and transfer schemas.
- `asgi.py`: FastAPI application factory and service entrypoint.
- `constants.py`: compatibility aggregate for validator constants; prefer importing from the owning domain file.
