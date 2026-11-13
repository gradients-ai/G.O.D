# Validator ops

Validator startup, auto-update, manual evaluation and Grafana setup.

## Contents

| Path | Purpose |
| --- | --- |
| `start_validator.sh` | Starts the validator processes. |
| `run_validator_auto_update.py` | Polls the repository and runs the validator update steps on a change. |
| `autoupdate_validator_steps.sh` | Validator update steps. Installs packages through `task install`. |
| `run_auditor_autoupdate.py` | Polls the repository and runs the auditor update steps on a change. |
| `autoupdate_auditor_steps.sh` | Auditor update steps. Installs packages through `task install`. |
| `run_evaluation.py` | Evaluates a task's submitted models by task id on local GPUs. |
| `setup_grafana.sh` | Grafana setup helper. |

## See also

- [Running a validator](../../docs/developer.md#running-a-validator): how the validator services are started.
- [Evaluations and manual tools](../../docs/developer.md#evaluations-and-manual-tools): manual evaluation commands.
