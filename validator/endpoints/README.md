# Validator endpoints

FastAPI route modules of the validator service.

## Contents

| Path | Purpose |
| --- | --- |
| `auditing.py` | Audit endpoints: recent tasks, task details, latest scores and duplicate reviews. |
| `grpo.py` | GRPO reward function endpoints. |
| `health.py` | Health checks. |
| `nodes.py` | Node registration endpoint. |
| `performance.py` | Latest tournament weights, weight projections and the last boss round. |
| `tasks.py` | Task creation, task details and dataset validation. |
| `tournament_analytics.py` | Tournament details, history, next start dates, active tournaments and GPU requirements. |
| `tournament_orchestrator.py` | Trainer registration and training stats. |
| `transfer_balances.py` | Coldkey balance and tournament fee endpoints. |

## See also

- [Results](../../docs/miner.md#results): the public endpoints a miner reads.
