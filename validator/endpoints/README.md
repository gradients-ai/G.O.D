# Validator Endpoints

FastAPI route modules for the validator service.

## Contents

- `auditing.py`: audit and task-history endpoints.
- `grpo.py`: GRPO reward function endpoints.
- `health.py`: health checks.
- `nodes.py`: node and metagraph endpoints.
- `performance.py`: performance and ranking endpoints, with the quant tournament's projection and performance fields next to text and image.
- `tasks.py`: task creation, task details, and task stats endpoints.
- `tournament_analytics.py`: tournament analytics and task detail endpoints. Responses that list tournament types (next dates, active, latest details, weights) carry an additive `quant` / `quant_*` field next to `text` and `image`. The next-dates response reports a start date for text, quant and image.
- `tournament_orchestrator.py`: tournament orchestration control endpoints.
- `transfer_balances.py`: tournament fee and balance endpoints. `GET /tournament/fees` returns `text_tournament_fee_rao`, `image_tournament_fee_rao` and `quant_tournament_fee_rao`.
