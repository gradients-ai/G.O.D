# Evaluation Flow and Debugging

This is a quick guide for checking validator evaluations.

## Flow

Tournament training and evaluation are separate. As each tournament task/hotkey finishes training with `training_status = 'success'`, the tournament orchestrator seeds a row in `evaluations`. It checks every 15 minutes and does not overwrite existing rows.

The validator task cycle picks up `pending` and `evaluating` rows from `evaluations`. It can run up to `10` evaluation rows at once across all tasks. Tournament rows can be evaluated while the task is still `TRAINING`.

Each completed row saves raw evaluation output into `task_nodes`: `test_loss` and `synth_loss`. The full task scoring/ranking happens later, after all evaluation rows for the task are terminal (`success` or `failure`) and tournament training is terminal. Final scoring reads the saved losses from `task_nodes`.

## Main Debugging Commands

List live Basilica deployments:

```bash
bs deploy ls
```

Check deployment phases:

```bash
bs deploy status <deployment_id> --show-phases
```

Follow deployment logs:

```bash
bs deploy logs -f <deployment_id>
```

In Grafana, search validator logs by:

- `task_id`
- hotkey / `miner_hotkey`
- expected repo name, such as `tournament-{tournament_id}-{task_id}-{hotkey_prefix}`

Turn on Basilica logs in Grafana when you need container output.

## Database Checks

Start with `evaluations`. This tells you whether the row is waiting, running, done, or failed.

```sql
select *
from evaluations
where task_id = '<task_id>'
order by created_at;
```

Useful fields:

- `evaluation_status`: `pending`, `evaluating`, `success`, or `failure`.
- `deployment_id`: Basilica deployment for a live or recently live evaluation.
- `created_at` / `updated_at`: tells you how long the row has been waiting.

Check saved losses in `task_nodes`:

```sql
select hotkey, expected_repo_name, test_loss, synth_loss, score_reason
from task_nodes
where task_id = '<task_id>'
order by test_loss desc;
```

Evaluation rows are only seeded for tournament hotkeys with `training_status = 'success'`.

## Timing

The evaluation loop checks for work every 30 seconds.

Each Basilica evaluation polls the result endpoint every `300` seconds. It stops when it gets a result, gets a failure, or reaches `16000` seconds, which is about 4 hours and 26 minutes.

Deployments are created with `ttl_seconds = 16000`, `timeout = 1800`, `cpu = 4`, and `memory = 64Gi`.

Failed evaluation attempts retry up to `3` times, with `900` seconds between retries. Poll timeouts are treated as final failures for that repo.

## Startup Recovery

On startup, the validator protects deployments already stored on `pending` or `evaluating` rows and deletes other lingering Basilica deployments.

Rows that remain `evaluating` (usually happens if they were interrupted in event of a validator restart) are picked up by the loop and if the stored deployment is healthy, polling resumes. If it is missing or unhealthy, the validator redeploys and saves the new deployment id.
