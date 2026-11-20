# Evaluation flow and debugging

A runbook for validator operators: how an evaluation moves through the validator, and what to check when one is stuck or failed.

## Flow

1. Every minute the tournament orchestrator adds an `evaluations` row for each tournament task and hotkey whose `training_status` is `success`. Existing rows are left alone.
2. Every 30 seconds the validator cycle picks up `pending` and `evaluating` rows. A tournament row is evaluated while its task is in `training`.
3. Each row runs in a remote deployment. Total evaluation GPUs are capped by `EVAL_MAX_GPUS` (`validator/lifecycle/constants.py`).
4. A finished row saves its raw output to `task_nodes` (`test_loss`, `synth_loss`).
5. The task is scored and ranked once every evaluation row and every training row of the task is terminal.

Failure classes and retries are defined in [validator/evaluation](../../validator/evaluation/README.md#whose-fault-an-evaluation-failure-is).

## Backends

`EVAL_BACKEND` selects the remote backend for all evaluations.

| Value | Backend | Deployment id in `evaluations.deployment_id` |
| --- | --- | --- |
| `basilica` (default) | Basilica deployments | The deployment name, prefixed `god-eval-` |
| `runpod` | dstack services scheduled on Runpod | `runpod:<run_name>`, run names prefixed `god-eval-` |

The Runpod backend reads `DSTACK_URL`, `DSTACK_TOKEN` and `DSTACK_PROJECT`. Optional overrides:

| Variable | Meaning | Default |
| --- | --- | --- |
| `EVAL_RUNPOD_DISK_SIZE` | Minimum disk size | `150GB` |
| `EVAL_RUNPOD_REGIONS` | Comma-separated regions | all regions |
| `EVAL_RUNPOD_GATEWAY` | dstack gateway for the services | none |

## Timing

Values are in `validator/evaluation/constants.py`.

| Setting | Value |
| --- | --- |
| Result poll interval | 300 s |
| Maximum time polling one deployment | 28,800 s (8 h) |
| Deployment ready timeout | 1,800 s |
| Attempts per evaluation | 3, with 30 s between attempts |
| Basilica deployment resources | 4 CPU, 64 Gi memory, TTL 28,800 s |
| Runpod service maximum duration | 7,200 s |
| Reconcile interval | 120 s |

On Basilica a poll timeout is not retried inside the same pass. On Runpod it counts as a failed attempt.

## Symptoms

### A row stays `pending`

| Check | Fix |
| --- | --- |
| Is the training row `success`? Rows are added only for successful trainings. | Wait for training, or fix the training. |
| Is the GPU cap reached? Sum `gpu_count` over `evaluations`. | Wait. Stale reservations are released by the reconciler. |
| Was a compression row re-queued after a counted error? It waits 10 minutes before the next dispatch. A validator restart clears the wait. | Wait. |

```sql
select hotkey, evaluation_status, deployment_id, gpu_count, infra_error_attempts, created_at, updated_at
from evaluations
where task_id = '<task_id>'
order by created_at;
```

### A row stays `evaluating`

1. Read `deployment_id` from the query above.
2. Inspect the deployment.

Basilica, with the Basilica CLI:

```bash
bs deploy ls
bs deploy status <deployment_id> --show-phases
bs deploy logs -f <deployment_id>
```

Runpod, with the dstack CLI:

```bash
dstack ps
dstack logs <run_name>
```

3. Search the validator logs in Grafana by `task_id`, by hotkey (`miner_hotkey`) or by the expected repo name. Remote container output is relayed to the validator logs as `[BASILICA_LOG]` lines on Basilica and `[REMOTE_EVAL_LOG]` lines on Runpod.

A healthy deployment is polled until it returns a result. A missing or unhealthy one is redeployed, and the row records the redeployed id.

### A row is `infra_held`

A compression task row that hit `MAX_EVAL_INFRA_ATTEMPTS` infrastructure errors is held. It is never dispatched or scored, and the task does not finalise while a row is held. The alert carries the SQL for both fixes and repeats every 4 hours.

| Cause | Fix |
| --- | --- |
| Infrastructure fault, fixed | Re-queue: set `evaluation_status='pending'` and `infra_error_attempts=0` on the held rows. |
| The entry cannot be evaluated | Fail the rows: set `evaluation_status='failure'`. |

### A row is `failure`, or a loss is missing

```sql
select hotkey, expected_repo_name, test_loss, synth_loss, score_reason, quant_failure
from task_nodes
where task_id = '<task_id>'
order by test_loss desc;
```

`score_reason` and `quant_failure` state why the entry failed. To evaluate a text or image task again by hand, see [Evaluations and manual tools](../../docs/developer.md#evaluations-and-manual-tools).

### Deployments or GPU reservations leak

The reconciler runs at startup and every 120 seconds. It protects deployments recorded on `pending` or `evaluating` rows. It deletes deployments that no row owns once they are past their startup grace. It releases reservations whose deployment is gone.

Normal completion and every failure path delete the deployment. If the cap looks wrong, compare the live deployment list with the `deployment_id` values in `evaluations`.

## See also

- [Evaluation internals](../../validator/evaluation/README.md): evaluation internals and failure classes.
- [Validator tournament](../../validator/tournament/README.md): crown rule, seeds, orchestration.
