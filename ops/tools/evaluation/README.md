# Evaluation Tools

Manual evaluation utilities and reward-function management scripts.

## Contents

- `add_affine_reward_functions.py`: add affine reward functions.
- `basilica_environment_eval.py`: Basilica environment evaluation helper.
- `basilica_swe_infinite_eval.py`: live Basilica SWE Infinite individual-eval smoke test.
- `instruct_eval_container.py`: instruct evaluation container probe.
- `local_environment_eval.py`: local environment evaluation runner.
- `local_swe_infinite_eval.py`: local SWE Infinite smoke test using host SGLang and the SWE Docker image.
- `manual_grpo_eval.py`: manual GRPO evaluation helper.
- `manually_add_grpo_rewards.py`: add GRPO rewards directly.
- `process_miners_pool_mixed_env_eval.py`: mixed environment pool evaluation tool.
- `pvp_anthropic_match.py`: run a PvP matchup with Anthropic models.
- `pvp_play.py`: manual PvP tool-calling harness stepper.
- `run_grpo_evaluation.py`: standalone GRPO evaluation runner.
- `run_image_evaluation_probe.py`: image evaluation probe.
- `run_text_evaluation_probe.py`: text evaluation probe.
- `simple_eval_grpo.sh`: shell GRPO evaluation example.
- `upload_grpo_model.sh`: upload helper for GRPO models.
- `__init__.py`: package marker.

## SWE Infinite Basilica Smoke Test

```bash
BASILICA_API_TOKEN=... SWE_INFINITE_SERVER_BASE_URL=https://affinetes.example \
  uv run --extra dev python -m ops.tools.evaluation.basilica_swe_infinite_eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --task-id 7 83 45
```

Use `--dry-run` to print the resolved Basilica image, model, and SWE env vars
without deploying.

This smoke test does not require validator database access. Basilica returns the
deployment URL used by the evaluator; the database is only used by production
validator flows for deployment resume/persistence bookkeeping.

SWE Infinite evaluation always requests Affinetes' MiniSWE agent; the smoke test
does not expose an agent selector.

## Local SWE Infinite Smoke Test

```bash
uv run --extra dev python -m ops.tools.evaluation.local_swe_infinite_eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-seeds 2 \
  --seed 42
```

This starts SGLang as a local host process and starts the Affinetes SWE Infinite
server from `gradientsio/swe-infinite:v1`. Because the SWE server runs in Docker,
the model URL sent to it defaults to `http://host.docker.internal:30000/v1`.
Use `--model-base-url` when your Docker runtime needs a different callback URL.
The live path requires a local SGLang installation and GPU access; use
`--sglang-start-cmd` for a custom launch command, or `--use-existing-sglang`
with `--sglang-base-url` to point at a server you already started.

Task selection is deterministic for a fixed `--seed`; running the command again
with the same seed, task range, and `--num-seeds` evaluates the same task IDs.
Use `--task-id 7` or `--task-id 7 83 45` to evaluate exact task IDs.

Useful options:

```bash
uv run --extra dev python -m ops.tools.evaluation.local_swe_infinite_eval \
  --model your-org/your-model-or-lora \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --task-id 7 83 45 \
  --swe-port 8000 \
  --swe-container-port 8000 \
  --dry-run
```

`--dry-run` prints the resolved task IDs, SGLang URL, Docker command, and SWE
payload-facing model URL without launching SGLang or Docker.
