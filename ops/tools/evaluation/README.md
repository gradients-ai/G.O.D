# Evaluation Tools

Manual evaluation utilities and reward-function management scripts.

## Contents

- `add_affine_reward_functions.py`: add affine reward functions.
- `basilica_environment_eval.py`: Basilica environment evaluation helper.
- `basilica_swe_infinite_eval.py`: live Basilica SWE Infinite individual-eval smoke test.
- `instruct_eval_container.py`: instruct evaluation container probe.
- `local_environment_eval.py`: local environment evaluation runner.
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
BASILICA_API_KEY=... SWE_INFINITE_SERVER_BASE_URL=https://affinetes.example \
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
