# Evaluation tools

Manual evaluation runs and reward function management.

## Contents

| Path | Purpose |
| --- | --- |
| `add_affine_reward_functions.py` | Adds the affine reward functions through the validator API. |
| `instruct_eval_container.py` | Runs an instruct evaluation of one model in the evaluation container. |
| `manual_grpo_eval.py` | Evaluates one model repo on a GRPO task by task id, in the evaluation container. |
| `manually_add_grpo_rewards.py` | Stores reward functions in the database and marks them as manual. |
| `run_grpo_evaluation.py` | Runs a GRPO evaluation in Docker from a dataset URL and a list of models. |
| `run_text_evaluation_probe.py` | Runs a text evaluation with fixed inputs as a smoke test. |
| `simple_eval_grpo.sh` | Shell example of a GRPO evaluation. |
| `upload_grpo_model.sh` | Uploads a GRPO model directory to the Hugging Face Hub. |

## See also

- [Evaluations and manual tools](../../../docs/developer.md#evaluations-and-manual-tools): manual evaluation commands.
