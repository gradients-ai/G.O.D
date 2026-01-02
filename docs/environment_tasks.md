# Environment Tasks & Rollout Functions

Gradients now supports **Environment Tasks**, leveraging the custom rollout capabilities within the TRL `GRPOTrainer`. This framework enables models to interact with external environments during the training loop to optimize based on dynamic feedback and sparse rewards.

## Evaluation Protocol

Post-training, Gradients assesses model performance by executing **500 episodes** within the target environment. The primary metric for success is the **average score** across this full evaluation set.

---

## Miner Requirements

During training, miners are provisioned with environment servers hosting specific task logic. To maximize throughput and minimize latency, one dedicated environment server is typically allocated per GPU.

### 1. Environment Connectivity

Server addresses are exposed via the `ENVIRONMENT_SERVER_URLS` environment variable as a comma-separated string.

**Python Implementation:**

```python
import os

# Extract and clean server URLs
raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]

```

### 2. The Rollout Function Workflow

Miners must implement a custom **Rollout Function** associated with the `dataset_type.environment_name`. This function manages the following lifecycle:

* **Sampling:** Generate model completions via `generate_rollout_completions`.
* **Execution:** Interface with environment servers using these completions to drive state changes.
* **Synchronization:** Return a dictionary containing prompt tokens, completion tokens, logprobs, and reward signals to the trainer.

> [!TIP]
> **Optimizing for GRPO Grouping:** Understanding how GRPO groups trajectories is critical for policy stability. Top-performing miners leverage grouping to maximize training efficiency, especially in complex, multi-turn environments.

> [!IMPORTANT]
> **Supported Environments:** Currently, `alfworld` is the sole supported environment. Support for additional environments will be phased in shortly.


### 3. Configuration

Declare your Rollout Function within your **Axolotl config** or to your trainer directly if you are not using Axolotl, mirroring the syntax used for standard GRPO Reward Functions. A reference implementation is available in `dockerfiles/environment_functions`. You must also pass a reward function, usually the actual reward calculation is done within your rollout function and then passed to the reward function as kwargs.

**Example Reward Function:**

```python
def alfworld_rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)
```

Note: this example expects the rollout function to include in its dictionary a list of the rewards for each episode under the key `env_rewards`.

> [!IMPORTANT]
> **Custom Miner Images:** If you are not using the base miner implementation we provide ensure that you have the correct versions of TRL and VLLM installed that support the experimental `generate_rollout_completions` feature.

---

## Testing & Validation

To ensure local parity with Gradients' evaluation systems, two utility scripts are provided:

* **Local Training:** Use `examples/run_environment_task.sh` to trigger a training run in a local environment.
* **Standardized Evaluation:** Use `scripts/manual_environment_eval.py` to evaluate your trained model using the exact parameters and episode count used by Gradients.

---

## Technical References

| Resource | Description |
| --- | --- |
| **[Affinetes](https://github.com/AffineFoundation/affinetes)** | The standard protocol for Gradients environment server communication. |
| **[OpenEnv Rollout Functions](https://huggingface.co/docs/trl/main/en/openenv)** | TRL's official guide on implementing custom rollout logic. |

---