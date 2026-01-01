# Environment Tasks & Rollout Functions

Gradients now supports **Environment Tasks**, leveraging new functionality within the TRL `GRPOTrainer`. This feature allows for custom rollout logic during training, enabling models to interact with external environments in real-time to receive dynamic rewards.

## Evaluation Protocol

Following training, Gradients evaluates the model by running **500 episodes** within the target environment. The final performance is determined by the **average score** across these episodes.

---

## Miner Requirements

During the training phase, miners are granted access to environment servers that host the specific task logic. To optimize throughput, one environment server is typically provided per GPU.

### 1. Connecting to Environment Servers

The server addresses are provided via the `ENVIRONMENT_SERVER_URLS` environment variable as a comma-separated string.

**Example Extraction:**

```python
import os

raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]

```

### 2. Implementing the Rollout Function

Miners must implement a custom **Rollout Function** for the environment specified in `dataset_type.environment_name`. The function is responsible for the following workflow:

* **Generation:** Produce model completions using `generate_rollout_completions`.
* **Interaction:** Use these completions to interface with the environment via the provided server URLs.
* **Data Return:** Return the prompt tokens, completion tokens, logprobs, and associated reward signals to the trainer.

> **Hint:** Pay close attention to how GRPO grouping works in order to ensure updates to the policy during training.

> **Note:** We are starting the rollout of these tasks with `alfworld` as the only supported environment. More will follow soon.

### 3. Configuration

The Rollout Function is defined within your **Axolotl configuration**, following a syntax similar to standard GRPO Reward Functions. A reference implementation used by the default miner can be found in `dockerfiles/environment_functions`.

---

## Technical References

| Resource | Description |
| --- | --- |
| **[Affinetes](https://github.com/AffineFoundation/affinetes)** | The standard protocol used by Gradients for running environment servers. |
| **[OpenEnv Rollout Functions](https://huggingface.co/docs/trl/main/en/openenv)** | TRL documentation regarding the implementation of custom rollout logic. |

---