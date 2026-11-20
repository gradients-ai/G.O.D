# Core constants

Constants that more than one runtime imports. Add a constant to the narrowest file that owns its domain.

## Contents

| Path | Purpose |
| --- | --- |
| `credentials.py` | Credentials and accounts read from the environment (Hugging Face, W&B, S3 bucket, YaRN upload account), and the tournament Hugging Face organisation. |
| `datasets.py` | Image-text pair limits and the default DPO and GRPO dataset fields. |
| `docker.py` | Evaluator image names, the env variable names passed to containers and the accepted text Dockerfile paths (`TEXT_DOCKERFILE_PATHS`). |
| `environments.py` | Evaluation type and training start point enums, and environment task configuration. |
| `network.py` | Subnet, chain and network constants. |
| `paths.py` | Adapter config and chat template file names, and the Hugging Face hub cache directory. |
| `quant.py` | Compression task constants: decoder limits, the decoded tensor dtype, the model size band and the output contract names. Imported by name; the package does not re-export it. |
| `training.py` | Valid YaRN scaling factors. |

## See also

- [Compression miner guide](../../docs/miner_compression.md): the limits in `quant.py` as a miner sees them.
