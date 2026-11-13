# Trainer model prep tests

Tests for the trainer's model-prep container code.

## Contents

| Path | Covers |
| --- | --- |
| `test_comprehensive_stats.py` | Baseline statistics on a small model. Marked `gpu`. |
| `test_merged_repo_name.py` | Merged-repo naming and publish target. |
| `test_quant_device_probes_gpu.py` | Compression device probes on a tiny in-memory model. Marked `gpu`. |
| `test_quant_device_probes_host.py` | Host probes (CPU, identity) and the `DeviceProbeStats` round trip. |
| `test_roundtrip.py` | Augmentation config, baseline statistics and the model-prep response survive JSON round trips. |

Tests marked `gpu` run only with `--run-gpu`.
