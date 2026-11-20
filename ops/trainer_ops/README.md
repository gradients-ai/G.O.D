# Trainer ops

Files that run the trainer as a system service on a trainer host.

## Contents

| Path | Purpose |
| --- | --- |
| `install_dcgm.sh` | Installs and starts NVIDIA DCGM. `task trainer`, `task bootstrap` and `trainer.service` run it; an already running hostengine is left alone. |
| `install_trainer_service.sh` | Installs, enables and starts `trainer.service`. Run as root. |
| `trainer.service` | systemd unit for the trainer. |

## See also

- [Running a trainer](../../docs/developer.md#running-a-trainer): trainer setup and configuration.
