# Trainer Ops

Trainer host setup and service scripts.

## Contents

- `install_dcgm.sh`: install and start NVIDIA DCGM. `task trainer`, `task bootstrap`, and `trainer.service` run it; an already running hostengine is left alone.
- `install_trainer_service.sh`: install trainer service on a host.
- `run_image_trainer.sh`: run image trainer manually.
- `run_text_trainer.sh`: run text trainer manually.
- `trainer.service`: systemd service file for trainer host operation.
