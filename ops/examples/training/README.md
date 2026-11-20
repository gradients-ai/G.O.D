# Training examples

Shell examples that run one training task in Docker: download, train, upload. Each script builds the downloader and uploader images from `ops/docker/`, and the trainer Dockerfile named below. Run it from the root of a directory that holds `ops/docker/` with the downloader and uploader Dockerfiles and the trainer Dockerfile named below. The text trainer Dockerfile comes from your training repository. The image toolkit one ships here.

## Contents

| Path | Purpose |
| --- | --- |
| `run_instruct_task.sh` | Instruct text task. Builds `ops/docker/standalone-text-trainer.dockerfile`. |
| `run_dpo_task.sh` | DPO task. Builds `ops/docker/standalone-text-trainer.dockerfile`. |
| `run_grpo_task.sh` | GRPO task. Builds `ops/docker/standalone-text-trainer.dockerfile`. |
| `run_image_task.sh` | Image task. Builds `ops/docker/standalone-image-toolkit-trainer.dockerfile`. |

## See also

- [Text guide: local testing](../../../docs/miner_text.md#local-testing): the text scripts as a miner uses them.
- [Image guide: local testing](../../../docs/miner_image.md#local-testing): the image script as a miner uses it.
