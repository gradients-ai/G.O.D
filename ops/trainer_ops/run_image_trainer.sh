#!/bin/bash
set -e
echo "[run_image_trainer.sh] Starting trainer with args: $@"
python3 /workspace/trainer/entrypoints/image_trainer.py "$@"