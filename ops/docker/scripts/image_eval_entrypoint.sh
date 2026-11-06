#!/bin/sh
set -eu

case "${MODEL_TYPE:-}" in
  flux2-klein-4b|qwen-image-edit-2511|hidream-o1|moss-image-base)
    exec python -m validator.evaluation.evaluators.diffusion_objective
    ;;
  *)
    exec python -m validator.evaluation.evaluators.diffusion
    ;;
esac
