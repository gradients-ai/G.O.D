#!/bin/sh
set -eu
exec python -m validator.evaluation.evaluators.diffusion
