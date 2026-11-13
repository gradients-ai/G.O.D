#!/bin/sh
# Runtime, not build: never linted.
pip install whatever-at-runtime
exec python train.py "$@"
