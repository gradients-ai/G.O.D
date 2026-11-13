# Docker Requirements

Requirements files used only inside Docker builds.

## Contents

- `quant-evaluator.txt`: quant evaluator image requirements (exact vLLM, torch, transformers, compressed-tensors and
  humming-kernels pins). compressed-tensors stays at vLLM 0.31.0's own pin (0.17.0); the image build asserts that
  and that vLLM reads the dense packed layout the structural pre-check expects.
- `validator.txt`: validator Docker image Python requirements.
