# Image for validator/evaluation/evaluators/quant.py.
# Scores quant submissions from vLLM prompt logprobs on H100.
# Build manually:
#   docker build -f ops/docker/quant-evaluator.dockerfile -t gradientsio/quant-evaluator:basilica .

# CUDA 13.0 user space to match torch 2.13.0+cu130 and humming-kernels[cu13]. The base image is
# enough: Humming compiles its kernels with NVRTC and the headers from the pip nvidia/cu13 wheels, and
# the only nvcc JIT on the path (FlashInfer's sampler) is turned off below. Host drivers older than
# 580 need the image's /usr/local/cuda/compat libraries (forward compatibility).
FROM nvidia/cuda:13.0.2-base-ubuntu24.04

WORKDIR /app

# Ubuntu 24.04 ships python3.12 as python3; the venv keeps `python`/`pip` on 3.12 and clear of the
# system packages. gcc, libc6-dev and python3-dev are for Triton, not nvcc: Triton builds its CUDA
# launcher stub (cuda_utils.c, includes Python.h) with the host cc at the first kernel launch, and
# vLLM 0.31 launches Triton kernels inside LLM(...) (the profile run's sampler, Inductor). Without
# them every scoring run dies with "Failed to find C compiler" before the first prompt.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev git libnuma1 \
        gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ops/docker/requirements/quant-evaluator.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir --no-deps .

# VLLM_*: keep the engine in-process (quant_vllm reads the loaded layers through LLM.apply_model) and
# skip FlashInfer's sampler, which JIT-compiles with nvcc. quant_vllm also sets both in-process.
# HUMMING_*: Humming's JIT kernel cache and its temp/lock files. Nothing is pre-warmed at build
# time: the cache is keyed by SKU, base-model shapes and install paths, and needs a live GPU. All
# repos of a task are scored in one container, so only the first load pays the JIT.
# LD_LIBRARY_PATH: NVRTC loads libnvrtc-builtins.so.13.0 from the pip nvidia/cu13 wheel.
ENV VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    HUMMING_CACHE_DIR=/opt/humming-cache \
    HUMMING_TMP_DIR=/opt/humming-tmp \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}

# Catch dependency skew at build time, before a deployment spends GPU time discovering it.
RUN python -c "import vllm, humming.tune; \
assert vllm.__version__ == '0.31.0', vllm.__version__; \
assert hasattr(humming.tune, '_get_heuristics_config'), 'stream-K patch target missing'; \
import validator.evaluation.quant_vllm, validator.evaluation.evaluators.single_quant"
# compressed-tensors is exactly vLLM's pin, and vLLM allocates the dense packed layout the structural
# pre-check (quant_structure.packed_layout) accepts, for every launch format. CPU only.
RUN python ops/docker/scripts/check_quant_packed_layout.py
# Triton's launcher build needs a working cc and the interpreter's headers; no GPU is needed to check them.
RUN cc --version >/dev/null && python -c "import os, sysconfig; \
assert os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')), 'Python.h missing'"

ENV MODELS=""
ENV ORIGINAL_MODEL=""
ENV DATASET_URL=""
ENV QUANT_SPEC=""
ENV QUANT_FORMAT=""

RUN mkdir /aplp
