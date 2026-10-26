# Text evaluator image (instruct / chat / DPO / GRPO evals). No axolotl: the eval owns its data path, so this
# image reproduces only the runtime the axolotl base (axolotlai/axolotl:main-20260701-py3.11-cu128-2.9.1) gave it -
# same CUDA 12.8 base, Python 3.11, torch 2.9.1 cu128 wheel and pinned HF stack - so loss numerics do not move.
# devel (not runtime) because causal-conv1d compiles with nvcc below, as it did on the axolotl base.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG PYTHON_VERSION=3.11
ARG MAMBA_SSM_VERSION=2.3.2.post1
ARG CAUSAL_CONV1D_VERSION=1.6.2.post1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs curl ca-certificates build-essential ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --skip-repo

COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /uvx /usr/local/bin/

# The torch wheels bundle their own CUDA/cuDNN libraries (pinned in the pins file), so numerics do not depend
# on the base image's CUDA toolkit. UV_TORCH_BACKEND makes uv take torch from the cu128 index, as the axolotl base did.
ENV UV_TORCH_BACKEND=cu128 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    VIRTUAL_ENV=/workspace/venv \
    PATH="/workspace/venv/bin:${PATH}"

RUN uv python install ${PYTHON_VERSION} \
    && uv venv --no-project --relocatable --python ${PYTHON_VERSION} /workspace/venv

COPY ops/docker/requirements/text-evaluator-pins.txt /tmp/text-evaluator-pins.txt
RUN uv pip sync --no-cache /tmp/text-evaluator-pins.txt

# mamba_ssm kernels (hybrid-SSM models such as Falcon-H1 take the same CUDA path they took on the axolotl base,
# which installed it the same way). Its runtime deps are already pinned; the constraint keeps it from moving them.
RUN MAMBA_SKIP_CUDA_BUILD=TRUE uv pip install --no-cache --no-build-isolation \
    -c /tmp/text-evaluator-pins.txt mamba_ssm==${MAMBA_SSM_VERSION}

WORKDIR /app

COPY ops/docker/requirements/validator.txt requirements.txt
RUN uv pip install --no-cache -c /tmp/text-evaluator-pins.txt -r requirements.txt

RUN uv pip uninstall textstat pyphen || true; \
    uv pip install --no-cache -c /tmp/text-evaluator-pins.txt --force-reinstall textstat==0.7.8

# causal-conv1d (quasar hybrid-arch load-time dep) must compile against the pinned torch (2.9.1), not
# an isolated newer torch, or the .so mismatches libc10_cuda's ABI at import. --no-build-isolation
# builds against the installed torch; the arch list covers Basilica A100 (8.0) + H100 (9.0).
RUN TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX" uv pip install --no-cache --no-build-isolation \
    -c /tmp/text-evaluator-pins.txt causal-conv1d==${CAUSAL_CONV1D_VERSION}


COPY . .

# Build-time smoke: no axolotl anywhere, the numerics stack is exactly the pinned one, and every text
# evaluator entrypoint imports on CPU (so a dependency the eval still needs but the image lacks fails the build).
RUN python -c "import importlib.util; assert importlib.util.find_spec('axolotl') is None, 'axolotl must not be installed'" \
    && python -c "\
from importlib.metadata import version; \
expected = {'transformers': '5.12.1', 'trl': '1.5.1', 'accelerate': '1.13.0'}; \
actual = {name: version(name) for name in expected}; \
assert actual == expected, actual; \
assert version('torch').split('+')[0] == '2.9.1', version('torch'); \
assert version('peft').startswith('0.19.'), version('peft'); \
assert version('datasets').startswith('4.8.'), version('datasets'); \
import importlib.util; \
missing = [m for m in ('mamba_ssm', 'causal_conv1d', 'fla') if importlib.util.find_spec(m) is None]; \
assert not missing, missing" \
    && python -c "import validator.evaluation.evaluators.instruct_text, validator.evaluation.evaluators.dpo, validator.evaluation.evaluators.grpo, validator.evaluation.evaluators.single_kl_divergence"

ENV JOB_ID=""
ENV DATASET=""
ENV MODELS=""
ENV ORIGINAL_MODEL=""
ENV DATASET_TYPE=""
ENV FILE_FORMAT=""
ENV TRANSFORMERS_ALLOW_TORCH_LOAD="true"
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV PYTORCH_ALLOC_CONF="expandable_segments:True"

RUN mkdir /aplp

# The CUDA base image's entrypoint prints a banner before exec; the axolotl base replaced it with a plain
# exec wrapper. Every caller (docker SDK, Basilica, Runpod) passes its own command.
ENTRYPOINT []
