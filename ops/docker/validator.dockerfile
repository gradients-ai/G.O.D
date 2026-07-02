FROM axolotlai/axolotl:main-20260701-py3.11-cu128-2.9.1

WORKDIR /app

COPY ops/docker/requirements/validator.txt requirements.txt
# axolotl base ships a uv-managed venv at /workspace/axolotl-venv with no `pip` on PATH.
RUN uv pip install --python /workspace/axolotl-venv/bin/python --no-cache -r requirements.txt

RUN uv pip uninstall --python /workspace/axolotl-venv/bin/python textstat pyphen || true; \
    uv pip install --python /workspace/axolotl-venv/bin/python --no-cache --force-reinstall textstat==0.7.8

# causal-conv1d (quasar hybrid-arch load-time dep) must compile against the BASE torch (2.9.1), not
# an isolated newer torch, or the .so mismatches libc10_cuda's ABI at import. --no-build-isolation
# builds against the installed torch; the arch list covers Basilica A100 (8.0) + H100 (9.0).
RUN TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX" uv pip install --python /workspace/axolotl-venv/bin/python \
    --no-cache --no-build-isolation causal-conv1d==1.6.2.post1


COPY . .

ENV JOB_ID=""
ENV DATASET=""
ENV MODELS=""
ENV ORIGINAL_MODEL=""
ENV DATASET_TYPE=""
ENV FILE_FORMAT=""
ENV TRANSFORMERS_ALLOW_TORCH_LOAD="true"

RUN mkdir /aplp
