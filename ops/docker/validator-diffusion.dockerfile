FROM python:3.11-slim

# Same Comfy revision and numerical libraries as the validated family experiments.
ARG COMFYUI_COMMIT=694815f498295080a0e15a1502edc9dba841b110
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN git init validator/evaluation/ComfyUI && \
    cd validator/evaluation/ComfyUI && \
    git remote add origin https://github.com/comfyanonymous/ComfyUI.git && \
    git fetch --depth 1 origin "${COMFYUI_COMMIT}" && git checkout FETCH_HEAD
COPY ops/docker/requirements/image-evaluator.txt /tmp/image-evaluator.txt
RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir -r validator/evaluation/ComfyUI/requirements.txt -r /tmp/image-evaluator.txt

COPY core core
COPY validator validator
COPY ops/docker/scripts/image_eval_entrypoint.sh /app/start.sh
RUN chmod +x /app/start.sh && mkdir -p /aplp && \
    python -c "import torch, diffusers, transformers; assert torch.__version__.startswith('2.9.1')"
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
CMD ["/app/start.sh"]
