FROM diagonalge/ai-toolkit:latest

ARG AI_TOOLKIT_COMMIT=e9c72b762748e4d31837fbeb3b87d01a5b143321
ARG MUSUBI_COMMIT=30c658c4f4b0bf05038b3346eff9670259b10fc7

WORKDIR /app/ai-toolkit
RUN git fetch origin "${AI_TOOLKIT_COMMIT}" && \
    git checkout "${AI_TOOLKIT_COMMIT}" && \
    pip install --no-cache-dir -r requirements.txt
COPY validator/evaluation/ai_toolkit_extensions/god_objective_evaluator \
    /app/ai-toolkit/extensions/god_objective_evaluator

WORKDIR /app
RUN git init musubi-tuner && \
    cd musubi-tuner && \
    git remote add origin https://github.com/kohya-ss/musubi-tuner.git && \
    git fetch --depth 1 origin "${MUSUBI_COMMIT}" && \
    git checkout FETCH_HEAD && \
    python3 -m venv /opt/musubi && \
    /opt/musubi/bin/pip install --upgrade pip && \
    /opt/musubi/bin/pip install --no-cache-dir \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && \
    /opt/musubi/bin/pip install --no-cache-dir -e . lycoris-lora

COPY core core
COPY validator validator
COPY ops/docker/scripts/image_eval_entrypoint.sh /app/start.sh
RUN chmod +x /app/start.sh && mkdir -p /aplp

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    AI_TOOLKIT_ROOT=/app/ai-toolkit \
    MUSUBI_ROOT=/app/musubi-tuner \
    MUSUBI_PYTHON=/opt/musubi/bin/python \
    MUSUBI_ACCELERATE=/opt/musubi/bin/accelerate

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/app/start.sh"]
