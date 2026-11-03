FROM diagonalge/ai-toolkit:latest

WORKDIR /app/ai-toolkit

ARG AI_TOOLKIT_COMMIT=e9c72b762748e4d31837fbeb3b87d01a5b143321
ARG MUSUBI_COMMIT=30c658c4f4b0bf05038b3346eff9670259b10fc7

RUN git fetch origin "${AI_TOOLKIT_COMMIT}" && \
    git checkout "${AI_TOOLKIT_COMMIT}" && \
    pip install --no-cache-dir -r requirements.txt

RUN git init /app/musubi-tuner && \
    cd /app/musubi-tuner && \
    git remote add origin https://github.com/kohya-ss/musubi-tuner.git && \
    git fetch --depth 1 origin "${MUSUBI_COMMIT}" && \
    git checkout FETCH_HEAD && \
    python3 -m venv /opt/musubi && \
    /opt/musubi/bin/pip install --upgrade pip && \
    /opt/musubi/bin/pip install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 && \
    /opt/musubi/bin/pip install --no-cache-dir -e . lycoris-lora

COPY core/training_templates /workspace/core/training_templates
COPY ops/docker/scripts/image_toolkit_entrypoint.py /usr/local/bin/image_toolkit_entrypoint.py

RUN chmod +x /usr/local/bin/image_toolkit_entrypoint.py

ENV MUSUBI_PYTHON=/opt/musubi/bin/python \
    MUSUBI_ACCELERATE=/opt/musubi/bin/accelerate

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python3", "/usr/local/bin/image_toolkit_entrypoint.py"]
