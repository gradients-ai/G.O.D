FROM python:3.10-slim

ARG COMFYUI_COMMIT=091b70edda0c062fc9338a1d7e8e2f94f4c0ad0b
ARG COMFYUI_TOOLING_NODES_COMMIT=5d3194f4d4158ab31df7a060e1e4c56fa03f320c

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git wget && rm -rf /var/lib/apt/lists/*

RUN mkdir /aplp

WORKDIR /app/validator/evaluation
RUN git init ComfyUI && \
    cd ComfyUI && \
    git remote add origin https://github.com/comfyanonymous/ComfyUI.git && \
    git fetch --depth 1 origin "${COMFYUI_COMMIT}" && \
    git checkout FETCH_HEAD

RUN pip install --no-cache-dir -r ComfyUI/requirements.txt
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --extra-index-url https://download.pytorch.org/whl/cu128
RUN cd ComfyUI/custom_nodes && \
    git init comfyui-tooling-nodes && \
    cd comfyui-tooling-nodes && \
    git remote add origin https://github.com/Acly/comfyui-tooling-nodes && \
    git fetch --depth 1 origin "${COMFYUI_TOOLING_NODES_COMMIT}" && \
    git checkout FETCH_HEAD && \
    cd .. && \
    if [ -f comfyui-tooling-nodes/requirements.txt ]; then \
        pip install --no-cache-dir -r comfyui-tooling-nodes/requirements.txt; \
    fi
   

RUN pip install --no-cache-dir docker diffusers huggingface_hub

ENV TEST_DATASET_PATH=""
ENV TRAINED_LORA_MODEL_REPOS=""
ENV BASE_MODEL_REPO=""
ENV BASE_MODEL_FILENAME=""
ENV LORA_MODEL_FILENAMES=""

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY ops/docker/requirements/validator.txt validator/requirements.txt
RUN pip install --no-cache-dir -r validator/requirements.txt

COPY . .

RUN echo '#!/bin/bash\n\
python /app/validator/evaluation/ComfyUI/main.py &\n\
python -m validator.evaluation.evaluators.diffusion' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
