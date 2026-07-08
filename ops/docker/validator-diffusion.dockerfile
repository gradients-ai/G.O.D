FROM python:3.10-slim

ARG COMFYUI_COMMIT=091b70edda0c062fc9338a1d7e8e2f94f4c0ad0b
ARG COMFYUI_TOOLING_NODES_COMMIT=5d3194f4d4158ab31df7a060e1e4c56fa03f320c

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git wget && rm -rf /var/lib/apt/lists/*

RUN mkdir /aplp

WORKDIR /app/validator/evaluation
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI && \
    cd ComfyUI && \
    git checkout --detach "${COMFYUI_COMMIT}"

RUN pip install --no-cache-dir -r ComfyUI/requirements.txt
RUN cd ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/Acly/comfyui-tooling-nodes && \
    cd comfyui-tooling-nodes && \
    git checkout --detach "${COMFYUI_TOOLING_NODES_COMMIT}" && \
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

WORKDIR /app

COPY ops/docker/requirements/validator.txt validator/requirements.txt
RUN pip install --no-cache-dir -r validator/requirements.txt

COPY . .

RUN echo '#!/bin/bash\n\
python /app/validator/evaluation/ComfyUI/main.py &\n\
python -m validator.evaluation.evaluators.diffusion' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
