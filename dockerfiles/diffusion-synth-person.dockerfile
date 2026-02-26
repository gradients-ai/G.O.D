FROM python:3.10-slim AS deps

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        python3-venv \
        python3-setuptools \
        python3-pip \
        git \
        build-essential libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /envs/comfyui && \
    python3 -m venv /envs/generate

WORKDIR /app/validator/tasks/image_synth
COPY validator/tasks/image_synth/requirements.txt ./requirements.txt
COPY validator/tasks/image_synth/requirements_llava.txt ./requirements_llava.txt

RUN . /envs/comfyui/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

RUN . /envs/generate/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements_llava.txt && \
    pip install websocket-client names runpod minio

RUN . /envs/generate/bin/activate && \
    python -c "from transformers import CLIPVisionModel; CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')"


FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=deps /envs/comfyui /envs/comfyui
COPY --from=deps /envs/generate /envs/generate

# Run heavyweight model setup in a cache-friendly layer.
WORKDIR /app/validator/tasks/image_synth
COPY validator/tasks/image_synth/setup.py ./setup.py
RUN /envs/comfyui/bin/python -c "import setup as s; s.setup_person_models()"

# Copy full source last so frequent code edits don't invalidate setup/model layers.
WORKDIR /app
COPY . .

ENV SKIP_MODEL_SETUP=1
CMD ["/envs/generate/bin/python", "-m", "validator.tasks.image_synth.generate_person"]
