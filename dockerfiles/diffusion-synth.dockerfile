FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        python3-venv \
        python3-setuptools \
        python3-pip \
        git wget \
        ffmpeg libsm6 libxext6 \
        build-essential libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*


COPY . .

RUN python3 -m venv /envs/comfyui && \
    python3 -m venv /envs/generate

WORKDIR /app/validator/tasks/image_synth/

RUN . /envs/comfyui/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

RUN . /envs/generate/bin/activate && \
    pip install -r requirements_llava.txt && \
    python -c "from transformers import CLIPVisionModel; CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')"

ENV SAVE_DIR=""

RUN pip install --no-cache-dir --upgrade pip setuptools wheel huggingface_hub

RUN python setup.py

WORKDIR /app

RUN echo '#!/bin/bash\n\
source /envs/comfyui/bin/activate\n\
python /app/validator/tasks/image_synth/ComfyUI/main.py &\n\
deactivate\n\
source /envs/generate/bin/activate\n\
if [ -n "$PROMPTS" ]; then\n\
    python -m validator.tasks.image_synth.generate_style\n\
else\n\
    python -m validator.tasks.image_synth.generate_person\n\
fi' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]