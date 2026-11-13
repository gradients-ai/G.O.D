# A build with a tagged base image and unhashed requirements: tagged runpod base, apt wget, inline pins, unhashed
# requirements, release-asset wheels, a fiber tag, and one hashed requirements file run fail-soft.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    vim zip tmux wget nano htop \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace /cache /workspace/scripts
WORKDIR /workspace/scripts

COPY scripts/training_requirements.txt /workspace/scripts/training_requirements.txt
COPY scripts/kernel_requirements.txt /workspace/scripts/kernel_requirements.txt

RUN pip install uv && \
    pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja && \
    uv pip install -r /workspace/scripts/training_requirements.txt --system && \
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl && \
    pip install "fiber @ git+https://github.com/rayonlabs/fiber.git@2.4.0"

RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0

RUN { pip install --no-deps --require-hashes -r /workspace/scripts/kernel_requirements.txt \
      && echo "kernels: present"; } \
    || { echo "kernels: ABSENT (fail-soft)"; }

COPY scripts /workspace/scripts
RUN chmod +x /workspace/scripts/entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
