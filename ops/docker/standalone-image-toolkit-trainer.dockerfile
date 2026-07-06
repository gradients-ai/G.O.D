FROM diagonalge/ai-toolkit:latest

WORKDIR /app/ai-toolkit

RUN git fetch origin 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    git checkout 99be3d96a2468d3a5228a4eb05ba67e63c586b4e && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

COPY core/training_templates /workspace/core/training_templates
COPY ops/docker/scripts/image_toolkit_entrypoint.py /usr/local/bin/image_toolkit_entrypoint.py

RUN chmod +x /usr/local/bin/image_toolkit_entrypoint.py

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python3", "/usr/local/bin/image_toolkit_entrypoint.py"]
