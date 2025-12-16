FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git wget && rm -rf /var/lib/apt/lists/*

RUN mkdir /aplp

WORKDIR /app/validator/evaluation
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI && \
    cd ComfyUI && \
    git fetch --depth 1 origin 9304e47351be8d178a093b30bcaf5d72c3a2baf5 && \
    git checkout 9304e47351be8d178a093b30bcaf5d72c3a2baf5 && \
    cd ..

RUN pip install -r ComfyUI/requirements.txt
RUN cd ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/Acly/comfyui-tooling-nodes && \
    cd comfyui-tooling-nodes && \
    git fetch --depth 1 origin e10daee9edea458fc709f60e725970a25567fca4 && \
    git checkout e10daee9edea458fc709f60e725970a25567fca4 && \
    cd ..

# Download Flux text encoders and VAE
RUN wget -O /app/validator/evaluation/ComfyUI/models/text_encoders/clip_l.safetensors \
    https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
    wget -O /app/validator/evaluation/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors \
    https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors && \
    wget -O /app/validator/evaluation/ComfyUI/models/vae/ae.safetensors \
    https://huggingface.co/Albert-zp/flux-vaesft/resolve/main/fluxVaeSft_aeSft.sft

# Download Z-Image text encoder and unet
RUN wget -O /app/validator/evaluation/ComfyUI/models/text_encoders/qwen_3_4b.safetensors \
    https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors && \
    wget -O /app/validator/evaluation/ComfyUI/models/unet/z_image_turbo_bf16.safetensors \
    https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors

# Download Qwen-Image text encoder, VAE, and unet
RUN wget -O /app/validator/evaluation/ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors && \
    wget -O /app/validator/evaluation/ComfyUI/models/vae/qwen_image_vae.safetensors \
    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors && \
    wget -O /app/validator/evaluation/ComfyUI/models/unet/qwen_image_fp8_e4m3fn.safetensors \
    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors

RUN pip install docker diffusers

ENV TEST_DATASET_PATH=""
ENV TRAINED_LORA_MODEL_REPOS=""
ENV BASE_MODEL_REPO=""
ENV BASE_MODEL_FILENAME=""
ENV LORA_MODEL_FILENAMES=""

WORKDIR /app

COPY validator/requirements.txt validator/requirements.txt
RUN pip install -r validator/requirements.txt

COPY . .

RUN echo '#!/bin/bash\n\
python /app/validator/evaluation/ComfyUI/main.py &\n\
python -m validator.evaluation.eval_diffusion' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
