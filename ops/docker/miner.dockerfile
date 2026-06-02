FROM axolotlai/axolotl:main-py3.11-cu124-2.5.1

RUN pip install mlflow
RUN pip install textstat==0.7.7
RUN pip install langcheck
RUN pip install detoxify

WORKDIR /workspace/axolotl
RUN mkdir -p /workspace/axolotl/configs \
    /workspace/axolotl/outputs \
    /workspace/input_data

ENV CONFIG_DIR="/workspace/axolotl/configs"
ENV OUTPUT_DIR="/workspace/axolotl/outputs"
ENV AWS_ENDPOINT_URL="https://5a301a635a9d0ac3cb7fcc3bf373c3c3.r2.cloudflarestorage.com"
ENV AWS_ACCESS_KEY_ID=d49fdd0cc9750a097b58ba35b2d9fbed
ENV AWS_DEFAULT_REGION="us-east-1"
ENV AWS_SECRET_ACCESS_KEY=02e398474b783af6ded4c4638b5388ceb8079c83bb2f8233d5bcef0e60addba6

RUN mkdir -p /root/.aws && \
    echo "[default]\naws_access_key_id=dummy_access_key\naws_secret_access_key=dummy_secret_key" > /root/.aws/credentials && \
    echo "[default]\nregion=us-east-1" > /root/.aws/config
