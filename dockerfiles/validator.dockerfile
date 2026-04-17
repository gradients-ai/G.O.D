FROM winglian/axolotl:main-20251113

WORKDIR /app

COPY validator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip uninstall -y textstat pyphen && \
    pip install --no-cache-dir --force-reinstall textstat==0.7.8


COPY . .

ENV JOB_ID=""
ENV DATASET=""
ENV MODELS=""
ENV ORIGINAL_MODEL=""
ENV DATASET_TYPE=""
ENV FILE_FORMAT=""
ENV TRANSFORMERS_ALLOW_TORCH_LOAD="true"

RUN mkdir /aplp
